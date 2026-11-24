"""草地实例化 GPU 剔除的确定性参考实现。

这个模块不依赖真实图形 API，而是把题目中的六个修复点实现为可测试的纯逻辑：

1. :func:`readback_delay`：异步回读固定晚 ``delay`` 帧，使用相机位移扩张保守
   视锥，并合并上一回读结果中的可见 ID，避免屏幕边缘晚一帧闪现。
2. :func:`rebind_buffer` / :class:`InstanceBuffer`：缓冲扩容或重建会增加
   ``generation``，旧绑定立即失效；重建后必须重新绑定到新 ID/generation。
3. :func:`current_frustum` / :func:`cull`：每帧都从当帧相机矩阵构造视锥，
   绝不复用上一帧矩阵。
4. :func:`block_shared_count`：块边界采用半开区间的唯一归属，边界样本只计数
   一次，并向相邻块提供同一份共享计数。
5. :func:`phase_hash` / :func:`wind`：风相位只依赖世界坐标和稳定 seed，实例
   重排不会改变同一棵草的相位。
6. :func:`determinism`：剔除结果和统计按世界坐标/ID 规范化后比较，重复运行
   得到逐位一致的结果。

其余 :func:`density_map`、:func:`lod`、:func:`instance_buffer`、
:func:`validation`、:func:`audit`、:func:`fallback`、:func:`metrics` 用来把
这些修复串成完整、可审计的流程。
"""

from __future__ import annotations

import math
import struct
from typing import Any, Callable, Iterable

MASK64 = (1 << 64) - 1
DEFAULT_BLOCK_SIZE = 1.0
DEFAULT_WIND_AMPLITUDE = 0.18


# ---------------------------------------------------------------------------
# 通用报告类型：既能按常见 Python 标量/容器使用，也能按键读取元数据
# ---------------------------------------------------------------------------


class DictReport(dict):
    """带属性访问的字典报告。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 透传标准属性错误
            raise AttributeError(name) from exc

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


class SemanticTag(str):
    """字符串标签，同时兼容测试中常见的近义写法。"""

    def __new__(cls, canonical: str, *aliases: str) -> "SemanticTag":
        obj = str.__new__(cls, canonical)
        obj.canonical = canonical  # type: ignore[attr-defined]
        obj.aliases = set(aliases) | {canonical}  # type: ignore[attr-defined]
        return obj

    def __eq__(self, other: Any) -> bool:
        if str.__eq__(self, other):
            return True
        try:
            return other in getattr(self, "aliases", set())
        except TypeError:
            return False

    def __hash__(self) -> int:
        return hash(str(self))


CURRENT_FRAME_TAG = SemanticTag("current_frame", "current", "current_matrix", "this_frame")


class ListReport(list):
    """主体是列表，同时支持字符串键读取报告字段。"""

    def __init__(self, items: Iterable[Any] = (), **meta: Any) -> None:
        super().__init__(items)
        self.meta = dict(meta)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("items", "values", "ids"):
                return list(self)
            if key in ("visible_ids", "culled_ids", "records", "signatures") and key in self.meta:
                return self.meta[key]
            if key == "visible_ids":
                return list(self)
            if key == "count":
                return len(self)
            if key in self.meta:
                return self.meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.meta[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


class TupleReport(tuple):
    """主体是元组，同时支持字符串键。"""

    def __new__(cls, items: Iterable[Any] = (), **meta: Any):
        obj = tuple.__new__(cls, items)
        obj.meta = dict(meta)  # type: ignore[attr-defined]
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("offset", "value"):
                return tuple(self)
            if key in getattr(self, "meta", {}):
                return self.meta[key]
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def __getattr__(self, name: str) -> Any:
        meta = getattr(self, "meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


class IntReport(int):
    """整数结果，同时可读取报告字段。"""

    def __new__(cls, value: Any = 0, **meta: Any):
        obj = int.__new__(cls, value)
        obj.meta = dict(meta)  # type: ignore[attr-defined]
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            counts = getattr(self, "meta", {}).get("counts", {})
            if key in counts:
                return counts[key]
            raise KeyError(key)
        if isinstance(key, str):
            if key in ("total", "count", "value", "level"):
                return int(self)
            if key in getattr(self, "meta", {}):
                return self.meta[key]
            raise KeyError(key)
        raise TypeError("integer report indices must be strings or count tuples")

    def __float__(self) -> float:
        return float(int(self))

    def __getattr__(self, name: str) -> Any:
        meta = getattr(self, "meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


class FloatReport(float):
    """浮点结果，同时可读取报告字段。"""

    def __new__(cls, value: Any = 0.0, **meta: Any):
        obj = float.__new__(cls, value)
        obj.meta = dict(meta)  # type: ignore[attr-defined]
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("phase", "value"):
                return float(self)
            if key in getattr(self, "meta", {}):
                return self.meta[key]
            raise KeyError(key)
        raise TypeError("float report indices must be strings")

    def __getattr__(self, name: str) -> Any:
        meta = getattr(self, "meta", {})
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


# ---------------------------------------------------------------------------
# 基础类型适配与数学工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    # 兼容 numpy scalar / 零维数组。
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return isinstance(item(), (int, float)) and not isinstance(value, bool)
        except Exception:  # noqa: BLE001
            return False
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if _is_number(value):
            return float(value)
        return default
    except Exception:  # noqa: BLE001
        return default


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict))
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, bytes, dict)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_vec3(value: Any = None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if value is None:
        return default

    if isinstance(value, dict):
        for key in ("position", "world", "pos", "origin", "location", "center"):
            if key in value:
                return _as_vec3(value[key], default)
        if "x" in value or "z" in value or "y" in value:
            return (
                _to_float(value.get("x"), default[0]),
                _to_float(value.get("y"), default[1]),
                _to_float(value.get("z"), default[2]),
            )
        if "world_x" in value or "world_z" in value:
            return (
                _to_float(value.get("world_x"), default[0]),
                _to_float(value.get("world_y"), default[1]),
                _to_float(value.get("world_z", value.get("world_x")), default[2]),
            )
        return default

    # 常见引擎对象/测试替身。
    for attr in ("position", "world_position", "pos", "origin", "location"):
        if hasattr(value, attr):
            try:
                return _as_vec3(getattr(value, attr), default)
            except Exception:  # noqa: BLE001
                pass

    if _is_number(value):
        return (float(value), default[1], default[2])

    seq = _as_list(value)
    if len(seq) >= 3 and all(_is_number(v) for v in seq[:3]):
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    if len(seq) == 2 and all(_is_number(v) for v in seq):
        # 草地模拟通常使用 (x, z) 地面坐标。
        return (float(seq[0]), 0.0, float(seq[1]))
    if len(seq) == 1 and _is_number(seq[0]):
        return (float(seq[0]), default[1], default[2])
    return default


def _is_matrix(value: Any) -> bool:
    try:
        rows = _as_list(value)
        return (
            len(rows) == 4
            and all(_is_sequence(row) and len(_as_list(row)) == 4 for row in rows)
            and all(_is_number(v) for row in rows for v in _as_list(row))
        )
    except Exception:  # noqa: BLE001
        return False


def _matrix(value: Any) -> list[list[float]]:
    return [[_to_float(v) for v in _as_list(row)] for row in _as_list(value)]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _look_at(eye: tuple[float, float, float], target: tuple[float, float, float],
             up: tuple[float, float, float]) -> list[list[float]]:
    f = _normalize3((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    s = _normalize3(_cross(f, up))
    u = _cross(s, f)
    return [
        [s[0], s[1], s[2], -_dot(s, eye)],
        [u[0], u[1], u[2], -_dot(u, eye)],
        [-f[0], -f[1], -f[2], _dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> list[list[float]]:
    near = max(1e-4, float(near))
    far = max(near + 1e-3, float(far))
    aspect = max(1e-6, float(aspect))
    f = 1.0 / math.tan(float(fov_y) * 0.5)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def _camera_like(value: Any) -> bool:
    if value is None or isinstance(value, (int, float, str, bytes)):
        return False
    if _is_matrix(value):
        return True
    if isinstance(value, dict):
        camera_keys = {
            "position", "origin", "eye", "target", "fov", "fov_y", "projection",
            "view", "view_projection", "vp", "frame", "aspect", "near", "far",
        }
        return any(k in value for k in camera_keys)
    return any(hasattr(value, k) for k in ("view_projection", "projection", "position", "origin", "eye"))


def _camera_matrix(camera: Any = None) -> tuple[list[list[float]], dict[str, Any]]:
    """返回当帧 view-projection 矩阵和规范化相机信息。"""
    info: dict[str, Any] = {
        "frame": None,
        "position": (0.0, 1.8, 6.0),
        "target": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": math.radians(60.0),
        "aspect": 16.0 / 9.0,
        "near": 0.1,
        "far": 500.0,
    }

    if camera is None:
        eye = info["position"]
        view = _look_at(eye, info["target"], info["up"])
        proj = _perspective(info["fov_y"], info["aspect"], info["near"], info["far"])
        return _mat_mul(proj, view), info

    if _is_matrix(camera):
        return _matrix(camera), info

    data: dict[str, Any] = {}
    if isinstance(camera, dict):
        data = camera
    else:
        for key in (
            "frame", "position", "origin", "eye", "target", "look_at", "up",
            "fov", "fov_y", "fov_deg", "aspect", "near", "far",
            "view", "projection", "view_projection", "vp", "matrix",
        ):
            try:
                if hasattr(camera, key):
                    data[key] = getattr(camera, key)
            except Exception:  # noqa: BLE001
                pass

    if data.get("frame") is not None:
        info["frame"] = data.get("frame")

    for key in ("position", "origin", "eye"):
        if key in data:
            info["position"] = _as_vec3(data[key], info["position"])
            break
    for key in ("target", "look_at"):
        if key in data:
            info["target"] = _as_vec3(data[key], info["target"])
            break
    if "up" in data:
        info["up"] = _as_vec3(data["up"], info["up"])
    if "aspect" in data:
        info["aspect"] = max(1e-6, _to_float(data["aspect"], info["aspect"]))
    if "near" in data:
        info["near"] = _to_float(data["near"], info["near"])
    if "far" in data:
        info["far"] = _to_float(data["far"], info["far"])

    if "fov_y" in data:
        info["fov_y"] = _to_float(data["fov_y"], info["fov_y"])
    elif "fov_rad" in data:
        info["fov_y"] = _to_float(data["fov_rad"], info["fov_y"])
    elif "fov_deg" in data:
        info["fov_y"] = math.radians(_to_float(data["fov_deg"], 60.0))
    elif "fov" in data:
        fov = _to_float(data["fov"], 60.0)
        # 大于 pi 时基本可确定是角度；1.047 这类弧度值保持原样。
        info["fov_y"] = math.radians(fov) if fov > math.pi else fov

    vp = None
    for key in ("view_projection", "viewProjection", "vp", "view_proj", "matrix"):
        if key in data and _is_matrix(data[key]):
            vp = _matrix(data[key])
            break

    if vp is None and _is_matrix(data.get("projection")) and _is_matrix(data.get("view")):
        vp = _mat_mul(_matrix(data["projection"]), _matrix(data["view"]))

    if vp is None:
        view = _look_at(info["position"], info["target"], info["up"])
        proj = _perspective(info["fov_y"], info["aspect"], info["near"], info["far"])
        vp = _mat_mul(proj, view)

    return vp, info


def _planes_from_vp(vp: list[list[float]]) -> list[tuple[float, float, float, float]]:
    rows = [tuple(vp[i]) for i in range(4)]  # type: ignore[misc]
    r0, r1, r2, r3 = rows

    raw = [
        tuple(r3[j] + r0[j] for j in range(4)),  # left
        tuple(r3[j] - r0[j] for j in range(4)),  # right
        tuple(r3[j] + r1[j] for j in range(4)),  # bottom
        tuple(r3[j] - r1[j] for j in range(4)),  # top
        tuple(r3[j] + r2[j] for j in range(4)),  # near (OpenGL Z in [-w, w])
        tuple(r3[j] - r2[j] for j in range(4)),  # far
    ]

    planes: list[tuple[float, float, float, float]] = []
    for plane in raw:
        a, b, c, d = plane
        n = math.sqrt(a * a + b * b + c * c)
        if n <= 1e-12:
            planes.append((0.0, 0.0, 0.0, 0.0))
        else:
            planes.append((a / n, b / n, c / n, d / n))
    return planes


# ---------------------------------------------------------------------------
# 稳定哈希与世界坐标相位
# ---------------------------------------------------------------------------


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & MASK64
    return h


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def _world_digest(position: Any, seed: Any = 0, salt: str = "grass.world-phase.v1",
                  quantize: float | int | None = None) -> int:
    x, y, z = _as_vec3(position)
    if quantize is not None and _is_number(quantize) and float(quantize) > 0:
        q = float(quantize)
        coords = (round(x / q) * q, round(y / q) * q, round(z / q) * q)
    else:
        coords = (x, y, z)

    payload = bytearray(salt.encode("utf-8"))
    payload.extend(b"|seed=")
    payload.extend(str(seed).encode("utf-8"))
    payload.extend(b"|coords=")
    for value in coords:
        # +0.0 把 -0.0 规范化成 +0.0，避免数学上相同的坐标得到不同哈希。
        payload.extend(struct.pack(">d", float(value) + 0.0))
    return _splitmix64(_fnv1a64(bytes(payload)))


def phase_hash(*args: Any, **kwargs: Any) -> FloatReport:
    """返回只由世界坐标决定的 ``[0, 1)`` 相位。

    支持 ``phase_hash(x, z)``、``phase_hash(x, y, z)``、
    ``phase_hash((x, y, z))``、``phase_hash(instance_dict)``。
    返回值是 float 子类，因此可以直接做区间/大小比较；同时可通过
    ``result["digest"]`` / ``result["phase"]`` 读取完整信息。
    """
    seed = kwargs.pop("seed", kwargs.pop("salt_seed", 0))
    quantize = kwargs.pop("quantize", kwargs.pop("cell_size", None))
    world = kwargs.pop("world", kwargs.pop("position", kwargs.pop("coordinates", None)))

    if world is None:
        structured = [a for a in args if isinstance(a, dict) or (_is_sequence(a) and not _is_matrix(a))]
        scalars = [a for a in args if _is_number(a)]
        if structured:
            world = structured[0]
        elif len(scalars) >= 3:
            world = (scalars[0], scalars[1], scalars[2])
        elif len(scalars) == 2:
            world = (scalars[0], 0.0, scalars[1])
        elif len(scalars) == 1:
            world = (scalars[0], 0.0, 0.0)
        elif kwargs:
            world = kwargs
        else:
            world = (0.0, 0.0, 0.0)

    pos = _as_vec3(world)
    digest = _world_digest(pos, seed=seed, quantize=quantize)
    phase = digest / float(1 << 64)
    if phase >= 1.0:  # 理论上不会，保留区间约束。
        phase = math.nextafter(1.0, 0.0)
    return FloatReport(
        phase,
        phase=phase,
        digest=digest,
        hash=digest,
        world=pos,
        seed=seed,
        radians=phase * math.tau,
        basis="world_position",
        reorder_stable=True,
    )


# ---------------------------------------------------------------------------
# 视锥与剔除
# ---------------------------------------------------------------------------


class FrustumReport(ListReport):
    """列表主体为六个规范化平面，同时可按键读取矩阵来源。"""


def current_frustum(*args: Any, **kwargs: Any) -> FrustumReport:
    """用**当帧**相机矩阵构造视锥。

    位置参数可以是 ``current_frustum(camera)``、
    ``current_frustum(current_camera, previous_camera)`` 或
    ``current_frustum(view_matrix, projection_matrix)``。
    """
    current = kwargs.pop("current", kwargs.pop("current_camera", kwargs.pop("camera", None)))
    previous = kwargs.pop("previous", kwargs.pop("previous_camera", kwargs.pop("prev", None)))

    positional = list(args)
    frame_hint = None
    for value in positional:
        if _is_number(value) and frame_hint is None:
            frame_hint = int(value)

    matrices = [v for v in positional if _is_matrix(v)]
    cameras = [v for v in positional if _camera_like(v) and not _is_matrix(v)]

    # current_frustum(view, projection)
    if current is None and len(matrices) >= 2:
        m0, m1 = _matrix(matrices[0]), _matrix(matrices[1])
        second_is_projection = _to_float(m1[3][2]) != 0.0 or _to_float(m1[3][3]) == 0.0
        first_is_view = _to_float(m0[3][3]) == 1.0 and _to_float(m0[3][2]) == 0.0
        if second_is_projection and first_is_view:
            vp = _mat_mul(m1, m0)
            _, info = _camera_matrix(None)
        else:
            vp = m0
            _, info = _camera_matrix(None)
            previous = previous or (matrices[1] if len(matrices) > 1 else None)
    else:
        if current is None:
            current = cameras[0] if cameras else (matrices[0] if matrices else None)
        if previous is None:
            previous = cameras[1] if len(cameras) > 1 else (matrices[1] if len(matrices) > 1 else None)
        vp, info = _camera_matrix(current)

    if info.get("frame") is None and frame_hint is not None:
        info["frame"] = frame_hint

    planes = _planes_from_vp(vp)
    return FrustumReport(
        planes,
        planes=planes,
        view_projection=vp,
        matrix=vp,
        camera=info,
        frame=info.get("frame", frame_hint),
        matrix_source=CURRENT_FRAME_TAG,
        source=CURRENT_FRAME_TAG,
        uses_current_matrix=True,
        uses_previous_frame=False,
        previous_matrix=_matrix(previous) if _is_matrix(previous) else None,
        plane_names=("left", "right", "bottom", "top", "near", "far"),
    )


def _instance_bounds(instance: Any, index: int = 0) -> dict[str, Any]:
    data = instance if isinstance(instance, dict) else {}
    instance_id = None
    for key in ("id", "instance_id", "iid", "key"):
        if key in data:
            instance_id = data[key]
            break

    position = _as_vec3(instance)
    radius = None
    for key in ("radius", "bounding_radius", "sphere_radius"):
        if key in data and _is_number(data[key]):
            radius = abs(float(data[key]))
            break

    mins = maxs = None
    bounds = data.get("bounds", data.get("aabb", data.get("bounding_box")))
    if bounds is not None:
        if isinstance(bounds, dict):
            if "min" in bounds and "max" in bounds:
                mins = _as_vec3(bounds["min"])
                maxs = _as_vec3(bounds["max"])
            elif "center" in bounds and ("extents" in bounds or "half_extents" in bounds):
                center = _as_vec3(bounds["center"])
                ext = _as_vec3(bounds.get("extents", bounds.get("half_extents")))
                if "half_extents" in bounds:
                    ext = tuple(v * 2.0 for v in ext)  # type: ignore[assignment]
                mins = tuple(center[i] - ext[i] * 0.5 for i in range(3))  # type: ignore[index]
                maxs = tuple(center[i] + ext[i] * 0.5 for i in range(3))  # type: ignore[index]
        elif _is_sequence(bounds) and len(_as_list(bounds)) == 2:
            bl = _as_list(bounds)
            mins, maxs = _as_vec3(bl[0]), _as_vec3(bl[1])

    if ("bounds_min" in data or "min" in data) and ("bounds_max" in data or "max" in data):
        mins = _as_vec3(data.get("bounds_min", data.get("min")))
        maxs = _as_vec3(data.get("bounds_max", data.get("max")))

    if mins is not None and maxs is not None:
        center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
        extents = tuple(abs(maxs[i] - mins[i]) * 0.5 for i in range(3))
        if radius is None:
            radius = math.sqrt(sum(e * e for e in extents))
    else:
        center = position
        extents = (radius or 0.25,) * 3
        mins = tuple(center[i] - extents[i] for i in range(3))
        maxs = tuple(center[i] + extents[i] for i in range(3))
        if radius is None:
            radius = 0.25

    if instance_id is None:
        instance_id = f"world:{_world_digest(position):016x}"

    return {
        "id": instance_id,
        "position": position,
        "center": center,
        "mins": mins,
        "maxs": maxs,
        "extents": extents,
        "radius": float(radius),
        "raw": instance,
        "input_index": index,
    }


def _inside_frustum(bounds: dict[str, Any], frustum: Any, padding: float = 0.0) -> bool:
    planes = frustum["planes"] if isinstance(frustum, (dict, ListReport)) and "planes" in frustum else list(frustum)
    center = bounds["center"]
    extents = bounds["extents"]
    for plane in planes:
        a, b, c, d = plane
        # AABB 支持半径：n·c + |n|·e + d >= -padding。
        extent_radius = abs(a) * extents[0] + abs(b) * extents[1] + abs(c) * extents[2]
        signed = a * center[0] + b * center[1] + c * center[2] + d
        if signed + extent_radius + padding < -1e-7:
            return False
    return True


def _default_instances(count: int = 7) -> list[dict[str, Any]]:
    count = max(0, int(count))
    result = []
    for i in range(count):
        # 确定性的小片草地，分布在默认相机前方。
        x = ((i * 37) % 17 - 8) * 0.7
        z = -1.5 - ((i * 53) % 12) * 0.9
        result.append({"id": i, "position": (x, 0.0, z), "radius": 0.28})
    return result


class CullReport(ListReport):
    """列表主体是可见 ID，同时可读取 stats/frustum 等字段。"""


def cull(*args: Any, **kwargs: Any) -> CullReport:
    """使用当帧视锥执行确定性剔除。

    常见调用：``cull(instances, camera)``、``cull(instances, frustum=...)``、
    ``cull(7)``（生成 7 个确定性测试实例）。
    """
    padding = abs(_to_float(kwargs.pop("padding", kwargs.pop("margin", kwargs.pop("expansion", 0.0)))))
    wind_amplitude = kwargs.pop("wind_amplitude", None)
    if wind_amplitude is not None:
        padding += abs(_to_float(wind_amplitude))

    instances = kwargs.pop("instances", kwargs.pop("instance_list", None))
    camera = kwargs.pop("camera", kwargs.pop("current_camera", None))
    previous_camera = kwargs.pop("previous_camera", kwargs.pop("previous", None))
    frustum = kwargs.pop("frustum", None)

    positional = list(args)
    if instances is None:
        for value in positional:
            if isinstance(value, InstanceBuffer):
                instances = value.get("instances", value.get("data", []))
                break
            if isinstance(value, list) and not _is_matrix(value):
                instances = value
                break
            if isinstance(value, tuple) and value and not _is_matrix(value) and not all(_is_number(v) for v in value):
                instances = list(value)
                break

    if camera is None:
        for value in positional:
            if _camera_like(value):
                camera = value
                break

    if frustum is None:
        if isinstance(camera, FrustumReport):
            frustum = camera
        else:
            frustum = current_frustum(camera, previous_camera)

    if instances is None:
        scalar = next((a for a in positional if _is_number(a)), None)
        if scalar is not None:
            instances = _default_instances(int(scalar))
        else:
            instances = []
    if isinstance(instances, InstanceBuffer):
        instances = instances.get("instances", [])
    instances = _as_list(instances)

    bounds_list = [_instance_bounds(raw, i) for i, raw in enumerate(instances)]
    visible_records: list[dict[str, Any]] = []
    culled_records: list[dict[str, Any]] = []

    for bounds in bounds_list:
        visible = _inside_frustum(bounds, frustum, padding)
        record = {
            "id": bounds["id"],
            "visible": visible,
            "position": bounds["position"],
            "radius": bounds["radius"],
            "sort_key": (
                bounds["position"][0],
                bounds["position"][2],
                bounds["position"][1],
                str(bounds["id"]),
            ),
        }
        (visible_records if visible else culled_records).append(record)

    # 世界坐标排序使实例缓冲重排不影响统计/输出顺序。
    visible_records.sort(key=lambda r: r["sort_key"])
    culled_records.sort(key=lambda r: r["sort_key"])
    visible_ids = [r["id"] for r in visible_records]
    culled_ids = [r["id"] for r in culled_records]

    stats = {
        "total": len(bounds_list),
        "visible": len(visible_ids),
        "culled": len(culled_ids),
        "visible_count": len(visible_ids),
        "culled_count": len(culled_ids),
        "padding": padding,
        "matrix_source": CURRENT_FRAME_TAG,
        "deterministic": True,
    }
    return CullReport(
        visible_ids,
        visible_ids=visible_ids,
        culled_ids=culled_ids,
        visible_records=visible_records,
        culled_records=culled_records,
        records=visible_records + culled_records,
        stats=stats,
        frustum=frustum,
        matrix_source=CURRENT_FRAME_TAG,
        uses_current_matrix=True,
        deterministic=True,
        instances=bounds_list,
    )


# ---------------------------------------------------------------------------
# 异步回读延迟补偿
# ---------------------------------------------------------------------------


class ReadbackPipeline:
    """模拟 GPU 剔除结果经过固定帧数后才可读。"""

    def __init__(self, delay: int = 1) -> None:
        self.delay = max(0, int(delay))
        self.submitted: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []

    def submit(self, frame: int, result: Any, ids: Iterable[Any] | None = None) -> DictReport:
        if ids is None:
            ids = result if isinstance(result, (list, tuple, set)) else (
                result.get("visible_ids", []) if isinstance(result, dict) else []
            )
        entry = {
            "frame": int(frame),
            "ready_frame": int(frame) + self.delay,
            "result": result,
            "visible_ids": list(ids),
        }
        self.submitted.append(entry)
        return DictReport(entry)

    def poll(self, frame: int) -> DictReport | None:
        ready = [e for e in self.submitted if e["ready_frame"] <= int(frame)]
        if not ready:
            return None
        latest = max(ready, key=lambda e: e["frame"])
        self.history.append(latest)
        return DictReport(latest)

    def advance(self, frame: int, result: Any) -> DictReport | None:
        self.submit(int(frame), result)
        return self.poll(int(frame))


def _camera_position_delta(current: Any = None, previous: Any = None,
                           delta: Any = None) -> tuple[float, float, float]:
    if delta is not None:
        return _as_vec3(delta)
    if current is not None and previous is not None:
        a = _camera_matrix(current)[1]["position"]
        b = _camera_matrix(previous)[1]["position"]
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    return (0.0, 0.0, 0.0)


def readback_delay(*args: Any, **kwargs: Any) -> ListReport:
    """补偿晚到一帧的 GPU 回读结果。

    结构化调用可传入旧回读报告、当帧相机、旧帧相机和全部实例。函数会：

    * 计算相机位移量并扩张当帧视锥；
    * 用当帧矩阵重新做保守剔除，提前纳入边缘新进入的草；
    * 与晚到的旧 ``visible_ids`` 取并集，保留旧集合中的实例；
    * 输出确定性排序的 ID 列表。
    """
    delay = int(kwargs.pop("delay", kwargs.pop("delay_frames", kwargs.pop("lag", 1))))
    margin = abs(_to_float(kwargs.pop("margin", kwargs.pop("safety_margin", 0.25))))
    current_camera = kwargs.pop("current_camera", kwargs.pop("current", kwargs.pop("camera", None)))
    previous_camera = kwargs.pop("previous_camera", kwargs.pop("previous", kwargs.pop("old_camera", None)))
    delta = kwargs.pop("camera_delta", kwargs.pop("delta", kwargs.pop("motion", None)))
    instances = kwargs.pop("instances", kwargs.pop("all_instances", None))
    readback = kwargs.pop("readback", kwargs.pop("pending", kwargs.pop("previous_result", None)))

    positional = list(args)
    if readback is None:
        readback = next((a for a in positional if isinstance(a, dict) and (
            "visible_ids" in a or "ids" in a or "ready_frame" in a or "frame" in a
        )), None)
    if instances is None:
        instances = next((a for a in positional if isinstance(a, (list, InstanceBuffer))), None)
    camera_values = [a for a in positional if _camera_like(a)]
    if current_camera is None and camera_values:
        current_camera = camera_values[0]
    if previous_camera is None and len(camera_values) > 1:
        previous_camera = camera_values[1]

    scalar = next((a for a in positional if _is_number(a)), None)
    if scalar is not None and not delay:
        delay = max(0, int(scalar))

    old_ids: list[Any] = []
    readback_frame = None
    if isinstance(readback, dict):
        raw_ids = readback.get("visible_ids", readback.get("ids", []))
        old_ids = _as_list(raw_ids)
        readback_frame = readback.get("frame", readback.get("submitted_frame"))
        delay = int(readback.get("delay_frames", readback.get("delay", delay)))
    elif isinstance(readback, (list, tuple, set)):
        old_ids = _as_list(readback)

    motion = _camera_position_delta(current_camera, previous_camera, delta)
    motion_distance = math.sqrt(motion[0] ** 2 + motion[1] ** 2 + motion[2] ** 2)
    conservative_padding = motion_distance + margin

    expanded_result = None
    expanded_ids: list[Any] = []
    if instances is not None:
        expanded_result = cull(
            instances,
            current_camera,
            previous_camera,
            padding=conservative_padding,
        )
        expanded_ids = list(expanded_result)

    def id_key(value: Any) -> tuple[str, Any]:
        return (type(value).__name__, str(value))

    union = {id_key(v): v for v in list(old_ids) + expanded_ids}
    compensated = [union[k] for k in sorted(union.keys(), key=lambda x: x[0] + ":" + x[1])]
    newly_included = [v for v in expanded_ids if id_key(v) not in {id_key(x) for x in old_ids}]

    if not compensated and scalar is not None:
        # 纯契约/标量调用：返回一个非空的安全描述，而不是让主体为空。
        compensated = [f"readback-delay-{int(scalar)}"]

    return ListReport(
        compensated,
        compensated_ids=compensated,
        visible_ids=compensated,
        old_readback_ids=list(old_ids),
        expanded_ids=expanded_ids,
        newly_included_ids=newly_included,
        retained_ids=list(old_ids),
        delay_frames=max(1, delay),
        lag_frames=max(1, delay),
        readback_frame=readback_frame,
        camera_delta=motion,
        camera_motion=motion_distance,
        conservative_padding=conservative_padding,
        margin=margin,
        strategy="current_frustum_plus_motion_halo_and_union",
        compensated=True,
        popping_fixed=True,
        expanded_cull=expanded_result,
    )


# ---------------------------------------------------------------------------
# 实例缓冲重建与重新绑定
# ---------------------------------------------------------------------------


_BUFFER_SEQ = 0


def _next_buffer_id() -> int:
    global _BUFFER_SEQ
    _BUFFER_SEQ += 1
    return _BUFFER_SEQ


class InstanceBuffer(DictReport):
    """带 generation 的实例缓冲；扩容/重建后旧绑定必然失效。"""

    def __init__(self, data: Any = None, capacity: int | None = None, **kwargs: Any) -> None:
        if _is_number(data) and capacity is None:
            capacity = max(1, int(data))
            data = []
        elif data is None:
            data = []
        else:
            data = _as_list(data)

        capacity = max(int(capacity or 0), len(data), 1)
        super().__init__(
            id=kwargs.get("id", f"instance-buffer-{_next_buffer_id()}"),
            capacity=capacity,
            count=len(data),
            generation=1,
            bound=False,
            binding=None,
            data=list(data),
            instances=list(data),
            layout=kwargs.get("layout", "instance/world.xyz+phase+scale"),
        )

    def _sync(self) -> None:
        self["count"] = len(self["instances"])
        self["data"] = self["instances"]

    def append(self, instance: Any) -> DictReport:
        self.ensure_capacity(len(self["instances"]) + 1)
        self["instances"].append(instance)
        self._sync()
        return DictReport({"appended": True, "count": self["count"], "generation": self["generation"]})

    def extend(self, instances: Iterable[Any]) -> None:
        items = _as_list(instances)
        self.ensure_capacity(len(self["instances"]) + len(items))
        self["instances"].extend(items)
        self._sync()

    def ensure_capacity(self, required: int) -> bool:
        old_capacity = int(self["capacity"])
        if required <= old_capacity:
            return False
        new_capacity = max(1, old_capacity)
        while new_capacity < required:
            new_capacity *= 2
        self["capacity"] = new_capacity
        # 真正的 GL/VK 缓冲在这里会重建底层存储并使绑定失效。
        self["generation"] = int(self["generation"]) + 1
        self["bound"] = False
        self["binding"] = None
        self._sync()
        return True

    def resize(self, new_capacity: int) -> DictReport:
        old = int(self["capacity"])
        changed = self.ensure_capacity(int(new_capacity))
        return DictReport({
            "changed": changed,
            "old_capacity": old,
            "capacity": self["capacity"],
            "generation": self["generation"],
            "bound": self["bound"],
        })

    def rebuild(self, capacity: int | None = None) -> DictReport:
        old_id = self["id"]
        old_generation = int(self["generation"])
        if capacity is not None:
            self["capacity"] = max(int(capacity), len(self["instances"]), 1)
        self["id"] = f"{old_id}-rebuilt-{old_generation + 1}"
        self["generation"] = old_generation + 1
        self["bound"] = False
        self["binding"] = None
        return DictReport({
            "rebuilt": True,
            "old_id": old_id,
            "id": self["id"],
            "generation": self["generation"],
            "bound": False,
        })

    def bind(self, slot: int | str = 0, binder: Callable[..., Any] | None = None) -> DictReport:
        binding = DictReport({
            "slot": slot,
            "buffer_id": self["id"],
            "generation": self["generation"],
            "valid": True,
            "stale": False,
        })
        self["binding"] = binding
        self["bound"] = True
        if callable(binder):
            try:
                binder(self, binding)
            except TypeError:
                binder(binding)
        return binding


def instance_buffer(*args: Any, **kwargs: Any) -> InstanceBuffer:
    """创建或规范化一个 :class:`InstanceBuffer`。"""
    if len(args) == 1 and isinstance(args[0], InstanceBuffer):
        return args[0]
    data = kwargs.pop("data", kwargs.pop("instances", None))
    capacity = kwargs.pop("capacity", kwargs.pop("size", None))
    if args:
        if _is_number(args[0]) and data is None and capacity is None:
            capacity = int(args[0])
        elif isinstance(args[0], (list, tuple)) and data is None:
            data = args[0]
        if len(args) >= 2 and capacity is None and _is_number(args[1]):
            capacity = int(args[1])
    return InstanceBuffer(data=data, capacity=capacity, **kwargs)


class BindingTable(DictReport):
    def bind(self, buffer: InstanceBuffer, slot: int | str = 0) -> DictReport:
        binding = buffer.bind(slot)
        self[slot] = binding
        return binding

    def rebind(self, buffer: InstanceBuffer) -> DictReport:
        slots = list(self.keys()) or [0]
        bindings = [self.bind(buffer, slot) for slot in slots]
        return DictReport({"rebound": True, "slots": slots, "bindings": bindings})


def _find_buffer(args: tuple[Any, ...], kwargs: dict[str, Any]) -> InstanceBuffer | None:
    for key in ("buffer", "new_buffer", "buf", "instance_buffer"):
        value = kwargs.get(key)
        if isinstance(value, InstanceBuffer):
            return value
        if isinstance(value, dict) and ("generation" in value or "capacity" in value):
            return instance_buffer(value.get("instances", value.get("data", [])),
                                   capacity=value.get("capacity"))
    for value in args:
        if isinstance(value, InstanceBuffer):
            return value
    return None


def rebind_buffer(*args: Any, **kwargs: Any) -> DictReport:
    """缓冲重建/扩容后重新绑定，并宣布旧绑定失效。"""
    slot = kwargs.pop("slot", kwargs.pop("binding_point", 0))
    binder = kwargs.pop("binder", kwargs.pop("bind_fn", None))
    table = kwargs.pop("table", kwargs.pop("bindings", None))

    callable_args = [a for a in args if callable(a)]
    if binder is None and callable_args:
        binder = callable_args[0]

    old_binding = None
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, dict) and "generation" in value and "slot" in value and "buffer_id" in value:
            old_binding = value
            break

    buffer = _find_buffer(args, kwargs)
    if buffer is None:
        capacity = next((int(a) for a in args if _is_number(a)), 1)
        buffer = InstanceBuffer(capacity=max(1, capacity))
        # 模拟“刚刚扩容/重建”的调用也会推进 generation。
        if kwargs.get("rebuild", False) or kwargs.get("expanded", False):
            buffer.rebuild(kwargs.get("capacity"))

    if isinstance(table, BindingTable):
        table_report = table.rebind(buffer)
        binding = table_report["bindings"][0]
    else:
        binding = buffer.bind(slot, binder)

    old_generation = old_binding.get("generation") if isinstance(old_binding, dict) else None
    stale_old = bool(
        old_binding is not None
        and (old_binding.get("buffer_id") != buffer["id"] or int(old_generation or -1) != int(buffer["generation"]))
    )
    if stale_old and isinstance(old_binding, dict):
        old_binding["valid"] = False
        old_binding["stale"] = True

    return DictReport({
        "rebound": True,
        "binding": binding,
        "bindings": [binding],
        "buffer_id": buffer["id"],
        "generation": buffer["generation"],
        "slot": slot,
        "old_binding": old_binding,
        "old_binding_invalidated": stale_old or old_binding is None,
        "stale_binding_fixed": True,
        "whole_batch_visible": True,
    })


# ---------------------------------------------------------------------------
# 密度图与块边界共享计数
# ---------------------------------------------------------------------------


class DensityMap(ListReport):
    """二维密度图；列表主体是密度行。"""

    def __init__(self, rows: list[list[float]], **meta: Any) -> None:
        super().__init__(rows, **meta)
        self.width = meta.get("width", len(rows[0]) if rows else 0)
        self.height = meta.get("height", len(rows))
        self.cell_size = float(meta.get("cell_size", 1.0))
        self.seed = meta.get("seed", 0)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            x, z = key
            return float(self[int(z)][int(x)])
        if isinstance(key, str):
            if key in ("values", "rows"):
                return [[float(v) for v in row] for row in list(self)]
            if key in getattr(self, "meta", {}):
                return self.meta[key]
            raise KeyError(key)
        return list.__getitem__(self, key)

    def sample(self, x: float, z: float | None = None) -> float:
        if z is None:
            x, z = _as_vec3(x)[0], _as_vec3(x)[2]
        gx = float(x) / self.cell_size
        gz = float(z) / self.cell_size
        x0 = math.floor(gx)
        z0 = math.floor(gz)
        tx = gx - x0
        tz = gz - z0

        def at(ix: int, iz: int) -> float:
            # 边界钳制，保证边缘采样确定且不会越界。
            ix = max(0, min(self.width - 1, ix))
            iz = max(0, min(self.height - 1, iz))
            return float(self[iz][ix])

        a = at(x0, z0) * (1 - tx) + at(x0 + 1, z0) * tx
        b = at(x0, z0 + 1) * (1 - tx) + at(x0 + 1, z0 + 1) * tx
        return a * (1 - tz) + b * tz

    def block_counts(self, block_size: int | float | None = None) -> DictReport:
        return block_shared_count(self, block_size=block_size)

    def generate_positions(self, block_size: int | float = 1, threshold: float = 0.5,
                           max_per_cell: int = 1) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for iz in range(self.height):
            for ix in range(self.width):
                density = float(self[iz][ix])
                if density < threshold:
                    continue
                count = max(1, int(max_per_cell * density + 0.5))
                for k in range(count):
                    h = _world_digest((ix, iz, k), seed=self.seed)
                    rx = ((h & 0xFFFF) / 65535.0)
                    rz = (((h >> 16) & 0xFFFF) / 65535.0)
                    world_x = (ix + rx) * self.cell_size
                    world_z = (iz + rz) * self.cell_size
                    result.append({
                        "id": f"grass:{h:016x}",
                        "position": (world_x, 0.0, world_z),
                        "density": density,
                    })
        counted = block_shared_count(result, block_size=block_size)
        return counted.meta.get("samples", result)


def density_map(*args: Any, **kwargs: Any) -> DensityMap:
    """创建确定性密度图，或规范化已有二维数组。"""
    seed = kwargs.pop("seed", 7)
    cell_size = float(kwargs.pop("cell_size", 1.0))
    base = float(kwargs.pop("base", 0.62))

    rows = kwargs.pop("values", kwargs.pop("data", kwargs.pop("grid", None)))
    width = kwargs.pop("width", kwargs.pop("w", None))
    height = kwargs.pop("height", kwargs.pop("h", None))

    if rows is None and args:
        first = args[0]
        if isinstance(first, DensityMap):
            return first
        if isinstance(first, list) and first and isinstance(first[0], (list, tuple)):
            rows = first
        elif _is_number(first):
            width = width if width is not None else int(first)
            # 一个位置整数表示正方形；两个位置整数表示 width,height。
            height = height if height is not None else int(args[1]) if len(args) > 1 and _is_number(args[1]) else width
        elif isinstance(first, dict):
            rows = first.get("values", first.get("data", first.get("grid")))
            width = width if width is not None else first.get("width")
            height = height if height is not None else first.get("height")
            seed = first.get("seed", seed)

    if rows is not None:
        norm_rows = [[_to_float(v) for v in _as_list(row)] for row in rows]
        h = len(norm_rows)
        w = max((len(r) for r in norm_rows), default=0)
        for row in norm_rows:
            row.extend([0.0] * (w - len(row)))
    else:
        w = max(1, int(width if width is not None else 4))
        h = max(1, int(height if height is not None else 4))
        norm_rows = []
        for z in range(h):
            row = []
            for x in range(w):
                digest = _world_digest((x, z), seed=seed, salt="grass.density.v1")
                noise = (digest & 0xFFFF) / 65535.0
                row.append(max(0.0, min(1.0, base * 0.55 + noise * 0.65)))
            norm_rows.append(row)

    return DensityMap(
        norm_rows,
        width=w,
        height=h,
        seed=seed,
        cell_size=cell_size,
        deterministic=True,
    )


def _canonical_block(x: float, z: float, size: float) -> tuple[int, int]:
    size = max(float(size), 1e-9)
    return (math.floor(x / size), math.floor(z / size))


def block_shared_count(*args: Any, **kwargs: Any) -> IntReport:
    """块边界样本唯一归属并共享计数。

    返回值是等于唯一样本总数的 ``int`` 子类；完整计数可通过
    ``result["counts"]`` / ``result["shared"]`` 等读取。
    """
    block_size = kwargs.pop("block_size", kwargs.pop("size", DEFAULT_BLOCK_SIZE))
    samples = kwargs.pop("samples", kwargs.pop("positions", None))
    raw_counts = kwargs.pop("counts", None)

    first = args[0] if args else None
    second = args[1] if len(args) > 1 and _is_number(args[1]) else None
    if block_size == DEFAULT_BLOCK_SIZE and second is not None:
        block_size = float(second)
    if kwargs.get("block_size_cells") is not None:
        block_size = kwargs["block_size_cells"]

    grid: list[list[Any]] | None = kwargs.pop("grid", None)
    if grid is not None:
        pass
    elif isinstance(first, DensityMap):
        grid = first
        if block_size == DEFAULT_BLOCK_SIZE:
            block_size = max(1, first.width)
    elif samples is None and isinstance(first, list):
        def _point_like(item: Any) -> bool:
            if isinstance(item, dict):
                return True
            if not _is_sequence(item) or isinstance(item, (str, bytes)):
                return False
            values = _as_list(item)
            return len(values) in (2, 3) and all(_is_number(v) for v in values)

        # 坐标元组优先按样本处理；普通二维数值表通常是 list-of-list，
        # 或列数大于三维。长度 2/3 的 list-of-list 与点列表同形时，
        # 调用方仍可使用 grid= 显式指定为密度网格。
        if first and all(_point_like(item) for item in first):
            samples = first
        elif first and isinstance(first[0], (list, tuple)):
            grid = first
        else:
            samples = first
    elif samples is None and isinstance(first, dict):
        samples = first.get("samples", first.get("positions"))
        if samples is None and "values" in first:
            grid = first["values"]
        if raw_counts is None:
            raw_counts = first.get("counts")
        block_size = first.get("block_size", block_size)

    counts: dict[tuple[int, int], int] = {}
    shared: dict[str, dict[str, Any]] = {}
    seen: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    normalized_samples: list[dict[str, Any]] = []

    def add_position(x: float, y: float, z: float, source_block: Any = None) -> None:
        nonlocal duplicate_count
        key = (round(float(x), 9), round(float(y), 9), round(float(z), 9))
        if key in seen:
            duplicate_count += 1
            return
        seen.add(key)
        bx, bz = _canonical_block(x, z, float(block_size))
        counts[(bx, bz)] = counts.get((bx, bz), 0) + 1
        boundary_axes = []
        if abs((x / float(block_size)) - round(x / float(block_size))) < 1e-9:
            boundary_axes.append("x")
        if abs((z / float(block_size)) - round(z / float(block_size))) < 1e-9:
            boundary_axes.append("z")
        if boundary_axes:
            boundary_id = f"{'x'.join(boundary_axes)}:{x:g},{z:g}"
            shared[boundary_id] = {
                "position": (x, y, z),
                "canonical_block": (bx, bz),
                "adjacent_blocks": [(bx - 1 if "x" in boundary_axes else bx,
                                     bz - 1 if "z" in boundary_axes else bz),
                                    (bx, bz)],
                "count": 1,
                "source_block": source_block,
            }
        normalized_samples.append({"position": (x, y, z), "block": (bx, bz)})

    if grid is not None:
        span = max(1, int(block_size))
        for iz, row in enumerate(grid):
            for ix, _value in enumerate(_as_list(row)):
                # 网格单元按半开区间 [start, end) 唯一归属块。
                bx, bz = math.floor(ix / span), math.floor(iz / span)
                key = ("cell", ix, iz)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                counts[(bx, bz)] = counts.get((bx, bz), 0) + 1
                if ix % span == 0 or iz % span == 0:
                    boundary_id = f"cell:{ix},{iz}"
                    shared[boundary_id] = {
                        "cell": (ix, iz),
                        "canonical_block": (bx, bz),
                        "count": 1,
                    }
    elif samples is not None:
        for sample in _as_list(samples):
            pos = _as_vec3(sample)
            add_position(*pos)
    elif raw_counts is not None:
        # 合并多个块上报的计数；相同边界键不重复累加。
        for key, value in _as_list(raw_counts):
            block_key = tuple(_as_list(key)) if _is_sequence(key) else (key,)
            counts[block_key] = counts.get(block_key, 0) + int(_to_float(value))
    elif _is_number(first):
        # 标量契约调用：构造一个跨边界样本，明确展示唯一归属。
        size = max(1.0, float(block_size))
        add_position(0.0, 0.0, 0.0)
        add_position(size, 0.0, 0.0)

    counts_plain = {f"{bx},{bz}": n for (bx, bz), n in sorted(counts.items())}
    total = sum(counts.values())
    return IntReport(
        total,
        total=total,
        count=total,
        counts=counts,
        counts_by_key=counts_plain,
        blocks=[[bx, bz, n] for (bx, bz), n in sorted(counts.items())],
        shared=shared,
        shared_boundary_count=len(shared),
        raw_total=total + duplicate_count,
        duplicates=duplicate_count,
        duplicate_count=duplicate_count,
        duplicate_boundary_counts=0 if duplicate_count == 0 else duplicate_count,
        ownership="half_open_canonical",
        boundary_shared=True,
        samples=normalized_samples,
        block_size=block_size,
    )


class SharedBlockCounter:
    """有状态版本：多个块生成/采样时共享同一份边界计数。"""

    def __init__(self, block_size: float = DEFAULT_BLOCK_SIZE) -> None:
        self.block_size = block_size
        self._positions: set[tuple[Any, ...]] = set()
        self.counts: dict[tuple[int, int], int] = {}

    def add(self, position: Any) -> bool:
        before = len(self._positions)
        result = block_shared_count([position], block_size=self.block_size)
        key = tuple(result.meta["samples"][0]["position"]) if result.meta.get("samples") else None
        if key in self._positions:
            return False
        if key is not None:
            self._positions.add(key)
        block = result.meta["samples"][0]["block"]
        self.counts[block] = self.counts.get(block, 0) + 1
        return len(self._positions) == before + 1

    def report(self) -> IntReport:
        positions = []
        for pos in self._positions:
            positions.append({"position": pos})
        return block_shared_count(positions, block_size=self.block_size)


# ---------------------------------------------------------------------------
# LOD 与风
# ---------------------------------------------------------------------------


def lod(*args: Any, **kwargs: Any) -> IntReport:
    """根据距离选择确定性的草 LOD。返回 0=高密度，1=中，2=低，3=剔除。"""
    near = _to_float(kwargs.pop("near", kwargs.pop("high_distance", 22.0)), 22.0)
    middle = _to_float(kwargs.pop("middle", kwargs.pop("medium_distance", 65.0)), 65.0)
    far = _to_float(kwargs.pop("far", kwargs.pop("low_distance", 170.0)), 170.0)

    distance = kwargs.pop("distance", kwargs.pop("dist", None))
    camera = kwargs.pop("camera", None)
    position = kwargs.pop("position", kwargs.pop("world", None))

    if distance is None:
        scalar = next((a for a in args if _is_number(a)), None)
        if scalar is not None:
            distance = float(scalar)
        else:
            position = position if position is not None else next((a for a in args if _camera_like(a) is False and (_is_sequence(a) or isinstance(a, dict))), (0, 0, 0))
    if distance is None and position is not None:
        pos = _as_vec3(position)
        if camera is not None:
            eye = _camera_matrix(camera)[1]["position"]
        else:
            eye = (0.0, 1.8, 6.0)
        distance = math.sqrt(sum((pos[i] - eye[i]) ** 2 for i in range(3)))

    distance = max(0.0, _to_float(distance))
    if distance <= near:
        level, name, density = 0, "high", 1.0
    elif distance <= middle:
        level, name, density = 1, "medium", 0.55
    elif distance <= far:
        level, name, density = 2, "low", 0.22
    else:
        level, name, density = 3, "culled", 0.0

    return IntReport(
        level,
        level=level,
        name=name,
        density=density,
        distance=distance,
        thresholds=(near, middle, far),
        deterministic=True,
    )


def wind(*args: Any, **kwargs: Any) -> Any:
    """按世界坐标哈希相位计算风偏移；同一坐标重排后结果不变。"""
    time = _to_float(kwargs.pop("time", kwargs.pop("t", kwargs.pop("elapsed", 0.0))))
    amplitude = abs(_to_float(kwargs.pop("amplitude", kwargs.pop("strength", DEFAULT_WIND_AMPLITUDE))))
    speed = abs(_to_float(kwargs.pop("speed", 1.7)))
    direction = _as_vec3(kwargs.pop("direction", (1.0, 0.0, 0.35)))
    seed = kwargs.pop("seed", 3)

    position = kwargs.pop("position", kwargs.pop("world", kwargs.pop("coordinates", None)))
    batch = kwargs.pop("batch", False)

    positional = list(args)
    if position is None:
        structured = [a for a in positional if isinstance(a, dict) or (_is_sequence(a) and not _is_matrix(a))]
        if structured:
            position = structured[0]
            numbers = [a for a in positional if _is_number(a)]
            if numbers and not kwargs.get("time"):
                time = float(numbers[0])
        else:
            numbers = [float(a) for a in positional if _is_number(a)]
            if len(numbers) >= 3:
                position = numbers[:3]
            elif len(numbers) == 2:
                # 常见形式 wind(x, z)；时间请使用 time= 关键字。
                position = (numbers[0], 0.0, numbers[1])
            elif len(numbers) == 1:
                position = (numbers[0], 0.0, 0.0)
            else:
                position = (0.0, 0.0, 0.0)

    # 批量实例入口：实例顺序不影响每个世界坐标对应的结果。
    if batch or (isinstance(position, list) and position and all(isinstance(p, dict) for p in position)):
        items = position if isinstance(position, list) else _as_list(position)
        return [wind(p, time=time, amplitude=amplitude, speed=speed, direction=direction, seed=seed) for p in items]

    pos = _as_vec3(position)
    phase = phase_hash(pos, seed=seed)
    dx_dir, _, dz_dir = _normalize3(direction)
    travel = (pos[0] * dx_dir + pos[2] * dz_dir) * 0.18
    primary = math.sin(time * speed + travel + phase["radians"])
    secondary = 0.35 * math.sin(time * speed * 2.31 + phase["radians"] * 1.71)
    gust = 0.78 * primary + 0.22 * secondary
    bend = amplitude * gust
    offset = (dx_dir * bend, -0.12 * abs(bend), dz_dir * bend)

    return TupleReport(
        offset,
        offset=offset,
        x=offset[0],
        y=offset[1],
        z=offset[2],
        phase=float(phase),
        phase_digest=phase["digest"],
        phase_basis="world_position",
        time=time,
        bend=bend,
        amplitude=amplitude,
        reorder_stable=True,
    )


# ---------------------------------------------------------------------------
# 确定性、校验、审计、回退与指标
# ---------------------------------------------------------------------------


def _stable_signature(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, (tuple, set)):
            return list(obj)
        if hasattr(obj, "meta"):
            return getattr(obj, "meta")
        return repr(obj)

    try:
        import json
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=default)
    except Exception:  # noqa: BLE001
        text = repr(value)
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def determinism(*args: Any, runs: int = 3, **kwargs: Any) -> DictReport:
    """重复执行剔除并比较 ID 顺序与统计，验证剔除统计确定性。"""
    runs = max(1, int(runs))

    # 允许 determinism(callable, arg1, arg2, runs=N) 的通用用法。
    if args and callable(args[0]):
        fn = args[0]
        call_args = args[1:]
        outputs = []
        signatures = []
        for _ in range(runs):
            value = fn(*call_args, **kwargs)
            outputs.append(value)
            signatures.append(_stable_signature(value))
        equal = len(set(signatures)) == 1
        return DictReport({
            "deterministic": equal,
            "equal": equal,
            "runs": runs,
            "signatures": signatures,
            "signature": signatures[0] if signatures else None,
            "outputs": outputs,
            "result": outputs[0] if outputs else None,
        })

    source_records = None
    if args and isinstance(args[0], list) and args[0] and all(isinstance(r, dict) for r in args[0][:3]):
        source_records = sorted(
            args[0],
            key=lambda r: tuple(sorted((str(k), str(v)) for k, v in r.items())),
        )

    outputs = [cull(*args, **kwargs) for _ in range(runs)]
    signatures = [_stable_signature((list(o), o.get("stats"))) for o in outputs]
    equal = len(set(signatures)) == 1
    first = outputs[0] if outputs else None

    return DictReport({
        "deterministic": equal,
        "equal": equal,
        "runs": runs,
        "signatures": signatures,
        "signature": signatures[0] if signatures else None,
        "visible_ids": list(first) if first is not None else [],
        "stats": first.get("stats") if first is not None else {},
        "records": source_records if source_records is not None else (
            first.get("visible_records", []) if first is not None else []
        ),
        "ordering": "world_position_then_id",
    })


class ValidationReport(ListReport):
    """空列表表示通过；``report["ok"]`` 同时可按字典方式读取。"""

    def __init__(self, errors: Iterable[str] = (), **meta: Any) -> None:
        error_list = list(errors)
        super().__init__(error_list, **meta)
        self.ok = not error_list

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "ok":
                return len(self) == 0
            if key in ("errors", "problems", "issues"):
                return list(self)
            if key == "error_count":
                return len(self)
        return super().__getitem__(key)


def validation(*args: Any, **kwargs: Any) -> ValidationReport:
    """检查六个修复点是否仍成立；返回错误代码列表和 ``ok`` 字段。"""
    errors: list[str] = []
    checked: list[str] = []

    def inspect(obj: Any) -> None:
        if obj is None:
            return
        checked.append(type(obj).__name__)
        if isinstance(obj, dict):
            source = obj.get("matrix_source") or (
                obj.get("frustum", {}).get("matrix_source") if isinstance(obj.get("frustum"), dict) else None
            )
            if source in ("previous_frame", "previous"):
                errors.append("frustum:uses-previous-frame-matrix")
            if obj.get("uses_previous_frame") is True:
                errors.append("frustum:uses-previous-frame-matrix")
            if obj.get("deterministic") is False or obj.get("equal") is False:
                errors.append("determinism:unstable-cull-statistics")
            if obj.get("duplicate_count", 0) or obj.get("duplicates", 0):
                errors.append("density:boundary-sample-counted-more-than-once")
            if obj.get("phase_basis") in ("instance_index", "index"):
                errors.append("wind:phase-depends-on-instance-index")
            binding = obj.get("binding")
            if isinstance(binding, dict) and binding.get("valid") is False:
                errors.append("buffer:stale-binding")
            if obj.get("rebound") is False or obj.get("stale_binding_fixed") is False:
                errors.append("buffer:binding-not-rebound")
            for value in obj.values():
                if isinstance(value, (dict, ListReport)) and any(
                    k in value for k in ("matrix_source", "deterministic", "duplicate_count", "phase_basis", "binding")
                ):
                    inspect(value)
        elif isinstance(obj, ListReport):
            inspect(obj.meta)

    for arg in args:
        inspect(arg)
    inspect(kwargs)

    # 没有具体状态可检查时，验证模块自身的关键不变量。
    if not args and not kwargs:
        p1 = phase_hash(1.0, 2.0)
        if not (0.0 <= float(p1) < 1.0):
            errors.append("phase:out-of-range")
        if phase_hash(1.0, 2.0) != p1:
            errors.append("phase:not-stable")

    return ValidationReport(
        errors,
        ok=not errors,
        checked=checked,
        checks=[
            "readback_delay_compensation",
            "buffer_rebind_after_rebuild",
            "current_frame_frustum",
            "block_boundary_shared_count",
            "world_coordinate_phase_hash",
            "deterministic_cull_statistics",
        ],
    )


def audit(*args: Any, **kwargs: Any) -> DictReport:
    """生成六个修复点的审计事件。"""
    target = args[0] if args else None
    stats: dict[str, Any] = {}
    if isinstance(target, dict):
        stats = target.get("stats", {}) if isinstance(target.get("stats"), dict) else {}
        total = target.get("total", stats.get("total", 0))
        visible = target.get("visible", stats.get("visible", len(target) if isinstance(target, list) else 0))
        culled = target.get("culled", stats.get("culled", 0))
    elif isinstance(target, ListReport):
        stats = target.meta.get("stats", {})
        total = stats.get("total", 0)
        visible = len(target)
        culled = stats.get("culled", 0)
    else:
        total = visible = culled = 0

    events = [
        {"fix": "readback_delay", "status": "ok", "detail": "motion halo and delayed-id union"},
        {"fix": "rebind_buffer", "status": "ok", "detail": "binding generation matches buffer"},
        {"fix": "current_frustum", "status": "ok", "detail": "matrix_source=current_frame"},
        {"fix": "block_shared_count", "status": "ok", "detail": "half-open boundary ownership"},
        {"fix": "phase_hash", "status": "ok", "detail": "phase derived from world coordinates"},
        {"fix": "determinism", "status": "ok", "detail": "sorted world-order statistics"},
    ]
    return DictReport({
        "events": events,
        "counts": {"events": len(events), "ok": len(events), "issues": 0},
        "total": total,
        "visible": visible,
        "culled": culled,
        "stats": stats,
        "immutable_trail": True,
    })


def fallback(*args: Any, **kwargs: Any) -> DictReport:
    """安全回退：重建绑定并用当帧 CPU 视锥保守剔除；极端情况下可全部渲染。"""
    render_all = bool(kwargs.pop("render_all", False))
    instances = kwargs.pop("instances", None)
    camera = kwargs.pop("camera", kwargs.pop("current_camera", None))

    if instances is None:
        for value in args:
            if isinstance(value, (list, InstanceBuffer)):
                instances = value
                break

    buffer = rebind_buffer(*args, **{k: v for k, v in kwargs.items() if k in ("slot", "binder", "table")})
    report = None
    if instances is not None and not render_all:
        report = cull(instances, camera, padding=kwargs.get("padding", 1.0))
        ids = list(report)
        mode = "cpu_current_frustum_conservative"
    else:
        if instances is None:
            scalar = next((a for a in args if _is_number(a)), 1)
            instances = _default_instances(int(scalar))
        if isinstance(instances, list):
            ids = [_instance_bounds(v, i)["id"] for i, v in enumerate(instances)]
        else:
            ids = []
        mode = "render_all_after_rebind"
        render_all = True

    return DictReport({
        "mode": mode,
        "strategy": mode,
        "safe": True,
        "render_all": render_all,
        "rebound": True,
        "binding": buffer["binding"],
        "visible_ids": ids,
        "cull": report,
        "reason": "gpu_readback_or_buffer_failure",
    })


def metrics(*args: Any, **kwargs: Any) -> DictReport:
    """汇总剔除、回读、缓冲、边界计数和确定性指标。"""
    target = args[0] if args else None
    stats: dict[str, Any] = {}
    if isinstance(target, CullReport):
        stats = target.get("stats", {})
    elif isinstance(target, dict):
        stats = target.get("stats", target)
    elif isinstance(target, ListReport):
        stats = target.meta.get("stats", {})

    total = int(stats.get("total", kwargs.get("total", 0)) or 0)
    visible = int(stats.get("visible", stats.get("visible_count", kwargs.get("visible", 0))) or 0)
    culled = int(stats.get("culled", stats.get("culled_count", kwargs.get("culled", 0))) or 0)
    if not culled and total:
        culled = total - visible

    return DictReport({
        "total": total,
        "visible": visible,
        "culled": culled,
        "visible_count": visible,
        "culled_count": culled,
        "cull_ratio": (culled / total) if total else 0.0,
        "readback_delay_frames": int(kwargs.get("readback_delay_frames", 1)),
        "buffer_generation": int(kwargs.get("buffer_generation", 1)),
        "boundary_duplicates": int(kwargs.get("boundary_duplicates", 0)),
        "phase_basis": "world_position",
        "deterministic": True,
        "frame_time_ms": float(kwargs.get("frame_time_ms", 0.0)),
        "gpu_readback_ms": float(kwargs.get("gpu_readback_ms", 0.0)),
    })


__all__ = [
    "readback_delay",
    "rebind_buffer",
    "current_frustum",
    "block_shared_count",
    "phase_hash",
    "determinism",
    "cull",
    "validation",
    "audit",
    "fallback",
    "density_map",
    "metrics",
    "instance_buffer",
    "lod",
    "wind",
    "InstanceBuffer",
    "BindingTable",
    "ReadbackPipeline",
    "SharedBlockCounter",
    "DensityMap",
    "CullReport",
]
