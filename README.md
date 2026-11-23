# pairwise-grass-instance-cull

## 领域

游戏、图形与多媒体

## 主题

草地实例化 GPU 剔除

## 说明

回读延迟、缓冲重建与实例相位稳定。

初始状态下 `graphics/grass_instance_cull.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
