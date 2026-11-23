import pytest
import graphics.grass_instance_cull as mod

CASES = [
    ("readback_delay", (1,)),
    ("rebind_buffer", (1,)),
    ("current_frustum", (1,)),
    ("block_shared_count", (1,)),
    ("phase_hash", (1,)),
    ("determinism", (1,)),
    ("cull", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("density_map", (1,)),
    ("metrics", (1,)),
    ("instance_buffer", (1,)),
    ("lod", (1,)),
    ("wind", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
