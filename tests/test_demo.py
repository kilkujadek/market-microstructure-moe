import numpy as np
import torch
from demo import DualMoE, arrays, chronological_indices, features, synthetic_l1

def test_generator_is_seeded():
    a, b = synthetic_l1(120, 4), synthetic_l1(120, 4)
    np.testing.assert_array_equal(a.mid, b.mid)

def test_features_are_prefix_causal():
    book = synthetic_l1(500, 3)
    prefix = type(book)(**{f:getattr(book,f)[:200] for f in book.__dataclass_fields__})
    np.testing.assert_allclose(features(book)[:200], features(prefix), atol=1e-6)

def test_split_has_no_shared_input_or_target_horizon():
    cuts = chronological_indices(1000, 12, 8)
    assert cuts["train"][-1]+8 < cuts["val"][0]-11
    assert cuts["val"][-1]+8 < cuts["test"][0]-11

def test_moe_entry_exit_heads_are_finite():
    model = DualMoE()
    output = model(torch.randn(6,12,7), torch.tensor([0,1,0,1,0,1]))
    assert output.shape == (6,)
    assert torch.isfinite(output).all()

def test_dataset_shapes():
    data, ids = arrays(synthetic_l1(900, 11))
    for s in ("train","val","test"):
        x,p,y = data[s]
        assert x.shape == (len(ids[s]),12,7)
        assert len(p) == len(y) == len(x)
        assert np.isfinite(x).all()
