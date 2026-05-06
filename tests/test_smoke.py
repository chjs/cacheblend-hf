"""Smoke tests: cheap import-only checks that run in CI without model downloads.

These guard against packaging or dependency regressions (e.g., the editable
install missing in CI that masked Phase 1's success). They must stay fast and
must not pull weights from the network.
"""


def test_import_cacheblend():
    import cacheblend
    assert hasattr(cacheblend, "__version__")


def test_import_layerwise_model():
    from cacheblend.model import LayerwiseModel
    assert LayerwiseModel.__name__ == "LayerwiseModel"


def test_import_torch_and_transformers():
    import torch
    import transformers
    assert torch.__version__
    assert transformers.__version__
