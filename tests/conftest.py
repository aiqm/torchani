import pytest


# Tests marked with pt2 are slow, so they are skipped unless selected with -m
def pytest_collection_modifyitems(config, items) -> None:
    if "pt2" in config.getoption("markexpr"):
        return
    skip_pt2 = pytest.mark.skip(reason="Slow torch.compile test, run with -m pt2")
    for item in items:
        if "pt2" in item.keywords:
            item.add_marker(skip_pt2)
