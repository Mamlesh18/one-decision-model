import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_MODEL = os.environ.get("LAYA_RUN_MODEL_TESTS") == "1"


def pytest_collection_modifyitems(config, items):
    """Tests marked `model` download ~1 GB and run inference; skip them unless asked."""
    if RUN_MODEL:
        return
    skip = pytest.mark.skip(reason="set LAYA_RUN_MODEL_TESTS=1 to run tests that load a checkpoint")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs a downloaded Laya checkpoint (slow)")


@pytest.fixture(scope="session")
def router():
    from shared import get_router
    return get_router()
