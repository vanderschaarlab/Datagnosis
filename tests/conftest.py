# stdlib
import shutil
from pathlib import Path
from typing import Generator

# third party
import pytest

# datagnosis absolute
from datagnosis.utils.reproducibility import clear_cache, enable_reproducible_results


@pytest.fixture(autouse=True, scope="session")
def run_before_tests() -> Generator:
    enable_reproducible_results(0)
    clear_cache()

    yield

    # cleanup after test
    workspace = Path("test_workspace")
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
