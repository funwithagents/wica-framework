import pytest

from wica.world import reset_world


@pytest.fixture(autouse=True)
def _reset_world():
    reset_world()
    yield
    reset_world()
