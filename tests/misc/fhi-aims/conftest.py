import pytest
from atomate2.utils.testing.aims import monkeypatch_aims


@pytest.fixture
def mock_aims(monkeypatch, test_dir):
    """
    This fixture allows one to mock (fake) running FHI-aims.
    """
    yield from monkeypatch_aims(monkeypatch, test_dir / "fhi-aims")
