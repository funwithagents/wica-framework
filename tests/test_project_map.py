import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_MD = _REPO_ROOT / "AGENTS.md"
_WICA_PKG = _REPO_ROOT / "src" / "wica"

# Matches link targets like `src/wica/content.py` anywhere in AGENTS.md.
_MODULE_LINK = re.compile(r"src/wica/([A-Za-z_][A-Za-z0-9_]*\.py)")


def _mapped_modules() -> set[str]:
    text = _AGENTS_MD.read_text(encoding="utf-8")
    return set(_MODULE_LINK.findall(text))


def _actual_modules() -> set[str]:
    return {path.name for path in _WICA_PKG.glob("*.py")}


def test_agents_md_maps_exactly_the_wica_modules():
    mapped = _mapped_modules()
    actual = _actual_modules()

    missing_from_map = actual - mapped
    stale_in_map = mapped - actual

    assert not missing_from_map, (
        f"src/wica modules not listed in the AGENTS.md project map: {sorted(missing_from_map)}. "
        "Add a row for each new module."
    )
    assert not stale_in_map, (
        f"AGENTS.md project map references modules that no longer exist: {sorted(stale_in_map)}. "
        "Remove or rename their rows."
    )
