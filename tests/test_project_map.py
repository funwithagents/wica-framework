import re
from pathlib import Path

from wica import Agent, Wica, WicaConfig, World

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_MD = _REPO_ROOT / "AGENTS.md"
_WICA_PKG = _REPO_ROOT / "src" / "wica"
_SPECS_DIR = _REPO_ROOT / "specs"

# Matches link targets like `src/wica/content.py` anywhere in AGENTS.md.
_MODULE_LINK = re.compile(r"src/wica/([A-Za-z_][A-Za-z0-9_]*\.py)")

# Package glue that isn't a spec'd concept, so it needs no owning spec.
_NON_CONCEPT_MODULES = {"__init__.py"}


def _mapped_modules() -> set[str]:
    text = _AGENTS_MD.read_text(encoding="utf-8")
    return set(_MODULE_LINK.findall(text))


def _actual_modules() -> set[str]:
    return {path.name for path in _WICA_PKG.glob("*.py")}


def _concept_modules() -> set[str]:
    return _actual_modules() - _NON_CONCEPT_MODULES


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


# --- Spec frontmatter drift guards ------------------------------------------
#
# Every spec declares the code/tests it governs in a YAML frontmatter block:
#
#     ---
#     code:
#       - src/wica/world.py
#     tests:
#       - tests/test_world.py
#     ---
#
# The spec-drift skills read this to scope what they diff, so it must stay
# honest: listed paths must exist, and every concept module must be governed by
# at least one spec. The mapping is many-to-many — agent.py is owned by both
# agent.md and commands.md, world.py by both world.md and inputs.md.
#
# Parsed with a tiny hand-rolled reader (not PyYAML, which is only a transitive
# dependency here) — the frontmatter format is authored in this repo and simple.

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_LIST_KEY = re.compile(r"^(code|tests):\s*$")
_LIST_ITEM = re.compile(r"^\s*-\s+(.+?)\s*$")


def _spec_files() -> list[Path]:
    # Concept specs only — skip index/scratch files (`_index.md`, `_todo.md`).
    return sorted(p for p in _SPECS_DIR.glob("*.md") if not p.name.startswith("_"))


def _parse_frontmatter(path: Path) -> dict[str, list[str]]:
    """Return {'code': [...], 'tests': [...]} from a spec's frontmatter block."""
    match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
    if match is None:
        return {}
    result: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key_match = _LIST_KEY.match(line)
        if key_match:
            current = result.setdefault(key_match.group(1), [])
            continue
        item_match = _LIST_ITEM.match(line)
        if item_match and current is not None:
            current.append(item_match.group(1))
    return result


def test_every_spec_declares_the_code_it_governs():
    missing: list[str] = []
    for spec in _spec_files():
        front = _parse_frontmatter(spec)
        if not front.get("code"):
            missing.append(spec.name)
    assert not missing, (
        f"specs missing a non-empty `code:` frontmatter list: {missing}. "
        "Add a frontmatter block naming the files this spec governs so the "
        "spec-drift skills know what to check."
    )


def test_spec_frontmatter_paths_all_exist():
    stale: list[str] = []
    for spec in _spec_files():
        front = _parse_frontmatter(spec)
        for rel in front.get("code", []) + front.get("tests", []):
            if not (_REPO_ROOT / rel).exists():
                stale.append(f"{spec.name} -> {rel}")
    assert not stale, (
        f"spec frontmatter points at paths that no longer exist: {sorted(stale)}. "
        "Update the `code:`/`tests:` lists when files are renamed or removed."
    )


def test_every_concept_module_is_governed_by_a_spec():
    governed: set[str] = set()
    for spec in _spec_files():
        for rel in _parse_frontmatter(spec).get("code", []):
            if rel.startswith("src/wica/"):
                governed.add(Path(rel).name)

    ungoverned = _concept_modules() - governed
    assert not ungoverned, (
        f"src/wica modules not named in any spec's `code:` frontmatter: {sorted(ungoverned)}. "
        "Add each to the frontmatter of the spec that governs it "
        f"(package glue exempt from this rule: {sorted(_NON_CONCEPT_MODULES)})."
    )


# --- Consumer-docs API drift guard ------------------------------------------
#
# README.md and INTEGRATING.md have twice lagged a rename while the tests stayed
# green, because nothing checked the documented signatures. This asserts that
# every `owner.method(` in those documents' python code fences names a method
# that actually exists on the public API.

_CONSUMER_DOCS = [_REPO_ROOT / "README.md", _REPO_ROOT / "INTEGRATING.md"]
_DOC_CALL = re.compile(r"\b(WicaConfig|Wica|wica|world|agent)\.([a-z_]+)\(")
_DOC_TARGETS = {
    "WicaConfig": WicaConfig,
    "Wica": Wica,
    "wica": Wica,
    "world": World,
    "agent": Agent,
}


def test_consumer_docs_only_call_methods_that_exist():
    missing: list[str] = []
    for doc in _CONSUMER_DOCS:
        text = doc.read_text(encoding="utf-8")
        fences = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
        for fence in fences:
            for owner, method in _DOC_CALL.findall(fence):
                if not hasattr(_DOC_TARGETS[owner], method):
                    missing.append(f"{doc.name}: {owner}.{method}()")
    assert not missing, (
        f"consumer docs call methods that do not exist: {sorted(set(missing))}"
    )
