"""Documentation drift guards: relative links and heading anchors resolve, the
examples a reader copies run as written, and the tables that mirror code match it
(specs/project.md, "Documentation checks")."""

from __future__ import annotations

import ast
import dataclasses
import re
import threading
import tomllib
from pathlib import Path
from typing import Any

import pytest

import wica
import wica.agent
from wica import AgentConfig, TextPart, Wica, WicaConfig, World
from wica.fake_model import FakeChatModel
from wica.testing import TestRun

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS = _REPO_ROOT / "docs"

# The Markdown a reader can reach from the repository: the root files and the
# tracked documentation trees. Left out: `plans/` (records of what was built,
# not rewritten when a file they link to later moves), `analysis/` (untracked
# notes), the benchmark's git-ignored `runs/` and the e2e `reports/`.
_ROOT_FILES = ["README.md", "AGENTS.md", "CONTRIBUTING.md"]
_DOC_DIRS = ["docs", "specs", "benchmarks", "examples", "tests", "tests-e2e"]
_IGNORED_PARTS = {"runs", "reports", "__pycache__"}


def _markdown_files() -> list[Path]:
    files = [_REPO_ROOT / name for name in _ROOT_FILES]
    for directory in _DOC_DIRS:
        for path in sorted((_REPO_ROOT / directory).rglob("*.md")):
            if not _IGNORED_PARTS & set(path.relative_to(_REPO_ROOT).parts):
                files.append(path)
    return files


# --- Relative links -----------------------------------------------------------

_FENCE = re.compile(r"^(```|~~~).*?^\1", re.DOTALL | re.MULTILINE)
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _prose(text: str) -> str:
    """The text with code blocks and code spans removed: a link inside code is
    an example, not a link."""
    return _CODE_SPAN.sub("", _FENCE.sub("", text))


def _slug(heading: str) -> str:
    """GitHub's heading anchor: markup dropped, lower-cased, punctuation
    removed, spaces turned into hyphens."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "")
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    for heading in _HEADING.findall(_FENCE.sub("", path.read_text(encoding="utf-8"))):
        slug = _slug(heading)
        candidate, n = slug, 0
        while candidate in anchors:  # GitHub numbers repeated headings
            n += 1
            candidate = f"{slug}-{n}"
        anchors.add(candidate)
    return anchors


def _broken_links(path: Path) -> list[str]:
    broken: list[str] = []
    for target in _LINK.findall(_prose(path.read_text(encoding="utf-8"))):
        if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (path.parent / file_part).resolve() if file_part else path
        if not resolved.exists():
            broken.append(f"{target} (missing file)")
        elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(f"{target} (no such heading)")
    return broken


@pytest.mark.parametrize(
    "path", _markdown_files(), ids=lambda p: p.relative_to(_REPO_ROOT).as_posix()
)
def test_relative_links_resolve(path: Path) -> None:
    assert _broken_links(path) == []


# --- Executed examples --------------------------------------------------------

_SUBHEADING = re.compile(r"^(#{1,6}) (.+?)\s*$")


def _python_block(path: Path, heading: str) -> str:
    """The first fenced `python` block of `path` under the Markdown heading
    `heading` (headings inside code blocks are not headings)."""
    in_section = in_fence = in_python = False
    level = 0
    block: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            if in_python:
                return "\n".join(block)
            in_fence = not in_fence
            in_python = in_fence and in_section and line.strip() == "```python"
            continue
        if in_python:
            block.append(line)
            continue
        match = None if in_fence else _SUBHEADING.match(line)
        if match:
            if match.group(2) == heading:
                in_section, level = True, len(match.group(1))
            elif in_section and len(match.group(1)) <= level:
                break
    raise AssertionError(f"no python block under {heading!r} in {path.name}")


def _run(path: Path, heading: str, namespace: dict[str, Any] | None = None) -> None:
    """Execute the block as a script; names it defines land in `namespace`."""
    code = compile(_python_block(path, heading), f"{path.name}:{heading}", "exec")
    globals_ = namespace if namespace is not None else {}
    globals_.setdefault("__name__", "__main__")
    exec(code, globals_)  # noqa: S102


def _fake_agent(script: list[dict[str, Any]], *, output_command: bool) -> Wica:
    """A started agent on the fake provider shaped like the docs' robot: a `speech_input`
    Input and, when asked, a `say` output Command."""
    config = WicaConfig(
        agent=AgentConfig(
            provider="fake",
            model="scripted",
            system_prompt="You are a test robot.",
            model_kwargs={"delay_s": 0, "script": script},
        )
    )
    agent = Wica.init(config, coalesce_window=0)
    if output_command:

        async def say(text: str) -> str:
            """Speak out loud to the person in front of you."""
            return "Said it."

        agent.set_output_command(say)
    agent.world.register(
        "speech_input",
        str,
        serialize_fn=lambda value, previous: [TextPart(f'The person said: "{value}"')],
        triggers_llm_call=True,
    )
    agent.start()
    return agent


def test_the_readme_quick_start_runs(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # The quick start names a real provider; a scripted model stands in for it, so the
    # program runs end to end, key-less, exactly as written.
    monkeypatch.setattr(
        wica.agent,
        "build_chat_model",
        lambda config: FakeChatModel(
            script=[{"text": "Hello! Nice to meet you."}], delay_s=0
        ),
    )
    _run(_REPO_ROOT / "README.md", "Quick start")
    assert "robot says: Hello! Nice to meet you." in capsys.readouterr().out


def test_the_getting_started_agent_runs_on_the_fake(capsys) -> None:
    _run(_DOCS / "getting-started.md", "Your first agent")
    out = capsys.readouterr().out
    assert out.index("robot dances") < out.index(
        "robot says: Hello! I just danced for you."
    )


def test_the_testing_guide_fake_provider_example_passes() -> None:
    _run(_DOCS / "guides" / "testing.md", "The `fake` provider")  # asserts inside


def test_the_testing_guide_harness_example_passes(capsys) -> None:
    script = [{"tool_calls": [{"name": "say", "args": {"text": "hi there"}}]}]
    _run(
        _DOCS / "guides" / "testing.md",
        "The harness: `TestRun`",
        {"make_my_agent": lambda: _fake_agent(script, output_command=True)},
    )
    assert re.fullmatch(r"\d+(\.\d+)?\n", capsys.readouterr().out)  # the median latency


def test_the_testing_guide_scripted_tester_example_passes() -> None:
    agent = _fake_agent(
        [{"tool_calls": [{"name": "say", "args": {"text": "hi there"}}]}],
        output_command=True,
    )
    namespace: dict[str, Any] = {"wica": agent}
    try:
        _run(_DOCS / "guides" / "testing.md", "A scripted tester", namespace)
        report = namespace["report"]
        assert report.ok, [f for f in report.findings if not f.ok]
        assert report.verdict == "pass"
    finally:
        namespace["run"].close()
        agent.close()


def test_the_latency_guide_subscription_prints_a_reaction(capsys) -> None:
    agent = _fake_agent([{"text": "Hello!"}], output_command=False)
    ended = threading.Event()
    agent.on_agent_reaction_ended.subscribe(lambda trace: ended.set())
    try:
        _run(
            _DOCS / "guides" / "latency-and-traces.md",
            "A reaction and its phases",
            {"wica": agent},
        )
        agent.world.update("speech_input", "hello")
        assert ended.wait(timeout=5)
    finally:
        agent.close()
    out = capsys.readouterr().out
    assert out.startswith("reaction 1: ok, latency=") and "tokens=" in out


# --- Tables that mirror code --------------------------------------------------


def _section(path: Path, heading: str) -> str:
    """The Markdown of `path` under `heading`, up to the next heading of the same or a
    higher level."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    level = 0
    for line in lines:
        match = _SUBHEADING.match(line)
        if match and match.group(2) == heading and not out:
            level = len(match.group(1))
            out.append(line)
        elif out:
            if match and len(match.group(1)) <= level:
                break
            out.append(line)
    assert out, f"no heading {heading!r} in {path.name}"
    return "\n".join(out)


def _table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if line.startswith("|") and not re.fullmatch(r"\|[-| ]+\|", line.strip()):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[1:]  # drop the header row


def _provider_extras() -> set[str]:
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return set(pyproject["project"]["optional-dependencies"]) - {"gradio"}


def test_configuration_providers_table_matches_the_extras() -> None:
    section = _section(_DOCS / "reference" / "configuration.md", "Providers")
    rows = _table_rows(section)
    assert {re.sub(r"\W", "", r[0].strip("`")) for r in rows} >= {"fake"}
    assert set(re.findall(r"wica\[([a-z-]+)\]", section)) == _provider_extras()
    assert any(r[0] == "`fake`" and r[1] == "none" for r in rows)


def test_readme_install_table_matches_the_extras() -> None:
    section = _section(_REPO_ROOT / "README.md", "Install")
    extras = {r[0].strip("`") for r in _table_rows(section)} - {"*(none)*"}
    assert extras == _provider_extras() | {"gradio"}


def test_configuration_field_table_matches_agent_config() -> None:
    section = _section(_DOCS / "reference" / "configuration.md", "The `agent` block")
    documented = {
        name
        for row in _table_rows(section)
        for name in re.findall(r"`agent\.([a-z_]+)`", row[0])
    }
    assert documented == {f.name for f in dataclasses.fields(AgentConfig)}


def test_api_imports_table_matches_the_public_api() -> None:
    section = _section(_DOCS / "reference" / "api.md", "Imports")
    documented = {
        name
        for row in _table_rows(section)
        for name in re.findall(r"`([A-Za-z_]+)`", row[0])
    }
    assert documented == set(wica.__all__)


def test_api_events_table_matches_wica() -> None:
    section = _section(_DOCS / "reference" / "api.md", "Events")
    documented = {
        name
        for row in _table_rows(section)
        for name in re.findall(r"wica\.(on_[a-z_]+)", row[0])
    }
    agent = Wica.init(
        WicaConfig(
            agent=AgentConfig(
                provider="fake", model="x", system_prompt="x", model_kwargs={}
            )
        )
    )
    try:
        assert documented == {name for name in vars(agent) if name.startswith("on_")}
    finally:
        agent.close()


# --- Documented calls exist ---------------------------------------------------
#
# The consumer docs have lagged a rename while the tests stayed green, because
# nothing checked the documented calls. Every `owner.method(` in their python
# fences must name a method that exists on the public API.

_DOC_CALL = re.compile(r"\b(WicaConfig|Wica|wica|world|agent|run)\.([a-z_]+)\(")
_DOC_TARGETS: dict[str, type] = {
    "WicaConfig": WicaConfig,
    "Wica": Wica,
    "wica": Wica,
    "world": World,
    "agent": wica.Agent,
    "run": TestRun,
}


def test_consumer_docs_only_call_methods_that_exist() -> None:
    missing: list[str] = []
    for doc in [_REPO_ROOT / "README.md", *sorted(_DOCS.rglob("*.md"))]:
        text = doc.read_text(encoding="utf-8")
        # Every fence, the signature listings included — they name calls too.
        for fence in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
            for owner, method in _DOC_CALL.findall(fence):
                if not hasattr(_DOC_TARGETS[owner], method):
                    missing.append(f"{doc.name}: {owner}.{method}()")
    assert not missing, f"docs call methods that do not exist: {sorted(set(missing))}"


def test_python_fences_parse() -> None:
    # Every python fence in the consumer docs is at least syntactically valid.
    for doc in [_REPO_ROOT / "README.md", *sorted(_DOCS.rglob("*.md"))]:
        for fence in re.findall(
            r"```python\n(.*?)```", doc.read_text(encoding="utf-8"), re.DOTALL
        ):
            ast.parse(fence, filename=doc.name)
