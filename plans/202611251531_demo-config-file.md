# The conversation demo from a config file, and a config app

**Status:** Done

Builds [specs/conversation-demo.md](../specs/conversation-demo.md) "The app: a folder or a file, a Setup tab, a session" as revised: the demo's config is a WICA document (a bare app), the demo runs a given file as it is (`--config FILE`), and a config app edits a file without running a robot.

**Why.** The example configs of `examples/configs/` are WICA documents, but the demo's folder held them under a `"wica"` key: two formats for one config, and no way to run an example config without pasting it into the Setup tab. Running a given file as it is (an example, a config prepared elsewhere, a test) and editing one without starting a session are app-setup's now (a config file as the config's source, bare apps, a config page); the demo takes them.

## Steps

1. **Dependency**: app-setup from its `config-file` branch (its PR #11: `ConfigFile`, `App(bare=True)`, `AppModel(app, source)`, `app_setup.launch`, `config_page`) until it is merged, then back to `main`. `uv lock --upgrade-package app-setup`.
2. **`examples/conversation_demo/app.py`**: `App(..., bare=True)`; `default_config.json` loses its `"wica"` wrapper. The command line is app-setup's `add_config_arguments` (`--folder DIR` | `--config FILE`, `--env PATH`) plus `--host`, `--port` and `--start` / `--no-start`; `open_source` opens the folder or the file, a `ConfigFileError` becomes `parser.error`. `AppModel(APP, source, locked=...)`; the persona is copied into a folder only, never next to a file; `starts_at_launch` starts a file's session at launch unless `--no-start`.
3. **`examples/config_app/`** (`__init__.py`, `__main__.py`): the demo's `APP`, `add_config_app_arguments`, `open_config_app`, `config_page(model).launch(...)`; a missing file is `parser.error`. Run as `uv run --group demo python -m examples.config_app FILE [--env PATH]`.
4. **`.gitignore`**: `.env`, so a key saved next to an example config is never committed.
5. **Docs**: the README's demo section — the three launches, where the `.env` is, an earlier folder's wrapped config deleted once; the spec's frontmatter names the config app and `examples/configs/fake.json`.

## Tests

`tests/test_conversation_demo.py`: the committed default is a bare WICA document, written as it is into an empty folder, and a save writes the document alone; the demo run from a file (its config and its persona relative to it, nothing copied, started at launch and opened on the Conversation tab, the model read-only and the file unchanged); `--no-start` from a file; the keys from the `.env` next to the file; a missing file stops the launch (exit 2, its path said, no page); every example config runs as a `--config` file (checked, its persona found, starting at launch); the config app saves a bare document to its file, takes `--env`, refuses a missing file; `--folder` and `--config` exclusive. The existing tests follow `bare` (the wrapped config only in the model's view) and the parser.
