# The conversation demo on app-setup's panels

**Status:** Done

Builds [specs/conversation-demo.md](../specs/conversation-demo.md) "The app: a folder, a Setup tab, a session", "The Conversation tab" and "Composition of a session", on top of [202611241200_setup-contract-and-keys.md](202611241200_setup-contract-and-keys.md).

**Why.** The demo read one hard-coded config file and fell back to an explore-only World without a key. A user had to edit JSON and export variables to try another provider. With `wica.setup`, app-setup's panels can choose the provider, install its extra, check and test the key, and save it to the app's folder.

## Steps

1. **The app.** `APP` (one section, `wica`, titled "Agent"; host `wica`; the scripted fake as the default config, the persona by its absolute path; `autostart=True`), `--folder`/`--host`/`--port` and `--start`/`--no-start` (`app_setup.autostart`), `AppModel` with `locked=conversation.locked`.
2. **The session** (`session.py`, Gradio-free): `Conversation` with `start(model)` (blockers first; then `open_session` from `used_config(saved)["wica"]`, the folder as `base_dir`; a failed build said) and `stop()` (closes the Wica); `locked()` from Start until the close finished.
3. **The composition** of one session, `open_session`: `build_system` (no explore-only fallback any more), the presenters, `wire`, `start`, and the Wica closed again on a later failure.
4. **The page** (`app_ui.py`): the Conversation tab (status line, Start, Stop, the reading guide, the surfaces in a `gr.render` keyed on the session's version), then `setup_panel` and `secrets_panel`; it opens on Setup.
5. **Dependencies.** `app-setup[ui]` in the `demo` and `dev` groups; the source on app-setup's `autostart` branch until its autostart API was merged, then back on `main`.
6. **Callers**: `tests-e2e/test_example_flow.py` and `benchmarks/bench.py` take the new `build_system` (a missing key or extra is a benchmark `Skip`).

## Tests

`tests/test_conversation_demo.py`: the default config needs nothing; Start locks until Stop; blockers are said and nothing is built; a failed build is said; a relative prompt file is read from the folder; the command line. The page was run live and driven in a headless browser: started at launch on the fake, a reply in the transcript, Stop, Start again, the Setup tab drawn, no page error.
