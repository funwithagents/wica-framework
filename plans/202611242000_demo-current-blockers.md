# The demo's status line: the saved config's current blockers, the unsaved notice

**Status:** Done

Builds [specs/conversation-demo.md](../specs/conversation-demo.md) "The Conversation tab" as revised: the status line shows the saved config's blockers as they are now, and the unsaved-changes notice sits next to Start.

**Why.** The status line kept the blockers of the last failed Start until a Start succeeded: after the user fixed a key (a new `api_key_env`, a key saved in Secrets), the Conversation tab still listed the old reasons. And with app-setup's draft and saved config apart, a user about to press Start needs to know it runs the saved one.

## Steps

1. **`session.py`**: `Conversation` keeps a failed build's error only (`error`), cleared by the next Start; the blockers a Start finds are returned, not kept.
2. **`app_ui.py`**: `status_text(conversation, blockers)` (the build error, then the blockers, or ready to start); `follow_blockers`, which recomputes the blockers when forced or when the secret store's revision, the session's version or its status changed; the tab's state per browser tab (the blockers, what they were computed against); recomputed on page load, tab select, `setup.saved.change`, Start and Stop, and on the 1 s tick only on a change. `unsaved_notice` next to Start on `setup.draft.change` and `setup.saved.change`.
3. **Dependency**: app-setup from its `draft-and-saved` branch (its PR #10, `unsaved_notice`, `SecretStore.revision`) until it is merged, then back to `main`.

## Tests

`tests/test_conversation_demo.py`: a key saved in the store clears the blocker without a Start; a saved config naming a set variable clears it when forced; no check on a tick with no change; a failed build's error kept while the blockers change, cleared by the next Start.
