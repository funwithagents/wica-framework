"""Optional, batteries-attached extras that are not part of core WICA.

Nothing here is re-exported from `wica`; each contrib is reached explicitly (e.g.
`from wica.contrib.gradio import world_state_panel`) and pulls its own dependencies through an
install extra (`wica[gradio]`). The coupling is one-directional: contrib imports core, never
the reverse. See specs/gradio-contrib.md ("Packaging").
"""
