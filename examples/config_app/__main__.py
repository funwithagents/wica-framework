"""Edit a WICA config file: the conversation demo's Setup and Secrets tabs over one file, with no
session (specs/conversation-demo.md, "Three launches").

    uv run --group demo python -m examples.config_app FILE [--env PATH] [--host ADDRESS] [--port N]

FILE is a WICA document (an example config of `examples/configs/`, say) and must exist. Save
writes it back, Revert reads it again; its keys are read from and saved to the `.env` in its
directory, unless `--env` names another. It runs in this project's environment, so its checks and
installs are those of the environment the config will run in.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from app_setup.config_file import ConfigFileError
from app_setup.launch import add_config_app_arguments, open_config_app
from app_setup.ui import config_page

from examples.conversation_demo.app import APP


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Edit a WICA config file.")
    add_config_app_arguments(parser)
    options = parser.parse_args(argv)
    try:
        model = open_config_app(options, APP)
    except ConfigFileError as exc:
        parser.error(str(exc))
    config_page(model).launch(server_name=options.host, server_port=options.port)


if __name__ == "__main__":
    main()
