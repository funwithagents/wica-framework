from pathlib import Path

import pytest

from support import PROVIDER_CONFIGS, real_chat_model


@pytest.mark.parametrize("config_path", PROVIDER_CONFIGS, ids=lambda p: p.stem)
def test_real_chat_model_responds(config_path: Path):
    response = real_chat_model(config_path).invoke("say hi")
    assert response.content
