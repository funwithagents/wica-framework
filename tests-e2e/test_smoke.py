from support import real_chat_model


def test_real_chat_model_responds():
    response = real_chat_model().invoke("say hi")
    assert response.content
