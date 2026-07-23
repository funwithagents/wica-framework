from wica.content import ImagePart, TextPart


def test_text_part_to_string_returns_text_verbatim():
    assert TextPart("hi").to_string() == "hi"


def test_image_part_to_string_returns_placeholder():
    assert ImagePart(b"\x89PNG", "image/png").to_string() == "[image image/png]"


def test_flattening_mixed_content_concatenates_part_strings():
    content = [
        TextPart("before "),
        ImagePart(b"...", "image/jpeg"),
        TextPart(" after"),
    ]
    flattened = "".join(part.to_string() for part in content)
    assert flattened == "before [image image/jpeg] after"
