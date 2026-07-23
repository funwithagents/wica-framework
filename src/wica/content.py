from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextPart:
    text: str

    def to_string(self) -> str:
        return self.text


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    media_type: str

    def to_string(self) -> str:
        return f"[image {self.media_type}]"


ContentPart = TextPart | ImagePart
Content = list[ContentPart]
