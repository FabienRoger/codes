from abc import ABC, abstractmethod
import base64
import random
from typing import Optional

import attrs

keep_chars = set("abcdefghijklmnopqrstuvwxyz ")


class Code(ABC):
    @abstractmethod
    async def encode(self, s: str, question: Optional[str] = None) -> str: ...
    @abstractmethod
    async def decode(self, s: str, question: Optional[str] = None) -> str: ...
    @property
    def name(self) -> str:
        return self.__class__.__name__


class Noop(Code):
    async def encode(self, s, question=None):
        return s

    async def decode(self, s, question=None):
        return s


class Base64(Code):
    async def encode(self, s, question=None):
        return base64.b64encode(s.encode("utf-8")).decode("utf-8")

    async def decode(self, s, question=None):
        return base64.b64decode(s).decode("utf-8")


class SpaceSepBase64(Code):
    async def encode(self, s, question=None):
        return " ".join(base64.b64encode(s.encode("utf-8")).decode("utf-8"))

    async def decode(self, s, question=None):
        return base64.b64decode("".join(s)).decode("utf-8")


alpha_names = [
    *("Albert", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Isaac", "Jenny"),
    *("Kevin", "Laura", "Michael", "Nancy", "Olivia", "Peter", "Quincy", "Rachel", "Steve"),
    *("Tina", "Ulysses", "Violet", "Walter", "Xavier", "Yolanda", "Zach"),
]

latin_sentences = [
    *("Mons aeterna viget.", "Lux animi.", "Flos vitae.", "Tempus fugit."),
    *("Vox populi.", "Astra incognita.", "Veritas vincit.", "Carpe noctem."),
    *("Fortuna audaces.", "Scientia potentia.", "Memento vivere.", "Ars longa."),
    *("Cogito ergo.", "Festina lente.", "Semper fidelis.", "Vivamus moriendum."),
    *("Nullius in verba.", "Ad astra.", "Virtute et armis.", "Dum spiro spero."),
    *("Acta non verba.", "Alea iacta.", "Caveat emptor.", "Cui bono."),
    *("Ego sum.", "Nunc aut nunquam.", "Sic itur."),
]


@attrs.frozen
class CharToStr(Code):
    name: str
    mapping: dict[str, str]

    def __attrs_post_init__(self):
        assert len(self.mapping) == len(keep_chars)
        assert set(self.mapping.keys()) == keep_chars

        map_to = list(self.mapping.values())
        for i, x in enumerate(map_to):
            for j, y in enumerate(map_to):
                if i != j:
                    assert x not in y

    async def encode(self, s, question=None):
        return "".join(self.mapping[c] for c in s)

    async def decode(self, s, question=None):
        for c, mapped in self.mapping.items():
            s = s.replace(mapped, c)
        assert all(c in keep_chars for c in s)
        return s

    @classmethod
    def names(cls):
        space_char = "."
        mapping = {n.lower()[0]: " " + name for name in alpha_names for n in name.lower() if n in keep_chars} | {
            " ": space_char,
        }
        return cls(mapping, name="CharToName")

    @classmethod
    def rdm_names(cls):
        space_char = "."

        letters = keep_chars - {" "}
        names = alpha_names.copy()
        random.Random(0).shuffle(names)

        mapping = {l: n for l, n in zip(letters, names)} | {
            " ": space_char,
        }
        return cls(mapping, name="CharToRdmName")

    @classmethod
    def latin(cls):
        assert len(latin_sentences) == len(keep_chars)

        mapping = {c: s for c, s in zip(keep_chars, latin_sentences)}
        return cls(mapping, name="CharToLatin")
