from abc import ABC, abstractmethod
import base64
import random
from typing import Optional, TypedDict

import attrs

from utils import asyncio_run

keep_chars = set("abcdefghijklmnopqrstuvwxyz ")


class Data(TypedDict):
    question: str
    answer: str


class Code(ABC):
    @abstractmethod
    async def encode(self, s: str, equestion: Optional[str] = None) -> str: ...
    @abstractmethod
    async def decode(self, s: str, equestion: Optional[str] = None) -> str: ...
    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def encode_data(self, d: Data) -> Data:
        encoded_question = await self.encode(d["question"])
        encoded_answer = await self.encode(d["answer"], equestion=encoded_question)
        return {"question": encoded_question, "answer": encoded_answer}

    async def decode_data(self, d: Data) -> Data:
        encoded_question = d["question"]
        decoded_question = await self.decode(encoded_question)
        decoded_answer = await self.decode(d["answer"], equestion=encoded_question)
        return {"question": decoded_question, "answer": decoded_answer}


class Noop(Code):
    async def encode(self, s, equestion=None):
        return s

    async def decode(self, s, equestion=None):
        return s


class Base64(Code):
    async def encode(self, s, equestion=None):
        return base64.b64encode(s.encode("utf-8")).decode("utf-8")

    async def decode(self, s, equestion=None):
        return base64.b64decode(s).decode("utf-8")


class SpaceSepBase64(Code):
    async def encode(self, s, equestion=None):
        return " ".join(base64.b64encode(s.encode("utf-8")).decode("utf-8"))

    async def decode(self, s, equestion=None):
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
                    assert x not in y, f"{x!r} in {y!r}"

    async def encode(self, s, equestion=None):
        return "".join(self.mapping[c] for c in s)

    async def decode(self, s, equestion=None):
        for c, mapped in self.mapping.items():
            s = s.replace(mapped, c)
        assert all(c in keep_chars for c in s)
        return s

    @classmethod
    def names(cls):
        space_char = "."
        mapping = {n.lower()[0]: " " + n for n in alpha_names} | {" ": space_char}
        return cls(mapping=mapping, name="CharToName")

    @classmethod
    def rdm_names(cls):
        space_char = "."

        letters = keep_chars - {" "}
        names = alpha_names.copy()
        random.Random(0).shuffle(names)

        mapping = {l: " " + n for l, n in zip(letters, names)} | {" ": space_char}
        return cls(mapping=mapping, name="CharToRdmName")

    @classmethod
    def latin(cls):
        assert len(latin_sentences) == len(keep_chars)

        mapping = {c: s + " " for c, s in zip(keep_chars, latin_sentences)}
        return cls(mapping=mapping, name="CharToLatin")


async def test():
    for cls in [CharToStr.names(), CharToStr.rdm_names(), CharToStr.latin()]:
        print(cls.name)
        for s in ["hello", "world", "hello world"]:
            encoded = await cls.encode(s)
            print(encoded)
            decoded = await cls.decode(encoded)
            print(s, decoded)
            assert s == decoded
        print()


if __name__ == "__main__":
    asyncio_run(test())
