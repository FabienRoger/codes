from abc import ABC, abstractmethod
import base64
import random
from typing import Optional, TypedDict

import attrs

from codes.utils import asyncio_run

keep_chars = set("abcdefghijklmnopqrstuvwxyz ")


class Data(TypedDict):
    question: str
    answer: str


class Code(ABC):
    @abstractmethod
    def encode(self, s: str) -> str: ...
    @abstractmethod
    def decode(self, s: str) -> str: ...
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def try_decode(self, s: str) -> Optional[str]:
        try:
            return self.decode(s)
        except Exception:
            return None


class Noop(Code):
    def encode(self, s):
        return s

    def decode(self, s):
        return s


class Base64(Code):
    def encode(self, s):
        return base64.b64encode(s.encode("utf-8")).decode("utf-8")

    def decode(self, s):
        return base64.b64decode(s).decode("utf-8")


class SpaceSepBase64(Code):
    def encode(self, s):
        return " ".join(base64.b64encode(s.encode("utf-8")).decode("utf-8"))

    def decode(self, s):
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

    def encode(self, s):
        return "".join(self.mapping[c] for c in s)

    def decode(self, s):
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

        letters = sorted(list(keep_chars - {" "}))
        names = alpha_names.copy()
        random.Random(0).shuffle(names)

        mapping = {l: " " + n for l, n in zip(letters, names)} | {" ": space_char}
        return cls(mapping=mapping, name="CharToRdmName")

    @classmethod
    def latin(cls):
        assert len(latin_sentences) == len(keep_chars)

        mapping = {c: s + " " for c, s in zip(sorted(list(keep_chars)), latin_sentences)}
        return cls(mapping=mapping, name="CharToLatin")


def test():
    for cls in [CharToStr.names(), CharToStr.rdm_names(), CharToStr.latin(), Noop(), Base64(), SpaceSepBase64()]:
        print(cls.name)
        for s in ["hello", "world", "hello world"]:
            encoded = cls.encode(s)
            print(encoded)
            decoded = cls.decode(encoded)
            print(s, decoded)
            assert s == decoded
        print()


if __name__ == "__main__":
    print(CharToStr.latin().mapping)
    test()
