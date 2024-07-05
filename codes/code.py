from abc import ABC, abstractmethod
import base64
import random
from typing import Optional, TypedDict

import attrs

from codes.utils import asyncio_run

ordered_keep_chars = "abcdefghijklmnopqrstuvwxyz "
keep_chars = set(ordered_keep_chars)


class Data(TypedDict):
    question: str
    answer: str
    category: str


class EncodedData(Data):
    equestion: str
    answer: str
    is_coded_q: bool
    is_coded_a: bool
    code_name: str


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


@attrs.frozen
class Spaced(Code):
    space_by: str = " "
    name: str = "Spaced"

    def encode(self, s):
        s = s.replace(" ", ".")
        return " ".join(s)

    def decode(self, s):
        return s.replace(" ", "").replace(".", " ")

    @classmethod
    def newline(cls):
        return cls(space_by="\n", name="SpacedNewline")


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
    *("Tina", "Ulysses", "Violet", "Walter", "Xavier", "Yolanda", "Zach", "."),
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

poetic_sentences = [
    "Autumn's whispered lullaby",
    "Beneath moonlit shadows",
    "Celestial dance eternal",
    "Dreaming in stardust",
    "Echoes of eternity",
    "Fading twilight's embrace",
    "Gossamer wings flutter",
    "Harmonious celestial spheres",
    "Iridescent dewdrops gleam",
    "Jasmine-scented reverie",
    "Kaleidoscope of memories",
    "Lingering velvet night",
    "Misty mountain song",
    "Nebulous ethereal whispers",
    "Ocean's rhythmic lullaby",
    "Pearlescent dawn breaks",
    "Quiet snowfall's grace",
    "Radiant solar flare",
    "Sylvan glade shimmers",
    "Timeless love's embrace",
    "Undulating golden fields",
    "Verdant meadow dreams",
    "Whispering willow boughs",
    "Xanadu's hidden treasures",
    "Yielding to destiny",
    "Zephyr's gentle caress",
    ".",
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
                    assert x.strip() not in y.strip(), f"{x!r} in {y!r}"

    def encode(self, s):
        return "".join(self.mapping[c] for c in s)

    def decode(self, s):
        for c, mapped in self.mapping.items():
            s = s.replace(mapped, c)
            s = s.replace(mapped.strip(), c)
        # assert all(c in keep_chars for c in s)
        return s

    @classmethod
    def names(cls):
        mapping = {c: " " + n for c, n in zip(ordered_keep_chars, alpha_names, strict=True)}
        return cls(mapping=mapping, name="CharToName")

    @classmethod
    def rdm_names(cls):
        names = alpha_names.copy()

        names_start = names[:-1]
        random.Random(0).shuffle(names_start)
        names[:-1] = names_start

        mapping = {c: " " + n for c, n in zip(ordered_keep_chars, names)}
        return cls(mapping=mapping, name="CharToRdmName")

    @classmethod
    def poetry(cls):
        mapping = {c: "\n" + n for c, n in zip(ordered_keep_chars, poetic_sentences, strict=True)}
        return cls(mapping=mapping, name="CharToPoetry")

    @classmethod
    def rdm_poetry(cls):
        names = poetic_sentences.copy()

        names_start = names[:-1]
        random.Random(0).shuffle(names_start)
        names[:-1] = names_start

        mapping = {c: "\n" + n for c, n in zip(ordered_keep_chars, names)}
        return cls(mapping=mapping, name="CharToRdmPoetry")


all_codes: list[Code] = [
    *(Noop(), Base64(), Spaced(), Spaced.newline(), SpaceSepBase64()),
    *(CharToStr.names(), CharToStr.rdm_names(), CharToStr.poetry(), CharToStr.rdm_poetry()),
]


def test():
    for cls in all_codes:
        print(cls.name)
        for s in ["hello", "world", "hello world", ""]:
            encoded = cls.encode(s)
            print(encoded)
            decoded = cls.decode(encoded)
            print(s, decoded)
            assert s == decoded
        print()


if __name__ == "__main__":
    test()
