import string
from dataclasses import dataclass
from functools import cache
from abc import ABC, abstractmethod

from afb.data.metadata import MetaData


@dataclass
class ImageID(ABC):
    @classmethod
    def instantiate_type(cls, image_id: str):
        """Tries to find the type of ImageID this is. Coerce to ChecksumID"""
        if isinstance(image_id, ImageID):
            return image_id.get_checksum_id()
        try:
            image_id = ChecksumID(image_id)
            return image_id
        except AssertionError:
            pass
        except ValueError:
            pass
        raise ValueError(f"Unknown ImageID value: {image_id}")

    def __hash__(self):
        return hash(str(self))

    @cache
    def get_lab_id(self):
        wsi = MetaData.specimen_lookup(item_id=self.get_checksum_id())
        return wsi.lab_id

    @abstractmethod
    def get_checksum_id(self):
        raise NotImplementedError

    def __eq__(self, other):
        a = self.get_checksum_id()
        b = other.get_checksum_id()
        return a.checksum == b.checksum


@dataclass(eq=False)
class ChecksumID(ImageID):
    checksum: str

    def __post_init__(self):
        assert isinstance(
            self.checksum, str
        ), f"Checksums should be strings, saw {type(self.checksum)}"
        assert (
            len(self.checksum) == 32
        ), f"Checksums should be 32 characters, saw {len(self.checksum)} chars in {self.checksum}"
        assert all(
            c in set(string.hexdigits) for c in self.checksum
        ), f"Checksums should be hexadecimal, saw non-hex chars in {self.checksum}"

    def __repr__(self):
        return f"<ChecksumID: {self.checksum}>"

    def __str__(self):
        return self.checksum

    def get_checksum_id(self):
        return self
