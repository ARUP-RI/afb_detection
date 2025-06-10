from typing import List

from dataclasses import dataclass
import logging
from typing import overload


@dataclass
class LabID:
    lab_id: str

    def __post_init__(self):
        self.lab_id = str(self.lab_id)
        if self.lab_id == "-1":
            logging.warning("Test Lab ID used. This is not a valid Lab ID.")
            return
        if self.lab_id == "-2":
            # Special case in our training data, not a real lab ID
            # "AO 3+ 2023-08-04 08.44.21.ndpi"
            return
        if self.lab_id.endswith("?"):
            logging.warning("Uncertain Lab ID used. This is not a valid Lab ID.")
            return
        # all ints, 11 characters long
        if len(self.lab_id) != 11:
            raise ValueError(f"Lab ID must be 11 characters long: `{self.lab_id}`")
        if not self.lab_id.isdigit():
            raise ValueError(f"Lab ID must be all digits: `{self.lab_id}`")

        # PE TODO NOTE we may only need warnings for bad lab ids, and instead of
        # raising an error, we could just return None

    def __repr__(self):
        return f"<LabID: {self.lab_id}>"

    def __str__(self):
        return self.lab_id

    def __hash__(self):
        return hash(str(self))

    @overload
    def __eq__(self, other: "LabID") -> bool:
        return str(self) == str(other)

    @overload
    def __eq__(self, other: str) -> bool:
        return self.lab_id == other

    def __eq__(self, other):
        return self.lab_id == str(other)

    def __lt__(self, other):
        return self.lab_id < str(other)
