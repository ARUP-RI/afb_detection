from enum import Enum
import pandas as pd
import numpy as np

AREA_1000X_MM = 0.21**2


def density_per_1000x(count, area_mm):
    """
    For reference, 0.21 mm, or 210 microns, is about the size of one FoV at
    100x, so this func converts from physical density (num per mm^2) to
    M48 scale of number of organisms per 1000x FoV
    """
    return count / (area_mm / AREA_1000X_MM)


class M48Scale(str, Enum):
    # https://clsi.org/standards/products/microbiology/documents/m48/

    UNKNOWN = "unknown"
    NEGATIVE = "negative"
    POSITIVE_PLUS_MINUS = "+/-"
    POSITIVE_1_PLUS = "1+"
    POSITIVE_2_PLUS = "2+"
    POSITIVE_3_PLUS = "3+"
    POSITIVE_4_PLUS = "4+"

    # NOTE PE this is slightly different than the original scale shown to
    # me by David Ng, adjusted to fit better with a logarithmic scale
    # the source scale probably differs and we may want to match it
    # exactly.
    @classmethod
    def logarithmic_from_density(cls, density: float):
        return (
            cls.POSITIVE_4_PLUS
            if density > 10
            else cls.POSITIVE_3_PLUS
            if density > 1
            else cls.POSITIVE_2_PLUS
            if density > 0.1
            else cls.POSITIVE_1_PLUS
            if density > 0.01
            else cls.POSITIVE_PLUS_MINUS
            if density > 0
            else cls.NEGATIVE
        )

    @classmethod
    def from_any(cls, s: any):
        if isinstance(s, M48Scale):
            return s
        if isinstance(s, np.integer):
            return {
                -5: M48Scale.UNKNOWN,
                -1: M48Scale.NEGATIVE,
                0: M48Scale.POSITIVE_PLUS_MINUS,
                1: M48Scale.POSITIVE_1_PLUS,
                2: M48Scale.POSITIVE_2_PLUS,
                3: M48Scale.POSITIVE_3_PLUS,
                4: M48Scale.POSITIVE_4_PLUS,
            }[s]
        if pd.isna(s):
            return M48Scale.UNKNOWN
        if not s:
            return M48Scale.UNKNOWN
        return {
            "negative": M48Scale.NEGATIVE,
            "neg": M48Scale.NEGATIVE,
            "-1": M48Scale.NEGATIVE,
            "positive_plus_minus": M48Scale.POSITIVE_PLUS_MINUS,
            "+/-": M48Scale.POSITIVE_PLUS_MINUS,
            "0": M48Scale.POSITIVE_PLUS_MINUS,
            "positive_1_plus": M48Scale.POSITIVE_1_PLUS,
            "1+": M48Scale.POSITIVE_1_PLUS,
            "positive_2_plus": M48Scale.POSITIVE_2_PLUS,
            "2+": M48Scale.POSITIVE_2_PLUS,
            "positive_3_plus": M48Scale.POSITIVE_3_PLUS,
            "3+": M48Scale.POSITIVE_3_PLUS,
            "positive_4_plus": M48Scale.POSITIVE_4_PLUS,
            "4+": M48Scale.POSITIVE_4_PLUS,
            "unk": M48Scale.UNKNOWN,
            "unknown": M48Scale.UNKNOWN,
        }.get(s.lower(), M48Scale.UNKNOWN)

    def __int__(self):
        return {
            M48Scale.UNKNOWN: -5,  # PE Maybe we don't want this
            M48Scale.NEGATIVE: -1,
            M48Scale.POSITIVE_PLUS_MINUS: 0,
            M48Scale.POSITIVE_1_PLUS: 1,
            M48Scale.POSITIVE_2_PLUS: 2,
            M48Scale.POSITIVE_3_PLUS: 3,
            M48Scale.POSITIVE_4_PLUS: 4,
        }[self]

    def is_positive(self):
        if self in [
            M48Scale.POSITIVE_PLUS_MINUS,
            M48Scale.POSITIVE_1_PLUS,
            M48Scale.POSITIVE_2_PLUS,
            M48Scale.POSITIVE_3_PLUS,
            M48Scale.POSITIVE_4_PLUS,
        ]:
            return True
        elif self == M48Scale.NEGATIVE:
            return False
        else:
            return None
