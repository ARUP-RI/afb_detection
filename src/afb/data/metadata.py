import functools
from pathlib import Path

import pandas as pd


class MetaData:
    """
    see here: https://rednafi.com/python/lru_cache_on_methods/
    lru_cache seems fine on classmethods, does exactly what I want, it's
    even reportedly threadsafe though I'm not sure if we need that or not

    Idea: From the outside then this'll look/feel a lot like the DjangoAPI
    that it replaces. Internally, we read the csv files from disk once and
    cache them at the class level, then the other classmethods can read
    from these internally cached tables. Obv then we can add classmethods for
    whatever lookups we think would be useful. This also let's us
    do all the necessary type cleanup, NaN handling, etc, just once when we
    first read the csvs from disk. I tried convert_dtypes() to automatically
    infer column dtypes but it always found edge cases to fail on, only
    solution I've found is to explicitly tell pandas the proper type for
    each column, and I'm using Pandas types that handle null values gracefully
    instead of numpy types that do strange things.
    """

    @classmethod
    # @cached_property would be more natural but doesn't seem to work w/ @classmethod
    @functools.lru_cache(maxsize=1)
    def _get_specimens(cls):
        spec_df = pd.read_csv(
            Path(__file__).parents[3] / "metadata/specimens.csv",
            dtype={
                "lab_id": "string",
                "clsi_m48": "Int64",
                "ao_pos": "boolean",
                "afb_positive": "boolean",
                "mgit_culture_positive": "boolean",
                "sample_type": "string",
                "sample_source": "string",
                "microbial_species": "string",
                "notes": "string",
            },
        )
        spec_df = spec_df[
            [
                "lab_id",
                "clsi_m48",
                "ao_pos",
                "afb_positive",
                "mgit_culture_positive",
                "sample_type",
                "sample_source",
                "microbial_species",
                "notes",
            ]
        ].copy()
        return spec_df

    @classmethod
    @functools.lru_cache(maxsize=1)
    def _get_wsis(cls):
        wsi_df = pd.read_csv(
            Path(__file__).parents[3] / "metadata/wsi.csv",
            dtype={
                "checksum_hash": "string",
                "lab_id": "string",
                "mpp_x": "Float64",
                "mpp_y": "Float64",
                "width": "Int64",
                "height": "Int64",
                "glass_slide_id": "Int64",
                "scanner": "string",
            },
        )
        wsi_df = wsi_df[
            [
                "checksum_hash",
                "lab_id",
                "mpp_x",
                "mpp_y",
                "width",
                "height",
                "glass_slide_id",
                "scanner",
            ]
        ].copy()
        spec_df = cls._get_specimens()
        assert (
            wsi_df.lab_id.dtypes == spec_df.lab_id.dtypes
        ), "Type mismatch for lab_id b/w specimen and wsi DataFrames"
        return wsi_df

    @classmethod
    def specimen_lookup(cls, lab_id=None, item_id=None):
        spec_df = cls._get_specimens()
        wsi_df = cls._get_wsis()
        if item_id is not None:
            item_id = str(item_id)
            row = wsi_df[wsi_df.checksum_hash == item_id]
            assert (
                len(row) == 1
            ), f"Should be exactly 1 entry for item_id {item_id} in metadata table but found {len(row)} entries!"
            if lab_id is None:
                lab_id = str(row.iloc[0].lab_id)
            else:
                lab_id = str(lab_id)
                assert (
                    row.iloc[0].lab_id == lab_id
                ), f"Provided lab_id {lab_id} doesn't match lab_id {row.iloc[0].lab_id} looked up from item_id {item_id}!"
        specimen = spec_df[spec_df.lab_id == lab_id]
        assert (
            len(specimen) == 1
        ), f"Should be exactly 1 entry for lab_id {lab_id} in metadata table but found {len(specimen)} entries!"
        # return Series instead of DataFrame for convenient downstream attribute access
        return specimen.iloc[0]

    @classmethod
    def scan_lookup(cls, item_id):
        wsi_df = cls._get_wsis()
        item_id = str(item_id)
        row = wsi_df[wsi_df.checksum_hash == item_id]
        assert (
            len(row) == 1
        ), f"Should be exactly 1 entry for item_id {item_id} in metadata table but found {len(row)} entries!"
        # return Series instead of DataFrame for convenient downstream attribute access
        return row.iloc[0]
