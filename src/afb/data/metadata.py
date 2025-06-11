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

    This class should let us get rid of DjangoAPI, but also Specimen,
    LabID, ImageID, and maybe ImageAnnotation(s). Without having to
    juggle checksums, wsi filepaths, etc, now IDs for scans and
    specimens can just always be str, and all lookups go through
    this class. Instead of a custom Specimen type, the metadata for
    one specimen can just be a Pandas Series, and with type conversion
    handled in one place (this class), that should be adequate.

    A corollary/aside on getting rid of Specimen: PatchesDataset currently
    returns a dict of image, target, item_id, extent, and specimen. As far
    as I can tell, we're only ever using specimen to lookup AO+/- and ground
    truth AFB+/- values. If the lookup provided by this class isn't too slow,
    we could possibly just drop the specimen from dataset __getitem__ in the
    new webdatasets and just lookup the specimen from the item_id as needed. I
    think extent is still necessary though? I don't see a way around that yet.
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
        # not sure if there"s still issues in our old text-based ao_pos column
        # but I know the clsi_m48 col is correct, so binarize that instead.
        # Note pd.NA values remain pd.NA as desired
        # spec_df["ao_pos"] = spec_df["clsi_m48"] > 0
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
            # print(f"spec lookup for item_id: {item_id}")
            item_id = str(item_id)
            row = wsi_df[wsi_df.checksum_hash == item_id]
            assert (
                len(row) == 1
            ), f"Should be exactly 1 entry for item_id {item_id} in metadata table but found {len(row)} entries!"
            if lab_id is None:
                lab_id = str(row.iloc[0].lab_id)
                # print(f"item_id {item_id} is linked to lab_id {lab_id}")
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
