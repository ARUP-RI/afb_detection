"""
Calling this a "schema" is being generous, but I constantly find myself
rewriting the same type coercions for pandas depending on the task at
hand, or the types get mangled after reading from file, etc.
"""

# from afb.data.image_id import ImageID

BBOX_SCHEMA = [
    "item_id",
    "ground_truth",
    "afb_label",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]
ITEM_SCHEMA = [
    "item_id",
    "density",
    "afb_count",
    "non_afb_count",
    "mimic_count",
    "unk_count",
    "ao_label",
    "wsi_positive",
    "total_patches",
    "patch_area_mm",
]


# def coerce_pandas_item_ids(df, coerce_type=ImageID):
#     _call = {ImageID: ImageID.instantiate_type, str: str}
#     _call = _call[coerce_type]
#     if not isinstance(df.iloc[0].item_id, coerce_type):
#         df["item_id"] = df["item_id"].apply(lambda x: _call(x))
#     return df
