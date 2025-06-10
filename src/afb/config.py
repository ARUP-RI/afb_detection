import logging

LOGLEVEL = logging.DEBUG

logging.basicConfig(
    format="[%(asctime)s] %(process)d  %(name)s  %(levelname)s  %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=LOGLEVEL,
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("everett").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.INFO)
# logging.getLogger("sentry_sdk").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.captureWarnings(True)

# RCNN Parameters
#################

# box IoU thresholds b/w proposals and gt boxes in classification
# head: both fg and bg thresholds default to 0.5, not sure but
# I think they need to be equal
BOX_IOU_THRESH = 0.2  # Low-ish seems good for AFB, since any box in the right area is probably 'good enough'


# Validation/Logging
####################
# comet logging box score thresholds, used for generating prec/recall curves
BOX_SCORE_THRESH_LOW = 0
BOX_SCORE_THRESH_HIGH = 1
BOX_SCORE_THRESH_STEPS = 100

# Inference Results Parameters
##############################

# box NMS threshold for use in inference. Serves double duty:
# inside faster rcnn models, it suppresses overlapping predicted
# bboxes. Default there is 0.5. Other use is as a
# threshold for combining bounding boxes across
# multiple possibly overlapping patches
NMS_IOU_THRESH = 0.3

# Minimum width or height of bounding box
# allowed in results output
MIN_BOX_LENGTH = 10
