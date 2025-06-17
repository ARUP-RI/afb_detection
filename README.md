# Object detection models for acid-fast bacilli screening

TODO: fill in details such as
- link to paper once it's posted
- basic outline of the code logic

## Models & Datasets
Our trained models, as used to generate the results in our paper, [are available on Huggingface](https://huggingface.co/arup-ri/afb), as are the [datasets they were trained and evaluated on](https://huggingface.co/datasets/arup-ri/kinyoun_afb_50k). Of special note is the validation set for WSI-level predictions, which is split into chunks because of its size(~100Gb). To reassemble them, <nobr>`cat *.tar.gz.* | tar xvfz -`</nobr>
or similar should work on most *nix-like systems (expect this extraction to take at least 15 min, even with fast disk io). Note you'll need over 200Gb of free space to download & then combine the chunks!
<!-- (per [this source](https://stackoverflow.com/a/38199694)) -->

## Python setup
To run the code in this repo, we recommend using `uv` for building a python environment. [Installation instructions can be found here](https://docs.astral.sh/uv/getting-started/installation/) if you haven't used `uv` before. Once `uv` is installed, running `uv sync` in your favorite terminal will regenerate a functioning `.venv` from the `uv.lock` file in this repo.

## Training
Once the python env is built, training can be run directly using the `afb` cli, typically from a bash script something like
```
# Find the path to the .env file and source it to set Comet workspace & API key
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
source "${SCRIPT_DIR}/../.env"

RUN_NAME='datestamp_and_descriptive_name_for_the_experiment'
RESULTS_ROOT=/some/path/to/save/experiments/in

mkdir -p $RESULTS_ROOT/$RUN_NAME

afb train main \
        /path/to/object/detection/training/dataset/ \
        /path/to/object/detection/test/dataset/ \
        $RESULTS_ROOT/$RUN_NAME \
        --run-name=$RUN_NAME \
```
See `src/afb/train.py` for complete listing of available cli options, and also see the `conf.yaml` stored with our trained models for the specific options used by the experiments reported in the paper.

We use comet.ml for tracking experiments, which requires the `COMET_API_KEY` and `COMET_WORKSPACE` env variables to be sourced.

_TODO_: remove comet dependency for easier reproducibility by others?

On an L40S GPU, training runs as reported in the paper can be done in about 1-1.5 hours, depending on choice of batch size, early stopping, etc.

## Inference
Similarly, a bash script something like below will take a trained model & generate predictions over a provided dataset:
```
MODEL_CHOICE='convnext_fcos' # see src/afb/inference.py

afb inference infer \
        $MODEL_CHOICE \
        /path/to/model/checkpoint \
        /path/to/dataset/ \
        --out-dir /path/to/dir/to/save/results/in/ \
```
Running inference on our slide predictions validation set (188 WSI with 10k tiles per slide) can take 2-3 hours on an L40S, depending on batch size, fileio speed, etc.
