# LifeQA

This repo contains the code for the LifeQA project.

To run it, setup a new environment with Conda and activate it:

```bash
conda env create -f environment.yml
conda activate lifeqa
```

## Data

Download the videos from Google Drive to `data/videos`, placing the files there without subdirectories.

## Baselines

There scripts under `run_scripts` provide example on how to run the baselines.

### Text

Obtain the transcriptions. You can fetch them with (put the credentials and remember there's a charge for each API
call):

```bash
cd transcribe
GOOGLE_APPLICATION_CREDENTIALS=... python gcp_transcribe.py
```

Then run `run_scripts/train_text_cnn.sh` or `run_scripts/train_text_lstm.sh`.

### Video (TGIF-QA)

You can [download the already extracted features](https://drive.google.com/drive/folders/1sV1IYoC1oIgjHfSVkIJ-p8GA2hOwx4u1?usp=sharing)
or do the following to extract them yourself.

1. Run `save_frames.sh` to extract the frames in the video files:

    ```bash
    bash feature_extraction/save_frames.sh
    ```

2. Download [pretrained weights from Sports1M for C3D](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)
and save it in `data/features/c3d.pickle`.
3. To extract the features (e.g. from an ImageNet pretrained ResNet152) and save them in heavy H5 files:

    ```bash
    mkdir data/features
    python feature_extraction/extract_features.py resnet
    ```

4. Run `run_scripts/train_tgif_qa.sh`.

### TVQA

TVQA repo content from commit `2c98044` was copied into `TVQA/` folder.

#### Changes from upstream

It has been changed to support 4 answer choices instead of 5.
Some other minor modifications have been done as well.

#### Setup

1. Convert LifeQA dataset to TVQA format

    ```bash
    python scripts/to_tvqa_format.py
    ```

2. Enter `TVQA/` directory:

    ```bash
    cd TVQA/
    ```

3. Setup the interpreter:

    ```bash
    conda env create -f environment.yml
    conda activate tvqa
    ```

#### Train on LifeQA dataset from scratch

```bash
python preprocessing.py --data_dir ../data/tvqa_format

# TODO: for each
python preprocessing.py --data_dir ../data/tvqa_format/fold${i}


mkdir cache_lifeqa
python tvqa_dataset.py \
  --input_streams sub \
  --no_ts \
  --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
  --train_path ../data/tvqa_format/lqa_train_processed.json \
  --valid_path ../data/tvqa_format/lqa_dev_processed.json \
  --test_path ../data/tvqa_format/lqa_test_processed.json \
  --word2idx_path cache_lifeqa/word2idx.pickle \
  --idx2word_path cache_lifeqa/idx2word.pickle \
  --vocab_embedding_path cache_lifeqa/vocab_embedding.pickle
python main.py \
  --input_streams sub vcpt \
  --no_ts \
  --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
  --train_path ../data/tvqa_format/lqa_train_processed.json \
  --valid_path ../data/tvqa_format/lqa_dev_processed.json \
  --test_path ../data/tvqa_format/lqa_test_processed.json \
  --word2idx_path cache_lifeqa/word2idx.pickle \
  --idx2word_path cache_lifeqa/idx2word.pickle \
  --vocab_embedding_path cache_lifeqa/vocab_embedding.pickle
python test.py --model_dir [results_dir] --mode test
```

#### Train on TVQA dataset and then on LifeQA dataset



## Flux

Run from this folder (cloned) the scripts under `scripts/flux`, such as: `qsub scripts/flux/tgif-qa.pbs`.
