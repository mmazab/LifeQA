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

4. Run `scripts/train_tgif_qa.sh`.

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

4. Do some pre-processing:

    ```bash
    python preprocessing.py --data_dir ../data/tvqa_format

    for i in 0 1 2 3 4; do
       python preprocessing.py --data_dir ../data/tvqa_format/fold${i}
    done

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
    ```

#### Train and test on LifeQA dataset from scratch

For train, dev and test partitions:

```bash
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

python test.py --model_dir $(ls -t results/ | head -1) --mode test
```

For 5-fold cross-validation:

```bash
for i in 0 1 2 3 4; do
    python main.py \
      --input_streams sub vcpt \
      --no_ts \
      --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
      --train_path ../data/tvqa_format/fold${i}/train_processed.json \
      --valid_path ../data/tvqa_format/fold${i}/validation_processed.json \
      --test_path ../data/tvqa_format/fold${i}/test_processed.json \
      --word2idx_path cache_lifeqa/word2idx.pickle \
      --idx2word_path cache_lifeqa/idx2word.pickle \
      --vocab_embedding_path cache_lifeqa/vocab_embedding.pickle

    python test.py --model_dir $(ls -t results/ | head -1) --mode test
done
```

#### Train on TVQA dataset

```bash
python preprocessing.py

mkdir cache_original
python tvqa_dataset.py \
  --input_streams sub \
  --no_ts \
  --word2idx_path cache_original/word2idx.pickle \
  --idx2word_path cache_original/idx2word.pickle \
  --vocab_embedding_path cache_lifeqa/vocab_embedding.pickle
python main.py \
  --input_streams sub vcpt \
  --no_ts
RESULTS_FOLDER_NAME=$(ls -t results/ | head -1)
```

The result from this part was saved in [results_2019_05_16_23_02_15](https://drive.google.com/drive/folders/1stvXP_38a4lLB22M8s1ye2pgbM23aoyA?usp=sharing).
Note it corresponds to S+V+Q, with cpt as the video feature and w/o ts.

##### Test on LifeQA dataset

For the test partition:

```bash
python test.py \
  --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
  --test_path ../data/tvqa_format/lqa_test_processed.json \
  --model_dir "${RESULTS_FOLDER_NAME}" \
  --mode test
```

For 5-fold cross-validation:

```bash
for i in 0 1 2 3 4; do
  python test.py \
    --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
    --test_path ../data/tvqa_format/fold${i}/test_processed.json \
    --model_dir "${RESULTS_FOLDER_NAME}" \
    --mode test
done
```

##### Fine-tune on LifeQA dataset

For train, dev and test partitions:

```bash
python main.py \
  --input_streams sub vcpt \
  --no_ts \
  --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
  --train_path ../data/tvqa_format/lqa_train_processed.json \
  --valid_path ../data/tvqa_format/lqa_dev_processed.json \
  --test_path ../data/tvqa_format/lqa_test_processed.json \
  --word2idx_path cache_original/word2idx.pickle \
  --idx2word_path cache_original/idx2word.pickle \
  --vocab_embedding_path cache_original/vocab_embedding.pickle \
  --pretrained_model_dir "${RESULTS_FOLDER_NAME}" \
  --new_word2idx_path cache_lifeqa/word2idx.pickle
```

For 5-fold cross-validation:

```bash
python main.py \
  --input_streams sub vcpt \
  --no_ts \
  --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
  --train_path ../data/tvqa_format/fold${i}/train_processed.json \
  --valid_path ../data/tvqa_format/fold${i}/validation_processed.json \
  --test_path ../data/tvqa_format/fold${i}/test_processed.json \
  --word2idx_path cache_original/word2idx.pickle \
  --idx2word_path cache_original/idx2word.pickle \
  --vocab_embedding_path cache_original/vocab_embedding.pickle \
  --pretrained_model_dir "${RESULTS_FOLDER_NAME}" \
  --new_word2idx_path cache_lifeqa/word2idx.pickle
```

TODO: where are these results saved? What result folder names?

## Flux

Run from this folder (cloned) the scripts under `scripts/flux`, such as: `qsub scripts/flux/tgif-qa.pbs`.
