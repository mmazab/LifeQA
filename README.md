# LifeQA data and code

This repo contains the data and PyTorch code that accompanies our LREC 2020 paper:

[LifeQA: A Real-life Dataset for Video Question Answering](https://www.aclweb.org/anthology/2020.lrec-1.536/)

[Santiago Castro](https://santi.uy),
[Mahmoud Azab](https://web.eecs.umich.edu/~mazab/),
[Jonathan C. Stroud](https://www.jonathancstroud.com/),
Cristina Noujaim,
Ruoyao Wang,
[Jia Deng](https://www.cs.princeton.edu/~jiadeng/),
[Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/)

More information is available at [the LifeQA website](https://lit.eecs.umich.edu/lifeqa).

## Setup

To run it, setup a new environment with Conda and activate it:

```bash
conda env create -f environment.yml
conda activate lifeqa
```

## Data

The dataset is under `data/`, in [`lqa_train.json`](data/lqa_train.json), [`lqa_dev.json`](data/lqa_dev.json),
and [`lqa_test.json`](data/lqa_test.json). Even though it's divided into train/dev/test, for most experiments we merge
them and use a five-fold cross-validation, with the folds indicated in [`data/folds`](data/folds).

### Visual features

You can [download the already extracted features](https://deepblue.lib.umich.edu/data/concern/data_sets/05741s53k)
or do the following to extract them yourself.

1. Download the videos. Due to YouTube's Terms of Service, we can't provide the video files. However, we provide the IDs
and timestamps to obtain the same data. Download the YouTube videos indicated in the field `parent_video_id` from the
JSON files, cut them based on the fields `start_time` and `end_time`, and save them based on the JSON key (e.g., `213`)
to `data/videos`, placing the files there without subdirectories.

2. Run `save_frames.sh` to extract the frames in the video files:

    ```bash
    bash feature_extraction/save_frames.sh
    ```

3. Download [pretrained weights from Sports1M for C3D](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)
and save it in `data/features/c3d.pickle`.

4. To extract the features (e.g. from an ImageNet-pretrained ResNet-152) and save them in big H5 files:

    ```bash
    mkdir data/features
    python feature_extraction/extract_features.py resnet
    ```

## Baselines

Check the scripts under `run_scripts` to run the available baselines.

### TVQA

Running the TVQA baseline is different from running the rest of the baselines.

We copied [TVQA's repo content from commit `2c98044`](https://github.com/jayleicn/TVQA/tree/2c98044b949b470d0d31c1cf25cdff60bc673fb8)
into the [`TVQA/`](TVQA) folder.

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

For train, dev, and test partitions:

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

The result from this part was saved in
[results_2019_05_16_23_02_15 in Google Drive](https://drive.google.com/drive/folders/1stvXP_38a4lLB22M8s1ye2pgbM23aoyA?usp=sharing).
Note it corresponds to S+V+Q, with cpt as the video feature and w/o ts.

##### Test on LifeQA dataset

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

For the test partition:

```bash
python test.py \
  --vcpt_path ../data/tvqa_format/det_visual_concepts_hq.pickle \
  --test_path ../data/tvqa_format/lqa_test_processed.json \
  --model_dir "${RESULTS_FOLDER_NAME}" \
  --mode test
```

##### Fine-tune on LifeQA dataset

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

For train, dev, and test partitions:

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

## Issues

If you encounter issues while using our data or code, please
[open an issue in this repo](https://github.com/mmazab/LifeQA/issues/new).

## Citation

[BibTeX entry](https://www.aclweb.org/anthology/2020.lrec-1.536.bib).
