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

## Flux

Run from this folder (cloned) the scripts under `flux`, such as: `qsub flux/tgif-qa.pbs`.
