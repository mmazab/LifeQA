# LifeQA

This repo contains the code for the LifeQA project.

Use Python 3.6 and Conda. Setup an environment from `environment.yml`, but also install torch 1.0.0, torchvision 0.2.1,
and an allennlp post-0.7.2 patched version (`pip install git+https://github.com/bryant1410/allennlp.git@patch-1`).

## Data

Download the videos from Google Drive to `data/videos`, placing the files there without subdirectories.

## Baselines

### Text

Obtain the transcriptions. You can fetch them with (put the credentials and remember there's a charge for each API
call):

```bash
cd transcribe
GOOGLE_APPLICATION_CREDENTIALS=... python gcp_transcribe.py
```

Then run `run_scripts/train_text_cnn.sh` or `run_scripts/train_text_lstm.sh`.

### Video (TGIF-QA)

1. Install ffmpeg.
2. Run `save_frames.sh` to extract the frames in the video files:

    ```bash
    bash feature_extraction/save_frames.sh
    ```

3. Download [pretrained weights from Sports1M for C3D](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)
and save it in `data/features/c3d.pickle`.
4. To extract the features and save them in heavy H5 files:

    ```bash
    python feature_extraction/extract_features.py
    ``` 
5. Run `run_scripts/train_tgif_qa.sh`.

## Flux

Run from this folder (cloned) the scripts under `flux`, such as: `qsub flux/tgif-qa.pbs`.
