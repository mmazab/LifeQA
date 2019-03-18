# LifeQA

This repo contains the code for the LifeQA project.

Use Python 3.6 and Conda. Setup an environment from `environment.yml`, but also install torch 1.0.0, torchvision 0.2.1,
and allennlp from command: `pip install git+https://github.com/allenai/allennlp@f8b10a9`.

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

## Caveats

### See deprecation warnings

In Python 3.6, [deprecation warnings are not shown by default](https://docs.python.org/3.6/library/warnings.html#warning-categories).
This behavior is changed in Python 3.7. If you want to see them, you need to enable them explicitly. As the main module
for running the training is 3rd party library code (AllenNLP's), we cannot just un-ignore that from code, but it can be
done from the command line. However, setting warning filters from the command line does not allow to use regular
expressions to specify the module, [as pointed out by a recent Python bug report](https://bugs.python.org/issue34624)
(which is misleading by reading [the Python 3.7 warning docs](https://docs.python.org/3.7/library/warnings.html#describing-warning-filters)).
So, just to check the deprecation warnings once in a while, it's recommend to set this env var:

```bash
export PYTHONWARNINGS="default::DeprecationWarning"
```

Which will show any deprecation warning (even those explicitly ignored). To unset it:

```bash
unset PYTHONWARNINGS
```
