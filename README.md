# LifeQA

This repo contains the code for the LifeQA project.

Use Python 3.6 and Conda.

## Data

Download the videos from Google Drive to `data/videos`.

## Baselines

### Text

Obtain or run the transcriptions (remember there's a charge for each API call):

```bash
cd transcribe
GOOGLE_APPLICATION_CREDENTIALS=<SET_VALUE_HERE> python gcp_transcribe.py
```

Then run `run_scripts/train_cnn.sh` or `run_scripts/train_lstm.sh`.

### Video

TODO
