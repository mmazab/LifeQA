# LifeQA Analysis

## Setup

1. Install the dependencies: `pip install -r requirements.txt`.
2. Download the questions to a file named `questions.txt`, one questions per line.
3. Tokenize it with `tokenizer.py`.
4. Follow the installation procedure from [jPTDP](https://github.com/datquocnguyen/jPTDP) with the `model256` model to convert the file and predict the dependency parsing, placing the output in `output/questions`.

## Run

Just run `./stats.py` or `./subject.py`. 
