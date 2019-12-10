# Amazon food reviews sentiment analysis

Data Analytics exam final project.

By [Matteo Coppola](https://github.com/matteocoppola), [Luca Palazzi](https://github.com/lucapalazzi), [Antonio Vivace](https://github.com/avivace).

## Run

Set up the a Python virtual environment and install required packages

```bash
# run this as sudo if it doesn't work
python3 -m spacy download en

cd scripts
python3 -m venv .
source bin/activate
pip3 install -r requirements.txt
```

Now, to run the full pipeline:
```bash
python3 main.py
```
