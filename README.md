# Analytics on Amazon Reviews

Data Analytics exam final project, [MSc in Computer Science](https://github.com/avivace/compsci).

By [Matteo Coppola](https://github.com/matteocoppola), [Luca Palazzi](https://github.com/lucapalazzi), [Antonio Vivace](https://github.com/avivace).

> Exploration, Sentiment Analysis, Topic Analysis (LDA) and a VueJS web application exposing the trained models.

[GO. PLAY. WITH THE PLOTS.](https://avivace.github.io/reviews-sentiment) (web demo deployment)

[Documentation](report.pdf)


#### Exploration

<img src="figures/1_rew_len_over_time.svg" width="50%"><img src="figures/1_avg_help_25_100_traffic.svg"  width="50%">

<img src="figures/1_ver_unver_time_traffic.svg"  width="50%"><img src="figures/1_correlation_words_opinion.svg"  width="50%">

#### Web demo

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp1.png">

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp2.png">

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp3.png">

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp4.png">

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp_plot2.png">

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp_plot1.png">

<img src="https://github.com/avivace/reviews-sentiment/blob/develop/figures/ext/webapp_plot3.png">



## Run

Set up the a Python virtual environment and install required packages

```bash
cd scripts
python3 -m venv .
source bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en
```

Optionally, install a ipynb kernel to use the venv packages
```bash
pip3 install --user ipykernel
python -m ipykernel install --user --name=myenv
# Check the installed kernels
jupyter kernelspec list
# Run Jupyter
jupyter lab
```


Now, to run the full pipeline:
```bash
python3 main.py
```

A Flask application exposes a simple API (on port 5000) allowing the trained models to be used on demand via simple HTTP requests (in main.py). The VueJS application needs a recent version of NodeJS and npm.

```bash
cd webapp
npm install
# serve the web application with hot reload at localhost:8080/reviews-sentiment
npm run serve
# builds the web application for production
npm run build
# deploys the build on the master branch, making github serve it on https://avivace.github.io/reviews-sentiment
npm run deploy
```


#### Antuz notes

Accent is `#B71C1C`, typeface is *Barlow* 500. On the plots and graphs, typeface is *Inter* 600, palette is `#4DAF4A`, `#FF7F00`, `#C73E31`.

#### Final notes from our supervisor, E.Fersini

Unverified/Spam "boom" happens around the first-publishing of some product, aggregating data from a category will hardly show this (there are papers on this)
