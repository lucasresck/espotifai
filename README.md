# espotifai
> Automatic Playlist Recommender.

We studied and implemented some algorithms to deal with the **playlist continuation** problem. Check out [our website](https://lucasresck.github.io/espotifai/) with the report of this work and our [screencast](https://youtu.be/w9jelBD4zy8).

This is our final project for [Foundations of Data Science](https://emap.fgv.br/disciplina/mestrado/fundamentos-de-ciencia-de-dados), a Mathematical Modelling Master's subject at Getulio Vargas Foundation (FGV).

Group: [Lucas Emanuel Resck Domingues](https://github.com/lucasresck) and [Lucas Machado Moschen](https://github.com/lucasmoschen).
Professor: [Dr. Jorge Poco](https://github.com/jpocom).

## Abstract

This repository contains our approach to the **playlist continuation** problem. We scraped data from Spotify and Last.fm and we made an exploratory data analysis. We also implemented models of playlist continuation and we saw good results. We develop a [website](https://lucasresck.github.io/espotifai/) to expose our work.

## Summary repository structure

```
├─ documents -------------------- Deliverables of our project
├─ images ----------------------- Images for our deliverables and README
├─ notebooks
│  ├─ data_scrapping ------------ Notebooks to scrap data
│  ├─ eda ----------------------- Notebooks of EDA
│  ├─ playlist_similarity_model - Model based on playlist similarity
│  └─ track_similarity_model ---- Model based on track similarity
├─ report ----------------------- Our website documents
└─ scripts ---------------------- Scripts to generate data
```

## Usage example

You can:
- Get a list of Last.fm users
- Scrap their public data in Last.fm and Spotify
- Make an exploratory data analysis of these datasets
- Analyse both recommendation models

All notebooks are very well documented and the models are explained in them.

### List of users

In a network propagation fashion, users are gathered from Last.fm. To do this, run:

```python
python generate_lastfm_users.py -h
```

### Data scraping

To scrap data from Spotify and Last.fm, run the notebooks of the folder [`notebooks/data_scraping/`](https://github.com/lucasresck/espotifai/tree/master/notebooks/data_scraping).

![alt text](https://raw.githubusercontent.com/lucasresck/espotifai/master/images/popular_artists.png)

### Exploratory Data Analysis

The template for an EDA of both datasets are inside the folder [`notebooks/eda/`](https://github.com/lucasresck/espotifai/tree/master/notebooks/eda). Fell free to edit and addapt it to your own needs.

![alt text](https://raw.githubusercontent.com/lucasresck/espotifai/master/images/sp_genres.png)

### Analyse the models

Three models are implemented and documented inside [`notebooks/`](https://github.com/lucasresck/espotifai/tree/master/notebooks).

The first is baseline model, with a random walk in a bipartite graph
(simplest similarity matrix). The second is a model based on
track similarity, and the third is based on playlist similarity. Each model
has its notebook detailing the math behind it, as well as the code.


![alt text](https://raw.githubusercontent.com/lucasresck/espotifai/master/images/evaluation.png)

## Development setup

We used the packages Spotipy and Pylast to scrap data from Spotify and Last.fm. Just install the requirements:

```sh
pip install -r requirements.txt
```
