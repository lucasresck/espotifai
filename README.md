# espotifai
> Automatic Playlist Recommender.

We studied and implemented some k-NN algorithms to deal with the **playlist continuation** problem. Check out [our website](https://lucasresck.github.io/espotifai/) with the report of this work.

This is our final project for [Foundations of Data Science](https://emap.fgv.br/disciplina/mestrado/fundamentos-de-ciencia-de-dados), a Mathematical Modelling Master's subject at Getulio Vargas Foundation (FGV).

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
python generate_users.py
```

### Data scraping

To scrap data from Spotify and Last.fm, run the notebooks of the folder [`notebooks/data_scraping/`](https://github.com/lucasresck/espotifai/tree/master/notebooks/data_scraping).

![alt text](https://raw.githubusercontent.com/lucasresck/espotifai/master/images/popular_artists.png)

### Exploratory Data Analysis

The template for an EDA of both datasets are inside the folder [`notebooks/eda/`](https://github.com/lucasresck/espotifai/tree/master/notebooks/eda). Fell free to edit and addapt it to your own needs.

![alt text](https://raw.githubusercontent.com/lucasresck/espotifai/master/images/sp_genres.png)

### Analyse the models

Two models are implemented and documented inside [`notebooks/`](https://github.com/lucasresck/espotifai/tree/master/notebooks).

Both of them are k-NN based models. The first is a model based on track similarity, and the second is based on playlist similarity. Each model has its notebook detailing the math behind it, as well as the code.

![alt text](https://raw.githubusercontent.com/lucasresck/espotifai/master/images/evaluation.png)

## Development setup

We used the packages Spotipy and Pylast to scrap data from Spotify and Last.fm. Just install the requirements:

```sh
pip install -r requirements.txt
```
