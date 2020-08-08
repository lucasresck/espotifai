# espotifai

An Automatic Playlist Recommender.

This is our final project for [Foundations of Data Science](https://emap.fgv.br/disciplina/mestrado/fundamentos-de-ciencia-de-dados), a Mathematical Modelling Master's subject at Getulio Vargas Foundation (FGV).

## Installation

`pip install pylast`

`pip install spotipy`

## Create datasets

In order to create the datasets, you have to register yourself at [Spotify](https://developer.spotify.com/documentation/web-api/) and [Last.fm](https://www.last.fm/api/). After that, follow this:

### Get users

`python scripts/generate_lastfm_users.py -k <api_key> -s <api_secret> -u <user_to_start> -n <level> -l <limit> -p <path: optional>`

### Get Last.fm data

Run `/notebooks/create_lastfm_datasets.ipynb`.

### Get Spotify data

Run `/notebooks/create_spotify_datasets.ipynb`.

### Create Last.fm analysis

Run `/notebooks/lastfm_data_analysis.ipynb`.

### Create Spotify analysis

Run `/notebooks/spotify_analysis.ipynb`.
