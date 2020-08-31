# Project evolution

Equally important to the final results is the process we passed through. It was as follows.

## The dataset problem

The most recommended dataset to be used was the Million Playlist Dataset, created by Spotify for the [ACM RecSys Challenge 2018](http://www.recsyschallenge.com/2018/). However, it wasn't available for us to use it.

There's no homogeneity in research community about playlist data. We saw that each researcher creates its own dataset. It was difficult to find playlist datasets on internet, and what we found showed us not to be very useful.

## Solution: dataset creation

Last.fm is a social network about music. Using the package pyLast, we gathered data from Last.fm users, in a network process fashion: starting with a few users, we walked through their followers recursively. At the end, we had public information about many users, tracks and artists.

Spotify is a music streaming service. Using the package Spotipy and the Spotify Web API we scrapped playlist data from many users. Because Spotify doesn't allow us to request user followers, we couldn't gather the data as with Pylast. So we tested the Last.fm users and we colleted their public playlists if available. At the end, he had many Spotify public playlists, as well as information about the tracks, such as the audio features, and about the artists too.

## Exploratory data analysis

The analysis of the data was an expected step in our project. We did a classic EDA (missing values, distributions etc.) and we also created some interesting visualizations, trying to answer questions like 'what's the genre with most tracks?'.

## The models

Music recommendation models are not so widespread as other data science models, so we had to search for these algorithms.

We tested some models, but we found computational problems. In order to solve this, we addapted them and also searched for more efficient algorithms. The two models presented are our approach to solve these problems.

We implemented the models, we chose the hyperparamers and we evaluated them. Although both models aren't much advanced, the results we found was good, comparing to top models from [ACM RecSys Challenge 2018](http://www.recsyschallenge.com/2018/).

## Documentation

Documenting our work was also part of the project. We created an [GitHub repository](https://github.com/lucasresck/espotifai) to store and version our code, and we documented it. The notebooks are also detailed and explain the models. Finally, we built this website in order to expose our work, using [MkDocs](https://www.mkdocs.org/) site generator, and we produced a [screencast](https://youtu.be/w9jelBD4zy8).
