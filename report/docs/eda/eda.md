# Spotify dataset analysis

Spotify has a web API, and this API has a translation to Python called Spotipy. It allows us to search for tracks, playlists and artists data. Among all the information available, there exists what is called *audio features*, that is, musical metrics like `danceability`, `loudness` and `instrumentalness`, which could be important for recommendation systems.

Data was captured using the API. The approach was:
1. We started with a big number of Last.fm users, because Last.fm allows network search (search for the friends of a user, and its friends, and so on...);
2. We select a subset of 1000 random users which also has a Spotify account;
3. We selected their public playlists;
4. We selected the tracks from these playlists;
5. We selected the audio features and the artists from these tracks.

It's important to note that songs don't have genres using this API. Who has genre is the artist of the track. So it's necessary to gather the artists of the tracks in order to have genre information.

At the end, we have:

- Users **tracks** dataset;
- List of **users**;
- **Playlists** of the tracks;
- **Audio features** of the tracks;
- **Artists** of the tracks.

We now make some exploration in order to get known the data and have insights for the recommendation models.


```python
from collections import Counter
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
```


```python
# Necessary to improve exploration
pd.set_option('display.max_columns', None)

# Beautiful Seaborn
sns.set()
```

## Initial exploration: datasets

We have many datasets, and we wanna know its variables. Let's sample them.

### Tracks dataset


```python
tracks = pd.read_pickle('../../data/sp_tracks_ready_999.pkl')
tracks.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added_at</th>
      <th>added_by</th>
      <th>is_local</th>
      <th>playlist_id</th>
      <th>available_markets</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15746</th>
      <td>2018-06-23 20:54:16+00:00</td>
      <td>isab3lla</td>
      <td>False</td>
      <td>3sBVA5SQr1uzHRSSzidYNy</td>
      <td>[SE]</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>



**Tracks** dataset is big enough to make we split it in many files. So we did it. At now we have 11 files just for the tracks. But the other datasets are compressed in a single file each one.


```python
list(tracks.columns)
```




    ['added_at',
     'added_by',
     'is_local',
     'playlist_id',
     'available_markets',
     'disc_number',
     'duration_ms',
     'explicit',
     'id',
     'name',
     'popularity',
     'track_number',
     'album_type',
     'album_available_markets',
     'album_id',
     'album_name',
     'album_release_date',
     'artists_ids',
     'artists_names',
     'album_artists_ids',
     'album_artists_names']



It's important to note that everything on Spotify have an ID. So when we have a song and know the ID of one playlist that contains it, we can use this playlist ID to search for the playlist data.

In the dataset, we see that a track have many features, which can be viewed as:

- **Playlist** features: information about the playlists, like when the song was added, who added it and the ID of the playlist.
- **Track** features: like `disc_number`, its duration, if its explicit, `id`, its name, popularity and its number.
- **Album** features, including artists of the album;
- **Artists** features: their IDs and names.

Many features have lists as values. It's because some of them have values which vary in length, like the `artists_ids`. We don't know if we will have 1, 2 or 32 artists. It will happen with other datasets too.

### Playlists dataset


```python
playlists = pd.read_pickle('../../data/sp_playlists.pkl')
playlists.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>collaborative</th>
      <th>description</th>
      <th>id</th>
      <th>name</th>
      <th>primary_color</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>915</th>
      <td>False</td>
      <td></td>
      <td>4cbOOTE5HPaKHd7dbgRk8L</td>
      <td>The Hellacopters Top Hits</td>
      <td>None</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>



Here we have basic information about a public playlist. One thing to note is that everything Spotipy returns is a JSON file, in the 'records' format. So in the column `tracks` we have a dict containing information about the tracks.


```python
playlists.loc[0, 'tracks']
```




    {'href': 'https://api.spotify.com/v1/playlists/0itjZK4e1qZzHL9fNInUJR/tracks',
     'total': 63}



It's little information, but it's not important. The tracks was gathered from the playlist using the ID of the playlist.

### Audio features dataset

Maybe only track information is not enough to make good recommendation systems. So we gather more information, this time more technician:


```python
audio_features = pd.read_pickle('../../data/sp_audio_features.pkl')
audio_features.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37769</th>
      <td>0.553</td>
      <td>0.67</td>
      <td>6</td>
      <td>-4.601</td>
      <td>1</td>
      <td>0.0302</td>
      <td>0.0213</td>
      <td>0.0</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>



Here we have the ID of the track and its features, like ` liveness` and `speechiness`. More information about this technical parte can be found on the [documentation of the Web API](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/).

Another thing to note is that some of `duration_ms` differs from the ones in the tracks database. Let's see:


```python
my_id = None
while pd.isnull(my_id):
    my_id = tracks.sample().id.iloc[0] 
```


```python
print(
    tracks[tracks.id == my_id].duration_ms.iloc[0],
    audio_features[audio_features.id == my_id].duration_ms.iloc[0]
)
```

    248266.0 248267


### Artists dataset


```python
artists = pd.read_pickle('../../data/sp_artists.pkl')
artists.sample()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>followers</th>
      <th>genres</th>
      <th>id</th>
      <th>name</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48244</th>
      <td>4560</td>
      <td>[australian metal, melodic progressive metal]</td>
      <td>1DuzOaU8hyIpzzRQFpAO9b</td>
      <td>Hemina</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



Artists datasets are simple and autoexplanative. It's important to note `genres` features, because each cell is a list of genres. As already said, the only way to classify a song in a genre is with the genres of its artists.

## Visualizations

### When a track was added to a playlist?


```python
data = pd.concat(
    [pd.read_pickle(file)['added_at'] for file in glob.glob('../../data/sp_tracks_ready_*.pkl')],
    ignore_index=True
)

sns.distplot(data.dt.year, kde=False)
plt.title("Histogram of added_at")
plt.xlabel("Year")
plt.show()
```


![png](output_28_0.png)


The max date is:


```python
data.max()
```




    Timestamp('2042-07-07 10:02:09+0000', tz='UTC')



We see some outliers, so we filter them:


```python
sns.distplot(data[(data > '2000') & (data < '2021')].dt.year, kde=False, bins=10)
plt.title("Histogram of added_at")
plt.xlabel("Year")
plt.show()
```


![png](output_32_0.png)


Most of the songs were added recently.

### How much time does a track take?


```python
data = pd.concat(
    [pd.read_pickle(file)['duration_ms'] for file in glob.glob('../../data/sp_tracks_ready_*.pkl')],
    ignore_index=True
)

sns.distplot(data[data < 1e6], kde=False)
plt.title("Distribution of the duration of the tracks (in ms)")
plt.show()
```


![png](output_35_0.png)


The mode of the songs has 3min20s.

### How danceable, louder, ... are the songs?

Audio features dataset has a lot of variables, so we explore some of them.


```python
for variable in ['danceability', 'loudness', 'instrumentalness', 'tempo']:
    sns.distplot(audio_features[variable], kde=False)
    plt.title('Distribution of ' + variable)
    if variable == 'loudness':
        plt.xlabel('loudness (dB)')
    elif variable == 'tempo':
        plt.xlabel('tempo (BPM)')
    plt.show()
```


![png](output_39_0.png)



![png](output_39_1.png)



![png](output_39_2.png)



![png](output_39_3.png)


The features above describe a confidence or a metric made by Spotify. We see that `danceability` (how danceable a track is) has a very nice distribution, that is, most of songs are quite danceable. `Loudness` (in dB) concentrates itself around -10dB.

The distribution of `instrumentalness` is curious: there are tracks we may say they are instrumental (around 0.9), there are tracks we would say it's not instrumental (close to, but not equal to, zero), and there are many many songs we are sure they are not instrumental (very close to or equal to zero). Maybe it's because it's easy to say a song is not instrumental (it's easy to recognize voice), but the reverse is not so true, that is, to ensure the song is instrumental. `temp` is another variable with a quite nice distribution. Songs vary it's tempo (in BPM) around 120.

One thing to say is that these graphs are quite compatible with those from [Spotify's Web API](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/). It suggests that we are in the right way with our data.

### How popular are the artists?


```python
sns.distplot(artists.popularity, kde=False)
plt.title('Distribution of artists popularity')
plt.show()
```


![png](output_43_0.png)


This distribution is very close to the distribution of the popularity of the songs, curiously.

### Finnaly, what are the genres with more tracks?


```python
# We will only sample data because the datasets are big
# This cell takes a while to run

data = pd.concat(
    [pd.read_pickle(file)['artists_ids'].sample(1000) for file in glob.glob('../../data/sp_tracks_ready_*.pkl')],
    ignore_index=True
)

artist_ids = []
for item in data.to_list():
    if type(item) == list:
        for artist_id in item:
            artist_ids.append(artist_id)

genres = []
for id in artist_ids:
    q = artists[artists.id == id].genres.iloc[0]
    if type(q) == list:
        genres.extend(q)

counter = Counter(genres)

ax = sns.barplot(
    list(list(zip(*counter.most_common()))[0][:15]),
    [i/len(genres) for i in list(zip(*counter.most_common()))[1]][:15]
)
plt.xticks(rotation=90)
plt.title("Genres with more tracks")

# https://stackoverflow.com/a/31357733
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

plt.show()
```


![png](output_46_0.png)


Again, that's pretty reasonable. Pop and rock was expected to be in this list, so as rap and hip hop. Again, maybe the percentages seem small, but we have a lot of them (only in the sample):


```python
len(set(genres))
```




    2297



Many of the genres as sub-sub-subgenres, as you can see:


```python
[genres[np.random.randint(len(genres))] for _ in range(15)]
```




    ['piseiro',
     'latin rock',
     'alternative dance',
     'instrumental rock',
     'baile pop',
     'indie soul',
     'rock',
     'swedish synthpop',
     'singer-songwriter',
     'alternative hip hop',
     'skate punk',
     'shimmer pop',
     'dance pop',
     'hip hop',
     'german underground rap']



### Are there correlations between numerical variables?

Let's analyse correlation in audio features dataset.


```python
sns.heatmap(audio_features.corr())
plt.title("Correlation between audio features")
plt.show()
```


![png](output_53_0.png)


We see strong correlations between

- `acousticness` and `energy`: they have strong negative correlation, because acoustic tracks in general are more calm;
- `acousticness` and `loudness`: the same as the latter;
- `energy` and `loudness`: the more energy the song, the louder it is;
- `valence` and `danceability`: the more valence (positiveness), the more danceable it is.
- `instrumental` and `loudness`: we found that a instrumental song tends to be less louder.
