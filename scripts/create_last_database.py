#!/usr/bin/env python
# coding: utf-8

import os
import json

import pylast 
import requests 
import pandas as pd

import time

class User:

    def __init__(self, network, filepath: str):

        self.network = network
        if not os.path.exists(filepath): 
            try: 
                with open(filepath, 'w') as f:
                    f.write('user_id,user_name\n')
            except: 
                raise Exception('Filename not found. Please, insert all path.')
        self.users_file = filepath

    def write_users_names(self, user_list: list, level: int = 5, limit: int = 20):
        '''From a list of usernames, write a netwotk list of users in file.
           The file should be a csv. '''
        with open(self.users_file, 'r') as f:
            line = f.readline()
            while line != '':                               # Get the last user id written
                last_line = line 
                line = f.readline()
            try: 
                self.last_user_id = int(last_line.split(',')[0])
            except ValueError:
                self.last_user_id = 0
        ids_users = [str(self.last_user_id + 1 + i) + ',' + user_list[i] for i in range(len(user_list))]
        self._write_in_file(ids_users)
        self.last_user_id += len(user_list)
        for user_name in user_list:
            user = self.network.get_user(username = user_name)
            self._build_network(user, level, limit)

    def _get_friends(self, user, limit: int) -> list:
        '''Get [limit] number of friends and return a list'''
        try: 
            friends = user.get_friends(limit = limit)
        except pylast.WSError: 
            return []
        name_friends = [link.name for link in friends]
        return name_friends 

    def _build_network(self, user, level: int, limit: int): 
        '''Get the frients and build a network'''
        friends = self._get_friends(user, limit = limit)
        id_and_friends = [str(self.last_user_id + i + 1) + ',' + friends[i] for i in range(len(friends))] 
        self._write_in_file(id_and_friends)
        self.last_user_id += len(friends)
        if level > 1:
            for friend in friends: 
                self._build_network(self.network.get_user(friend), level - 1, limit)
        else:
            return
        if self.last_user_id % 1000 == 0: 
            print('Last User is {}'.format(self.last_user_id))

    def _write_in_file(self, lst: list):
        '''Write a list in a filepath. Auxliary funcion'''
        with open(self.users_file, 'a') as f: 
            for i in lst:
                f.write(i)
                f.write('\n')

    def get_id_by_name(self, user_name):
        '''Given a user name return a user_id if it
        exists, or create one if it does not+'''

        users = pd.read_csv(self.users_file, index_col='user_id')
        df_artist = users[users.artist_name == user_name]
        if len(df_artist) == 1:
            user_id = user_id.index[0]
        elif len(df_artist) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            user_id = len(users) + 1
            with open(self.users_file, 'a') as f:
                f.write(str(user_id) + ',' + user_name + '\n')
        return int(user_id)

    def get_user_info(self, user_name: str, limit: int = 20) -> dict: 
        '''Get a lot of user information from last.fm and build it in a dictionary'''

        website = 'http://ws.audioscrobbler.com/2.0/?method=user.getinfo&user='
        website += user_name 
        website += '&api_key='
        website += self.network.api_key
        website += '&format=json'
        get_json = requests.get(website)
        try: 
            user_info_json = json.loads(get_json.content)['user']
        except KeyError: 
            print(user_name + ': ' + json.loads(get_json.content)['message'])
            return {}

        
        user = self.network.get_user(user_name)
        user_info = {}
        user_info['name'] = user_name
        user_info['subscriber'] = int(user_info_json['subscriber'])
        user_info['playcount'] = user_info_json['playcount']
        user_info['registered_since'] = user_info_json['registered']['unixtime']
        user_info['country'] = user_info_json['country']
        user_info['age'] = int(user_info_json['age'])
        user_info['playlists'] = user_info_json['playlists']
        user_info['gender'] = user_info_json['gender']

        tracks = Track(self.network)
        artists = Artist(self.network)
        albums = Album(self.network)
        tags = Tag(self.network)

        user_info['loved_tracks'] = [tracks.get_id_by_name(loved.track.title, loved.track.artist.name) 
                                     for loved in user.get_loved_tracks()]
        try: 
            user_info['recent_tracks'] = [tracks.get_id_by_name(recent.track.title, recent.track.artist.name) 
                                        for recent in user.get_recent_tracks()]
        except pylast.WSError:
            user_info['recent_tracks'] = None
        user_info['top_tracks'] = [(tracks.get_id_by_name(top.item.title, top.item.artist.name),top.weight) 
                                     for top in user.get_top_tracks(limit = limit)]
        user_info['top_tags'] = [(tags.get_id_by_name(top.item.name), top.weight) 
                                  for top in user.get_top_tags(limit = limit)]
        user_info['top_albums'] = [(albums.get_id_by_name(top.item.title, top.item.artist.name), top.weight)
                                     for top in user.get_top_albums(limit = limit)]
        user_info['top_artists'] = [(artists.get_id_by_name(top.item.name), top.weight) 
                                     for top in user.get_top_artists(limit = limit)]
        
        tracks.write_to_csv()
        artists.write_to_csv()
        albums.write_to_csv()
        tags.write_to_csv()
        
        return user_info

class Track:

    def __init__(self, network, filepath: str = '../data/lastfm-api/tracks.csv'):

        self.network = network
        self.tracks_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('track_id\tartist_name\ttrack_name\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')
        self.tracks_df = pd.read_csv(self.tracks_file, index_col='track_id', sep = '\t')
            
    def _set_to_date(self, date: str) -> str:
        '''Given [day] [month] [year] [hour] pattern, return year-mm-dd'''
        day, month, year, _ = date.split() 
        year = year[:4]
        month = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 
                 'Mai':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
                 'Sep':'07', 'Oct':'10', 'Nov':'11', 'Dec':'12' }[month]
        return year + '-' + month + '-' + day

    def write_to_csv(self):

        self.tracks_df.to_csv(self.tracks_file, sep = '\t')

    def get_id_by_name(self, track_name: str, artist_name: str) -> int: 
        '''Given an artist name and track name, return a track id if it
        exists, or create one if it does not'''

        df_track = self.tracks_df[(self.tracks_df.artist_name == artist_name) 
                                & (self.tracks_df.track_name == track_name)]
        if len(df_track) == 1:
            track_id = df_track.index[0]
        elif len(df_track) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            track_id = len(self.tracks_df) + 1
            self.tracks_df.loc[track_id] = [artist_name, track_name]
        return int(track_id)

    def get_track_info(self, track_name: str, artist_name: str, limit: int = 20) -> dict:
        '''Given an artist name and track name, get track info'''
        track_info = {}
        track = self.network.get_track(artist_name, track_name)
        track_info['id'] = self.get_id_by_name(track_name, artist_name)
        track_info['name'] = track_name
        track_info['artist'] = artist_name
        track_info['duration'] = track.get_duration()
        track_info['listeners'] = track.get_listener_count()
        track_info['playcount'] = track.get_playcount()
        track_info['album'] = track.get_album().title
        track_info['published'] = self._set_to_date(track.get_wiki_published_date())

        tags = Tag(self.network)
        track_info['top_tags'] = [(tags.get_id_by_name(tag.item.name), tag.weight) 
                                  for tag in track.get_top_tags(limit = limit)]
        track_info['similar'] = [(similar.item.title, 
                                  similar.item.artist.name, 
                                  similar.match) for similar in track.get_similar(limit = limit)]
        
        return track_info

class Artist:

    def __init__(self, network, filepath: str = '../data/lastfm-api/artists.csv'):

        self.network = network
        self.artists_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('artist_id\tartist_name\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')
        self.artists_df = pd.read_csv(self.artists_file, index_col='artist_id', sep = '\t')
        
    def write_to_csv(self):

        self.artists_df.to_csv(self.artists_file, sep = '\t')
            
    def _set_to_date(self, date: str) -> str:
        '''Given [day] [month] [year] [hour] pattern, return year-mm-dd'''
        day, month, year, _ = date.split() 
        year = year[:4]
        month = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 
                 'Mai':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
                 'Sep':'07', 'Oct':'10', 'Nov':'11', 'Dec':'12' }[month]
        return year + '-' + month + '-' + day

    def get_id_by_name(self, artist_name):
        '''Given an artist name return a artist_id if it
        exists, or create one if it does not+'''

        df_artist = self.artists_df[self.artists_df.artist_name == artist_name]
        if len(df_artist) == 1:
            artist_id = df_artist.index[0]
        elif len(df_artist) > 1:
            raise Exception('Two artists with same name')
        else: 
            artist_id = len(self.artists_df) + 1
            self.artists_df.loc[artist_id] = [artist_name]
        return int(artist_id)

    def get_artist_info(self, artist_name: str, limit: int = 20) -> dict:
        '''Given an artist name, get artist info'''
        artist_info = {}
        artist = self.network.get_artist(artist_name)
        artist_info['id'] = self.get_id_by_name(artist_name)
        artist_info['name'] = artist_name
        artist_info['listeners'] = artist.get_listener_count()
        artist_info['plays'] = artist.get_playcount()
        artist_info['published'] = self._set_to_date(artist.get_bio_published_date())

        tags = Tag(self.network)
        albums = Album(self.network)
        tracks = Track(self.network)

        artist_info['topalbums'] = [(albums.get_id_by_name(top.item.title, top.item.artist.name), top.weight) 
                                    for top in artist.get_top_albums(limit = limit)]
        artist_info['toptags'] = [(tags.get_id_by_name(top.item.name), top.weight) 
                                  for top in artist.get_top_tags(limit = limit)]
        artist_info['toptracks'] = [(tracks.get_id_by_name(top.item.title, top.item.artist.name), top.weight) 
                                    for top in artist.get_top_tracks(limit = limit)]
        artist_info['similar'] = [(similar.item.name, 
                                  similar.match) for similar in artist.get_similar(limit = limit)]
        
        return artist_info

class Album:

    def __init__(self, network, filepath: str = '../data/lastfm-api/albums.csv'):

        self.network = network
        self.albums_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('album_id\tartist_name\talbum_name\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')
        self.albums_df = pd.read_csv(self.albums_file, index_col='album_id', sep = '\t')
        
    def write_to_csv(self):

        self.albums_df.to_csv(self.albums_file, sep = '\t')
            
    def _set_to_date(self, date: str) -> str:
        '''Given [day] [month] [year] [hour] pattern, return year-mm-dd'''
        day, month, year, _ = date.split() 
        year = year[:4]
        month = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 
                 'Mai':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
                 'Sep':'07', 'Oct':'10', 'Nov':'11', 'Dec':'12' }[month]
        return year + '-' + month + '-' + day
            
    def get_id_by_name(self, album_name, artist_name):
        '''Given an artist name and album name, return a album id if it
        exists, or create one if it does not+'''

        df_album = self.albums_df[(self.albums_df.artist_name == artist_name) 
                            & (self.albums_df.album_name == album_name)]
        if len(df_album) == 1:
            album_id = df_album.index[0]
        elif len(df_album) > 1:
            raise Exception('Two albums with same artist name and same album name')
        else: 
            album_id = len(self.albums_df) + 1
            self.albums_df.loc[album_id] = [artist_name, album_name]
        return int(album_id)

    def get_album_info(self, album_name: str, artist_name: str, limit: int = 20) -> dict:
        '''Given an artist name and album name, get album info'''
        album_info = {}
        album = self.network.get_album(artist_name, album_name)
        album_info['id'] = self.get_id_by_name(album_name, artist_name)
        album_info['name'] = album_name
        album_info['artist'] = artist_name
        album_info['listeners'] = album.get_listener_count()
        album_info['playcount'] = album.get_playcount()
        album_info['published'] = self._set_to_date(album.get_wiki_published_date())
        
        tags = Tag(self.network)
        tracks = Track(self.network)

        album_info['tracks'] = [tracks.get_id_by_name(track.title, track.artist.name) for track in album.get_tracks()]
        album_info['toptags'] = [(tags.get_id_by_name(top.item.name), top.weight) for top in album.get_top_tags(limit = limit)]
        
        return album_info

class Tag:

    def __init__(self, network, filepath: str = '../data/lastfm-api/tags.csv'):

        self.network = network
        self.tags_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('tag_id,tag\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')
        self.tags_df = pd.read_csv(self.tags_file, index_col='tag_id', sep = '\t')

    def write_to_csv(self):

        self.tags_df.to_csv(self.tags_file, sep = '\t')
            
    def _set_to_date(self, date: str) -> str:
        '''Given [day] [month] [year] [hour] pattern, return year-mm-dd'''
        day, month, year, _ = date.split() 
        year = year[:4]
        month = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 
                 'Mai':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
                 'Sep':'07', 'Oct':'10', 'Nov':'11', 'Dec':'12' }[month]
        return year + '-' + month + '-' + day
            
    def get_id_by_name(self, tag):
        '''Given a tag, return a tag id if it
        exists, or create one if it does not+'''

        df_tag = self.tags_df[self.tags_df.tag == tag]
        if len(df_tag) == 1:
            tag_id = df_tag.index[0]
        elif len(df_tag) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            tag_id = len(self.tags_df) + 1
            self.tags_df.loc[tag_id] = [tag]
        return int(tag_id)

    def get_tag_info(self, tag_name: str, limit: int = 20) -> dict:
        '''Given an artist name and tag name, get tag info'''
        tag_info = {}
        website = 'http://ws.audioscrobbler.com/2.0/?method=tag.getinfo&tag=' 
        website += tag_name + '&api_key=' + self.network.api_key + '&format=json'
        get_json = requests.get(website)
        tag_info_json = json.loads(get_json.content)

        tag = self.network.get_tag(tag_name)
        tag_info['id'] = self.get_id_by_name(tag_name)
        tag_info['name'] = tag_name
        tag_info['reached'] = tag_info_json['reach']
        tag_info['taggings'] = tag_info_json['total']
        tag_info['published'] = self._set_to_date(tag.get_wiki_published_date())
        
        artists = Artist(self.network)
        albums = Album(self.network)
        tracks = Track(self.network)

        tag_info['toptracks'] = [(tracks.get_id_by_name(top.item.title, top.item.artist.name), top.weight) 
                                  for top in tag.get_top_tracks(limit = limit)]
        tag_info['topartists'] = [(artists.get_id_by_name(top.item.name), top.weight) 
                                   for top in tag.get_top_artists(limit = limit)]
        tag_info['topalbums'] = [(albums.get_id_by_name(top.item.title, top.item.artist.name), top.weight) 
                                  for top in tag.get_top_albums(limit = limit)]
        
        return tag_info

class Library: 

    def __init__(self, network):
        
        self.network = network

    def get_library(self, user_name: str) -> dict:  
        '''A paginated list of all the artists in a user's library, with play
        counts and tag counts'''
        user_library = {}
        artists = Artist(self.network)

        website = 'http://ws.audioscrobbler.com/2.0/?method=library.getartists&api_key='
        website += self.network.api_key + '&user=' + user_name + '&format=json&page='
        get_json = requests.get(website)
        library_json = json.loads(get_json.content)
        pages = int(library_json['artists']['@attr']['totalPages'])

        for page in range(1,pages + 1):
            website = website.replace('page=','page='+str(page))
            get_json = requests.get(website)
            library_json = json.loads(get_json.content)
            for artist in library_json['artists']['artist']:
                user_library[artists.get_id_by_name(artist['name'])] = int(artist['playcount'])
            time.sleep(5)
            
        return user_library

class Geo:

    def __init__(self, network):

        self.network = network

    def get_country_top(self, country: str, limit: int = 50) -> dict:
        '''Get top tracks and artists in the country'''

        artists = Artist(self.network)
        tracks = Track(self.network)

        country_info = {}
        country_info['toptracks'] = [(tracks.get_id_by_name(top.item.title, top.item.artist.name), top.weight) 
                                  for top in self.network.get_geo_top_tracks(country, limit = limit)]
        country_info['topartists'] = [(artists.get_id_by_name(top.item.name), top.weight) 
                                   for top in self.network.get_geo_top_artists(country, limit = limit)]
        return country_info