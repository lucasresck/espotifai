#!/usr/bin/env python
# coding: utf-8

import os
import json

import pylast 
import requests 
import pandas as pd

class User:

    def __init__(self, network):

        self.network = network

    def write_users_names(self, user_list: list, filepath: str, level: int = 5, limit: int = 20):
        '''From a list of usernames, write a netwotk list of users in file.
           The file should be a csv. '''
        if not os.path.exists(filepath): 
            try: 
                with open(filepath, 'w') as f:
                    f.write('user_id, user_name\n')
            except: 
                raise Exception('Filename not found. Please, insert all path.')
        with open(filepath, 'r') as f:
            line = f.readline()
            while line != '':                               # Get the last user id written
                last_line = line 
                line = f.readline()
            try: 
                self.last_user_id = int(last_line.split(',')[0])
            except ValueError:
                self.last_user_id = 0
        ids_users = [str(self.last_user_id + 1 + i) + ',' + user_list[i] for i in range(len(user_list))]
        self._write_in_file(ids_users, filepath)
        self.last_user_id += len(user_list)
        for username in user_list:
            user = self.network.get_user(username = username)
            self._build_network(user, level, limit, filepath)

    def _get_friends(self, user, limit: int) -> list:
        '''Get [limit] number of friends and return a list'''
        try: 
            friends = user.get_friends(limit = limit)
        except pylast.WSError: 
            return []
        name_friends = [link.name for link in friends]
        return name_friends 

    def _build_network(self, user, level: int, limit: int, filepath: str): 
        '''Get the frients and build a network'''
        friends = self._get_friends(user, limit = limit)
        id_and_friends = [str(self.last_user_id + i + 1) + ',' + friends[i] for i in range(len(friends))] 
        self._write_in_file(id_and_friends, filepath)
        self.last_user_id += len(friends)
        if level > 1:
            for friend in friends: 
                self._build_network(self.network.get_user(friend), level - 1, limit, filepath)
        else:
            return
        print('Level {} compĺeted'.format(level))

    def _write_in_file(self, lst: list, filepath: str):
        '''Write a list in a filepath. Auxliary funcion'''
        with open(filepath, 'a') as f: 
            for i in lst:
                f.write(i)
                f.write('\n')

    def get_user_info(self, username: str) -> dict: 
        '''Get a lot of user information from last.fm and build it in a dictionary'''

        website = 'http://ws.audioscrobbler.com/2.0/?method=user.getinfo&user='
        website += username 
        website += '&api_key='
        website += self.network.api_key
        website += '&format=json'
        get_json = requests.get(website)
        user_info_json = json.loads(get_json.content)['user']
        
        user = self.network.get_user(username)
        user_info = {}
        user_info['name'] = username
        user_info['subscriber'] = int(user.is_subscriber())
        user_info['playcount'] = user.get_playcount()
        user_info['registered_since'] = user.get_registered()
        user_info['country'] = None
        user_info['age'] = int(user_info_json['age'])
        user_info['playlists'] = user_info_json['playlists']
        user_info['gender'] = user_info_json['gender']
        if user.get_country():
            user_info['country'] = user.get_country().name

        tracks = Track(self.network)
        artists = Artist(self.network)
        albums = Album(self.network)
        tags = Tag(self.network)

        user_info['loved_tracks'] = [tracks.get_id_by_name(loved.track.title, loved.track.artist.name) 
                                     for loved in user.get_loved_tracks()]
        user_info['recent_tracks'] = [tracks.get_id_by_name(recent.track.title, recent.track.artist.name) 
                                     for recent in user.get_recent_tracks()]
        user_info['top_tracks'] = [(tracks.get_id_by_name(top.item.title, top.item.artist.name),top.weight) 
                                     for top in user.get_top_tracks()]
        user_info['top_tags'] = [(tags.get_id_by_name(top.item.name), top.weight) 
                                  for top in user.get_top_tags()]
        user_info['top_albums'] = [(albums.get_id_by_name(top.item.title, top.item.artist.name), top.weight)
                                     for top in user.get_top_albums()]
        user_info['top_artists'] = [(artists.get_id_by_name(top.item.name), top.weight) 
                                     for top in user.get_top_artists()]
        
        return user_info

class Track:

    def __init__(self, network, filepath: str = '../data/tracks.csv'):

        self.network = network
        self.tracks_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('track_id,artist_name,track_name\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')

    def get_id_by_name(self, track_name, artist_name):
        '''Given an artist name and track name, return a track id if it
        exists, or create one if it does not;'''

        tracks = pd.read_csv(self.tracks_file, index_col='track_id')
        df_track = tracks[(tracks.artist_name == artist_name) & (tracks.track_name == track_name)]
        if len(df_track) == 1:
            track_id = df_track.index[0]
        elif len(df_track) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            track_id = len(tracks) + 1
            with open(self.tracks_file, 'a') as f:
                f.write(str(track_id) + ',' + artist_name + ',' + track_name + '\n')
        return track_id

class Artist:

    def __init__(self, network, filepath: str = '../data/artists.csv'):

        self.network = network
        self.artists_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('artist_id,artist_name\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')

    def get_id_by_name(self, artist_name):
        '''Given an artist name return a artist_id if it
        exists, or create one if it does not;'''

        artists = pd.read_csv(self.artists_file, index_col='artist_id')
        df_artist = artists[artists.artist_name == artist_name]
        if len(df_artist) == 1:
            artist_id = artist_id.index[0]
        elif len(df_artist) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            artist_id = len(artists) + 1
            with open(self.artists_file, 'a') as f:
                f.write(str(artist_id) + ',' + artist_name + '\n')
        return artist_id

class Album:

    def __init__(self, network, filepath: str = '../data/albums.csv'):

        self.network = network
        self.albums_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('album_id,artist_name,album_name\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')
            
    def get_id_by_name(self, album_name, artist_name):
        '''Given an artist name and album name, return a album id if it
        exists, or create one if it does not;'''

        albums = pd.read_csv(self.albums_file, index_col='album_id')
        df_album = albums[(albums.artist_name == artist_name) & (albums.album_name == album_name)]
        if len(df_album) == 1:
            album_id = df_album.index[0]
        elif len(df_album) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            album_id = len(albums) + 1
            with open(self.albums_file, 'a') as f:
                f.write(str(album_id) + ',' + artist_name + ',' + album_name + '\n')
        return album_id

class Tag:

    def __init__(self, network, filepath: str = '../data/tags.csv'):

        self.network = network
        self.tags_file = filepath
        if not os.path.exists(filepath):
            try: 
                with open(filepath, 'w') as f:
                    f.write('tag_id,tag\n')
            except: 
                raise Exception('Problem in creating file. Maby the folder does not exist.')
            
    def get_id_by_name(self, tag,):
        '''Given a tag, return a tag id if it
        exists, or create one if it does not;'''

        tags = pd.read_csv(self.tags_file, index_col='tag_id')
        df_tag = tags[tags.tag == tag]
        if len(df_tag) == 1:
            tag_id = df_tag.index[0]
        elif len(df_tag) > 1:
            raise Exception('Two tracks with same artist name and same track name')
        else: 
            tag_id = len(tags) + 1
            with open(self.tags_file, 'a') as f:
                f.write(str(tag_id) + ',' + tag + '\n')
        return tag_id