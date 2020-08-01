#!/usr/bin/env python
# coding: utf-8

from create_last_database import User
import sys, getopt
import pylast
import pandas as pd

def main(argv):
    api_key = ''
    api_secret = ''
    users = []
    level = None
    limit = None
    try:
        opts, _ = getopt.getopt(argv,"hk:s:u:n:l:",["api_key=","api_secret=", "user=", "level=", "limit="])
    except getopt.GetoptError:
        print('generate_lastfm_users.py -k <api_key> -s <api_secret> -u <user_to_start>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('generate_lastfm_users.py -k <api_key> -s <api_secret> -u <user_to_start>')
            sys.exit()
        elif opt in ("-k", "--api_key"):
            api_key = arg
        elif opt in ("-s", "--api_secret"):
            api_secret = arg
        elif opt in ("-u", "--user"):
            users.append(arg)
        elif opt in ('-n', '--level'):
            level  = arg
        elif opt in ('-l','--limit'):
            limit = arg
    return api_key, api_secret, users, level, limit

def get_unique(users_path):

    users_df = pd.read_csv(users_path)
    users = users_df['user_name'].unique()
    users_unique_df = pd.DataFrame({'user_id': list(range(1, len(users)+1)), 
                                    'user_name': users})
    users_unique_df.to_csv(users_path, index = False)
    print('DONE')    

if __name__ == '__main__':

    API_KEY, API_SECRET, users, level, limit = main(sys.argv[1:])
    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)
    if level:
        level = int(level)
    if limit:
        limit = int(limit)

    #write_users = User(network, '../data/lastfm-api/users_lastfm.csv')
    #write_users.write_users_names(users, level, limit)

    get_unique('../data/lastfm-api/users_lastfm.csv')