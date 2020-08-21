#!/usr/bin/env python
# coding: utf-8

from create_last_database import User
import sys, getopt, os
import pylast
import pandas as pd
from __init__ import ROOT_DIR

def main(argv):
    ''' 
    Function that receive the system arguments and organize for the scrip.
    Input -h to understand the arguments. 
    '''
    api_key = ''
    api_secret = ''
    users = []
    level = None
    limit = None
    helping = '''generate_lastfm_users.py -k <api_key> -s <api_secret> -u <user_to_start> -n <level> -l <limit> '''
    try:
        opts, _ = getopt.getopt(argv,"hk:s:u:n:l:",["api_key=","api_secret=", "user=", "level=", "limit="])
    except getopt.GetoptError:
        print()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helping)
            print('Generate a csv of usernames from last.fm')
            print('----------------------------------------')
            print('api_key: provided by last.fm api')
            print('api_secret: provided by last.fm api')
            print('user_to_start: username of last.fm to start from your choice. Insert how many -u you want')
            print('level: how many levels to walk on the network. If level = 2, friends from friends are considered.')
            print('limit: the limit of friends to get from an user.')
            print('---------------------------------------- ')
            print('It will be taken approx (limit^level) users.')
            print('Remember: 50^4 = 6250000 takes long!')
            print('----------------------------------------')
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
    
    path = os.path.join(ROOT_DIR, '../data/lastfm-api')
    
    if not os.path.exists('../data'):
        os.mkdir('../data')
    if not os.path.exists(path):
        os.mkdir(path)
    
    write_users = User(network, os.path.join(path, 'users_lastfm.csv'))
    write_users.write_users_names(users, level, limit)

    get_unique(os.path.join(path, 'users_lastfm.csv'))