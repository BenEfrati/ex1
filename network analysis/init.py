from requests_oauthlib import OAuth1Session
import json
import csv
import time
from time import gmtime, strftime
from random import randint


customer_key='your-customer-key'
customer_secret='your-customer-secret'
token='your-token'
token_secret='your-token-secret'

followers_list = []


def get_twitter_ids():
    #reading the file list with the 120 kneset members
    with open('kneset_members.txt') as f:
        keneset_members = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    keneset_members = [x.strip().replace('\\','').replace("'",'') for x in keneset_members]

    #getting the twitter id for each keneset member by the result of the first search on search api
    kneset_members_with_id=[]
    for member in keneset_members:
        url = 'https://api.twitter.com/1.1/users/search.json?q='+member+'&count=1'
    print(str(len(kneset_members_with_id))+' members with id are found')
    save_to_csv('kneset_members_with_id',kneset_members_with_id)

def save_to_csv(fileName,list_to_save):
    with open(fileName+'.csv', 'wb') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['id', 'name'])
        for row in list_to_save:
            csv_out.writerow(row)

def read_csv_tuples_file(fileName):
    with open(fileName+'.csv', 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)
        list = map(tuple, reader)
    return list

def get_twitter_followers():
    counter=1
    list = read_csv_tuples_file("mk_id")
    for id in list:
        #if(counter%15==0):
        #    time.sleep(910)
        followers_url='https://api.twitter.com/1.1/friends/ids.json?user_id='+id[0]
        twitter = OAuth1Session(customer_key, customer_secret, token, token_secret)
        r = twitter.get(followers_url)
        d = json.loads(r._content)
        print("Doing now id "+str(id[0])+" number: "+str(counter))
        save_to_csv('friends'+"_"+str(id), followers_list)
        skipped=False
        print("time before sleep: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        while 'ids' not in d:
            print("sleeping..")
            timeToWait=randint(1,20)+920
            if (r._content=='{"request":"\\/1.1\\/friends\\/ids.json","error":"Not authorized."}'):
                break
            if (r._content=='{"errors":[{"message":"Rate limit exceeded","code":88}]}'):
                time.sleep(timeToWait)
                r = twitter.get(followers_url)
                d = json.loads(r._content)
            else:
                print ("skipped: "+str(id))
                skipped=True
                break;


        print("time after sleep: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        # if 'ids' not in d:
        #     break
        if (r._content == '{"request":"\\/1.1\\/friends\\/ids.json","error":"Not authorized."}'):
            continue
        if (skipped==True):
            continue
        for f_id in d['ids']:
            followers_list.append((f_id,id[0]))
        counter=counter+1
            ##getNextByCursor(id[0],d['next_cursor'],followers_url,twitter)
        print ("number of followers: " + str(len(d['ids'])) + " , followers list size: " + str(len(followers_list)))
def getNextByCursor(id,cursor_num,followers_url,twitter):
    if (cursor_num > 0):
        cursor_url = followers_url + '&cursor=' + str(cursor_num)
        cr = twitter.get(cursor_url)
        d = json.loads(cr._content)
        if len(d) > 0:
            for f_id in d:
                followers_list.append((f_id,id))
        if 'next_cursor' in d:
            getNextByCursor(d['next_cursor'], followers_url)

def getCSVfilteredByID():
    list_ids = read_csv_tuples_file("mk_id")
    #getting only the id without the name
    new_ids_list=list(list_ids[i][0] for i in range(len(list_ids)))
    list_grapg = read_csv_tuples_file("friends")
    #converting list to dict
    new_id_dict=dict((key, key) for key in new_ids_list)
    filteredList=[]
    for v in list_grapg:
      if(v[0] in new_id_dict and v[1] in new_id_dict):
          filteredList.append(v)
    save_to_csv('only_kneset_graph',filteredList)
    print str(len(set(filteredList)))
#Main:
#get_twitter_ids()
#get_twitter_followers()
#save_to_csv('friends',followers_list)
#getCSVfilteredByID()
print("done")
