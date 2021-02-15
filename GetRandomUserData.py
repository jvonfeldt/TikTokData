import datetime as dt
import pickle
from TikTokApi import TikTokApi
verifyFp="verify_kl5mqu5r_pQYvivd9_FhZG_4ohY_AjIs_eqBY0W5xdRfa"
api = TikTokApi.get_instance(custom_verifyFp=verifyFp)

starter1 = api.getUser(username='gatorchris1')
starter2 = api.getUser(username='marawaters')
starter3 = api.getUser(username='galeriaojosdelarte')
starter4 = api.getUser(username='jasonderulo')
starter5 = api.getUser(username='odgkenzo')
starter6 = api.getUser(username='katiejlinz')
starter7 = api.getUser(username='laurrenmf')
starter8 = api.getUser(username='kidlyza')
starter9 = api.getUser(username='nicoleolivayt')
starter10 = api.getUser(username='marahwaters')
starter11 = api.getUser(username='nicky19822020')
starter12= api.getUser(username='sarahmagusara')
starter13= api.getUser(username='zoelaverne')

##starters = [starter1,starter2,starter3]
##starters = [starter4,starter5,starter6]
starters = [starter1,starter2,starter3,starter4,starter5,starter6,starter7,starter8,starter9,starter10,starter11,starter12,starter13]


starting_author_info = {}
for user in starters:
    author_info = {}
    tiktokid = user['userInfo']['user']['id']
    author_info['stats'] = user['userInfo']['stats']
    author_info['username'] = user['uniqueId']
    starting_author_info[tiktokid] = author_info

all_authors = {}


for user in starting_author_info:
    suggested_users = api.getSuggestedUsersbyIDCrawler(count=30,startingId=user)
    for sug_users in suggested_users:
        # print(sug_users)
        tiktokid = sug_users['id']
        fancount = sug_users['extraInfo']['fans']
        secuid = sug_users['extraInfo']['secUid']
        all_authors[tiktokid] = {'fans': fancount,'secuid': secuid}

all_tiktoks = {}
starttime = dt.datetime.strptime('2020-01-01','%Y-%m-%d').timestamp()

for user in all_authors:
    uid = user
    secuid = all_authors[user]['secuid']
    fancount = all_authors[user]['fans']
    tiktoklist = api.userPosts(userID=uid,secUID=secuid,count=50,minCursor=starttime)
    #print(tiktoklist)
    for t in tiktoklist:
        likes = t['stats']['diggCount']
        views = t['stats']['playCount']
        vidid = t['id']
        posttime = t['createTime']
        all_tiktoks[vidid] = {'user': uid,'fans': fancount, 'views': views,'posttime': posttime}

with open('tiktokmegalist.pickle', 'wb') as handle:
    pickle.dump(all_tiktoks, handle, protocol=pickle.HIGHEST_PROTOCOL)




