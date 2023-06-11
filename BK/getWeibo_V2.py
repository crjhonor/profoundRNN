"""
my UID: 1936637103
UID: 1875034341 华尔街见闻app
UID: 1640337222 环球市场播报

URL: https://m.weibo.cn/api/container/getIndex?type=uid&value=1936637103&containerid=1076031936637103
URL: https://m.weibo.cn/api/container/getIndex?containerid=102803&openApp=0
"""
from pyarrow import null

"""
At version 2.0, I am selecting UID:1630447222 as the most interested content to generate weibo features.
"""

"""
=PART I. Daily maintenance.=============================================================================================
=Crawling all the interested UIDs, and every card is stored into database. MID is the primary key to generate integrity=
========================================================================================================================
"""

import datetime
import requests
import json
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
# # Try to use mariaDB to store the weibo data.=========================================================================
import pymysql

class mariaDBstorage:
    def __init__(self, uid):
        self.uid = uid

    def databaseShowall(self, uid):
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd="275699", db='weibo')
        cur = conn.cursor()
        cur.execute("SELECT * FROM weibo_tbl WHERE UID = '" + str(uid) + "'")
        readList = []
        for read in cur:
            readList.append(read)
        cur.close()
        conn.close()
        return readList

    def weiboInsert(self, uid, mid, created_at, text):
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd="275699", db='weibo')
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO weibo_tbl(UID, MID, CREATED_AT, TEXT) VALUES (%s, %s, %s, %s)"
                try:
                    cursor.execute(sql, (uid, mid, created_at, text))
                except pymysql.err.IntegrityError:
                    conn.commit()
                finally:
                    conn.commit()
        finally:
            conn.close()

    def getTodate(self):
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd="275699", db='weibo')
        databaseRead = []
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM weibo_tbl WHERE UID = '" + str(uid) + "' ORDER BY CREATED_AT DESC"
                cursor.execute(sql)
                for read in cursor:
                    databaseRead.append(read)
            conn.commit()
        finally:
            conn.close()
        toDate = databaseRead[0][2]
        return toDate

# # Crawling back to a certain date.====================================================================================
class CrawlWeibo:
    def __init__(self, uid, fromDate, toDate, mode):
        self.fromDate = fromDate
        self.toDate = toDate
        self.uid = uid
        self.mode = mode
        self.ms = mariaDBstorage(self.uid)
        if mode == "update":
            self.toDate = self.ms.getTodate()
        self.results = self.getCards(self.toDate)
        self.returnResults = self.storeTodatabase()

    def getCards(self, toDate):
        id = self.uid
        page = 0
        list_cards = []
        while True:
            page = page + 1
            #print('Crawling through page{} cards'.format(page))
            url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id + '&containerid=107603' \
                  + id + '&page=' + str(page)
            response = requests.get(url)
            ob_json = json.loads(response.text)
            list_cards.append(ob_json['data']['cards'])
            time.sleep(2)
            #print('Wait for 2 seconds.')

            try:
                for i in ob_json['data']['cards']:
                    if pd.to_datetime(i['mblog']['created_at']).date() == toDate.date() + datetime.timedelta(
                            days=-1) and page != 1:
                        return list_cards
                        break
            except IndexError:
                return list_cards
                break

    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub('r[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

    def storeTodatabase(self):
        results = self.results
        fromDate = self.fromDate
        toDate = self.toDate
        mode = self.mode
        if mode == 'startup':
            for page in results:
                for card in page:
                    try:
                        x = pd.to_datetime(card['mblog']['created_at'])
                        if x.date() >= toDate.date() and x.date() <= fromDate.date() + datetime.timedelta(days=-1):
                            uid = card['mblog']['user']['id']
                            mid = card['mblog']['mid']
                            created_at = pd.to_datetime(card['mblog']['created_at']).to_pydatetime()
                            text = self.strip_html_tags(card['mblog']['text'])
                            self.ms.weiboInsert(uid, mid, created_at, text)
                    except KeyError:
                        continue
        if mode == 'update':
            # get the day before today as fromDate.
            fromDate = datetime.datetime.now() + datetime.timedelta(days=-1)
            try:
                assert toDate.date() < fromDate.date()
                for page in results:
                    for card in page:
                        try:
                            x = pd.to_datetime(card['mblog']['created_at'])
                            if x.date() >= toDate.date() and x.date() <= fromDate.date():
                                uid = card['mblog']['user']['id']
                                mid = card['mblog']['mid']
                                created_at = pd.to_datetime(card['mblog']['created_at']).to_pydatetime()
                                text = self.strip_html_tags(card['mblog']['text'])
                                self.ms.weiboInsert(uid, mid, created_at, text)
                        except KeyError:
                            continue
            except AssertionError:
                print('\nYou might have already done the update before as the database is up to date.\n')
        print("\nUpdate from UID:{} is DONE!".format(self.uid), "."*200, "\n")
        return null

'''
If in the startup of mariaDB, run the following codes.
'''
"""
uid = '1640337222'
fromDate = pd.to_datetime("2022-06-11")
toDate = pd.to_datetime('2022-05-15')
cl_one = CrawlWeibo(uid, fromDate, toDate, mode = 'startup')
uid = '1875034341'
cl_two = CrawlWeibo(uid, fromDate, toDate, mode = 'startup')
"""

'''
In daily maintenance of the database, run the following codes as only inserting data into the database.
'''
uid = '1640337222'
fromDate = datetime.datetime.now().date() - datetime.timedelta(60)
toDate = pd.to_datetime('2022-07-15')
cl_one = CrawlWeibo(uid, fromDate, toDate, mode='update')
uid = '1875034341'
cl_two = CrawlWeibo(uid, fromDate, toDate, mode='update')

"""
=PART II. Return Weibo Features.========================================================================================
=Recreate the returnFeatures class inherited from CrawlingWeibo class and using mariaDBstorage class as well. But this =
=class will generate the suitable return features to return for parent code.                                           =
========================================================================================================================
"""
class returnResults(CrawlWeibo):
    def __init__(self, uid, fromDate, toDate, mode):
        super().__init__(uid, fromDate, toDate, mode)

    def storeTodatabase(self):
        results = self.results
        fromDate = self.fromDate
        toDate = self.toDate
        # Read those already existed in database.
        readDB = self.ms.databaseShowall(self.uid)
        # Join with fresh read from weibo.
        crawlWeibo = []
        for page in results:
            for card in page:
                try:
                    x = pd.to_datetime(card['mblog']['created_at'])
                    if x.date() > toDate.date():
                        uid = card['mblog']['user']['id']
                        mid = card['mblog']['mid']
                        created_at = pd.to_datetime(card['mblog']['created_at']).to_pydatetime()
                        text = self.strip_html_tags(card['mblog']['text'])
                        crawlWeibo.append([uid, mid, created_at, text])
                except KeyError:
                    continue
        # reverse the fresh crawl as the order is reversive.
        crawlWeibo.reverse()
        returnWeibo = readDB.copy()
        for itm in crawlWeibo:
            returnWeibo.append(itm)

        return returnWeibo

import datetime
uid = '1640337222'
fromDate = pd.to_datetime(datetime.date.today())
toDate = pd.to_datetime('2022-07-15')
rR = returnResults(uid, fromDate, toDate, mode='update')
returnWeibo = rR.returnResults