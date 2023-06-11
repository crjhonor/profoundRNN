"""
Run this cook code first in Mission II to obtain email and weibo features as they are very much time-consuming.

For the convenience of applying multiple models, amid the prolonging processes of reading and analysing data, this code
is to perform all the necessary steps to obtain the data and save them into files within the /dataForlater directory for
modeling.

Of course, several improvements are made:
1. Now I am not reading all the emails from the server, instead, using mariDB, I am only reading emails within the
current month from the email server and read the previous ones form the mariDB.
2. Date back to which started the reading of email and weibo is now automatically calculated for a month and so on tp
improve efficiency.
"""
import os
from pathlib import Path
import json

dataDirName = "dataForlater"
forLaterfilename = Path(Path(os.getcwd()), dataDirName, "II_seq2seq_moon2sun_cook_email_feature_forlater.json")

# Reading email, saving to database and sentiment analysis.
import II_sentimentCode as SC
totalResults = SC.totalResults

import II_sentiAnalysis as sa
saReturn = sa.dataProcess(totalResults)
emailFeatures_df = saReturn.dataFinal
emailFeatures_df.to_json(forLaterfilename)
print("EMAIL FEATURE IS SAVED INTO ", forLaterfilename, "!", "."*10)

# I am going to add weibo as more features.
dataDirName = "dataForlater"
forLaterfilename = Path(Path(os.getcwd()), dataDirName, "II_seq2seq_moon2sun_cook_weibo_feature_forlater.json")

import II_weiboSentimentcode as wsc
weiboTotalresults = wsc.weiboTotalresults

import II_weiboSentianalysis as wsa
wsaReturn = wsa.weiboDataprocess(weiboTotalresults)
weiboFeatures_df = wsaReturn.dataFinal
weiboFeatures_df.to_json(forLaterfilename)
print("WEIBO FEATURE IS SAVED INTO ", forLaterfilename, "!", "."*10)
print("Done!", "."*180)

