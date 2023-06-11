"""
========================================================= PART I =======================================================
=Using getWeibo_V2.py code to crawl down weibo contents, datas are stored in mariaDB database. There are several UIDs  =
=which I am interested in, but at this version, only one UID with the most appropriate contents are selected to derive =
=feature results for machine learning.                                                                                 =
========================================================================================================================
"""
import II_getWeibo as GW
returnWeibo = GW.returnWeibo

"""
======================================================== PART II =======================================================
=Using baidu paddlehub's bilstm module to sentiment the weibo contents. Before sentiment, contents of every single wei-=
=bo is split using functions in <textNormalizer_V2.py>. First of all, I think of the method provided by nltk package   =
=with Chinese kernel. Let's try this method first.
========================================================================================================================
"""

import paddlehub as hub
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from BK import textNormalizer_V2 as tn


class weiboSentimentcode:
    def __init__(self, sentimentData):
        self.sentimentData = sentimentData
        self.senta_bilstm = hub.Module(name="senta_bilstm")

    def testText(self): # Preparing the test text for sentiment test
        rawData = self.sentimentData
        return_testText = []
        for singleList in rawData:
            # Break down content text into sentences using textNormalizer_V2.py
            tmp_x = singleList[3]
            sentences = tn.tn_preProcessing.weiboCutsentence(tmp_x)
            sentimentSingledict = {
                "DATE": singleList[2],
                "TEXT": sentences
            }
            return_testText.append(sentimentSingledict)
        return return_testText

    def totalResults(self):
        testText = self.testText()
        return_totalResult = []
        print("\n", "Beginning to sentiment weibo contents.", "."*200, "\n")
        for i in tqdm(range(len(testText)), ncols=150, desc="Weibos Sentiment", colour="magenta"):
            singleText = testText[i]
            input_dict = singleText['TEXT']
            bilstm_results = self.senta_bilstm.sentiment_classify(texts=input_dict,
                                                                  use_gpu=True,
                                                                  batch_size=1)
            singleResult = {
                "DATE":singleText['DATE'],
                "RESULTS":bilstm_results
            }
            return_totalResult.append(singleResult)
        print("\n", "Weibo sentiment is DONE!", "."*200, "\n")
        return return_totalResult

weiboSenti = weiboSentimentcode(returnWeibo)
weiboTotalresults = weiboSenti.totalResults()
#print(weiboTotalresults)
