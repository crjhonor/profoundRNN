"""
========================================================= PART I =======================================================
=Using receiveEmail_V2.py code to receive Email contents, and the contents are stored in database, but for the purpose =
=clearance, I am not directly fetching email contents from the database. Instead, I am still retrieving email contents =
=from email server.                                                                                                    =
========================================================================================================================
"""
import receiveEmail_V2 as RE
allEmailsList = RE.allEmailsList

"""
======================================================== PART II =======================================================
=Using baidu ernie module to sentiment the email contents. It is a direct pre-training module included in paddlehub.   =
=Although the ernie module within paddlehub package has a few backdrops, including occupying huge amount of memory, but=
=it works fine with English words.                                                                                     =
========================================================================================================================
"""

import paddlehub as hub
positive_significantLevel = 0.6 # Set a significant level for sentiment results
negative_significantLevel = 0.6
import re
from bs4 import BeautifulSoup
import nltk
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import textNormalizer_V2 as tn

class sentimentCode:
    def __init__(self, sentimentData, posi_sigLevel, nega_sigLevel): # sentimentData is a list
        self.senta = hub.Module(name="senta_bilstm")
        self.sentimentData = sentimentData
        self.posi_sigLevel = posi_sigLevel
        self.nega_sigLevel = nega_sigLevel

    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, 'html.parser')
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub('r[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

    def testText(self): # Preparing the test text for sentiment test
        rawData = self.sentimentData
        re_testText = []
        default_st = nltk.sent_tokenize
        for singleList in rawData:
            # print(singleList, "..........")
            # ... dealing with plain text ...
            # ... Break down into sentences using nltk ...
            try:
                tmp_x = singleList['plainText']
            except KeyError:
                #print('No Plain Text')
                continue
            plainText_sentences = tn.tn_preProcessing.pre_process_document(tmp_x)
            try:
                sentimentSingledict = {"DATE":singleList['Date'],
                                       "TEXT":plainText_sentences}
            except IndexError:
                sentimentSingledict = {"DATE":singleList['Date'],
                                       "TEXT":'NaN'}
                continue
            re_testText.append(sentimentSingledict)
            # ... dealing with html text ...
            try:
                content = singleList['htmlText']
            except KeyError:
                #print("No HTML Text")
                continue
            clean_content = self.strip_html_tags(content)
        return re_testText

    def totalResult(self):
        test_Text = self.testText()
        re_totalResult = []
        for i in tqdm(range(len(test_Text)), ncols=150, desc="Emails Sentiment", colour="green"):
            singleText = test_Text[i]
            input_dict = singleText['TEXT']
            results = self.senta.sentiment_classify(texts=input_dict, use_gpu = True, batch_size=1)
            singleTestresult={"DATE":singleText['DATE'],
                              "RESULTS":results}
            re_totalResult.append(singleTestresult)
        print("\n", "Email contents sentiment is DONE!", "."*200, "\n")
        return re_totalResult

senti = sentimentCode(allEmailsList, positive_significantLevel, negative_significantLevel)
totalResults = senti.totalResult()
#print(totalResults)