# # Using receiveEmail.py code to receive Email contents into a list.===================================================
import receiveEmail as RE
allEmailsList = RE.allEmailsList


# # Using sentimenCode.py to sentiment the contents of emails.=========================================================
import paddlehub as hub
positive_significantLevel = 0.6 # Set a significant level for sentiment results
negative_significantLevel = 0.6
import re
from bs4 import BeautifulSoup
import nltk
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import textNormalizer as tn

class sentimentCode:
    def __init__(self, sentimentData, posi_sigLevel, nega_sigLevel): # sentimentData is a list
        self.senta = hub.Module(name="ernie_skep_sentiment_analysis")
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
                print('No Plain Text')
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
                print("No HTML Text")
                continue
            clean_content = self.strip_html_tags(content)
        return re_testText

    def totalResult(self):
        print('Sentimenting begins!..........')
        test_Text = self.testText()
        prt_t = len(test_Text)
        prt_i = 1
        re_totalResult = []
        for singleText in test_Text:
            input_dict = singleText['TEXT']
            results = self.senta.predict_sentiment(texts=input_dict, use_gpu = True)
            singleTestresult={"DATE":singleText['DATE'],
                              "RESULTS":results}
            re_totalResult.append(singleTestresult)
            print(f"...{prt_i}/{prt_t} IS DONE...........\n".format(prt_i, prt_t))
            prt_i += 1
        return re_totalResult

senti = sentimentCode(allEmailsList, positive_significantLevel, negative_significantLevel)
totalResults = senti.totalResult()
# print(totalResults)