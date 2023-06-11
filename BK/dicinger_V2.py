"""
=PART I, Features extraction.===========================================================================================
= 1. Email contents sentiment results are derived from <sentimentCode_V2.py> and then sent to <sentiAnalysis_V2.py> to =
= process in order to get the emailFeatures.                                                                           =
= 2. Then contents of weibo is crawled and downloaded using <weiboSentimentcode_V2.py>, inside which will import       =
= <getWeibo_V2.py> to crawl down weibo user's contents with known uids and store then into database. Consider of capac-=
= ity and necessity, I am only using one UID currently to derive contents and results. Using also                      =
= <weiboSentianalysis_v2.py> to process the features to generate weiboFeatures.                                        =
= I save all the features to json file for later analysis as R is better for data analysis I assumed, and it has plenty=
= of deep learning modules. I should try it later.                                                                     =
= 3. Yields features are read using <getYields_V2.py>. This code directly read from yields.xls stored in the data dire-=
= ctory and it also calculates the log return on daily bases.
========================================================================================================================
"""

import sentimentCode_V2 as SC
totalResults = SC.totalResults

# Save email sentiment results out to json file and leave the analysis job to R.
import os
from pathlib import Path
import json

dataDirName = "dataForlater"
outputResultfilename = Path(Path(os.getcwd()), dataDirName, "emailSentimentresult.json")
saveTofile = totalResults.copy()
# Dumping to json file needs a list
resultJsondata = json.dumps(saveTofile)

try:
    with open(outputResultfilename, 'w') as fileObject:
        json.dump(resultJsondata, fileObject)
except FileNotFoundError:
    os.mkdir(Path(os.getcwd(), dataDirName))
    with open(outputResultfilename, 'w') as fileObject:
        json.dump(resultJsondata, fileObject)
print("EMAIL SENTIMENT RESULTS ARE SAVED INTO ", outputResultfilename, "!")

# sentiAnalysis_V2.py is mostly coded for Results preprocessing after the sentiment. And, further deep learning is
# approached with label and feature designing and machine learning.

import sentiAnalysis_V2 as sa
saReturn = sa.dataProcess(totalResults)
emailFeatures_df = saReturn.dataFinal

# in version 2, I am going to add weibo as more features.
import weiboSentimentcode_V2 as wsc
weiboTotalresults = wsc.weiboTotalresults

outputResultfilename = Path(Path(os.getcwd()), dataDirName, "weiboSentimentresult.json")
saveTofile = []
# datetime object is not dump-able to json file so I have to convert them into strings.
for itm in weiboTotalresults:
    saveTofile_singlelist = {"DATE":str(itm['DATE']),
                             "RESULTS":itm['RESULTS']}

# Dumping to json file needs a list
resultJsondata = json.dumps(saveTofile)

try:
    with open(outputResultfilename, 'w') as fileObject:
        json.dump(resultJsondata, fileObject)
except FileNotFoundError:
    os.mkdir(Path(os.getcwd(), dataDirName))
    with open(outputResultfilename, 'w') as fileObject:
        json.dump(resultJsondata, fileObject)
print("WEIBO SENTIMENT RESULTS ARE SAVED INTO ", outputResultfilename, "!")
# Getting the weiboTotalresults undergoing analysis to generate weibo features.
import weiboSentianalysis_V2 as wsa
wsaReturn = wsa.weiboDataprocess(weiboTotalresults)
weiboFeatures_df = wsaReturn.dataFinal

# in version 1, I was adding bond yields as more features.
import getYields_V2 as gt
yieldsWanted = ['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry']
gtReturn = gt.readingYields(yieldsWanted)
featuresYieldsDL_df = gtReturn.returnFeatures

"""
=PART II, Deep Learning.================================================================================================
=There are three different types of deep learning network models to be implemented. Although they are quite pre-mature =
=I am counting on it to generate odds for dicing.                                                                      =
=1.<Simple Linear Model.>                                                                                              =
=2.<Simple Complete Learning Network Model.>                                                                           =
=3.<Simple Convolutional Network Model.>
========================================================================================================================
"""
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'PP0', 'L0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ["SCM", 'BU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'MA0', 'RU0']

import getLabels_V2 as gl
from tqdm import tqdm
import simpleDeeplearning_V2 as sdl
import simpleCompleteln_V2 as scln
import simpleConvolutionnetwork_V2 as scnn
import tkinter

def run_CU0():
    countThreshold = var_countThreshold.get()
    countType = var_countType.get()
    indexWanted = indexWanted_CU0
    maxCount = 0
    while True:
        positiveCount = 0
        negativeCount = 0
        # Implement simple linear model.
        sdlResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in sdlReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            sdlResults.append(sdlReturn.results)

        # Implement simple complete learning network.
        sclnResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in sclnReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            sclnResults.append(sclnReturn.results)

        # The 3rd network, simple convolution network.
        """
        I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
        data, [batch_size, 1, 3, 4], the '1' is a channel.
        """
        scnnResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in scnnReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            scnnResults.append(scnnReturn.results)

        # Showing the counting results in the text box.
        countResults = "\nCounting Results:\n" +\
                       "Positive Count: " + str(positiveCount) + "\n" + \
                       "Negative Count: " + str(negativeCount) + "\n"
        print(countResults)
        print("Total Count: " + str(positiveCount*(1-countType)+negativeCount*countType) + "\n")
        if positiveCount*(1-countType)+negativeCount*countType > maxCount:
            """
            =PART III, Deep Learning Results Output.====================================================================
            =Using tkinter for a more pretty output.                                                                   =
            ============================================================================================================
            """
            # In order to use tkinter for pretty output, I need to package all the results into one string.
            outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType==1 else 'POSITIVE') \
                           + '\nTOTAL COUNT: ' + str(positiveCount*(1-countType)+negativeCount*countType)
            for i in range(len(indexWanted)):
                singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                               '*' * 100,
                                               str(sdlResults[i]),
                                               '.' * 100,
                                               str(sclnResults[i]),
                                               str(scnnResults[i]),
                                               "=" * 100])
                outputString = "\n".join([outputString, singleIndexresult])
            text.delete(1.0, "end")
            text.insert("end", outputString)
            maxCount = positiveCount*(1-countType)+negativeCount*countType

        if positiveCount*(1-countType)+negativeCount*countType >= countThreshold:
            break

def run_RB0():
    countThreshold = var_countThreshold.get()
    countType = var_countType.get()
    indexWanted = indexWanted_RB0
    maxCount = 0
    while True:
        positiveCount = 0
        negativeCount = 0
        # Implement simple linear model.
        sdlResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in sdlReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            sdlResults.append(sdlReturn.results)

        # Implement simple complete learning network.
        sclnResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in sclnReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            sclnResults.append(sclnReturn.results)

        # The 3rd network, simple convolution network.
        """
        I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
        data, [batch_size, 1, 3, 4], the '1' is a channel.
        """
        scnnResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in scnnReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            scnnResults.append(scnnReturn.results)

        # Showing the counting results in the text box.
        countResults = "\nCounting Results:\n" +\
                       "Positive Count: " + str(positiveCount) + "\n" + \
                       "Negative Count: " + str(negativeCount) + "\n"
        print(countResults)
        print("Total Count: " + str(positiveCount*(1-countType)+negativeCount*countType) + "\n")
        if positiveCount*(1-countType)+negativeCount*countType > maxCount:
            """
            =PART III, Deep Learning Results Output.====================================================================
            =Using tkinter for a more pretty output.                                                                   =
            ============================================================================================================
            """
            # In order to use tkinter for pretty output, I need to package all the results into one string.
            outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType==1 else 'POSITIVE') \
                           + '\nTOTAL COUNT: ' + str(positiveCount*(1-countType)+negativeCount*countType)
            for i in range(len(indexWanted)):
                singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                               '*' * 100,
                                               str(sdlResults[i]),
                                               '.' * 100,
                                               str(sclnResults[i]),
                                               str(scnnResults[i]),
                                               "=" * 100])
                outputString = "\n".join([outputString, singleIndexresult])
            text.delete(1.0, "end")
            text.insert("end", outputString)
            maxCount = positiveCount*(1-countType)+negativeCount*countType

        if positiveCount*(1-countType)+negativeCount*countType >= countThreshold:
            break

def run_SCM():
    countThreshold = var_countThreshold.get()
    countType = var_countType.get()
    indexWanted = indexWanted_SCM
    maxCount = 0
    while True:
        positiveCount = 0
        negativeCount = 0
        # Implement simple linear model.
        sdlResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE LINEAR NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            sdlReturn = sdl.simpleDeeplearning(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in sdlReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            sdlResults.append(sdlReturn.results)

        # Implement simple complete learning network.
        sclnResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE COMPLETE NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            sclnReturn = scln.simpleCompleteln(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in sclnReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            sclnResults.append(sclnReturn.results)

        # The 3rd network, simple convolution network.
        """
        I need a 4x3 features inorder to use the convolution network. And features array should be transformed into a 4D 
        data, [batch_size, 1, 3, 4], the '1' is a channel.
        """
        scnnResults = []
        for i in tqdm(range(len(indexWanted)), ncols=100, desc="SIMPLE CONVOLUTION NETWORK", colour="blue"):
            ind = [indexWanted[i]]
            glReturn = gl.getLabels(indexWanted=ind)
            labelsDL_df = glReturn.returnLabels
            scnnReturn = scnn.simpleConvolutionnetwork(ind, emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
            for var in scnnReturn.results.iloc[0].values[2:]:
                if var >= 0:
                    positiveCount = positiveCount + 1
                else:
                    negativeCount = negativeCount + 1
            scnnResults.append(scnnReturn.results)

        # Showing the counting results in the text box.
        countResults = "\nCounting Results:\n" +\
                       "Positive Count: " + str(positiveCount) + "\n" + \
                       "Negative Count: " + str(negativeCount) + "\n"
        print(countResults)
        print("Total Count: " + str(positiveCount*(1-countType)+negativeCount*countType) + "\n")
        if positiveCount*(1-countType)+negativeCount*countType > maxCount:
            """
            =PART III, Deep Learning Results Output.====================================================================
            =Using tkinter for a more pretty output.                                                                   =
            ============================================================================================================
            """
            # In order to use tkinter for pretty output, I need to package all the results into one string.
            outputString = '\nCOUNT TYPE: ' + ('NEGATIVE' if countType==1 else 'POSITIVE') \
                           + '\nTOTAL COUNT: ' + str(positiveCount*(1-countType)+negativeCount*countType)
            for i in range(len(indexWanted)):
                singleIndexresult = '\n'.join(["\nDeep Learning results of " + indexWanted[i],
                                               '*' * 100,
                                               str(sdlResults[i]),
                                               '.' * 100,
                                               str(sclnResults[i]),
                                               str(scnnResults[i]),
                                               "=" * 100])
                outputString = "\n".join([outputString, singleIndexresult])
            text.delete(1.0, "end")
            text.insert("end", outputString)
            maxCount = positiveCount*(1-countType)+negativeCount*countType

        if positiveCount*(1-countType)+negativeCount*countType >= countThreshold:
            break

win = tkinter.Tk()
win.title('Deep Learning Results')
width = 1600
height = 1000
screenwidth = win.winfo_screenwidth()
screenheight = win.winfo_screenheight()
root_str = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
win.geometry(root_str)
win.maxsize(1600, 1000)

tkinter.Label(win, text = 'ACTIONS => ').place(relx=0.0, rely=0.0, relwidth=0.1, relheight=0.1)
btnCU0 = tkinter.Button(win, text="RUN_COPPER", command=run_CU0)
btnCU0.place(relx=0.1, rely=0.0, relwidth=0.3, relheight=0.1)
btnRB0 = tkinter.Button(win, text="RUN_REBAR", command=run_RB0)
btnRB0.place(relx=0.4, rely=0.0, relwidth=0.3, relheight=0.1)
btnSCM = tkinter.Button(win, text="RUN_CRUDE", command=run_SCM)
btnSCM.place(relx=0.7, rely=0.0, relwidth=0.3, relheight=0.1)

tkinter.Label(win, text = 'COUNT THRESHOLD => ').place(relx=0.1, rely=0.1, relwidth=0.2, relheight=0.1)
var_countThreshold = tkinter.IntVar()
var_countThreshold.set(value=39)
entry_countThreshold = tkinter.Entry(win, textvariable=var_countThreshold, font=('Arial', 20))
entry_countThreshold.place(relx=0.3, rely=0.1, relwidth=0.1, relheight=0.1)

tkinter.Label(win, text = 'COUNT TYPE CHECKED => ').place(relx=0.5, rely=0.1, relwidth=0.2, relheight=0.1)
var_countType = tkinter.IntVar()
var_countType.set(value=0)
cBox = tkinter.Checkbutton(win, text="NEGATIVE", variable=var_countType, onvalue=1, offvalue=0, font=('Arial', 20))
cBox.place(relx=0.7, rely=0.1, relwidth=0.2, relheight=0.1)

text = tkinter.Text(win)
text.place(rely=0.2, relwidth=1, relheight=0.8)
win.mainloop()
print('\nPRETTY DONE AS WELL!', '='*200)
