# # Reading emails and getting contents sentimented by trained deep learning.==========================================
"""
Currently I am using Baidu's paddlepaddle and paddlehub to sentiment the content received by email.
"""

import sentimentCode as SC
totalResults = SC.totalResults

# # save out to json file and leave the analysis job to R ==============================================================
"""
R is better for data analysis I assumed, and it has plenty of deep learning modules. I should try it later.
"""
import os
from pathlib import Path
import json

dataDirName = "dataForlater"
outputResultfilename = Path(Path(os.getcwd()), dataDirName, "sentimentsResult.json")
saveTofile = totalResults
# Dumping to json file needs a list
resultJsondata = json.dumps(saveTofile)

try:
    with open(outputResultfilename, 'w') as fileObject:
        json.dump(resultJsondata, fileObject)
except FileNotFoundError:
    os.mkdir(Path(os.getcwd(), dataDirName))
    with open(outputResultfilename, 'w') as fileObject:
        json.dump(resultJsondata, fileObject)

print("SENTIMENTINGS RESULTS ARE SAVED INTO ", outputResultfilename, "!")

# # Performing Deep Learning and Try to Predict ========================================================================
"""
sentiAnalysis.py is mostly coded for Results preprocessing after the sentiment. And, further deep learning is approached
with label and feature designing and machine learning. 
"""
# Define a dataProcess class including functions for data preprocessing after sentiment.
import sentiAnalysis as sa
saReturn = sa.dataProcess(totalResults)
featuresDL_df = saReturn.dataFinal

# in this version, I am adding bond yields as more features.
import getYields as gt
yieldsWanted = ['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry']
gtReturn = gt.readingYields(yieldsWanted)
featuresYieldsDL_df = gtReturn.returnFeatures


# And I also need the labels.
indexWanted = ['RU0', 'RB0', 'TA0', 'P0', 'I0', 'SR0', 'M0', "AL0", "ZN0", "BU0", "FU0"]

import getLabels as gl
indexWanted = ['RU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
"""
Given the features and labels, I pass them into the deep learning scripts to generate DL modules. Features and labels 
engineering is needed for deep learning. And Let me try to impose the most simple neutral network.
"""

sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_1 = sdlReturn.results

# # MORE wanted index

indexWanted = ['RB0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_2 = sdlReturn.results

indexWanted = ['TA0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_3 = sdlReturn.results

indexWanted = ['P0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_4 = sdlReturn.results

indexWanted = ['I0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_5 = sdlReturn.results

indexWanted = ['SR0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_6 = sdlReturn.results

indexWanted = ['M0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_7 = sdlReturn.results

indexWanted = ['AL0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_8 = sdlReturn.results

indexWanted = ['ZN0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_9 = sdlReturn.results

indexWanted = ['BU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_10 = sdlReturn.results

indexWanted = ['FU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sdlReturn = sdl.simpleDeeplearning(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sdlResults_11 = sdlReturn.results

# # More accurate model of Simple Complete Learning Nework.=================================================
import simpleCompleteln as scln
indexWanted = ['RU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_1 = sclnReturn.results

indexWanted = ['RB0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_2 = sclnReturn.results

indexWanted = ['TA0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_3 = sclnReturn.results

indexWanted = ['P0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_4 = sclnReturn.results

indexWanted = ['I0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_5 = sclnReturn.results

indexWanted = ['SR0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_6 = sclnReturn.results

indexWanted = ['M0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_7 = sclnReturn.results

indexWanted = ['AL0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_8 = sclnReturn.results

indexWanted = ['ZN0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_9 = sclnReturn.results

indexWanted = ['BU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_10 = sclnReturn.results

indexWanted = ['FU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
sclnReturn = scln.simpleCompleteln(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
sclnResults_11 = sclnReturn.results

# # The 3rd network, simple convolution network.========================================================================
"""
I need a 4x2 features inorder to use the convolution network. And features array should be transformed into a 4D data, 
[batch_size, 1, 2, 4], the '1' is a channel.
"""

from BK import simpleConvolutionnetwork as scnn

indexWanted = ['RU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_1 = scnnReturn.results

indexWanted = ['RB0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_2 = scnnReturn.results

indexWanted = ['TA0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_3 = scnnReturn.results

indexWanted = ['P0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_4 = scnnReturn.results

indexWanted = ['I0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_5 = scnnReturn.results

indexWanted = ['SR0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_6 = scnnReturn.results

indexWanted = ['M0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_7 = scnnReturn.results

indexWanted = ['AL0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_8 = scnnReturn.results

indexWanted = ['ZN0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_9 = scnnReturn.results

indexWanted = ['BU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_10 = scnnReturn.results

indexWanted = ['FU0']
glReturn = gl.getLabels(indexWanted=indexWanted)
labelsDL_df = glReturn.returnLabels
scnnReturn = scnn.simpleConvolutionnetwork(indexWanted, featuresDL_df, featuresYieldsDL_df, labelsDL_df)
scnnResults_11 = scnnReturn.results

# # Deep Learning Results output.=======================================================================================
"""
Now I am having the idea that, If I own 3 different kinds of network and put features and lables seperately into these 
networks and generate all 3 results, then I should have the odds of the results.
"""

print('\n'*5, 'FINAL RESULTS', '='*200, '\n'*2)
print('For Target one', '.'*200, '\n')
print(sdlResults_1)
print(sclnResults_1)
print(scnnResults_1)
print('\n'*2, 'For Target two', '.'*200, '\n')
print(sdlResults_2)
print(sclnResults_2)
print(scnnResults_2)
print('\n'*2, 'For Target three', '.'*200, '\n')
print(sdlResults_3)
print(sclnResults_3)
print(scnnResults_3)
print('\n'*2, 'For Target four', '.'*200, '\n')
print(sdlResults_4)
print(sclnResults_4)
print(scnnResults_4)
print('\n'*2, 'For Target five', '.'*200, '\n')
print(sdlResults_5)
print(sclnResults_5)
print(scnnResults_5)
print('\n'*2, 'For Target six', '.'*200, '\n')
print(sdlResults_6)
print(sclnResults_6)
print(scnnResults_6)
print('\n'*2, 'For Target seven', '.'*200, '\n')
print(sdlResults_7)
print(sclnResults_7)
print(scnnResults_7)
print('\n'*2, 'For Target eight', '.'*200, '\n')
print(sdlResults_8)
print(sclnResults_8)
print(scnnResults_8)
print('\n'*2, 'For Target nine', '.'*200, '\n')
print(sdlResults_9)
print(sclnResults_9)
print(scnnResults_9)
print('\n'*2, 'For Target ten', '.'*200, '\n')
print(sdlResults_10)
print(sclnResults_10)
print(scnnResults_10)
print('\n'*2, 'For Target eleven', '.'*200, '\n')
print(sdlResults_11)
print(sclnResults_11)
print(scnnResults_11)

print('\nDONE!', '='*200)

# # SOME Pretty output.=================================================================================================
import tkinter
win = tkinter.Tk()
win.title('Deep Learning Results')
width = 1600
height = 800
screenwidth = win.winfo_screenwidth()
screenheight = win.winfo_screenheight()
root_str = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
win.geometry(root_str)
win.maxsize(1600, 800)

text = tkinter.Text(win, width = 1000, height = 600)
scroll = tkinter.Scrollbar()
scroll.pack(side=tkinter.RIGHT, fill=tkinter.Y)

scroll.config(command=text.yview)
text.config(yscrollcommand=scroll.set)

text.pack()
str1 = '\n'.join(["\nDeep Learning results of RU0\n" + '.' * 100,
                  str(sclnResults_1), str(scnnResults_1),
                  "=" * 100,
                  "\nDeep Learning results of RB0\n" + '.' * 100,
                  str(sclnResults_2), str(scnnResults_2),
                  "=" * 100,
                  "\nDeep Learning results of TA0\n" + '.' * 100,
                  str(sclnResults_3), str(scnnResults_3),
                  "=" * 100,
                  "\nDeep Learning results of P0\n" + '.' * 100,
                  str(sclnResults_4), str(scnnResults_4),
                  "=" * 100,
                  "\nDeep Learning results of I0\n" + '.' * 100,
                  str(sclnResults_5), str(scnnResults_5),
                  "=" * 100,
                  "\nDeep Learning results of SR0\n" + '.' * 100,
                  str(sclnResults_6), str(scnnResults_6),
                  "=" * 100,
                  "\nDeep Learning results of M0\n" + '.' * 100,
                  str(sclnResults_7), str(scnnResults_7),
                  "=" * 100,
                  "\nDeep Learning results of AL0\n" + '.' * 100,
                  str(sclnResults_8), str(scnnResults_8),
                  "=" * 100,
                  "\nDeep Learning results of ZN0\n" + '.' * 100,
                  str(sclnResults_9), str(scnnResults_9),
                  "=" * 100,
                  "\nDeep Learning results of BU0\n" + '.' * 100,
                  str(sclnResults_10), str(scnnResults_10),
                  "=" * 100,
                  "\nDeep Learning results of FU0\n" + '.' * 100,
                  str(sclnResults_11), str(scnnResults_11),
                  "=" * 100])

text.insert(tkinter.INSERT, str1)
win.mainloop()

print('\nPRETTY DONE AS WELL!', '='*200)
