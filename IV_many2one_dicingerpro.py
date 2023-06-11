"""
Mission IV: DicingerPro
In the previous verssion, features includes only 12 raw data, 4 from email sentiment, 4 from weibo sentiment and another
4 yields.
"""
import os
from pathlib import Path
import pandas as pd

# Part I:
# Prepare the features and labels for models.===========================================================================
dataDirName = "dataForlater"
emailReadfilename = Path(Path(os.getcwd()), dataDirName, "II_seq2seq_moon2sun_cook_email_feature_forlater.json")
weiboReadfilename = Path(Path(os.getcwd()), dataDirName, "II_seq2seq_moon2sun_cook_weibo_feature_forlater.json")
emailFeatures_df = pd.read_json(emailReadfilename)
weiboFeatures_df = pd.read_json(weiboReadfilename)

"""
Get labels and process both the features and labels for deep learning.
"""
# Also read the indexes
TD_indexes = pd.read_csv('/home/crjLambda/PRO80/DailyTDs/ref_TD.csv')
TD_yields_indexes = pd.read_csv('/home/crjLambda/PRO80/DailyTDs/ref_yields.csv')
TD_Currency_indexes = pd.read_csv('/home/crjLambda/PRO80/DailyTDs/ref_Currency.csv')

# And generate wanted dataset
indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
# indexesAll = indexesAll.join(TD_yields_indexes, rsuffix='_yields')
indexesAll_ind = indexesAll.iloc[0,]

"""
To get labels for deep learning.
"""
import os
from pathlib import Path
import xlrd
import pandas as pd
import numpy as np

# class to get labels and yields feature.----------
class getLabels:
    def __init__(self, indexWanted):
        self.indexWanted = indexWanted
        self.dataDirName = 'data'
        self.readFilename = Path(os.getcwd(), self.dataDirName, self.indexWanted+'.xls')
        self.workSheet = self.readFiles()
        self.returnLabels = self.generateLabels()

    def readFiles(self):
        labelWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = labelWorkbook.sheet_by_index(0)
        return workSheet

    def generateLabels(self):
        workSheet = self.workSheet
        returnLabels = pd.DataFrame(columns=['DATE', 'LABELS'])
        returnLabels['DATE'] = [xlrd.xldate_as_datetime(dt.value, 0) for dt in workSheet.col(2)[2:-6]]
        # Using daily log return as the labels.
        close_t = [cl.value for cl in workSheet.col(6)[2:-6]]
        close_t1 = [cl.value for cl in workSheet.col(6)[1:-7]]
        logr = [np.log(close_t[i]/close_t1[i]) for i in range(len(close_t))]
        returnLabels['LABELS'] = logr
        return returnLabels

class readingYields:
    def __init__(self, yieldsWanted):
        self.yieldsWanted = yieldsWanted
        self.dataDirName = "data"
        self.readFilename = Path(os.getcwd(), self.dataDirName, 'yields.xls')
        self.workSheet = self.readFiles()
        self.returnFeatures = self.generateFeatures()

    def readFiles(self):
        yieldsWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = yieldsWorkbook.sheet_by_index(0)
        return workSheet

    def generateFeatures(self):
        workSheet = self.workSheet
        yieldLambda = 1 # Try to fine tune.
        # # Loading the data.
        yieldsRead = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        for i in yieldsRead.columns:
            if(i == 'DATE'):
                yieldsRead[i] = [pd.Timestamp(dt.value) for dt in workSheet.col(0)[4:-7]]
            elif(i == 'US_10yry'):
                # locate the feature's col number
                for j in range(workSheet.ncols):
                    if (workSheet.row(0)[j].value == i):
                        tmp_x = j
                # Dealing with the data lagging for 1 day if there is any.
                tmp_y = [i.value for i in workSheet.col(tmp_x)[4:-7]]
                if (tmp_y[0] == ''):
                    tmp_y[0] = tmp_y[1]
                yieldsRead[i] = tmp_y
            else:
                # locate the feature's col number
                for j in range(workSheet.ncols):
                    if (workSheet.row(0)[j].value == i):
                        tmp_x = j
                yieldsRead[i] = [i.value for i in workSheet.col(tmp_x)[4:-7]]
        def f(x):
            if x == '':
                return np.nan
            else:
                return x
        yieldsRead = yieldsRead.applymap(f)
        yieldsRead = yieldsRead.dropna()
        # # Generate the yield features.
        returnFeatures = pd.DataFrame(columns=['DATE']).join(pd.DataFrame(columns=[i for i in self.yieldsWanted]))
        for i in returnFeatures.columns:
            if (i=='DATE'):
                returnFeatures[i] = yieldsRead[i][:-1]
            else:
                close_t = np.array(yieldsRead[i][:-1])
                close_tsub1 = np.array(yieldsRead[i][1:])
                returnFeatures[i] = [np.log(close_t[j]/close_tsub1[j])*yieldLambda for j in range(len(close_t))]
        print("\n", "Getting yields DONE!", "."*200, "\n")
        return returnFeatures

yieldsWanted = ['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry']
featuresYieldsDL_df = readingYields(yieldsWanted).returnFeatures

indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'JM0', 'UR0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ["SCM", 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'M0', 'LUM', 'RU0']
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

# Generate the overall dataset using individual target and using correlation matrix for the orders.
def datetimeProcessing(dataToproc):
    # functions to reprocess the DATE formate
    dataToproc['DATE_'] = ''
    for i in range(len(dataToproc)):
        dataToproc['DATE_'].iloc[i] = dataToproc['DATE'].iloc[i].to_pydatetime().date()
    dataToproc.index = dataToproc['DATE_']
    return dataToproc

def fnl_reProcessing(emailFeatures, weiboFeatures, featuresYields, labels):
    # function to combine the features and labels.
    emailFeatures = datetimeProcessing(emailFeatures)
    weiboFeatures = datetimeProcessing(weiboFeatures)
    featuresYields = datetimeProcessing(featuresYields)
    labels = datetimeProcessing(labels)
    # join the emailFeatures and labels into one dataframe
    fnl_fd = labels.join(emailFeatures, rsuffix='_other')
    # remove NAs
    fnl_fd = fnl_fd.dropna()
    fnl_fd = fnl_fd.drop(['DATE', 'DATE_other', 'DATE__other'], axis="columns")
    # join the weiboFeatures
    fnl_fd = fnl_fd.join(weiboFeatures, rsuffix='_weibo')
    fnl_fd = fnl_fd.dropna()
    fnl_fd = fnl_fd.drop(['DATE', 'DATE__weibo'], axis="columns")
    # more features yields
    fnl_fd = fnl_fd.join(featuresYields, rsuffix='_other')
    fnl_fd = fnl_fd.dropna()
    fnl_fd = fnl_fd.drop(['DATE', 'DATE__other'], axis="columns")
    return fnl_fd

labels, features = [], []
for ind in indexList:
    labelsDL_df = getLabels(indexWanted=ind).returnLabels
    fnl = fnl_reProcessing(emailFeatures_df, weiboFeatures_df, featuresYieldsDL_df, labelsDL_df)
    # Reorder of futures using correlation matrix.
    fnl_corr = fnl.corr()
    newOrder = fnl_corr.iloc[0, :].sort_values(axis=0, ascending=False)
    # Having label and all the features ready, now it is time to generate the label dataset and feature dataset.
    fnl = fnl[newOrder.index.to_list()]
    for i in range(fnl.shape[0]):
        label = fnl.iloc[i, 0]
        feature = fnl.iloc[i, 1:].to_numpy(np.float)
"""
NOT FINISHED YET.
"""


"""
=PART II, Deep Learning.================================================================================================
=There are three different types of deep learning network models to be implemented. Although they are quite pre-mature =
=I am counting on it to generate odds for dicing.                                                                      =
=1.<Simple Linear Model.>                                                                                              =
=2.<Simple Complete Learning Network Model.>                                                                           =
=3.<Simple Convolutional Network Model.>
========================================================================================================================
"""
# in version 1, I was adding bond yields as more features.
from BK import getYields_V2 as gt, getLabels_V2 as gl, simpleDeeplearning_V2 as sdl, simpleCompleteln_V2 as scln, \
    simpleConvolutionnetwork_V2 as scnn

yieldsWanted = ['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry']
gtReturn = gt.readingYields(yieldsWanted)
featuresYieldsDL_df = gtReturn.returnFeatures

from tqdm import tqdm
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
        countResults = "\nCounting Results:\n" + \
                       'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
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
            # Saving output string to text
            saving_to_file = open(Path(Path(os.getcwd()).parents[1], dataDirName, "outputStringCU0.txt"), 'w')
            saving_to_file.write(countResults + "Total Count: " +
                                 str(positiveCount*(1-countType)+negativeCount*countType) + "\n" +
                                 outputString)
            saving_to_file.close()
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
        countResults = "\nCounting Results:\n" + \
                       'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
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
            # Saving output string to text
            saving_to_file = open(Path(Path(os.getcwd()).parents[1], dataDirName, "outputStringRB0.txt"), 'w')
            saving_to_file.write(countResults + "Total Count: " +
                                 str(positiveCount*(1-countType)+negativeCount*countType) + "\n" +
                                 outputString)
            saving_to_file.close()
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
        countResults = "\nCounting Results:\n" + \
                       'COUNT TYPE: ' + ('NEGATIVE' if countType == 1 else 'POSITIVE') + '\n' + \
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
            # Saving output string to text
            saving_to_file = open(Path(Path(os.getcwd()).parents[1], dataDirName, "outputStringSCM.txt"), 'w')
            saving_to_file.write(countResults + "Total Count: " +
                                 str(positiveCount*(1-countType)+negativeCount*countType) + "\n" +
                                 outputString)
            saving_to_file.close()
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
