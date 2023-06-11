"""
Get bond yields to be additional yields. As fast as the China's bond yields are updated, I am trying to read
'cn_10yry', 'cn_5yry' and 'cn_2yry' . Not their values but their log returns.
"""

import os
from pathlib import Path
import xlrd, xlwt
import pandas as pd
import numpy as np
dataDirName = "data"

class readingYields:
    def __init__(self, yieldsWanted):
        self.yieldsWanted = yieldsWanted
        self.readFilename = Path(os.getcwd(), dataDirName, 'yields.xls')
        self.workSheet = self.readFiles()
        self.returnFeatures = self.generateFeatures()

    def readFiles(self):
        yieldsWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = yieldsWorkbook.sheet_by_index(0)
        print('Reading yields FILE...<DONE!>...')
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
        return returnFeatures

# yR = readingYields(['CN_10yry', 'US_10yry', 'CN_5yry', 'CN_2yry'])
# print(yR.returnFeatures)