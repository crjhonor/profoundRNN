"""
To get labels for deep learning.
"""
import os
from pathlib import Path
import xlrd, xlwt
import pandas as pd
import numpy as np
dataDirName = "data"

class getLabels:
    def __init__(self, indexWanted):
        self.indexWanted = indexWanted
        self.readFilename = Path(os.getcwd(), dataDirName, self.indexWanted[0]+'.xls')
        self.workSheet = self.readFiles()
        self.returnLabels = self.generateLabels()

    def readFiles(self):
        labelWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = labelWorkbook.sheet_by_index(0)
        # print('Reading LABEL FILE...<DONE!>...')
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

"""
indexWanted = ['RU0']
glReturn = getLabels(indexWanted=indexWanted)
labels = glReturn.returnLabels
"""