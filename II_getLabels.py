"""
To get labels for deep learning. In version 3, I used the classes of [0, 1] instead of log return, which is 0 if price
decline and 1 if price rise.
"""
import os
from pathlib import Path
import xlrd, xlwt
import pandas as pd
import numpy as np
import math
dataDirName = "data"

class getLabels:
    def __init__(self, indexWanted, num_classes=4):
        self.indexWanted = indexWanted
        self.num_classes = num_classes
        self.readFilename = Path(os.getcwd(), dataDirName, self.indexWanted[0]+'.xls')
        self.workSheet = self.readFiles()
        self.returnLabel, self.classesTable = self.generateLabels()

    def readFiles(self):
        labelWorkbook = xlrd.open_workbook(self.readFilename)
        workSheet = labelWorkbook.sheet_by_index(0)
        # print('Reading LABEL FILE...<DONE!>...')
        return workSheet

    def _to_class(self, x, classes=4):
        """
        Let's make it more precisely, that log return are devided into n classes provided as the parameter. And the
        classes should begin from 0 and to n-1, they will represent the range from most negative to most positive
        number. In return, both the transformed dataset and the classesTable are returned.
        I have to save for numbers for special use in classes, they are:
        0, padding number;
        1, not used;
        2, BOS;
        3, EOS;
        """
        x_dvi = 2 * x.std() / (classes - 2)
        x_mean = x.mean()
        x_ceil = np.array([math.ceil(m / x_dvi) for m in x])
        x_range = np.array([m for m in range(-int(classes / 2 - 1), int(classes / 2 + 1))])
        # Adjust the tail
        for i in range(len(x_ceil)):
            if x_ceil[i] < x_range.min():
                x_ceil[i] = x_range.min()
            elif x_ceil[i] > x_range.max():
                x_ceil[i] = x_range.max()
        # Now adjust more
        x_ceil_min = x_ceil.min()
        x_ceil = x_ceil + abs(x_ceil_min) + 4
        # Generate the classTable for returning.
        cT_range = np.arange(4, (classes+4), 1)
        cT_shift = np.arange(x_mean - ((classes-2) / 2) * abs(x_dvi),
                             x_mean + ((classes-2) / 2 + 1) * abs(x_dvi),
                             x_dvi)
        cT_label = []
        for i in range(len(cT_range)):
            if i == 0:
                cT_label.append(''.join(["x < ", str(round(cT_shift[i], ndigits=4))]))
            elif i == (len(cT_range) - 1):
                cT_label.append(''.join([str(round(cT_shift[i-1], ndigits=4)), ' <= x']))
            else:
                cT_label.append(''.join([str(round(cT_shift[i-1], ndigits=4)), ' <= x < ',
                                         str(round(cT_shift[i], ndigits=4))]))
        cT_range = np.append(np.array([1, 2, 3]), cT_range)
        cT_label = ['not used', 'BOS', 'EOS'] + cT_label
        classesTable = pd.DataFrame({
            "class": cT_range,
            "stand for": cT_label
        })
        classesTable.index = classesTable['class']
        return x_ceil, classesTable

    def generateLabels(self):
        workSheet = self.workSheet
        returnLabels = pd.DataFrame(columns=['DATE', 'LABELS'])
        returnLabels['DATE'] = [xlrd.xldate_as_datetime(dt.value, 0) for dt in workSheet.col(2)[2:-6]]
        # Using daily log return as the labels.
        close_t = [cl.value for cl in workSheet.col(6)[2:-6]]
        close_t1 = [cl.value for cl in workSheet.col(6)[1:-7]]
        logr = [np.log(close_t[i]/close_t1[i]) for i in range(len(close_t))]
        logr_array = np.array(logr)
        x_classes, x_classesTable = self._to_class(logr_array, classes=self.num_classes)
        returnLabels['LABELS'] = x_classes.astype(np.int32)
        return returnLabels, x_classesTable

"""
indexWanted = ['AG0']
glReturn = getLabels(indexWanted=indexWanted)
labels = glReturn.returnLabels
"""