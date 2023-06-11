"""
Script to read and preprocess the raw data into seasonal format.
"""
import pandas as pd
import numpy as np
import math
import datetime as dt

class rawdataRead:
    def __init__(self, indexWanted, suffix="Close"):
        self.rawdataFilepath = "/home/crjLambda/PRO80/DEEPLEARN/TD_All.csv"
        self.TDindexFilepath = "/home/crjLambda/PRO80/DailyTDs/ref_TD.csv"
        self.yieldsindexFilepath = "/home/crjLambda/PRO80/DailyTDs/ref_yields.csv"
        self.currencyindexFilepath = "/home/crjLambda/PRO80/DailyTDs/ref_Currency.csv"
        self.indexWanted = indexWanted
        self.suffix = suffix

    def readingRawdata(self):
        TD_all_dataset = pd.read_csv(self.rawdataFilepath)
        colNames = TD_all_dataset.columns.values
        colNames[0] = "DATE"
        TD_all_dataset.columns = colNames
        TD_all_dataset["DATE"] = pd.to_datetime(TD_all_dataset["DATE"].values)
        TD_all_dataset.index = TD_all_dataset["DATE"]
        indexWanted = ["DATE"]
        [indexWanted.append(''.join([i, self.suffix])) for i in self.indexWanted]
        TD_indexes = pd.read_csv(self.TDindexFilepath)
        TD_yields_indexes = pd.read_csv(self.yieldsindexFilepath)
        TD_Currency_indexes = pd.read_csv(self.currencyindexFilepath)
        indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
        indexesAll = indexesAll.join(TD_yields_indexes, rsuffix="_yields")
        indexesAll_ind = indexesAll.iloc[0, ]
        returnRawdata = TD_all_dataset[indexWanted]
        return returnRawdata

    def generate_MultiClassesLabel(self, dataset, classes=4, isDATE=True):
        """
        function to transform continuous data into discrete data
        :param classes:
        :param isDATE:
        :return:
        """
        assert classes > 2, "classes must > 2 "
        if isDATE:
            dataset_DATE = dataset['DATE']
            dataset_noDATE = dataset.drop(columns=['DATE'])
        else:
            dataset_noDATE = dataset

        def to_classes(x, classes=4):
            """
            Let's make it more precisely, that log return are devided into n classes provided as the parameter. And the
            classes should begin from 0 and to n-1, they will represent the range from most negative to most positive
            number. In return, both the transformed dataset and the classesTable are returned.
            :param x:
            :param classes:
            :return:
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
            x_ceil = x_ceil + abs(x_ceil_min)
            # Generate the classTable for returning.
            cT_range = np.arange(0, classes, 1)
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
            classesTable = pd.DataFrame({
                "class": cT_range,
                "stand for": cT_label
            })
            return x_ceil, classesTable

        x_array = dataset_noDATE.values
        x_classes, x_classesTable = to_classes(x_array, classes=classes)
        dataset_noDATE = x_classes

        if isDATE:
            dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
            returnDataset = dataset_DATE.join(dataset_noDATE)
        else:
            returnDataset = dataset_noDATE
        return returnDataset, x_classesTable

    def monthlyLogr(self, dataset):
        """
        The function to transfer daily data into monthly data. Resample the raw data to obtain the last day's data of
        every month.
        :param dataset:
        :return:
        """
        monthlyLast = dataset.resample("m").last()
        logr = np.log(monthlyLast.iloc[:, 1]/((monthlyLast.iloc[:, 1] - monthlyLast.diff(1).iloc[:, 1])))
        logr = logr.dropna()
        return logr

    def generateFnL(self, dataset, frequency=12, difference=1, l_classes=4, f_classes=4):
        """

        :param dataset: input dataset
        :param frequency: number of length of the feature
        :param difference: label is n days ahead
        :param l_classes: Number of classes of the label that I would like to use
        :param f_classes: Can see it as the vocab_size
        :return: feature, label, predict_X
        """
        ### Create feature and label data frame
        numRows = len(dataset) - frequency - (difference - 1)
        numColumns = frequency + 1
        colNames = ['_'.join(["feature", str(i+1)]) for i in range(12)]
        colNames.append("label")
        # Getting the y_train converting into multiclass
        returnFnL = pd.DataFrame(columns=colNames)
        for i in range(numRows):
            x = dataset.iloc[i:i+frequency, ]
            x = x.append(dataset.iloc[(i+frequency-1+difference):(i+frequency-1+difference+1), ])
            returnFnL.loc[i] = x.values
        # X_train = returnFnL.iloc[:, :(numColumns-1)]
        y_train = returnFnL.iloc[:, (numColumns-1):]
        # Still need more preprocessing to the y_train to transform them from continuous data to discrete data.
        # Transform the y_train data into discrete data.
        y_train, y_classesTable = self.generate_MultiClassesLabel(y_train, classes=l_classes, isDATE=False)

        # Getting the X_train and X_predict converting into multiclass
        # Transform the dataset into multiclass with parameter f_classes
        l_dataset, X_classesTable = self.generate_MultiClassesLabel(dataset, classes=f_classes, isDATE=False)
        dataset=pd.Series(l_dataset, index=dataset.index)
        returnFnL = pd.DataFrame(columns=colNames)
        for i in range(numRows):
            x = dataset.iloc[i:i+frequency, ]
            x = x.append(dataset.iloc[(i+frequency-1+difference):(i+frequency-1+difference+1), ])
            returnFnL.loc[i] = x.values
        X_train = returnFnL.iloc[:, :(numColumns-1)]
        ### Generate the last day's features as the feature for prediction X
        X_predict = dataset.iloc[(len(dataset) - frequency):len(dataset), ]
        X_predict = X_predict.values

        return X_train, y_train, X_predict, y_classesTable, X_classesTable

    def generateOnetomanyFnL(self, dataset, frequency=12, nb_classes=4):
        reDataset, reClassesTable = self.generate_MultiClassesLabel(dataset, classes=nb_classes, isDATE=False)
        ### Create feature and label data frame
        numRows = len(reDataset) - frequency
        colNames = ['_'.join(["feature", str(i + 1)]) for i in range(frequency)]
        [colNames.append('_'.join(["label", str(i + 1)])) for i in range(frequency)]
        returnFnL = pd.DataFrame(columns=colNames)
        for i in range(numRows):
            x = reDataset[i:i+frequency]
            y = reDataset[i + 1:i + frequency + 1]
            returnFnL.loc[i] = x.tolist() + y.tolist()
        X_train = returnFnL.iloc[:, :frequency]
        y_train = returnFnL.iloc[:, frequency:]
        X_predict = y_train.iloc[-1]
        return X_train, y_train, X_predict, reClassesTable


#rr = rawdataRead(["CU0"], suffix="Close")
#rawData = rr.readingRawdata()
#logr = rr.monthlyLogr(rawData)
#newFnL, predict_x = rr.generateFnL(logr, frequency=12, difference=1)
#print(rawData)