"""
Script to read and preprocess the raw data into seasonal format.
"""
import pandas as pd
import numpy as np
import math
import datetime as dt

class rawdataRead:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.rawdataFilepath = "/home/crjLambda/PRO80/DEEPLEARN/TD_All.csv"
        self.TDindexFilepath = "/home/crjLambda/PRO80/DailyTDs/ref_TD.csv"
        self.yieldsindexFilepath = "/home/crjLambda/PRO80/DailyTDs/ref_yields.csv"
        self.currencyindexFilepath = "/home/crjLambda/PRO80/DailyTDs/ref_Currency.csv"

    def _readingRawdata(self):
        TD_all_dataset = pd.read_csv(self.rawdataFilepath)
        colNames = TD_all_dataset.columns.values
        colNames[0] = "DATE"
        TD_all_dataset.columns = colNames
        TD_all_dataset["DATE"] = pd.to_datetime(TD_all_dataset["DATE"].values)
        TD_all_dataset.index = TD_all_dataset["DATE"]
        TD_indexes = pd.read_csv(self.TDindexFilepath)
        TD_yields_indexes = pd.read_csv(self.yieldsindexFilepath)
        TD_Currency_indexes = pd.read_csv(self.currencyindexFilepath)
        indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
        indexesAll = indexesAll.join(TD_yields_indexes, rsuffix="_yields")
        indexesAll_ind = indexesAll.iloc[0, ]
        indexWanted = ["DATE"]
        [indexWanted.append(''.join([i, 'Close'])) for i in indexesAll_ind] # Getting all Daily Close Data
        returnRawdata = TD_all_dataset[indexWanted]
        return returnRawdata, indexesAll_ind

    def _generate_MultiClassesLabel(self, dataset, classes=4, isDATE=True):
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
            cT_range = np.arange(4, (classes + 4), 1)
            cT_shift = np.arange(x_mean - ((classes - 2) / 2) * abs(x_dvi),
                                 x_mean + ((classes - 2) / 2 + 1) * abs(x_dvi),
                                 x_dvi)
            cT_label = []
            for i in range(len(cT_range)):
                if i == 0:
                    cT_label.append(''.join(["x < ", str(round(cT_shift[i], ndigits=4))]))
                elif i == (len(cT_range) - 1):
                    cT_label.append(''.join([str(round(cT_shift[i - 1], ndigits=4)), ' <= x']))
                else:
                    cT_label.append(''.join([str(round(cT_shift[i - 1], ndigits=4)), ' <= x < ',
                                             str(round(cT_shift[i], ndigits=4))]))
            cT_range = np.append(np.array([1, 2, 3]), cT_range)
            cT_label = ['not used', 'BOS', 'EOS'] + cT_label
            classesTable = pd.DataFrame({
                "class": cT_range,
                "stand for": cT_label
            })
            classesTable.index = classesTable['class']
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

    def _monthlyLogr(self, dataset):
        """
        The function to transfer daily data into monthly data. Resample the raw data to obtain the last day's data of
        every month.
        :param dataset:
        :return:
        """
        monthlyLast = dataset.resample("m").last()
        logr = np.log(monthlyLast/((monthlyLast - monthlyLast.diff(1))))
        logr = logr.dropna()
        logr.index = logr.index.strftime("%Y-%m")
        return logr

    def read_load(self):
        rawData, indexes = self._readingRawdata()
        # Now I still have to convert the daily trade data into monthly trade data.
        # Respectively, using every target to generate the feature and label for the one to many model. Then I will have
        # a pretty larger dataset for modeling.
        NUM_CLASSES = 1000
        FREQNENCY = 12 # similar to maxlen and 12 means months based.
        colNames = ['_'.join(["feature", str(i + 1)]) for i in range(FREQNENCY)]
        colNames.append("label")
        returnFnL = pd.DataFrame(columns=colNames)
        monthlyLogr = pd.DataFrame()
        for ind in indexes:
            _d = rawData[''.join([ind, 'Close'])]
            _m = self._monthlyLogr(_d)
            _m = _m.dropna()
            # Dealing with some miss leading data.
            _m = _m.map(lambda x: 0 if x == float('-inf') else x)
            _m_classes, _m_classesTable = self._generate_MultiClassesLabel(dataset=_m, classes=NUM_CLASSES, isDATE=False)
            _m = pd.DataFrame(_m_classes, index=_m.index, columns=[_m.name])
            numRows = len(_m) - FREQNENCY
            numColumns = FREQNENCY + 1
            for i in range(numRows):
                _x = _m.iloc[i:i+FREQNENCY, ]
                _x = _x.append(_m.iloc[(i+FREQNENCY):(i+FREQNENCY+1), ])
                _x = pd.DataFrame(_x.T)
                _x.columns = colNames
                returnFnL = returnFnL.append(_x, ignore_index=True)
        # Buiding up X_train and y_train for return.
        X_train = returnFnL.iloc[:, :FREQNENCY]
        y_train = returnFnL.iloc[:, 1:]
        return X_train, y_train

    def read_predict(self, ind):
        rawData, indexes = self._readingRawdata()
        # Now I still have to convert the daily trade data into monthly trade data.
        # Respectively, using every target to generate the feature and label for the one to many model. Then I will have
        # a pretty larger dataset for modeling.
        NUM_CLASSES = 1000
        FREQNENCY = 12 # similar to maxlen and 12 means months based.
        colNames = ['_'.join(["feature", str(i + 1)]) for i in range(FREQNENCY)]
        colNames.append("label")
        returnFnL = pd.DataFrame(columns=colNames)
        monthlyLogr = pd.DataFrame()
        ind = ind[0]
        _d = rawData[''.join([ind, 'Close'])]
        _m = self._monthlyLogr(_d)
        _m = _m.dropna()
        # Dealing with some miss leading data.
        _m = _m.map(lambda x: 0 if x == float('-inf') else x)
        _m_classes, _m_classesTable = self._generate_MultiClassesLabel(dataset=_m, classes=NUM_CLASSES, isDATE=False)
        _m = pd.DataFrame(_m_classes, index=_m.index, columns=[_m.name])
        numRows = len(_m) - FREQNENCY
        numColumns = FREQNENCY + 1
        for i in range(numRows):
            _x = _m.iloc[i:i+FREQNENCY, ]
            _x = _x.append(_m.iloc[(i+FREQNENCY):(i+FREQNENCY+1), ])
            _x = pd.DataFrame(_x.T)
            _x.columns = colNames
            returnFnL = returnFnL.append(_x, ignore_index=True)
        return returnFnL.iloc[-1, 1:], _m.index[-1]