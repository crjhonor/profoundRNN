"""
Script to read and preprocess the raw data into seasonal format.
"""
import os
import pandas as pd
import numpy as np
import math
import datetime as dt

class rawdataRead:
    def __init__(self, indexes, nb_classes):
        self.rawdataFilepath = "./data"
        self.indexes = indexes
        self.nb_classes = nb_classes

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

        x_array = dataset_noDATE.values
        x_classes, x_classesTable = to_classes(x_array, classes=classes)
        dataset_noDATE = x_classes

        if isDATE:
            dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
            returnDataset = dataset_DATE.join(dataset_noDATE)
        else:
            returnDataset = dataset_noDATE
        return returnDataset, x_classesTable

    def readingRawdata(self):
        indexes = self.indexes

        def hourlyLogr(dataset):
            hourly = dataset['close']
            logr = np.log(hourly / (hourly - hourly.diff(1)))
            logr = logr.dropna()
            logr.name = "logr"
            return logr

        def readSingle(index):
            readFilepath = os.path.join(self.rawdataFilepath, ''.join([index, '_60.xls']))
            TD_all_dataset = pd.read_excel(readFilepath)
            returnData = TD_all_dataset[['证券代码', '交易时间', '收盘价']]
            returnData = returnData.dropna()
            colNames = ['contract', 'time', 'close']
            returnData.columns = colNames

            # To make feather aligning of all the data, I need to modify the date time index
            year = [x.year for x in returnData['time']]
            month = [x.month for x in returnData['time']]
            day = [x.day for x in returnData['time']]
            hour = [x.hour for x in returnData['time']]
            minute = [x.minute for x in returnData['time']]
            # Now do a little adjustment to the minute so that different trading data the same hour, which may have a
            # different close minute can be aligned.
            minute = [0 if x==0 else 1 for x in minute]
            # we make the dataframe index
            dfIndex = [dt.datetime(year=y, month=mo, day=d, hour=h, minute=m)
                       for y, mo, d, h, m in zip(year, month, day, hour, minute)]
            dfIndex = pd.to_datetime(dfIndex)
            returnData.index = dfIndex
            return returnData

        readLists = []
        for i in indexes:
            read = readSingle(i)
            readLogr = hourlyLogr(read)
            # It will be even more convenience to convert to discrete data here I think.
            readLogrclasses, classesTable = self.generate_MultiClassesLabel(readLogr,
                                                                            classes=self.nb_classes, isDATE=False)
            readList = pd.DataFrame(readLogrclasses, columns=[''.join([i, '_60'])], index=read.index[1:])
            readLists.append(readList)
            if i == indexes[0]:
                returnClassestable = classesTable

        # Then align all the log return data obtained from hourly trading records.
        for i in range(len(readLists)):
            if i == 0:
                returnDataset = readLists[i]
            else:
                returnDataset = returnDataset.join(readLists[i])
        returnDataset_nona = returnDataset.dropna()
        returnDataset_nona = returnDataset_nona.astype(np.int64)
        return returnDataset_nona, returnClassestable

    def download_and_read(self, dataset):
        maxlen_en = dataset.shape[1]
        maxlen_fr = maxlen_en
        sents_en, sents_fr_in, sents_fr_out = [], [], []
        for i in range(dataset.shape[0]-maxlen_en):
            sent_en = dataset.iloc[i, :].to_numpy()
            sent_fr = dataset.iloc[i:i+maxlen_fr, 0].to_numpy()
            last_feature = dataset.index[i]
            last_label_time = dataset.index[i+maxlen_fr]
            # add 2 in front of the label feature as the BOS
            sent_fr_in = np.append(np.array(2), sent_fr)
            sent_fr_out = np.append(sent_fr, np.array(3))
            sents_en.append(sent_en)
            sents_fr_in.append(sent_fr_in)
            sents_fr_out.append(sent_fr_out)
        # Still have to generate the X_predict
        X_predict = dataset.iloc[(dataset.shape[0]-1), :].to_numpy()
        X_predict_time = dataset.index[-1]

        return sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time