"""
Similar to sentiAnalysis_V2.py code, this code is to receive weiboTotalresults list and process the sentiment data into
daily based features.
"""

import pandas as pd
import re
import datetime as dt

class weiboDataprocess:
    def __init__(self, dataToproc):
        self.dataToproc = dataToproc
        self.dataFinal = self.dataProc()

    def resultsProc(self, results):
        results_df = pd.DataFrame(results)
        resultsGroups = results_df.groupby(['sentiment_key'])
        # Getting the statistic results.
        try:
            x1_p = resultsGroups.size().loc['positive']
        except KeyError:
            x1_p = 0

        try:
            x1_n = resultsGroups.size().loc['negative']
        except KeyError:
            x1_n = 0

        try:
            x2_p = resultsGroups.mean().lookup(['positive'], ['positive_probs'])[0]
        except KeyError:
            x2_p = 0

        try:
            x2_n = resultsGroups.mean().lookup(['negative'], ['negative_probs'])[0]
        except KeyError:
            x2_n = 0

        statisticReturn = {'result_numbers': resultsGroups.size().sum(),
                           'positive_numbers': x1_p,
                           'negative_numbers': x1_n,
                           'positive_probs_avg': x2_p,
                           'negative_probs_avg': x2_n}
        return statisticReturn

    # Function to modify the data into groups with results.
    def dataMoreproc(self, dataToproc):
        # Turn the data into dataframe
        dataToproc = pd.DataFrame(dataToproc)
        dataToproc['year'] = ''
        dataToproc['month'] = ''
        dataToproc['day'] = ''
        dataToproc['result_numbers'] = ''
        dataToproc['positive_numbers'] = ''
        dataToproc['negative_numbers'] = ''
        dataToproc['positive_probs_avg'] = ''
        dataToproc['negative_probs_avg'] = ''
        for i in range(len(dataToproc)):
            dataToproc['year'][i] = dataToproc['DATE'][i].year
            dataToproc['month'][i] = dataToproc['DATE'][i].month
            dataToproc['day'][i] = dataToproc['DATE'][i].day
            tmp_x = dataToproc['RESULTS'][i] # obtain the result dictionary object.
            dataToproc['result_numbers'][i] = tmp_x['result_numbers']
            dataToproc['positive_numbers'][i] = tmp_x['positive_numbers']
            dataToproc['negative_numbers'][i] = tmp_x['negative_numbers']
            dataToproc['positive_probs_avg'][i] = tmp_x['positive_probs_avg']
            dataToproc['negative_probs_avg'][i] = tmp_x['negative_probs_avg']
        dataReturn = dataToproc.drop(columns = 'RESULTS')
        return(dataReturn)

    def dataGroupby(self, dataToproc):
        groups = dataToproc.groupby(['year', 'month', 'day'])
        x1 = groups.mean()
        x2 = groups.sum()
        # Generate the feature data.
        x2['positive_result_pct'] = x2['positive_numbers'] / x2['result_numbers']
        x2['negative_result_pct'] = x2['negative_numbers'] / x2['result_numbers']
        returnGroups = x1[['positive_probs_avg', 'negative_probs_avg']]
        returnGroups['positive_result_pct'] = x2['positive_result_pct']
        returnGroups['negative_result_pct'] = x2['negative_result_pct']
        # Creating 'DATE' columns for returnGroups.
        returnGroups['DATE'] = ''
        for i in range(len(returnGroups)):
            ind = returnGroups.index[i]
            returnGroups['DATE'][i] = pd.to_datetime("-".join([str(j) for j in ind]))
        return (returnGroups)

    def dataProc(self):
        datasets = self.dataToproc.copy()
        returnData = []
        for singleDataset in datasets:
            # For already formatized DATE, I don't have to modify the DATE here.
            # and for the RESULTS, process them to generate daily based features.
            singleDataset['RESULTS'] = self.resultsProc(singleDataset['RESULTS'])
            returnData.append(singleDataset)
        returnData = self.dataMoreproc(returnData)
        returnData = self.dataGroupby(returnData)
        return returnData

