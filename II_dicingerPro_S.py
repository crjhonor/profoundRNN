
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
========================================================================================================================
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import datetime

# Part I:
# Prepare the features and labels for models.---------------------------------------------------------------------------
dataDirName = "dataForlater"
emailReadfilename = Path(Path(os.getcwd()), dataDirName, "II_seq2seq_moon2sun_cook_email_feature_forlater.json")
weiboReadfilename = Path(Path(os.getcwd()), dataDirName, "II_seq2seq_moon2sun_cook_weibo_feature_forlater.json")
emailFeatures_df = pd.read_json(emailReadfilename)
weiboFeatures_df = pd.read_json(weiboReadfilename)

"""
Get labels and process both the features and labels for deep learning.
"""
import pandas as pd
import numpy as np
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'JM0', 'UR0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ["SCM", 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'M0', 'LUM', 'RU0']
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

def datetimeProcessing(dataToproc):
    # functions to reprocess the DATE formate
    dataToproc['DATE_'] = ''
    for i in range(len(dataToproc)):
        dataToproc['DATE_'].iloc[i] = dataToproc['DATE'].iloc[i].to_pydatetime().date()
    dataToproc.index = dataToproc['DATE_']
    return dataToproc

def fnl_reProcessing(emailFeatures, weiboFeatures, labels):
    # function to combine the features and labels.
    emailFeatures = datetimeProcessing(emailFeatures)
    weiboFeatures = datetimeProcessing(weiboFeatures)
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
    return fnl_fd

import II_getLabels_V3 as gl
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import II_sentimentElseDeepLearning

outputs_df = pd.DataFrame({'DATE': ['later', 'later', 'later'],
                           'MODULE': ['linear', 'CNN', 'RNN']})
for i in tqdm(range(len(indexList)), ncols=100, desc="somethingElse", colour="blue"):
    ind = [indexList[i]]
    glReturn = gl.getLabels(indexWanted=ind)
    labelsDL_df = glReturn.returnLabels
    fnl = fnl_reProcessing(emailFeatures=emailFeatures_df,
                           weiboFeatures=weiboFeatures_df,
                           labels=labelsDL_df)
    """
    Generating the X_train, X_test, y_train, y_test, X_predict and X_predict_DATE
    """
    labels = fnl['LABELS']
    features = fnl.drop(['DATE_', 'LABELS'], axis="columns")
    X = np.array(features.iloc[:-1, :])
    y = np.array(labels[1:])
    X_predict = np.array(features.iloc[-1, :])
    X_predict_DATE = fnl['DATE_'][-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    """
    Passing through the all the data to sentimentElseDeepLearning.py
    """
    sedl = II_sentimentElseDeepLearning.sentimentElseDeepLearning(X_train, X_test, y_train, y_test, X_predict,
                                                               ind, X_predict_DATE)
    output = []
    output.append(sedl.deeperLinearModel_train(learning_rate=0.001, num_epochs=200))
    output.append(sedl.CNN_model_train(learning_rate=0.001, num_epochs=50))
    output.append(sedl.RNN_model_train(learning_rate=0.001, num_epochs=50))
    output_df = pd.DataFrame(output, columns=['_'.join([ind[0], j]) for j in ['acc.', 'y']])
    outputs_df = outputs_df.join(output_df)
outputs_df['DATE'][0] = X_predict_DATE
outputs_df['DATE'][1] = X_predict_DATE
outputs_df['DATE'][2] = X_predict_DATE

"""
Saving fo the finalProba project.
"""
import os
from pathlib import Path
dataDirName = "finalProba/allProba"
outputs_filename = Path(Path(os.getcwd()).parents[1], dataDirName, "dicingerpro_S.csv")
outputs_df.to_csv(outputs_filename)

print('DONE!')
