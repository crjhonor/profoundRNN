"""
This code is including preprocessing method needed for somethingElse main code.
"""
import pandas as pd
import numpy as np
import math

def generate_logr(dataset, isDATE=True):
    if isDATE:
        dataset_DATE = dataset['DATE']
        dataset_noDATE = dataset.drop(columns=['DATE'])
    else:
        dataset_noDATE = dataset

    dataset_noDATE_pct_change = dataset_noDATE.pct_change(periods=1)
    dataset_noDATE_pct_change = dataset_noDATE_pct_change.iloc[1:]
    dataset_noDATE_logr = dataset_noDATE_pct_change.applymap(lambda x: np.log(x + 1))

    if isDATE:
        dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
        returnDataset = dataset_DATE.join(dataset_noDATE_logr)
    else:
        returnDataset = dataset_noDATE_logr
    return returnDataset

def generate_onehot(dataset, isDATE=True):
    if isDATE:
        dataset_DATE = dataset['DATE']
        dataset_noDATE = dataset.drop(columns=['DATE'])
    else:
        dataset_noDATE = dataset

    dataset_noDATE_pct_change = dataset_noDATE.pct_change(periods=1)
    dataset_noDATE_pct_change = dataset_noDATE_pct_change.iloc[1:]
    dataset_noDATE_onehot = dataset_noDATE_pct_change.applymap(lambda x: int(1) if x >= 0 else int(0))

    if isDATE:
        dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
        returnDataset = dataset_DATE.join(dataset_noDATE_onehot)
    else:
        returnDataset = dataset_noDATE_onehot
    return returnDataset

def generate_MultiClassesLabel(dataset, classes=4, isDATE=True):
    assert classes > 2, "classes must > 2 "
    if isDATE:
        dataset_DATE = dataset['DATE']
        dataset_noDATE = dataset.drop(columns=['DATE'])
    else:
        dataset_noDATE = dataset

    dataset_noDATE_pct_change = dataset_noDATE.pct_change(periods=1)
    dataset_noDATE_pct_change = dataset_noDATE_pct_change.iloc[1:]
    # GR10_yryClose and INA_10yryClose have some unsuitable data.
    dataset_noDATE_pct_change = dataset_noDATE_pct_change.drop(columns=['GR_10yryClose', 'INA_10yryClose'])

    def to_classes(x, classes=4):
        x_div = 2*x.std()/(classes-2)
        x_ceil = np.array([math.ceil(m / x_div) for m in x])
        x_range = np.array([m for m in range(-int(classes/2-1), int(classes/2+1))])
        # Adjust the tail
        for i in range(len(x_ceil)):
            if x_ceil[i] < x_range.min():
                x_ceil[i] = x_range.min()
            elif x_ceil[i] > x_range.max():
                x_ceil[i] = x_range.max()
        return x_ceil

    for col in dataset_noDATE_pct_change.columns:
        x_array = dataset_noDATE_pct_change[col].values
        x_classes = to_classes(x_array)
        dataset_noDATE_pct_change[col] = x_classes

    if isDATE:
        dataset_DATE = pd.DataFrame(dataset_DATE.iloc[1:])
        returnDataset = dataset_DATE.join(dataset_noDATE_pct_change)
    else:
        returnDataset = dataset_noDATE_pct_change
    return returnDataset

def oneDaylagging(dataset, indexWanted, suffix='Close'):
    assert suffix in ['Close', '_hpft', '_hpfc'], "suffix can only be Close, _hpft or _hpfc"
    label = dataset[indexWanted[0]+suffix]
    feature = dataset.drop(columns=['DATE', indexWanted[0]+suffix])
    # Generate one day lagging.
    label = np.array(label[1:])
    feature = np.array(feature.iloc[:-1, :])
    return label, feature

def oneDay_pred_feature(dataset, indexWanted, suffix='Close'):
    assert suffix in ['Close', '_hpft', '_hpfc'], "suffix can only be Close, _hpft or _hpfc"
    feature = dataset.drop(columns=['DATE', indexWanted[0]+suffix])
    # Generate one day predicting feature
    feature = np.array(feature.iloc[-1])
    return feature
