"""
The initiative of this code is to use more profound technic which currently used in the Recurrent Neural Network deep
learning and, try to use them upon hourly trading dataset. This dataset will not only involve log return of a target
commodity hourly trading data but will also involve a basket of commodities with highly correlations.

During the daily trading, it is essential to obtain more accurate model for hourly return of the target commodity which
should be able to guide the buy in and sell off actions. The purpose is to reduce the loss and cost coming from
volatility and uncertainty.
"""
import nltk
import I_seq2seq_mimic_Preproc as Is2smp
import I_seq2seq_mimic_tk as Is2smtk
import numpy as np
import re
import shutil
import tensorflow as tf
import os
import unicodedata
import zipfile
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

"""
Obtaining dataset PART =================================================================================================

"""
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'JM0', 'UR0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ['SCM', 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'LUM', 'RU0']

# Include all the interested commodity indexes and making the target index as the first one.
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

"""
tkinter and visualizing PART ===========================================================================================

"""
mimicApp = Is2smtk.Application(indexes=indexList)
mimicApp.mainloop()

print("Done")