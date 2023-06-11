"""
MISSION II:
What am I going to is to use the sequence to sequence RNN network models to predict a series of returns ahead with the
features of collected data up to present day.
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

def datetimeProcessing(dataToproc):
    # functions to reprocess the DATE formate
    dataToproc['DATE_'] = ''
    for i in range(len(dataToproc)):
        dataToproc['DATE_'].iloc[i] = dataToproc['DATE'].iloc[i].to_pydatetime().date()
    dataToproc.index = dataToproc['DATE_']
    return dataToproc


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


def fnl_reProcessing(emailFeatures, weiboFeatures, TDFeatures, labels, num_classes=4):
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
    # join the TDFeatures
    fnl_fd = fnl_fd.join(TDFeatures, rsuffix='_TD')
    # More processing to standardize the features.
    for item in ['positive_probs_avg', 'negative_probs_avg', 'positive_result_pct', 'negative_result_pct',
                 'positive_probs_avg_weibo', 'negative_probs_avg_weibo',
                 'positive_result_pct_weibo', 'negative_result_pct_weibo']:
        bf = fnl_fd[item]
        bf_array = np.array(bf)
        bf_array = bf_array - bf_array.mean()
        aft, _ = to_classes(bf_array, classes=num_classes)
        fnl_fd[item] = aft.astype(np.int32)
    fnl_fd = fnl_fd.drop(columns="DATE")
    return fnl_fd

indexWanted_up = ['AL0', 'I0', 'CF0', 'FG0', 'CF0']  # Later will have the pretty tkinter interface.
indexWanted_dn = ['ZN0', 'L0', 'SA0', 'V0', 'TA0']
indexWanted = list(np.unique(indexWanted_up + indexWanted_dn))

# Also read the indexes
TD_indexes = pd.read_csv('/home/crjLambda/PRO80/DailyTDs/ref_TD.csv')
TD_yields_indexes = pd.read_csv('/home/crjLambda/PRO80/DailyTDs/ref_yields.csv')
TD_Currency_indexes = pd.read_csv('/home/crjLambda/PRO80/DailyTDs/ref_Currency.csv')

# And generate wanted dataset
indexesAll = TD_indexes.join(TD_Currency_indexes, rsuffix='_Currency')
# indexesAll = indexesAll.join(TD_yields_indexes, rsuffix='_yields')
indexesAll_ind = indexesAll.iloc[0,]

NUM_CLASSES = 1000

import II_getLabels as gl
import II_getTD as gt
from tqdm import tqdm

"""
What I am thinking right now is that, if, targeted commodities are correlated, then, perhaps. The prices of each target
's close will also be correlated. And then, although the size of the original dataset is very limited, but I can use 
different labels with features to enlarge the size of the dataset.

Now it will be very insteresting to generate the label sequences and feature sequences with different target commodity
as label. With current dataset size uptodate, I can generate a lot larger dataset.

One model for all.
"""
# Generating the features.----------------------------------------------------------------------------------------------
maxlen_en = 65  # the length of the total features' length including emailFeatures, weiboFeatures and TDFeatures
maxlen_fr = 25  # stand for 25 trading days which are very close to one month
sents_en, sents_fr_in, sents_fr_out = [], [], []

for i in tqdm(range(len(indexesAll_ind)), ncols=100, desc="Generating dataset", colour="blue"):
    ind = [indexesAll_ind[i]]
    # Including sentiments results................................................
    glReturn = gl.getLabels(indexWanted=ind, num_classes=NUM_CLASSES)
    labelsDL_df, labels_classesTable = glReturn.returnLabel, glReturn.classesTable
    # Including other trade data results..........................................
    TDFeatures = gt.getDataclose(num_classes=NUM_CLASSES)
    TDFeatures = TDFeatures.drop(columns="".join([ind[0], "Close"]))
    fnl = fnl_reProcessing(emailFeatures=emailFeatures_df,
                           weiboFeatures=weiboFeatures_df,
                           TDFeatures=TDFeatures,
                           labels=labelsDL_df,
                           num_classes=NUM_CLASSES)
    fnl = fnl.dropna()
    fnl = fnl.drop(columns='DATE_')
    # Self attention that matters, I need to rearrange the order of columns according to correlations.
    fnl_corr = fnl.corr()
    newOrder = fnl_corr.iloc[0, :].sort_values(axis=0, ascending=False)
    # Having label and all the features ready, now it is time to generate the label dataset and feature dataset.
    fnl = fnl[newOrder.index.to_list()]
    for j in range(fnl.shape[0] - maxlen_en):
        sent_en = fnl.iloc[j, 1:].to_numpy().astype(np.int32)
        sent_fr = fnl.iloc[j:j + maxlen_fr, 0].to_numpy().astype(np.int32)
        # add 2 in front of the label as the BOS
        sent_fr_in = np.append(np.array(2), sent_fr).astype(np.int32)
        # add 3 at the end of the label as the EOS
        sent_fr_out = np.append(sent_fr, np.array(3)).astype(np.int32)
        sents_en.append(sent_en)
        sents_fr_in.append(sent_fr_in)
        sents_fr_out.append(sent_fr_out)

"""
RNN MODEL BUILD UP.
"""
import II_seq2seq_moon2sun_model as IIs2smm

moon2sun_model = IIs2smm.moon2sun_model(
    sents_en=sents_en,
    sents_fr_in=sents_fr_in,
    sents_fr_out=sents_fr_out,
    num_classes=NUM_CLASSES,
    num_epochs=100
)

"""
Time to design the pretty face with tkinter GUI
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
matplotlib.style.use('Solarize_Light2')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

# More definition of commodity indexes
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'JM0', 'UR0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ['SCM', 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'LUM', 'RU0']

# Include all the interested commodity indexes and making the target index as the first one.
indexList = list(np.unique(indexWanted_CU0 + indexWanted_RB0 + indexWanted_SCM))

outputs_dataDirName = "finalProba/allProba"

# Define extra tkinter class
class BoundText(tk.Text):
    def __init__(self, *args, textvariable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._variable = textvariable
        if self._variable:
            self.insert('1.0', self._variable.get())
            self._variable.trace_add('write', self._set_content)
            self.bind('<<Modified>>', self._set_var)

    def _set_content(self, *_):
        self.delete('1.0', tk.END)
        self.insert('1.0', self._variable.get())

    def _set_var(self, *_):
        if self.edit_modified():
            content = self.get('1.0', 'end-1chars')
            self._variable.set(content)
            self.edit_modified(False)

class LabelInput(tk.Frame):
    def __init__(self, parent, label, var, input_class=tk.Entry,
                 input_args=None, label_args=None, **kwargs):
        super().__init__(parent, **kwargs)
        input_args = input_args or {}
        label_args = label_args or {}
        self.variable = var
        self.variable.label_widget = self

        if input_class in (ttk.Checkbutton, ttk.Button):
            input_args['text'] = label
        else:
            self.label = ttk.Label(self, text=label, **label_args)
            self.label.grid(row=0, column=0, sticky=(tk.W + tk.E))

        if input_class in (
                ttk.Checkbutton, ttk.Button, ttk.Radiobutton
        ):
            input_args['variable'] = self.variable
        else:
            input_args['textvariable'] = self.variable

        # setup the input
        if input_class == ttk.Radiobutton:
            # for Radiobutton, create one input per value
            self.input = tk.Frame(self)
            for v in input_args.pop('values', []):
                button = ttk.Radiobutton(
                    self.input, value=v, text=v, **input_args
                )
                button.pack(side=tk.LEFT, ipadx=10, ipady=2, expand=True, fill='x')
        else:
            self.input = input_class(self, **input_args)

        self.input.grid(row=1, column=0, sticky=(tk.W + tk.E))
        self.columnconfigure(0, weight=1)

    def grid(self, sticky=(tk.E + tk.W), **kwargs):
        super().grid(sticky=sticky, **kwargs)

"""
Widget and frame creation.
"""
class processingWindow(ttk.Frame):
    def __init__(self, *args, indexes=None, indexWanted_up=None, indexWanted_dn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self.indexWanted_up = indexWanted_up
        self.indexWanted_dn = indexWanted_dn
        self._vars = {
            'UP1': tk.StringVar(),
            'UP2': tk.StringVar(),
            'UP3': tk.StringVar(),
            'UP4': tk.StringVar(),
            'UP5': tk.StringVar(),
            'DN1': tk.StringVar(),
            'DN2': tk.StringVar(),
            'DN3': tk.StringVar(),
            'DN4': tk.StringVar(),
            'DN5': tk.StringVar(),
            'Number of Epochs': tk.IntVar(),
            'Results': tk.StringVar()
        }

        self.columnconfigure(1, weight=1)

        self._vars['UP1'].set(self.indexWanted_up[0])
        self._vars['UP2'].set(self.indexWanted_up[1])
        self._vars['UP3'].set(self.indexWanted_up[2])
        self._vars['UP4'].set(self.indexWanted_up[3])
        self._vars['UP5'].set(self.indexWanted_up[4])
        self._vars['DN1'].set(self.indexWanted_dn[0])
        self._vars['DN2'].set(self.indexWanted_dn[1])
        self._vars['DN3'].set(self.indexWanted_dn[2])
        self._vars['DN4'].set(self.indexWanted_dn[3])
        self._vars['DN5'].set(self.indexWanted_dn[4])

        left_frame = self._add_frame(label='Commodity Index Action', cols=1)
        left_frame.grid(row=0, column=0, sticky=(tk.N + tk.S))


        left_t_frame = ttk.LabelFrame(master=left_frame, text='Indexes UP')
        left_t_frame.grid(row=0, column=0, sticky=(tk.W + tk.E))
        LabelInput(left_t_frame, "UP1", input_class=ttk.Combobox,
                   var=self._vars['UP1'],
                   input_args={'values': self.indexes}
                   ).grid(row=0, column=0, pady=5)
        UP1_predict_btn = ttk.Button(
            left_t_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['UP1'].get())
        )
        UP1_predict_btn.grid(row=0, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_t_frame, "UP2", input_class=ttk.Combobox,
                   var=self._vars['UP2'],
                   input_args={'values': self.indexes}
                   ).grid(row=1, column=0, pady=5)
        UP2_predict_btn = ttk.Button(
            left_t_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['UP2'].get())
        )
        UP2_predict_btn.grid(row=1, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_t_frame, "UP3", input_class=ttk.Combobox,
                   var=self._vars['UP3'],
                   input_args={'values': self.indexes}
                   ).grid(row=2, column=0, pady=5)
        UP3_predict_btn = ttk.Button(
            left_t_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['UP3'].get())
        )
        UP3_predict_btn.grid(row=2, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_t_frame, "UP4", input_class=ttk.Combobox,
                   var=self._vars['UP4'],
                   input_args={'values': self.indexes}
                   ).grid(row=3, column=0, pady=5)
        UP4_predict_btn = ttk.Button(
            left_t_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['UP4'].get())
        )
        UP4_predict_btn.grid(row=3, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_t_frame, "UP5", input_class=ttk.Combobox,
                   var=self._vars['UP5'],
                   input_args={'values': self.indexes}
                   ).grid(row=4, column=0, pady=5)
        UP5_predict_btn = ttk.Button(
            left_t_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['UP5'].get())
        )
        UP5_predict_btn.grid(row=4, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        ALLUPs_predict_btn = ttk.Button(
            left_t_frame,
            text='Predict all UPs',
            command=lambda: self._all_predict(
                ind=[
                    self._vars['UP1'].get(),
                    self._vars['UP2'].get(),
                    self._vars['UP3'].get(),
                    self._vars['UP4'].get(),
                    self._vars['UP5'].get()
                ]
            )
        )
        ALLUPs_predict_btn.grid(row=5, column=0, columnspan=2, sticky=(tk.W + tk.S + tk.E), pady=5)

        left_b_frame = ttk.LabelFrame(master=left_frame, text='Indexes DN')
        left_b_frame.grid(row=1, column=0, sticky=(tk.W + tk.E))
        LabelInput(left_b_frame, "DN1", input_class=ttk.Combobox,
                   var=self._vars['DN1'],
                   input_args={'values': self.indexes}
                   ).grid(row=0, column=0, pady=5)
        DN1_predict_btn = ttk.Button(
            left_b_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['DN1'].get())
        )
        DN1_predict_btn.grid(row=0, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_b_frame, "DN2", input_class=ttk.Combobox,
                   var=self._vars['DN2'],
                   input_args={'values': self.indexes}
                   ).grid(row=1, column=0, pady=5)
        DN2_predict_btn = ttk.Button(
            left_b_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['DN2'].get())
        )
        DN2_predict_btn.grid(row=1, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_b_frame, "DN3", input_class=ttk.Combobox,
                   var=self._vars['DN3'],
                   input_args={'values': self.indexes}
                   ).grid(row=2, column=0, pady=5)
        DN3_predict_btn = ttk.Button(
            left_b_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['DN3'].get())
        )
        DN3_predict_btn.grid(row=2, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_b_frame, "DN4", input_class=ttk.Combobox,
                   var=self._vars['DN4'],
                   input_args={'values': self.indexes}
                   ).grid(row=3, column=0, pady=5)
        DN4_predict_btn = ttk.Button(
            left_b_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['DN4'].get())
        )
        DN4_predict_btn.grid(row=3, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        LabelInput(left_b_frame, "DN5", input_class=ttk.Combobox,
                   var=self._vars['DN5'],
                   input_args={'values': self.indexes}
                   ).grid(row=4, column=0, pady=5)
        DN5_predict_btn = ttk.Button(
            left_b_frame,
            text='Predict',
            command=lambda: self._single_predict(ind=self._vars['DN5'].get())
        )
        DN5_predict_btn.grid(row=4, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)
        ALLDNs_predict_btn = ttk.Button(
            left_b_frame,
            text='Predict all DNs',
            command=lambda: self._all_predict(
                ind=[
                    self._vars['DN1'].get(),
                    self._vars['DN2'].get(),
                    self._vars['DN3'].get(),
                    self._vars['DN4'].get(),
                    self._vars['DN5'].get()
                ]
            )
        )
        ALLDNs_predict_btn.grid(row=5, column=0, columnspan=2, sticky=(tk.W + tk.S + tk.E), pady=5)

        right_frame = self._add_frame(label='Results Windows')
        right_frame.grid(row=0, column=1, sticky=(tk.N + tk.S))
        # Visualizing outputs
        ttk.Label(right_frame, text='Visualizing outputs').grid(row=0, column=0, sticky=(tk.W + tk.E))
        self.figure = Figure(figsize=(18, 10), dpi=100)
        self.canvas_tkagg = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas_tkagg.get_tk_widget().grid(row=1, column=0, sticky=(tk.W + tk.E))

    def _add_frame(self, label, cols=3):
        frame = ttk.LabelFrame(self, text=label)
        frame.grid(sticky=(tk.W + tk.E))
        for i in range(cols):
            frame.columnconfigure(i, weight=1)
        return frame

    def get(self):
        data = dict()
        for key, variable in self._vars.items():
            data[key] = ''
        return data

    def _generate_xy(self, output):
        nor_eta = NUM_CLASSES
        y = output - 3 - nor_eta / 2
        x = np.arange(0, len(output))
        radiant = np.linspace(1, 0, len(output))
        y_p_radiant = np.round([a * b for a, b in zip(y, radiant)], 0)
        y_radiant = np.cumsum(y_p_radiant)
        return x, y, y_radiant

    def _single_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', ind]))

        # Generating prediction feature, the X_predict and X_predict_date
        # Including sentiments results................................................
        glReturn = gl.getLabels(indexWanted=[ind], num_classes=NUM_CLASSES)
        labelsDL_df, labels_classesTable = glReturn.returnLabel, glReturn.classesTable
        # Including other trade data results..........................................
        TDFeatures = gt.getDataclose(num_classes=NUM_CLASSES)
        TDFeatures = TDFeatures.drop(columns="".join([ind, "Close"]))
        fnl = fnl_reProcessing(emailFeatures=emailFeatures_df,
                               weiboFeatures=weiboFeatures_df,
                               TDFeatures=TDFeatures,
                               labels=labelsDL_df,
                               num_classes=NUM_CLASSES)
        fnl = fnl.dropna()
        fnl = fnl.drop(columns='DATE_')
        # Self attention that matters, I need to rearrange the order of columns according to correlations.
        fnl_corr = fnl.corr()
        newOrder = fnl_corr.iloc[0, :].sort_values(axis=0, ascending=False)
        # Having label and all the features ready, now it is time to generate the label dataset and feature dataset.
        fnl = fnl[newOrder.index.to_list()]
        X_predict = fnl.iloc[-1, 1:].to_numpy().astype(np.int32)
        X_predict_date = pd.to_datetime(fnl.index[-1]).date()

        predict_output = moon2sun_model.predictXone(X_predict, X_predict_date)

        # Visualize results output
        self.figure.clf()
        self.fig1 = self.figure.add_subplot(2, 3, (1, 2))
        x, y, y_radiant = self._generate_xy(predict_output)
        line1 = self.fig1.plot(x, y, color='black')
        color = 'red' if y_radiant[-1] >= y_radiant[0] else 'green'
        fill1 = self.fig1.fill(x, y_radiant, color=color, alpha=0.8)
        self.fig1.set_xlabel('TIME')
        self.fig1.set_ylabel('CLASSES')
        self.fig1.set_title('PREDICTION HEAT MAP OF {:s}.'.format(ind))
        self.fig1.legend(['Prediction', 'Heating'], loc='upper left')
        self.fig2 = self.figure.add_subplot(2, 3, 3)
        text2 = self.fig2.text(x=0, y=0.5,
                               ha='left', va='center', color='black',
                               bbox=dict(facecolor='red', alpha=0.5),
                               fontsize=18,
                               s=f'Script: II_seq2seq_moon2sun \n'
                                 f'Model: GRU \n'
                                 f'Target: {ind} \n'
                                 f'At Date: {X_predict_date}')
        self.fig2.axis('off')
        self.canvas_tkagg.draw()

        # Save to file.
        outputs_filename = Path(Path(os.getcwd()).parents[1],
                                outputs_dataDirName,
                                '-'.join([str(datetime.datetime.now().date()), 'II_s2s_moon2sun', ind, 'save.jpg']))
        self.figure.savefig(outputs_filename)

    def _all_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', " ".join(ind)]))

        # Visualize results output
        self.figure.clf()
        self.axes = self.figure.subplots(2, 3)
        for i in range(len(ind)):
            # Generating prediction feature, the X_predict and X_predict_date
            # Including sentiments results................................................
            glReturn = gl.getLabels(indexWanted=[ind[i]], num_classes=NUM_CLASSES)
            labelsDL_df, labels_classesTable = glReturn.returnLabel, glReturn.classesTable
            # Including other trade data results..........................................
            TDFeatures = gt.getDataclose(num_classes=NUM_CLASSES)
            TDFeatures = TDFeatures.drop(columns="".join([ind[i], "Close"]))
            fnl = fnl_reProcessing(emailFeatures=emailFeatures_df,
                                   weiboFeatures=weiboFeatures_df,
                                   TDFeatures=TDFeatures,
                                   labels=labelsDL_df,
                                   num_classes=NUM_CLASSES)
            fnl = fnl.dropna()
            fnl = fnl.drop(columns='DATE_')
            # Self attention that matters, I need to rearrange the order of columns according to correlations.
            fnl_corr = fnl.corr()
            newOrder = fnl_corr.iloc[0, :].sort_values(axis=0, ascending=False)
            # Having label and all the features ready, now it is time to generate the label dataset and feature dataset.
            fnl = fnl[newOrder.index.to_list()]
            X_predict = fnl.iloc[-1, 1:].to_numpy().astype(np.int32)
            X_predict_date = pd.to_datetime(fnl.index[-1]).date()

            predict_output = moon2sun_model.predictXone(X_predict, X_predict_date)

            x, y, y_radiant = self._generate_xy(predict_output)
            self.axes[i // 3, i % 3].plot(x, y, color='black')
            color = 'red' if y_radiant[-1] >= y_radiant[0] else 'green'
            self.axes[i // 3, i % 3].fill(x, y_radiant, color=color, alpha=0.8)
            self.axes[i // 3, i % 3].set_xlabel('TIME')
            self.axes[i // 3, i % 3].set_ylabel('CLASSES')
            self.axes[i // 3, i % 3].set_title('PREDICTION HEAT MAP OF {:s}.'.format(ind[i]))
            self.axes[i // 3, i % 3].legend(['Prediction', 'Heating'], loc='upper left')

        self.axes[1, 2].text(x=0, y=0.5,
                             ha='left', va='center', color='black',
                             bbox=dict(facecolor='red', alpha=0.5),
                             fontsize=18,
                             s=f'Script: II_seq2seq_moon2sun \n'
                               f'Model: GRU \n'
                               f'Target: {" ".join(ind)} \n'
                               f'At Date: {X_predict_date}')
        self.axes[1, 2].axis('off')
        self.canvas_tkagg.draw()

        # Save to file.
        outputs_filename = Path(Path(os.getcwd()).parents[1],
                                outputs_dataDirName,
                                '-'.join([str(datetime.datetime.now().date()), 'II_s2s_moon2sun', '-'.join(ind), 'save.jpg']))
        self.figure.savefig(outputs_filename)

class Application(tk.Tk):
    def __init__(self, *args, indexes=None, indexWanted_up=None, indexWanted_dn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self.indexWanted_up = indexWanted_up
        self.indexWanted_dn = indexWanted_dn
        self.title('II_seq2seq_moon2sun')
        self.columnconfigure(0, weight=1)
        ttk.Label(
            self,
            text='SEQUENCE TO SEQUENCE MOON 2 SUN ANALYSIS TO DAILY DATA',
            font=('TkDefault', 16)
        ).grid(row=0, padx=10)

        self.processingWindow = processingWindow(self,
                                                 indexes=indexes,
                                                 indexWanted_up=indexWanted_up,
                                                 indexWanted_dn=indexWanted_dn)
        self.processingWindow.grid(row=1, padx=10, sticky=(tk.W + tk.E))

        self.status = tk.StringVar()
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=tk.W + tk.E)

    def _to_status(self, text):
        self.status.set(text)


indexWanted = list(np.unique(indexWanted_up + indexWanted_dn))

App = Application(indexes=indexList, indexWanted_up=indexWanted_up, indexWanted_dn=indexWanted_dn)
App.mainloop()
