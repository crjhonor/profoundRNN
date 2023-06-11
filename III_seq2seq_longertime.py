"""
This code is aiming to analysis monthly data and to predict a necessary movement of the coming month. First of all, the
monthly data is derived from daily trading data, and, of course, from the macroeconomic data.
How am I going to cook the raw data depends on two parts. One is the trading data, I will derive the raw data in multi-
target generated label and feature for the one to many model. Then, multi-generated label and feature for the many to
many model is going to be derived from the label with months ahead data sequence and macroeconomic data with correlation
matrix self attention. But, the macroeconomic data need to be transformed into normalized forms. And I still have to
find a way to do that.
"""
import os
from pathlib import Path
import datetime

"""
One to many model-------------------------------------------------------------------------------------------------------
"""
# Obtaining the dataset.
import III_seq2seq_longertimepreproc as IIIs2spp
NUM_CLASSES = 1000
rr = IIIs2spp.rawdataRead(NUM_CLASSES)
X_dataset, y_dataset = rr.read_load()

# Creating and training the model.
import III_seq2seq_longertime_RNN as IIIs2sltrnn
longertime_model = IIIs2sltrnn.onetoManyRNN(
    X_dataset=X_dataset,
    y_dataset=y_dataset,
    num_classes=NUM_CLASSES,
    num_epochs=10000
)

"""
Then let me design the tkinter for pretty face.
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.style.use('ggplot')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
FigureCanvasTkAgg,
NavigationToolbar2Tk
)

# More definition of commodity indexes
indexWanted_CU0 = ['CU0', 'P0', 'Y0', 'AG0', 'BU0', 'ZN0', 'C0', 'AL0', 'RM0', 'M0', 'CF0']
indexWanted_RB0 = ['RB0', 'HC0', 'I0', 'V0', 'BU0', 'PP0', 'L0', 'FG0', 'MA0', 'SA0', 'SR0']
indexWanted_SCM = ['SCM', 'AU0', 'PG0', 'EB0', 'FU0', 'TA0', 'PP0', 'L0', 'V0', 'MA0', 'RU0']

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
            self.label.grid(row=0, column=0, sticky=(tk.W+tk.E))

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

        self.input.grid(row=1, column=0, sticky=(tk.W+tk.E))
        self.columnconfigure(0, weight=1)

    def grid(self, sticky=(tk.E + tk.W), **kwargs):
        super().grid(sticky=sticky, **kwargs)

"""
Widget and frame creation.
"""

class processingWindow(ttk.Frame):
    def __init__(self, *args, indexes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self._vars = {
            'TARGET': tk.StringVar()
        }

        self.columnconfigure(1, weight=1)

        self._vars['TARGET'].set('RU0')

        top_frame = self._add_frame(label='Results Windows')
        top_frame.grid(row=0, column=0, sticky=(tk.E + tk.W))
        # Visualizing outputs
        ttk.Label(top_frame, text='Visualizing outputs').grid(row=0, column=0, sticky=(tk.W+tk.E))
        self.figure = Figure(figsize=(16, 8), dpi=100)
        self.canvas_tkagg = FigureCanvasTkAgg(self.figure, top_frame)
        self.canvas_tkagg.get_tk_widget().grid(row=1, column=0, sticky=(tk.W+tk.E))

        middle_frame = self._add_frame(label='Commodity Index Action')
        middle_frame.grid(row=1, column=0, sticky=(tk.E + tk.W))

        LabelInput(middle_frame, "SELECT THE TARGET", input_class=ttk.Combobox,
                   var=self._vars['TARGET'],
                   input_args={'values': self.indexes}
        ).grid(row=0, column=0, pady=5)
        target_predict_btn = ttk.Button(
            middle_frame,
            text='Target Predict',
            command=lambda: self._single_predict(ind=self._vars['TARGET'].get())
        )
        target_predict_btn.grid(row=0, column=1, sticky=(tk.W + tk.S + tk.E), pady=5)

        bottom_frame = self._add_frame(label='Multiple Action')
        bottom_frame.grid(row=2, column=0, sticky=(tk.E + tk.W))
        all_predict_btn = ttk.Button(
            bottom_frame,
            text='Predict All',
            command=lambda: self._all_predict(ind=self.indexes)
        )
        all_predict_btn.grid(row=0, column=0, sticky=(tk.E + tk.W))

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
        x = y.index
        radiant = np.linspace(1, 0, len(output))
        y_p_radiant = np.round([a * b for a, b in zip(y, radiant)], 0)
        y_radiant = np.cumsum(y_p_radiant)
        return x, y, y_radiant

    def _single_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', ind]))

        X_predict, X_predict_date = rr.read_predict([ind])
        predict_output = longertime_model.predictXone(X_predict, X_predict_date)

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
                               s=f'Script: III_seq2seq_longertime \n'
                                 f'Model: one to Many GRU \n'
                                 f'Target: {ind} \n'
                                 f'At Date: {X_predict_date}')
        self.fig2.axis('off')
        self.canvas_tkagg.draw()

        # Save to file.
        outputs_filename = Path(Path(os.getcwd()).parents[1],
                                outputs_dataDirName,
                                '-'.join([str(X_predict_date), 'III_s2s_longertime', ind, 'save.jpg']))
        self.figure.savefig(outputs_filename)

    def _all_predict(self, ind):
        # Setting the master's status
        self.master._to_status(" ".join(['Prediction of single target to', " ".join(ind)]))

        for item in ind:
            self._single_predict(item)
            self.master.update()

class Application(tk.Tk):
    def __init__(self, *args, indexes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self.title('III_seq2seq_longertime')
        self.columnconfigure(0, weight=1)
        ttk.Label(
            self,
            text='SEQUENCE TO SEQUENCE LONGER TIME ANALYSIS TO MONTHLY DATA',
            font=('TkDefault', 16)
        ).grid(row=0, padx=10)

        self.processingWindow = processingWindow(self,
                                                 indexes=indexes)
        self.processingWindow.grid(row=1, padx=10, sticky=(tk.W+tk.E))

        self.status = tk.StringVar()
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=tk.W+tk.E)

    def _to_status(self, text):
        self.status.set(text)

App = Application(indexes=indexList)
App.mainloop()