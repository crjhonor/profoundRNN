"""
Code to design the control panel using tkinter and visualizing the output result with matplotlib.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import os
import csv
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
FigureCanvasTkAgg,
NavigationToolbar2Tk
)
import I_seq2seq_mimic_Preproc as Is2smp
import I_seq2seq_mimic_model as Is2smm

"""
Define extra tkinter class
"""
outputs_dataDirName = "finalProba/allProba"

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
            'Commodity Index': tk.StringVar(),
            'Number of Epochs': tk.IntVar(),
            'Results': tk.StringVar()
        }

        self.columnconfigure(0, weight=1)

        top_frame = self._add_frame('Select The Commodity Index')
        LabelInput(top_frame, "Commodity Index", input_class=ttk.Combobox,
                   var=self._vars['Commodity Index'],
                   input_args={'values': self.indexes}
        ).grid(row=0, column=0)
        self._vars['Number of Epochs'].set(150)
        LabelInput(
            top_frame, 'Number of Epochs',  input_class=ttk.Spinbox, var=self._vars['Number of Epochs'],
            input_args={'from_': 0, 'to': 200, 'increment': 10}
        ).grid(row=0, column=1)

        middle_frame = self._add_frame('Actions')
        standard_train_predict_btn = ttk.Button(
            middle_frame,
            text='Standard_Train_Predict',
            command=self._standard_train_predict
        )
        standard_train_predict_btn.grid(row=0, column=0, sticky=(tk.W+tk.E))
        extra_train_predict_btn = ttk.Button(
            middle_frame,
            text='Extra_Train_Predict',
            command=self._extra_train_predict
        )
        extra_train_predict_btn.grid(row=0, column=1, sticky=(tk.W+tk.E))

        bottom_frame = self._add_frame('Results and Outputs')
        self.resultsText = LabelInput(
            bottom_frame, 'Results',
            input_class=BoundText,
            var=self._vars['Results'],
            input_args={'width': 100, 'height': 10}
        )
        self.resultsText.grid(sticky=(tk.W+tk.E), row=0, column=0, columnspan=3)

        # Visualizing outputs
        ttk.Label(bottom_frame, text='Visualizing outputs').grid(row=1, column=0, sticky=(tk.W+tk.E))
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.canvas_tkagg = FigureCanvasTkAgg(self.figure, bottom_frame)
        self.canvas_tkagg.get_tk_widget().grid(row=2, column=0, sticky=(tk.W+tk.E))

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

    def _generate_xy(self, output, classesTable):
        nor_eta = classesTable.shape[0]
        y = output - nor_eta / 2
        x = np.arange(0, len(output))
        radiant = np.linspace(1, 0, len(output))
        y_p_radiant = np.round([a * b for a, b in zip(y, radiant)], 0)
        y_radiant = np.cumsum(y_p_radiant)
        return x, y, y_radiant

    def _standard_train_predict(self):
        """
        Obtaining dataset PART =========================================================================================

        """
        if self._vars['Commodity Index'].get() == '':
            self.master._to_status('Please select a commodity index first.')
            return
        else:
            indexWanted = [self._vars['Commodity Index'].get()]
            self.master._to_status('Obtaining dataset with wanted index of ' + indexWanted[0] + ' ......')

        indexesRearrange = indexWanted.copy()
        for e in self.indexes:
            if e != indexWanted[0]:
                indexesRearrange.append(e)

        # Obtain the overall dataset.
        num_epochs = self._vars['Number of Epochs'].get()
        nb_classes = 1000
        rr = Is2smp.rawdataRead(indexesRearrange, nb_classes)
        rawDataset, classesTable = rr.readingRawdata()  # rawData should be the log return already
        sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time = rr.download_and_read(rawDataset)

        """
        RNN model build up and predict PART ============================================================================

        """
        standard_mimic_model = Is2smm.mimic_model(
            indexWanted, sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time, nb_classes, classesTable
        )
        self.output = standard_mimic_model.model_train(num_epochs=num_epochs)

        """
        Generating Results and Outputs PART ============================================================================

        """
        # Text results output
        returnResults = f"With index {indexWanted[0]} selected: \n" \
                        f"The X_predict_time is at time {X_predict_time} and \nthe feature extracted is \n " \
                        f"{str(X_predict)} \n The prediction is \n" \
                        f"{str(self.output)}"

        self._vars['Results'].set(returnResults)

        # Visualize results output
        self.figure.clf()
        self.fig1 = self.figure.add_subplot(1, 3, (1, 2))
        x, y, y_radiant = self._generate_xy(self.output, classesTable)
        line1 = self.fig1.plot(x, y, color='black')
        color = 'red' if y_radiant[-1] >= y_radiant[0] else 'green'
        fill1 = self.fig1.fill(x, y_radiant, color=color, alpha=0.8)
        self.fig1.set_xlabel('TIME')
        self.fig1.set_ylabel('CLASSES')
        self.fig1.set_title('STANDARD PREDICTION HEAT MAP of {:s}'.format(indexWanted[0]))
        self.fig1.legend(['Prediction', 'Heating'], loc='lower center')
        self.fig2 = self.figure.add_subplot(1, 3, 3)
        text2 = self.fig2.text(x=0, y=0.5,
                               ha='left', va='center', color='black',
                               bbox=dict(facecolor='red', alpha=0.5),
                               fontsize=18,
                               s=f'Script: I_seq2seq_mimic \n'
                                 f'Model: Many to Many GRU \n'
                                 f'Target: {indexWanted[0]} \n'
                                 f'At Date: {X_predict_time}')
        self.fig2.axis('off')
        self.canvas_tkagg.draw()
        # Save to file.
        outputs_filename = Path(Path(os.getcwd()).parents[1],
                                outputs_dataDirName,
                                '-'.join([str(datetime.now().date()), 'I_s2s_mimic_standard', indexWanted[0], 'save.jpg']))
        self.figure.savefig(outputs_filename)


    def _extra_train_predict(self):
        """
        Obtaining dataset PART =========================================================================================

        """
        if self._vars['Commodity Index'].get() == '':
            self.master._to_status('Please select a commodity index first.')
            return
        else:
            indexWanted = [self._vars['Commodity Index'].get()]
            self.master._to_status('Extra RNN model training with target of ' + indexWanted[0] + ' ......')

        indexesRearrange = indexWanted.copy()
        for e in self.indexes:
            if e != indexWanted[0]:
                indexesRearrange.append(e)

        # Obtain the overall dataset.
        num_epochs = self._vars['Number of Epochs'].get()
        nb_classes = 1000
        rr = Is2smp.rawdataRead(indexesRearrange, nb_classes)
        rawDataset, classesTable = rr.readingRawdata_extra()
        print('Done')
        # rawData should be the log return already
        sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time = rr.download_and_read(rawDataset)

        """
        RNN model build up and predict PART ============================================================================

        """
        extra_mimic_model = Is2smm.mimic_model(
            indexWanted, sents_en, sents_fr_in, sents_fr_out, X_predict, X_predict_time, nb_classes, classesTable
        )
        self.output_extra = extra_mimic_model.model_train(num_epochs=num_epochs)
        """
        Generating Results and Outputs PART ============================================================================

        """
        # Text results output
        returnResults = f"With index {indexWanted[0]} selected: \n" \
                        f"The X_predict_time is at time {X_predict_time} and \nthe feature extracted is \n " \
                        f"{str(X_predict)} \n The prediction is \n" \
                        f"{str(self.output_extra)}"

        self._vars['Results'].set(returnResults)

        # Visualize results output
        self.figure.clf()
        self.fig1 = self.figure.add_subplot(1, 3, (1,2))
        x, y, y_radiant = self._generate_xy(self.output_extra, classesTable)
        line1 = self.fig1.plot(x, y, color='black')
        color = 'red' if y_radiant[-1] >= y_radiant[0] else 'green'
        fill1 = self.fig1.fill(x, y_radiant, color=color, alpha=0.8)
        self.fig1.set_xlabel('TIME')
        self.fig1.set_ylabel('CLASSES')
        self.fig1.set_title('EXTRA RNN Model PREDICTION HEAT MAP of {:s}'.format(indexWanted[0]))
        self.fig1.legend(['Prediction', 'Heating'], loc='lower center')
        self.fig2 = self.figure.add_subplot(1, 3, 3)
        text2 = self.fig2.text(x=0, y=0.5,
                               ha='left', va='center', color='black',
                               bbox=dict(facecolor='red', alpha=0.5),
                               fontsize=18,
                               s=f'Script: I_seq2seq_mimic \n'
                                 f'Model: extra Many to Many GRU \n'
                                 f'Target: {indexWanted[0]} \n'
                                 f'At Date: {X_predict_time}')
        self.fig2.axis('off')
        self.canvas_tkagg.draw()
        # Save to file.
        outputs_filename = Path(Path(os.getcwd()).parents[1],
                                outputs_dataDirName,
                                '-'.join([str(datetime.now().date()), 'I_s2s_mimic_extra', indexWanted[0], 'save.jpg']))
        self.figure.savefig(outputs_filename)

class Application(tk.Tk):
    def __init__(self, *args, indexes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexes = indexes
        self.title('I_seq2seq_mimic')
        self.columnconfigure(0, weight=1)
        ttk.Label(
            self,
            text='SEQUENCE TO SEQUENCE MIMIC RNN ANALYSIS TO HOURLY TRADING DATA',
            font=('TkDefault', 16)
        ).grid(row=0, padx=10)

        self.processingWindow = processingWindow(self, indexes=indexes)
        self.processingWindow.grid(row=1, padx=10, sticky=(tk.W+tk.E))

        self.status = tk.StringVar()
        ttk.Label(
            self, textvariable=self.status
        ).grid(row=99, padx=10, sticky=tk.W+tk.E)

    def _to_status(self, text):
        self.status.set(text)