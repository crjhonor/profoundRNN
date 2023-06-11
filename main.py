"""
Inspired by sentiCourse project and amid the increasing skillful of the deep learning coding, I am deciding on fixed
selection of the RNN model, either sequence to one or sequence to sequence, with transformer technic as the model to
analysis the trading data, both hourly and daily. Besides, yields, currencies and macro data are also considered to make
predictions. Seasonal models are also developed as they can be useful for target selecting, but prediction may be less
precise due to the small size of seasonal dataset.
"""

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""
MISSION I:
I_seq2seq_mimic.py code is designed to handel the hourly trading dataset. Along with:
I_seq2seq_mimic_Preproc.py: The code to obtain and preprocessing dataset.
I_seq2seq_mimic_model.py: The code to build the RNN model and predict.
I_seq2seq_mimic_tk.py: The code for control panel and output visualization.
Improvements:
RNN model build up and training are transfer into the I_seq2seq_mimic_model_bk.py for importing, thus make the original
code more clearly for viewing.
The encoder and decoder parameters of every commodities respectively are saved into ./dl_data folder and while using the
I_seq2seq_mimic_predict.py they can be loaded into prediction model and make the predictions.
With more friendly GUI using tkinter, I am trying good looking as a big improvement of all the codes.
What's more, I created two datasets to implement the RNN model, one is the dataset will only close data involved; and
the second, dataset will include hodrick prescott filter generated trends and cycles of every commodity. 
"""

"""
Mission II:
II_seq2seq_moon2sun.py: code is designed to analyse the daily trading data. The premature RNN model I used earlier will 
also be used for dicing predictions and I believe this method maybe helpful.
II_seq2seq_moon2sun_cook.py: For the convenience of applying multiple models, amid the prolonging processes of reading 
and analysing data, this code is to perform all the necessary steps to obtain the data and save them into files within 
the ./dataForlater directory for modeling.
II_seq2seq_moon2sun_model: The model code, and I am trying to implement one model for all data which makes a log larger
dataset by re targeting label and rearrange the features according to correlation matrix. Which I think it is similar 
to self attention method.
II_dicingerPro_S.py: the code designed in 2022 which generate many to one prediction results using 3 deep learning
models and the outputs are used in table calculation. With sentiment results.
II_dicingerPro_TD.py: the code designed in 2022 which generate many to one prediction results using 3 deep learning
models and the outputs are used in table calculation. With only daily trading data.
"""

"""
Mission III:
III_seq2seq_longertime.py: similar to Mission II, I am implementing the similar strategy of using RNN network to monthly
trading data to predict the monthly change.
III_seq2seq_longertimepreproc.py: code to prepare and process the dataset in Mission III.
III_seq2seq_longertime_RNN.py: code to build up deep learning model and predict.
III_seq2seq_longertime_MACRO.py: additional code that are trying to analysis the monthly trading data together with the 
macro economic data.
"""

"""
Mission IV:
IV_many2one_dicingerpro.py: This is the idea that using many to one model, I am able to predict in massive way that may
generate the odds of either going up or down. I will use the old model I developed in the middle of last year to
generate the results.
"""

"""
Mission V:
Always have to add new things in the tail that Maybe I will be able to solve the probability matrix of trading. First
but not least, I am thinking about using the FED's releases as input to transformers like GPT-3 and etc, let's see what
results we can get. 
"""