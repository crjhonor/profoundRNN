import paddlehub as hub
# initial GPU for paddlehub
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define functions for sentiment testing................................................................................
def sentimentParralletest(keyboardInput):
    """
    # First sentiment model is ernie_skep_sentiment_analysis from paddlehub.............................................
    senta_essa = hub.Module(name="ernie_skep_sentiment_analysis")
    inputdict = {keyboardInput}
    essa_results = senta_essa.predict_sentiment(texts = inputdict, use_gpu = True)
    """

    # Second sentiment model is bilstm..................................................................................
    senta_bilstm = hub.Module(name="senta_bilstm")
    test_text = [keyboardInput]
    bilstm_results = senta_bilstm.sentiment_classify(texts = test_text,
                                            use_gpu = True,
                                             batch_size = 1)

    """
    # Third sentiment model is emotion_detection_textcnn................................................................
    senta_edtcnn = hub.Module(name = "emotion_detection_textcnn")
    test_text = [keyboardInput]
    edtcnn_results = senta_edtcnn.emotion_classify(texts = test_text, use_gpu = True)
    """

    # Fourth sentiment model is bow.....................................................................................
    senta_bow = hub.Module(name = "senta_bow")
    test_text = [keyboardInput]
    bow_results = senta_bow.sentiment_classify(texts = test_text, use_gpu = True, batch_size = 1)

    # Fifth sentiment model is cnn......................................................................................
    senta_cnn = hub.Module(name="senta_cnn")
    test_text = [keyboardInput]
    cnn_results = senta_cnn.sentiment_classify(texts = test_text, use_gpu = True, batch_size = 1)

    # Sixth sentiment model is GRU......................................................................................
    senta_gru = hub.Module(name="senta_gru")
    test_text = [keyboardInput]
    gru_results = senta_gru.sentiment_classify(texts = test_text, use_gpu = True, batch_size = 1)

    # Seventh sentiment model is LSTM...................................................................................
    senta_lstm = hub.Module(name="senta_lstm")
    test_text = [keyboardInput]
    lstm_results = senta_lstm.sentiment_classify(texts = test_text, use_gpu = True, batch_size = 1)

    returnList = [bilstm_results,
                  bow_results,
                  cnn_results,
                  gru_results,
                  lstm_results]
    return returnList

# Get some text, from keyboard..........................................................................................
# # Using tkinter for pretty little input thing.
from tkinter import *
def run():
    keyboardInput = inp.get()
    sentimentResults = sentimentParralletest(keyboardInput)
    s = str()
    for i in sentimentResults:
        s = '\n'.join([s, str(i[0])])
    txt.insert(END, s)
    inp.delete(0, END)

inputWin = Tk()
inputWin.title('INPUT WINDOW')
width = 1000
height = 400
screenwidth = inputWin.winfo_screenwidth()
screenheight = inputWin.winfo_screenheight()
win_str = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
inputWin.geometry(win_str)

lb = Label(inputWin, text = "Enter a text to sentiment.")
lb.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.1)
inp = Entry(inputWin)
inp.place(relx=0.1, rely=0.2, relwidth=0.8, relheight=0.1)
btn = Button(inputWin, text="Sentiment", command=run)
btn.place(relx=0.1, rely=0.3, relwidth=0.3, relheight=0.1)
txt = Text(inputWin)
txt.place(rely=0.6, relwidth=1, relheight=0.6)
inputWin.mainloop()
