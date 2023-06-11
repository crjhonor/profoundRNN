"""
This scrip is trying to implement RNN model on monthly data to detect and analysis its seasonal factor.
"""

import seasonalPreproc as spp

# Obtain feature and label data.========================================================================================
indexWanted = ["RU0"]
l_classes = 4 # Number of classes of the label that I would like to use
f_classes = 100 # Can see it as the vocab_size
# Obtain the overall dataset.
rr = spp.rawdataRead(indexWanted, suffix="Close")
rawData = rr.readingRawdata()
monthlyLogr = rr.monthlyLogr(rawData)
# Before heading toward machine learning, more feature and label engineering aiming at transform the continuous data
# into discrete data. Here I try to implement onehot encoding.
X_train, y_train, X_predict, y_classesTable, X_classesTable = rr.generateFnL(
    monthlyLogr, frequency=12, difference=1, l_classes=l_classes, f_classes=f_classes)
# monthlyLogr.plot()
# plt.show()

# Implement some deep learning neural network model. ===================================================================
import seasonalRecurrentneuralnetwork
sadl = seasonalRecurrentneuralnetwork.seasonalDeeplearning(
    X_train, y_train, X_predict, indexWanted, l_classes, f_classes, y_classesTable, X_classesTable)
# sadl.tensorflowRNN(learning_rate=0.001, num_epochs=200)

"""
Above is copying from the many to one example in the book of tensorflow, but it is not working for my dataset, it seems
that the loss function cannot work fine with this classification task. Then I will have to move back a little to the 
one to may example. Then I have to regenerate the dataset.
"""
# Before heading toward machine learning, more feature and label engineering aiming at transform the continuous data
# into discrete data.
nb_classes = 100
X_train, y_train, X_predict, LogrClassestable = rr.generateOnetomanyFnL(monthlyLogr, frequency=12, nb_classes=nb_classes)
sadl.onetoManyRNN(X_train, y_train, X_predict,
                  vocab_size=nb_classes, classesTable=LogrClassestable, learning_rate=0.001, num_epochs=700)
print("Done")

