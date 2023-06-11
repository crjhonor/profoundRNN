"""
Including features and labels engineering and imposing of simple convolution neutral network.
"""
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

# Using gpu
paddle.device.set_device('gpu:0')

class simpleCNN(paddle.nn.Layer):
    def __init__(self):
        super(simpleCNN, self).__init__()

        # Define convolution network with out_channels to be 20, kernel size to be 5...
        self.conv1 = Conv2D(in_channels=1, out_channels=10, kernel_size=2, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=10, out_channels=10, kernel_size=2, stride=1, padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.fc = Linear(in_features=90, out_features=1)

    # Define the net forward function
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x

"""
First of all, the feature's date and label's date should not be in the same as my purpose is to predict trading 
direction of the future. Thus I am adjusting them by lagging the feature by 2 days.

The feature and label are trying to be engineered by scikit-learn package sklearn's transformation functions.
"""
import numpy as np
import datetime as dt
import pandas as pd
from sklearn import preprocessing


class simpleConvolutionnetwork:
    def __init__(self, indexWanted, emailFeatures, weiboFeatures, featuresYields, labels):
        self.indexWanted = indexWanted
        self.emailFeatures = emailFeatures
        self.weiboFeatures = weiboFeatures
        self.featuresYields = featuresYields
        self.labels = labels
        self.results = pd.DataFrame(index=['_'.join([self.indexWanted[0], 'scnn'])],
                                    columns=['VOLATILITY', 'STD', 'T+1 RESULT', 'T+2 RESULT'])
        self.fnl_fd = self.fnl_reProcessing()
        self.scnn_modeling()
        # print(self.results)


    def datetimeProcessing(self, dataToproc):
        # functions to reprocess the DATE formate
        dataToproc['DATE_'] = ''
        for i in range(len(dataToproc)):
            dataToproc['DATE_'].iloc[i] = dataToproc['DATE'].iloc[i].to_pydatetime().date()
        dataToproc.index = dataToproc['DATE_']
        return dataToproc

    def fnl_reProcessing(self):
        # function to combine the features and labels.
        self.emailFeatures = self.datetimeProcessing(self.emailFeatures)
        self.weiboFeatures = self.datetimeProcessing(self.weiboFeatures)
        self.featuresYields = self.datetimeProcessing(self.featuresYields)
        self.labels = self.datetimeProcessing(self.labels)
        # join the emailFeatures and labels into one dataframe
        fnl_fd = self.labels.join(self.emailFeatures, rsuffix = '_other')
        # remove NAs
        fnl_fd = fnl_fd.dropna()
        fnl_fd = fnl_fd.drop(['DATE', 'DATE_other', 'DATE__other'], axis = "columns")
        # join the weiboFeatures
        fnl_fd = fnl_fd.join(self.weiboFeatures, rsuffix='_weibo')
        fnl_fd = fnl_fd.dropna()
        fnl_fd = fnl_fd.drop(['DATE', 'DATE__weibo'], axis="columns")
        # more features yields
        fnl_fd = fnl_fd.join(self.featuresYields, rsuffix = '_other')
        fnl_fd = fnl_fd.dropna()
        fnl_fd = fnl_fd.drop(['DATE', 'DATE__other'], axis = "columns")
        return fnl_fd

    def scnn_train(self, datas): # features and labels
        scnn_model = simpleCNN()
        opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=scnn_model.parameters()) # 0.001
        EPOCH_NUM = 250 #250
        features = datas[0]
        labels = datas[1]
        for epoch in range(EPOCH_NUM):
            for i in range(labels.shape[0]):
                feature = features[i].reshape((1, 1, 3, 4)).astype('float32')
                label = labels[i].reshape((1, 1)).astype('float32')

                # calculate forward
                predict = scnn_model(feature)

                # calculate loss
                loss = F.square_error_cost(predict, label)
                avg_loss = paddle.mean(loss)

                # backwarding
                avg_loss.backward()
                opt.step()
                opt.clear_grad()
            # print("EPOCH_NUM: {}, loss is: {}".format(epoch, avg_loss.numpy()))
        return scnn_model

    def scnn_predict(self, datas):
        # for day t+1
        feature_t_1 = datas[0]
        feature_t_1_tensor = paddle.to_tensor(feature_t_1).reshape((1, 1, 3, 4)).astype('float32')
        result_t_1 = self.scnn_model(feature_t_1_tensor)
        # print('Prediction for DAY T + 1:', result_t_1.numpy())
        self.results['T+1 RESULT'] = result_t_1.numpy()
        # for day t+2
        feature_t_2 = datas[1]
        feature_t_2_tensor = paddle.to_tensor(feature_t_2).reshape((1, 1, 3, 4)).astype('float32')
        result_t_2 = self.scnn_model(feature_t_2_tensor)
        # print('Prediction for DAY T + 2:', result_t_2.numpy())
        self.results['T+2 RESULT'] = result_t_2.numpy()


    def scnn_modeling(self):
        fnl_fd = self.fnl_fd
        # transform the dataset into more desired features and labels.
        labels = self.fnl_fd['LABELS']
        features = self.fnl_fd.drop(['DATE_', 'LABELS'], axis="columns")
        labels_scaled = labels
        # using StandardScaler of preprocessing in sklearn to scale the features
        self.scaler = preprocessing.StandardScaler().fit(features)
        features_scaled = self.scaler.transform(features)

        # in order to predict result in 2 days, I am lagging the features by 2.
        labels_train = labels_scaled[2:]
        features_train = features_scaled[:-2:]
        # prepare the labels and features for Deep Learning.
        labels_array = np.array(labels_train).reshape(len(labels_train), 1)
        features_array = np.array(features_train)
        labels_tensor = paddle.to_tensor(labels_array)
        features_tensor = paddle.to_tensor(features_array)

        # implement the training function.
        datas = [features_tensor, labels_tensor]
        self.scnn_model = self.scnn_train(datas)
        # paddle.save(self.scnn_model.state_dict(), './scnn_model.pdparams')

        # Try to predict from new features.
        features_predict = features_scaled[-2:]
        self.scnn_predict(features_predict)

        # Other results
        self.results['VOLATILITY'] = abs(labels).mean()
        self.results['STD'] = labels.std()