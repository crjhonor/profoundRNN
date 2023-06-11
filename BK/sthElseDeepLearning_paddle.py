
"""
At this code, X_train, X_test, y_train, y_test, X_predict (to predict one day ahead), X_predict_DATE (the DATE of
predict feature derived from and the day ahead is to be predicted) has been transfered and preprocessed further and
implemented many deep learning models. Data should firstly be assembled into a DataLoader.

I am trying to use paddlepaddle instead of pytorch, but I encountered some problems with paddlepaddle. And that's why I
walked back into the pytorch and leave paddle later.
"""
import numpy as np
import paddle
from paddle.io import TensorDataset
from paddle.io import DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
import matplotlib.pyplot as plt

class linearModel(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(linearModel, self).__init__()
        self.layer1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.sigmoid(x)
        x = self.layer2(x)
        x = F.softmax(x)
        return x

class somethingElseDeepLearning:
    def __init__(self, X_train, X_test, y_train, y_test, X_predict, indexWanted, X_predict_DATE):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_predict = X_predict
        self.indexWanted = indexWanted
        self.X_predict_DATE= X_predict_DATE
        self.train_dl = self.to_DataLoader()

    def to_DataLoader(self):
        X_train = paddle.to_tensor(self.X_train, dtype=paddle.float)
        y_train = paddle.to_tensor(self.y_train)
        train_ds = TensorDataset([self.X_train, self.y_train])
        batch_size = 2
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        return train_dl

    def linerModel_train(self, learning_rate=0.001, num_epochs=100):
        """

        :param learning_rate: The Learning Rate
        :param num_epochs: The Number of Epoches
        :return:
        """
        input_size = self.X_train.shape[1]
        hidden_size = 16
        output_size = len(np.unique(self.y_train))
        log_epochs = 10
        model = linearModel(input_size, hidden_size, output_size)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters() )
        loss_hist = [0] * num_epochs
        accuracy_hist = [0] * num_epochs
        train_dl = self.train_dl
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist[epoch] += is_correct.mean()
            loss_hist[epoch] /= len(train_dl.dataset)
            accuracy_hist[epoch] /= len(train_dl.dataset)

        # Ploting accuracies
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(loss_hist, lw=3)
        ax.set_title('Training loss', size=15)
        ax.set_xlabel('Epoch', size=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(accuracy_hist, lw=3)
        ax.set_title('Training accuracy', size=15)
        ax.set_xlabel('Epoch', size=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.show()

        # Evaluating the trained model on the test dataset
        X_test = torch.tensor(self.X_test, dtype=torch.float)
        y_test = torch.from_numpy(self.y_test)
        pred_test = model(X_test)
        correct = (torch.argmax(pred_test, dim=1) == y_test).float()
        accuracy = correct.mean()
        print(f'Test Acc.: {accuracy:.4f}')

        # Predict one day ahead
        X_predict = torch.tensor(self.X_predict, dtype=torch.float)
        X_predict = torch.reshape(X_predict, (-1, ))
        y_predict = model(X_predict)
        y_predict_class = np.argmax(y_predict.detach().numpy(), axis=0)
        print(f'Prediction of the {self.indexWanted} at DATE {self.X_predict_DATE.values} '
              f'then one day ahead will be {y_predict_class}.')

