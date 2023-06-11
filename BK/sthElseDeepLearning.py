
"""
At this code, X_train, X_test, y_train, y_test, X_predict (to predict one day ahead), X_predict_DATE (the DATE of
predict feature derived from and the day ahead is to be predicted) has been transfered and preprocessed further and
implemented many deep learning models. Data should firstly be assembled into a DataLoader.

I am trying to use paddlepaddle instead of pytorch, but I encountered some problems with paddlepaddle. And that's why I
walked back into the pytorch and leave paddle later.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

class linearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax()(x)
        return x

class tryRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.rnn = nn.LSTM(embedding_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        out = self.embedding(x)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

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
        X_train = torch.tensor(self.X_train, dtype=torch.float)
        y_train = torch.from_numpy(self.y_train)
        train_ds = TensorDataset(X_train, y_train)
        batch_size = 2
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_dl

    def linearModel_train(self, learning_rate=0.001, num_epochs=100):
        """

        :param learning_rate: The Learning Rate
        :param num_epochs: The Number of Epoches
        :return: [accuracy, predict class of one day ahead]
        """
        input_size = self.X_train.shape[1]
        hidden_size = 16
        output_size = len(np.unique(self.y_train))
        log_epochs = 10
        model = linearModel(input_size, hidden_size, output_size)
        device = torch.device("cuda:0")
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_hist = [0] * num_epochs
        accuracy_hist = [0] * num_epochs
        train_dl = self.train_dl
        for epoch in range(num_epochs):
            loss_hist_train = 0
            for x_batch, y_batch in train_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist[epoch] += is_correct.mean().cpu()
            loss_hist[epoch] /= len(train_dl.dataset)
            accuracy_hist[epoch] /= len(train_dl.dataset)
            loss_hist_train += loss.item()
            if epoch % log_epochs==0:
                print(f'Epoch {epoch}  Loss '
                      f'{loss_hist_train/len(train_dl):.4f}')

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
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        pred_test = model(X_test)
        correct = (torch.argmax(pred_test, dim=1) == y_test).float()
        accuracy = correct.mean().cpu()
        print(f'Test Acc.: {accuracy:.4f}')

        # Predict one day ahead
        torch.cuda.synchronize()
        model_cpu = model.cpu()
        X_predict = torch.tensor(self.X_predict, dtype=torch.float)
        X_predict = torch.reshape(X_predict, (-1, ))
        y_predict = model_cpu(X_predict)
        y_predict_class = torch.argmax(y_predict, dim=0).numpy()
        print(f'Prediction of the {self.indexWanted} at DATE {self.X_predict_DATE} '
              f'then one day ahead will be {y_predict_class}.')
        return [np.array(accuracy), y_predict_class]

    def CNN_model_train(self, learning_rate=0.001, num_epochs=50):
        """

        :param learning_rate: The Learning Rate
        :param num_epochs: The Number of Epoches
        :return: [accuracy, predict class of one day ahead]
        """
        device = torch.device("cuda:0")
        # Preparing the dataloader
        train_dl = self.train_dl
        data_reshape = (2, 1, 4, 14)
        output_size = len(np.unique(self.y_train))
        # CNN model construction
        model = nn.Sequential()
        model.add_module(
            'conv1',
            nn.Conv2d(
                in_channels=1, out_channels=16,
                kernel_size=3, padding=1
            )
        )
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
        model.add_module(
            'conv2',
            nn.Conv2d(
                in_channels=16, out_channels=32,
                kernel_size=3, padding=1
            )
        )
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc1', nn.Linear(96, 32))
        model.add_module('relu3', nn.ReLU())
        model.add_module('dropout', nn.Dropout(p=0.5))
        model.add_module('fc2', nn.Linear(32, output_size))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = model.to(device)
        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        for epoch in range(num_epochs):
            model.train()
            for x_batch, y_batch in train_dl:
                x_batch = x_batch.reshape(data_reshape)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.sum().cpu()

            loss_hist_train[epoch] /= len(train_dl.dataset)
            accuracy_hist_train[epoch] /= len(train_dl.dataset)

            print(f'Eopch {epoch+1} accuracy: '
                  f'{accuracy_hist_train[epoch]:.4f}')

        hist = [loss_hist_train, accuracy_hist_train]
        # Visualize the learning curves
        x_arr = np.arange(len(hist[0])) + 1
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, hist[0], '-o', label='Train loss')
        ax.legend(fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, hist[1], '-o', label='Train acc.')
        ax.legend(fontsize=15)
        ax.set_xlabel('Epoch', size=15)
        ax.set_ylabel('Accuracy', size=15)
        plt.show()

        # evaluate the trained model on test dataset:
        model.eval()
        X_test = torch.tensor(self.X_test, dtype=torch.float)
        X_test = X_test.reshape((X_test.shape[0:1] + data_reshape[1:]))
        y_test = torch.from_numpy(self.y_test)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        pred_test = model(X_test)
        is_correct = (torch.argmax(pred_test, dim=1) == y_test).float()
        accuracy = is_correct.mean().cpu()
        print(f'Test Acc.: {accuracy:.4f}')

        # Predict one day ahead
        torch.cuda.synchronize()
        model_cpu = model.cpu()
        X_predict = torch.tensor(self.X_predict, dtype=torch.float)
        X_predict = X_predict.reshape(((1,) + data_reshape[1:]))
        y_predict = model_cpu(X_predict)
        y_predict_class = torch.argmax(y_predict, dim=1).numpy()
        print(f'Prediction of the {self.indexWanted} at DATE {self.X_predict_DATE} '
              f'then one day ahead will be {y_predict_class}.')
        return [np.array(accuracy), y_predict_class]

    def RNN_model_train(self, learning_rate=0.001, num_epochs=50):
        """

        :param learning_rate: The Learning Rate
        :param num_epochs: The Number of Epoches
        :return: [accuracy, predict class of one day ahead]
        """
        # Recreating the DataLoader.
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        class CustomDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __getitem__(self, item):
                return self.X[item], self.y[item]

            def __len__(self):
                return len(self.X)

        # wrap the encode and transformation function
        def collate_batch(batch):
            label_list, text_list, lengths = [], [], []
            for _text, _label in batch:
                processed_text = torch.tensor(_text, dtype=torch.int64)
                text_list.append(processed_text)
                label_list.append(_label)
                lengths.append(processed_text.size(0))
            padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
            lengths = torch.tensor(lengths)
            label_list = torch.tensor(label_list)
            return padded_text_list, label_list, lengths

        train_ds = CustomDataset(X_train, y_train)
        test_ds = CustomDataset(X_test, y_test)
        batch_size = 4
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        device = torch.device("cuda:0")
        num_embeddings=10
        embedding_dim=3
        rnn_hidden_size = 64
        fc_hidden_size = 64
        torch.manual_seed(1)
        model = tryRNN(num_embeddings, embedding_dim, rnn_hidden_size, fc_hidden_size)
        model.to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def train(dataloader):
            device = torch.device("cuda:0")
            model.train()
            total_acc, total_loss = 0, 0
            for text_batch, label_batch, lengths in dataloader:
                optimizer.zero_grad()
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)
                lengths = lengths.to(device)
                pred = model(text_batch, lengths)[:, 0]
                loss = loss_fn(pred, label_batch.float())
                loss.backward()
                optimizer.step()
                total_acc += (
                    (pred >= 0.5).float() == label_batch
                ).float().sum().item()
                total_loss += loss.item() * label_batch.size(0)
            return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

        def evaluate(dataloader):
            device = torch.device("cuda:0")
            model.eval()
            total_acc, total_loss = 0, 0
            with torch.no_grad():
                for text_batch, label_batch, lengths in dataloader:
                    text_batch = text_batch.to(device)
                    label_batch = label_batch.to(device)
                    lengths = lengths.to(device)
                    pred = model(text_batch, lengths)[:, 0]
                    loss = loss_fn(pred, label_batch.float())
                    total_acc += (
                        (pred >= 0.5).float() == label_batch
                    ).float().sum().item()
                    total_loss += loss.item() * label_batch.size(0)
            return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

        torch.manual_seed(1)
        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        for epoch in range(num_epochs):
            acc_train, loss_train = train(train_dl)
            loss_hist_train[epoch] = loss_train
            accuracy_hist_train[epoch] = acc_train
            print(f'Epoch {epoch} accuracy: {acc_train:.4f}')
        hist = [loss_hist_train, accuracy_hist_train]
        # Visualize the learning curves
        x_arr = np.arange(len(hist[0])) + 1
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x_arr, hist[0], '-o', label='Train loss')
        ax.legend(fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_arr, hist[1], '-o', label='Train acc.')
        ax.legend(fontsize=15)
        ax.set_xlabel('Epoch', size=15)
        ax.set_ylabel('Accuracy', size=15)
        plt.show()

        acc_test, _ = evaluate(test_dl)
        print(f'test_accuracy: {acc_test:.4f}')

        # Predict one day ahead
        torch.cuda.synchronize()
        model_cpu = model.cpu()
        X_predict = np.array([self.X_predict] * 4)
        y_predict = np.array([0, 0, 0, 0])
        predict_ds = CustomDataset(X_predict, y_predict)
        predict_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        for X, y, lengths in predict_dl:
            pred = model_cpu(X, lengths)[:, 0]
        y_predict_class = (pred.mean() >= 0.5).int().numpy()
        print(f'Prediction of the {self.indexWanted} at DATE {self.X_predict_DATE} '
              f'then one day ahead will be {y_predict_class}.')

        return [np.array(acc_test), y_predict_class]
