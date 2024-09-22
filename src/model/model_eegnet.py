import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score


class EEGNet(nn.Module):
    def __init__(self, signal_length, num_class=1, F1=16, D=2, F2=16, kernel_size_1=(1, 63),
                 kernel_size_2=(128, 1), kernel_size_3=(1, 16), kernel_size_4=(1, 1)):
        super(EEGNet, self).__init__()

        signal_length = 127

        # layer 1
        self.conv1 = nn.Conv2d(1, F1, kernel_size_1, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(F1)
        # layer 2
        self.conv2 = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(D * F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.MaxPool2d((1, 4))
        self.Dropout = nn.Dropout2d(0.2)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d(D * F1, D * F1, kernel_size_3,
                                                padding='same', groups=D * F1)
        self.Separable_conv2D_point = nn.Conv2d(D * F1, F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(F2)
        self.Average_pooling2D_2 = nn.MaxPool2d((1, 8))

        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.LazyLinear(num_class)
        self.Softmax = nn.Sigmoid()

    def forward(self, x):

        y = self.batchnorm1(self.conv1(x))  # .relu()
        # layer 2
        y = self.batchnorm2(self.conv2(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)

        """
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.sigmoid(self.fc1(x))
        """

        return y

    def fit(self, data_train, y_train, data_val, y_val, batch_size, optimizer, criterion, epochs):
        self.criterion = criterion
        self.metric_monitor = Metrics()
        self.metric_monitor.initiate_history()
        best_val_loss = np.inf
        for epoch in range(epochs):
            self.train()
            permutation = torch.randperm(data_train.shape[0])
            self.metric_monitor.reset_temp()
            for i in range(0, data_train.shape[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = torch.Tensor(data_train[indices]), torch.Tensor(y_train[indices])
                batch_x, batch_y = Variable(batch_x.cuda(0)), Variable(batch_y.cuda(0))
                output = self(batch_x)
                loss = criterion(output, batch_y.float())
                loss.backward()
                optimizer.step()
                self.metric_monitor.add_metrics(batch_y.cpu().numpy(), output.cpu().detach().numpy(), loss.item(),
                                                train=True)
            num_iterations = data_train.shape[0] // batch_size + 1
            self.metric_monitor.update_metrics(num_iterations=num_iterations, train=True)
            self.eval()
            with torch.no_grad():
                for i in range(0, data_val.shape[0], batch_size):
                    batch_x, batch_y = torch.Tensor(data_val[i:i + batch_size]), torch.Tensor(y_val[i:i + batch_size])
                    batch_x, batch_y = Variable(batch_x.cuda(0)), Variable(batch_y.cuda(0))
                    output = self(batch_x)

                    loss = criterion(output, batch_y.float())
                    self.metric_monitor.add_metrics(batch_y.cpu().numpy(), output.cpu().detach().numpy(), loss.item(),
                                                    train=False)
            num_iterations = data_val.shape[0] // batch_size + 1
            self.metric_monitor.update_metrics(num_iterations=num_iterations, train=False)

            print(
                f'Epoch {epoch + 1}/{epochs} - Loss: {self.metric_monitor.history["loss_train"][-1]} - '
                f'Accuracy: {self.metric_monitor.history["accuracy_train"][-1]} - '
                f'Val Loss: {self.metric_monitor.history["loss_val"][-1]} - V'
                f'al Accuracy: {self.metric_monitor.history["accuracy_val"][-1]}')

            if self.metric_monitor.history['loss_val'][-1] < best_val_loss:
                best_val_loss = self.metric_monitor.history['loss_val'][-1]
                torch.save(self.state_dict(), 'best_model.pth')
        torch.load('best_model.pth')
        return self.metric_monitor.history

    def predict(self, data, batch_size):
        self.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, data.shape[0], batch_size):
                batch_x = torch.Tensor(data[i:i + batch_size])
                batch_x = Variable(batch_x.cuda(0))
                output = self(batch_x)
                predictions.append(output.cpu().detach().numpy())
        return predictions

    def evaluate(self, data, y_true, batch_size):
        y_predicted = self.predict(data, batch_size)
        y_predicted = np.concatenate(y_predicted)
        y_true = np.array(y_true)
        loss = self.criterion(torch.Tensor(y_predicted), torch.Tensor(y_true))
        accuracy = accuracy_score(y_true, y_predicted > 0.5)
        precision = precision_score(y_true, y_predicted > 0.5)
        recall = recall_score(y_true, y_predicted > 0.5)
        f1 = f1_score(y_true, y_predicted > 0.5)
        roc_auc = roc_auc_score(y_true, y_predicted)
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }


class Metrics:
    def __init__(self):
        self.history = {
            'accuracy_train': [],
            'precision_train': [],
            'recall_train': [],
            'f1_train': [],
            'roc_auc_train': [],
            'loss_train': [],
            'accuracy_val': [],
            'precision_val': [],
            'recall_val': [],
            'f1_val': [],
            'roc_auc_val': [],
            'loss_val': []
        }
        self.temp = {
            'accuracy_train': 0,
            'precision_train': 0,
            'recall_train': 0,
            'f1_train': 0,
            'roc_auc_train': [],
            'loss_train': [],
            'accuracy_val': [],
            'precision_val': [],
            'recall_val': [],
            'f1_val': [],
            'roc_auc_val': [],
            'loss_val': []
        }

    def reset_temp(self):
        self.temp = {
            'accuracy_train': 0,
            'precision_train': 0,
            'recall_train': 0,
            'f1_train': 0,
            'roc_auc_train': 0,
            'loss_train': 0,
            'accuracy_val': 0,
            'precision_val': 0,
            'recall_val': 0,
            'f1_val': 0,
            'roc_auc_val': 0,
            'loss_val': 0
        }

    def add_metrics(self, y_true, y_pred, loss, train=False):
        if train:
            suffix = 'train'
        else:
            suffix = 'val'

        self.temp['loss_' + suffix] += loss
        self.temp['accuracy_' + suffix] += accuracy_score(y_true, y_pred > 0.5)
        self.temp['precision_' + suffix] += precision_score(y_true, y_pred > 0.5)
        self.temp['recall_' + suffix] += recall_score(y_true, y_pred > 0.5)
        self.temp['f1_' + suffix] += f1_score(y_true, y_pred > 0.5)

        try:
            self.temp['roc_auc_' + suffix] += roc_auc_score(y_true, y_pred)
        except:
            self.temp['roc_auc_' + suffix] += 50

    def update_metrics(self, num_iterations, train=False):
        if train:
            suffix = 'train'
        else:
            suffix = 'val'
        self.history['loss_' + suffix].append(self.temp['loss_' + suffix] / num_iterations)
        self.history['accuracy_' + suffix].append(self.temp['accuracy_' + suffix] / num_iterations)
        self.history['precision_' + suffix].append(self.temp['precision_' + suffix] / num_iterations)
        self.history['recall_' + suffix].append(self.temp['recall_' + suffix] / num_iterations)
        self.history['f1_' + suffix].append(self.temp['f1_' + suffix] / num_iterations)
        self.history['roc_auc_' + suffix].append(self.temp['roc_auc_' + suffix] / num_iterations)

        return self.history

    def initiate_history(self):
        history = {
            'accuracy_train': [],
            'precision_train': [],
            'recall_train': [],
            'f1_train': [],
            'roc_auc_train': [],
            'accuracy_val': [],
            'precision_val': [],
            'recall_val': [],
            'f1_val': [],
            'roc_auc_val': []
        }
        return history
