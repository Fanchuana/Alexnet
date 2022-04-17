import torch
from torch import nn
from network import MyAlexNet
import numpy as np
from torch.optim import lr_scheduler
import os
import sys
import time

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

ROOT_TRAIN = r'E:/Program Files/Alexnet/data/train'
ROOT_TEST = r'E:/Program Files/Alexnet/data/val'

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TRAIN, train_transform)
val_dataset = ImageFolder(ROOT_TEST, val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

def train(dataloader, model, loss_fn, optimizer):
    loss, current, n=0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image = x.to(device)
        y= y.to(device)
        output = model(image)
        current_loss = loss_fn(output, y)
        pred = torch.max(output, axis=1)[1]
        current_accuracy = torch.sum(y == pred )/output.shape[0]

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        current +=current_accuracy.item()
        n +=1
        print("\r", end="")
        print("Training progress: {5}.{2}%: ".format(n*100/len(dataloader)), "▋" * (n*20//len(dataloader)), end="")
        sys.stdout.flush()
        time.sleep(0.05)

    train_loss = loss/ n
    train_accuracy = current / n
    print('train_loss' + str(train_loss))
    print('train_accuracy' + str(train_accuracy))
    return train_loss, train_accuracy


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
       for batch, (x, y) in enumerate(dataloader):
          image = x.to(device)
          y= y.to(device)
          output = model(image)
          current_loss = loss_fn(output, y)
          pred = torch.max(output, axis=1)[1]
          current_accuracy = torch.sum(y == pred) / output.shape[0]
          loss += current_loss.item()
          current += current_accuracy.item()
          n += 1
          print("\r", end="")
          print("Valing progress: {5}.{2}%: ".format(n*100/len(dataloader)), "▋" * (n *20//len(dataloader)), end="")
    val_loss = loss / n
    val_accuracy = current / n
    print('val_loss' + str(val_loss))
    print('val_accuracy' + str(val_accuracy))
    return val_loss, val_accuracy

def plt_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Loss tables')
    plt.show()

def plt_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train acc')
    plt.plot(val_acc, label='validation acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Accuracy tables')
    plt.show()
#start training
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 20
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t+1}\n---------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    if val_acc >max_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        max_acc = val_acc
        print(f"save best model, the {t+1} round")
        torch.save(model.state_dict(), 'save_model/best_model.pth')
    if t == epoch-1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')
print('done!')
plt_loss(loss_train, loss_val)
plt_acc(acc_train, acc_val)