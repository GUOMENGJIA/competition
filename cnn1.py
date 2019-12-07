import torch
import os
import csv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',\
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',\
        '0','1','2','3','4','5','6','7','8','9']


class Data(Dataset):
        def __init__(self,root,transfroms=None,train=True):
                images = os.listdir(root)
                self.images = [os.path.join(root, image) for image in images]
                self.transfroms = transfroms
                self.train=train

        def __getitem__(self, index):
                image_path = self.images[index]  # 图片完整的路径
                pil_image = Image.open(image_path)  # 生成了图像所对应的对象
                data=self.transfroms(pil_image)

                if self.train:
                        names = image_path.split(os.sep)[-1].split('.')[0]
                        label = [labels.index(names[0]), labels.index(names[1]), labels.index(names[2]), labels.index(names[3]),
                                 labels.index(names[4])]
                        return data,label
                else:
                        return data, image_path
        def __len__(self):
                return len(self.images)



class CNN(nn.Module):
        def __init__(self,kinds,charNumber):
                super(CNN,self).__init__()
                self.kinds=kinds
                self.charNumber=charNumber
                self.conv=nn.Sequential(

                        nn.Conv2d(3, 16, 3, padding=(1, 1)),
                        nn.MaxPool2d(2, 2),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),

                        nn.Conv2d(16, 64, 3, padding=(1, 1)),
                        nn.MaxPool2d(2, 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),

                        nn.Conv2d(64, 512, 3, padding=(1, 1)),
                        nn.MaxPool2d(2, 2),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),

                        nn.Conv2d(512, 512, 3, padding=(1, 1)),
                        nn.MaxPool2d(2, 2),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                )

                self.fc=nn.Linear(4608,self.kinds*self.charNumber)

        def forward(self,x):
                x=self.conv(x)
                x=x.view(x.size(0),-1)
                x1=self.fc(x)
                x2 = self.fc(x)
                x3 = self.fc(x)
                x4= self.fc(x)
                x5 = self.fc(x)

                return x1,x2,x3,x4,x5

def write_data(reslut):
        number = []
        lable = []
        j = 1
        for i in reslut:
                lable.append(i)
                number.append(j)
                j = j + 1
        data = pd.DataFrame({'id': number, 'y': lable})
        data.to_csv('submission-test.csv', index=False, sep=',')
        print("预测结果已写入csv文件")

def model_train():
        batch_size,epoches,learning_rate = 128,10,0.01
        data = Data("train/train", transform)
        trainloader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        model = CNN(len(labels), 5)
        if (os.path.exists('model.pkl')):
                model.load_state_dict(torch.load('model.pkl'))
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        for i in range(1, epoches + 1):
                t = 0
                for index, (data, target) in enumerate(trainloader):
                        y1, y2, y3, y4, y5 = target[0], target[1], target[2], target[3], target[4]
                        # 导数清零
                        optimizer.zero_grad()
                        x1, x2, x3, x4, x5 = model(data)
                        loss1, loss2, loss3, loss4, loss5 = criterion(x1, y1), criterion(x2, y2), criterion(x3, y3), criterion(x4, y4), criterion(x5, y5)
                        loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
                        # 反向传播
                        loss.backward()
                        optimizer.step()
                        t =t+ loss.item()
                        if index % 10 == 9:
                                print('Train Epoch:{}[{}/{}]\t Loss:{:.6f}'.format(i, index * len(data),len(trainloader.dataset), t / 10))
                                t = 0
        torch.save(model.state_dict(), 'model.pkl')


def predict():
    s1=[0 for i in range(20000)]
    label=list()
    cnn =CNN()
    cnn.load_state_dict(torch.load('model.pkl'))
    cnn.eval()
    testset = Data('test\test',transform,False)
    testloader = DataLoader(testset)

    kinds = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    with torch.no_grad():
        for i, (inputs,path) in enumerate(testloader):
            y1,y2,y3,y4,y5 = cnn(inputs)
            preds1 = torch.argmax(y1,1)
            preds2 = torch.argmax(y2,1)
            preds3 = torch.argmax(y3,1)
            preds4 = torch.argmax(y4,1)
            preds5 = torch.argmax(y5,1)
            kind = kinds[preds1]+kinds[preds2]+kinds[preds3]+kinds[preds4]+kinds[preds5]
            num = int(path[0].split(os.sep)[1].split('.')[0])
            s1[num] = kind
        for i in s1:
            label.append(i)

    return label


transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
if __name__=='__main__':
        model_train()
        label=predict()
        write_data(label)




