import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import model
from visdom import Visdom
import time
import numpy as np
viz = Visdom()


EPOCH = 20
BATCH_SIZE = 32
LR = 0.01
USE_GPU = True

if USE_GPU:
    gpu_statue = torch.cuda.is_available()
else:
    gpu_statue = False


def visualization():
    global time_p, tr_acc, ts_acc, loss_p, sum_step, sum_loss, sum_acc
    test_acc = 0
    for (test_data, test_label) in testloader:
        if gpu_statue:
            test_data = test_data.cuda()
        test_out = net(test_data)
        pred_ts = torch.max(test_out, 1)[1].cpu().data.squeeze()
        acc = (pred_ts==test_label).sum().item()/test_label.size(0)
        test_acc += acc
    print("epoch: [{}/{}] | Loss: {:.4f} | TR_acc: {:.4f} | TS_acc: {:.4f} | Time: {:.1f}".format(epoch + 1, EPOCH,
                            sum_loss / (sum_step),sum_acc / (sum_step),test_acc/len(testloader),time.time() - start_time))
    # 可视化部分
    time_p.append(time.time() - start_time)
    tr_acc.append(sum_acc / sum_step)
    ts_acc.append(test_acc/len(testloader))
    loss_p.append(sum_loss / sum_step)
    viz.line(X=np.column_stack((np.array(time_p), np.array(time_p), np.array(time_p))),
             Y=np.column_stack((np.array(loss_p), np.array(tr_acc), np.array(ts_acc))),
             win=line,
             opts=dict(legend=["Loss", "TRAIN_acc", "TEST_acc"],
                       xlabel='Time / s',
                       ylabel='Precision and Loss',
                       title='Convergence on FashionMNIST'))
    # visdom text 支持html语句
    viz.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
             "<p style='color:BlueViolet'>TRAIN_acc:{:.4f}</p><br><p style='color:orange'>TEST_acc:{:.4f}</p><br>"
             "<p style='color:green'>Time:{:.2f}</p>".format(epoch+1, sum_loss / sum_step, sum_acc / sum_step, test_acc/len(testloader),
                                                             time.time() - start_time),
             win=text)


line = viz.line(np.arange(10))
text = viz.text("<h1>convolution Nueral Network</h1>")
# Loading and normalizing FashionMNIST
train_transform = transforms.Compose(
    [transforms.Resize(64),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose(
    [transforms.Resize(64),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# Define a Convolution Neural Network
net = model.vgg16(num_classes=10)

if gpu_statue:
    net = net.cuda()
    print('*'*26, '使用gpu', '*'*26)
else:
    print('*'*26, '使用cpu', '*'*26)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

tr_acc, ts_acc, loss_p, time_p = [], [], [], []

# Train and test the network
start_time = time.time()
for epoch in range(EPOCH):
    sum_loss, sum_acc, sum_step = 0., 0., 0.
    for i, (data, label) in enumerate(trainloader):
        if gpu_statue:
            data, label = data.cuda(), label.cuda()
        out = net(data)
        loss = criterion(out, label)
        sum_loss += loss.item()*len(label)
        pred_tr = torch.max(out, 1)[1]
        sum_acc += sum(pred_tr==label).item()
        sum_step += label.size(0)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch [{}/{}]'.format(i, len(trainloader)))
        # 可视化
        if i % 200 == 0:
            visualization()
            sum_loss, sum_acc, sum_step = 0., 0., 0.,
    f = open('./result.txt', 'a')
    f.write('epoch: ' + str(epoch) + '\n' +
            'tr_acc: ' + str(tr_acc) + '\n' +
            'ts_acc: ' + str(ts_acc) + '\n' +
            'time_p: ' + str(time_p) + '\n')
    f.close()
print('hellow')