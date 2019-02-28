## Training Script for Neural Network

## Import Statements
from tools import *

from Unet import Unet #Model Import
from dataload import PolypLoader


## Local  Variables
epochs = 50
batch_size = 32

## Model Loading
model = Unet(<insert Params you lazy Fuck>)
net = model.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss =  

## Creating Save Directories

curr_dir = os.getcwd()
out_dir = os.path.join(os.getcwd(),'results')


## Logging

log = Logger()
log.open(out_dir+'/log.train.txt',mode='a')


## Saving Model


## Load Data
transforms = [transforms.resize(), transforms.ToTensor(), ]

data_path = os.path.join(os.getcwd(),'data')
train_path = os.path.join(data_path, 'training')
valid_path = os.path.join(data_path, 'validation')
test_path = os.path.join(dat_path, 'training')


train_data = PolypLoader(train_path, transforms = transform)
train = DataLoader(train_data, batch_size = batch_size, shuffle=True)

valid_data = PolypLoader(valid_path, transforms = transform)
valid = DataLoader(valid_data, batch_size = batch_size, shuffle=True)

test_data = PolypLoader(test_path, transforms = transform)
test= DataLoader(test_data, batch_size = batch_size, shuffle=True)

## Training Script
for i in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        ## Forward
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        ## Backward
        loss.backward()
        optimizer.step()
## Saving Model

print("Finished Training")

