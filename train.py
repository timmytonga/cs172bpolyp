## Training Script for Neural Network

## Import Statements
from common import *
from 

## Local  Variables
epochs = 50
data = optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss = 

## Logging





## Load Data
data = 

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


print("Finished Training")

