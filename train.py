from tools import *
def train_net(net, optimizer, trainloader,criterion, epoch, log, lr):
    """
    Function for Training Pass of Neural Network on dataset
    """
    epoch_train = []
    num_it = len(trainloader)
    for it, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        #print("Input SHape: ", inputs.shape)
        #print("Label Shape: ", labels.shape)

        ## Training Pass
        net.train() ## Enter training mode

        ## Forward
        optimizer.zero_grad()
        

        outputs = net(inputs)
        #print("Print Output Shape ", outputs.shape)
        #print("Label Shape ", labels.shape)
        loss = criterion(outputs, labels)
       
         ## Training Loss
        temp_train_loss = loss.item()
        #print("Training Loss", train_loss)
        epoch_train.append(temp_train_loss)

        ## Backward
        loss.backward()
        optimizer.step()
        
        ## Write to Log
        log.write('\tTraining: %5.1f   |   loss:%0.4f  ... \n' % \
                        (epoch +(it/num_it), lr, temp_train_loss))



    epoch_train = np.array(epoch_train)
    train_loss = np.mean(epoch_train)
    return train_loss

