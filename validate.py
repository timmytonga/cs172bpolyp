from tools import *

def validate(model, validloader, criterion, epoch, log):

    ## Enter Evalution Mode
    net.eval()
   
    ## Validation Pass
    epoch_valid = []
    with torch.no_grad():
        num_it = len(validloader)
        for it, data in enumerate(validloader,0):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            pred = model(inputs)
            loss = criterion(pred, labels)

            temp_valid_loss = loss.item()
            epoch_valid.append(temp_valid_loss)
            
            ## Write Validation Loss
            log.write('\tValidation: %5.1f      %0.4f  | %0.4f  ... \n' % \
                        (epoch +(it/num_it), lr, temp_valid_loss))


        epoch_valid = np.array(epoch_valid)
        valid_loss = np.mean(epoch_valid)
    return valid_loss

