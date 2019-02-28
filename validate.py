from tools import *

def validate(val_loader, model, criterion, args, log):

    ## Enter Eval Mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            train, label = data

            in_data = train.cuda()
            out_data = label.cuda()

            pred = model(in_data)
            loss = criterion(output, target)

            # Measure Accuracy
            valid_acc = 
            valid_loss = loss.data[0]
