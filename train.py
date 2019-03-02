## Training Script for Neural Network

## Import Statements
from tools import *

from Unet import Unet #Model Import
from dataload import PolypLoader
from torchvision import transforms
H = 224
W = 224
D = 3

## Local  Variables
epochs = 10
batch_size = 2
lr = 0.01

epoch_save = list(range(epochs)[::5])
transform = [transforms.Resize((H,W)), transforms.ToTensor()]

## Model Loading
model = Unet((3,H,W))
net = model.cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
criterion = nn.BCEWithLogitsLoss().cuda()

## Creating Save Directories

curr_dir = os.getcwd()
out_dir = os.path.join(os.getcwd(),'results')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir+'/backup', exist_ok=True)
os.makedirs(out_dir+'/checkpoint', exist_ok=True)
os.makedirs(out_dir+'/snap', exist_ok=True)
os.makedirs(out_dir+'/train', exist_ok=True)
os.makedirs(out_dir+'/valid', exist_ok=True)
os.makedirs(out_dir+'/test',  exist_ok=True)
os.makedirs(out_dir+'/graph', exist_ok = True)
## Logging

log = Logger()
log.open(os.path.join(out_dir, 'log_train.txt'),mode='a')
log.write('\n--- [Start %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
log.write('** some project setting **\n')
log.write('\tSEED = %u\n' % SEED)
log.write('\tfile = %s\n' % __file__)
log.write('\tout_dir = %s\n' % out_dir)
log.write('\n')


## Saving Model
##backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/train.code.zip')

## Load Data
#transforms_train = [transforms.resize((H,W)), transforms.ToTensor(), ]
print("Load Data Start")
log.write('** dataset setting **\n')

data_path = os.path.join(os.getcwd(),'data')
train_path = os.path.join(data_path, 'training')
valid_path = os.path.join(data_path, 'validation')
test_path = os.path.join(data_path, 'training')


train_data = PolypLoader(train_path, transform = transform)
trainloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

valid_data = PolypLoader(valid_path, transform = transform)
validloader = DataLoader(valid_data, batch_size = batch_size, shuffle=True)

test_data = PolypLoader(test_path, transform = transform)
testloader= DataLoader(test_data, batch_size = batch_size, shuffle=True)

print("Load Data Complete")

## Training Script
log.write('** net setting **\n')
log.write('%s\n\n' % (type(net)))

# Variables
log.write('** start training! ** \n')
log.write('\n')

log.write('epoch   iter   rate  |   train_loss   ... \n')
log.write('--------------------------------------------------------------------------------------------------\n')

running_loss = 0.0
smooth_loss = 0.0
train_loss = []
train_acc = np.nan
test_loss = np.nan
test_acc = np.nan
valid_loss = []
time = 0

epoch_train = []
epoch_loss = []
log.write("Start Training")
for epoch in range(epochs):
    start = timer() 

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
        log.write('\tTraining: %5.1f      %0.4f  | %0.4f  ... \n' % \
                        (epoch +(it/num_it), lr, temp_train_loss))



    epoch_train = np.array(epoch_train)
    train_loss.append(np.mean(epoch_train))

    ## Validation Pass
    net.eval()
   
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
        valid_loss.append(np.mean(epoch_valid))

    ## Saving Model
    if epoch in epoch_save:
        save_name = '%03d.pth'%epoch
        torch.save(net.state_dict(), os.path.join(out_dir,'snap', save_name))
        torch.save({
            'state_dict':net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            }, os.path.join(out_dir, 'checkpoint', save_name))
    
    ## Outputing Results
    log.write('  Epoch Total: %5.1f      %0.4f   |  %0.4f  | %0.4f  ... \n' % \
                        (epoch, lr, np.mean(epoch_valid), np.mean(epoch_valid)))

    ## Adjust Learning Rate
    adjust_learning_rate(optimizer, epoch, lr) ## Reduces Learning rate every 30 epochs

## Saving Model
torch.save(net.state_dict(),out_dir +'/snap/final.pth')

##
model_name = "Unet"
train_name = '%s_train.npy'%model_name
valid_name = '%s_valid.npy'%model_name
np.save(os.path.join(out_dir, 'graph', train_name), np.array(train_loss))
np.save(os.path.join(out_dir, 'graph', valid_name), np.array(valid_loss))

print("Finished Training")

