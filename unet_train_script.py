## Training Script for Neural Network

## Import Statements
from tools import *

from models.Unet import Unet #Model Import
from dataload import PolypLoader
from train import train
from validate import validate
H = 224
W = 224
D = 3

## Local  Variables
epochs = 10
batch_size = 2
lr = 0.01

epoch_save = list(range(epochs)[::5])
train_transform = [transforms.Resize((H,W)),
        transforms.RandomAffine(degrees = (-45,45), translate = (0.3,0.3), scale = (0.5,2)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

valid_transform = [transforms.Resize((H,W)),
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]



## Model Loading
model = Unet((3,H,W))
net = model
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


train_data = PolypLoader(train_path, transform = train_transform)
trainloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

valid_data = PolypLoader(valid_path, transform = valid_transform)
validloader = DataLoader(valid_data, batch_size = batch_size, shuffle=True)

test_data = PolypLoader(test_path, transform = None)
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
log.write("Start Training")
for epoch in range(epochs):
    start = timer() 
   
    ## Training Pass
    temp_train_loss = train(net, optimizer, trainloader, criterion, epoch, log)
    train_loss.append(temp_train_loss)

    ## Validation Pass
    temp_valid_loss = validate(net, validloader, criterion, epoch, log, lr)
    valid_loss.append(temp_valid_loss)

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
                        (epoch, lr, temp_train_loss, temp_valid_loss))

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

