## Training Script for Neural Network

## Import Statements
from tools import *

from Unet import Unet #Model Import
from dataload import PolypLoader

H = 224
W = 224
D = 3

## Local  Variables
epochs = 50
batch_size = 32

epoch_save = []


## Model Loading
model = Unet((3,H,W))
net = model.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
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
backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/train.code.zip')

## Load Data
#transforms_train = [transforms.resize((H,W)), transforms.ToTensor(), ]

log.write('** dataset setting **\n')

data_path = os.path.join(os.getcwd(),'data')
train_path = os.path.join(data_path, 'training')
valid_path = os.path.join(data_path, 'validation')
test_path = os.path.join(dat_path, 'training')


train_data = PolypLoader(train_path, transforms = transform)
trainloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

valid_data = PolypLoader(valid_path, transforms = transform)
valid = DataLoader(valid_data, batch_size = batch_size, shuffle=True)

test_data = PolypLoader(test_path, transforms = transform)
test= DataLoader(test_data, batch_size = batch_size, shuffle=True)


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
train_loss = np.nan
train_acc = np.nan
test_loss = np.nan
test_acc = np.nan
time = 0


for i in range(epochs):
    start = timer()       
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
    
        ## Training Pass
        net.train() ## Enter training mode

        ## Forward
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        ## Backward
        loss.backward()
        optimizer.step()

        ## Validation Pass
        net.eval()

        with torch.no_grad():
            for it, data in enumerate(validloader,0):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                pred = model(inputs)
                loss = criterion(pred, labels)

                valid_loss = loss.data[0]

        ## Saving Model
        if epoch in epoch_save:
            save_name = '%03d.pth'%epoch
            torch.save(net.state_dict(), os.path.join(out_dir,'snap', save_name))
            torch.save({
                'state_dict':net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                }, os.path.join(out_dir, 'checkpoint', save_name))

## Saving Model

print("Finished Training")

