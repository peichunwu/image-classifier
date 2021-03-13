import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

from dataset import dataset_voc
from model import train_model, test_model

data_dir = './data'
batch_size = 32
num_epochs = 15
input_size = 224
class_num = 20

config = dict()

config['use_gpu'] = True # Change this to True for training on the cluster
config['lr'] = 0.0001
config['batchsize_train'] = 16
config['batchsize_val'] = 64
config['maxnumepochs'] = 35
config['scheduler_stepsize'] = 10
config['scheduler_factor'] = 0.3

mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# datasets
image_datasets={}
image_datasets['train'] = dataset_voc(data_dir,trvaltest = "train", transform = data_transforms['train'])
image_datasets['val'] = dataset_voc(data_dir,trvaltest = 'val', transform = data_transforms['val'])

# dataloaders
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], 
        batch_size = batch_size, shuffle = True, num_workers = 8)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], 
        batch_size = batch_size, shuffle = False, num_workers = 1)

# device
device = torch.device('cuda:0') if config['use_gpu'] else torch.device('cpu')

# model
model = models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, class_num)
model.to(device)

# loss function
loss_fn = nn.BCEWithLogitsLoss(reduction = 'sum')

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = config['lr'], momentum = 0.9)

# scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['scheduler_stepsize'], eta_min = 0, last_epoch = -1)

# train the model
# model = train_model(model, dataloaders, loss_fn, optimizer, config['maxnumepochs'], scheduler, device, class_num)

# save the model
# torch.save(obj = model.state_dict(), f = "./models/model.pth")

# load the model
model.load_state_dict(torch.load('./models/model.pth', map_location='cpu'))

# test the model
_ = test_model(model, dataloaders['val'], loss_fn, device, 20, saveResult = False)
