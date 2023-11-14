import argparse
import os

import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_builder import load_model
from training import train, validate
import torch

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("--save_hist", action="store_true", 
                    help="save the weights after each epoch") 
parser.add_argument("--num_epochs", type=int, default=20, 
                    help="stop after n_epochs epoch") 
parser.add_argument("--early_stop", type=int, default=0, 
                    help="stop after n epochs of not improving") 


args = parser.parse_args()

print("Enabled history saving: ", args.save_hist)

filepath = args.config

model = load_model(filepath)
# for i in range(1, len(model.retina)):
#     for param in model.retina[i].parameters():
#         param.requires_grad = False
# for i in range(len(model.fc)-1):
#     for param in model.fc[i].parameters():
#         param.requires_grad = False
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-6)


transf = [transforms.ToTensor()]
if model.in_channels == 1:
    transf.append(transforms.Grayscale())

if args.dataset == "stl10":
    if model.img_size != 96:
        transf.append(transforms.Resize(model.img_size, antialias=True))
    train_data = datasets.STL10(
        root="../data", split="train", download=True, transform=transforms.Compose(transf)
    )
    test_data = datasets.STL10(
        root="../data", split="test", download=True, transform=transforms.Compose(transf)
    )
else:
    if model.img_size != 32:
        transf.append(transforms.Resize(model.img_size, antialias=True))
    train_data = datasets.CIFAR10(
        root="../data", train=True, download=True, transform=transforms.Compose(transf)
    )
    test_data = datasets.CIFAR10(
        root="../data", train=False, download=True, transform=transforms.Compose(transf)
    )

if args.save_hist:
    if not os.path.exists(filepath):
        os.mkdir(filepath)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, prefetch_factor=4)

prev_best_acc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
model.to(device)

i_epoch = 0
stop=False
_path_dir, _file_name = os.path.split(filepath)
log_file = os.path.join("../models", _path_dir,"logs", _file_name+".yaml")

log_dict =  {}
log_dict['model_config'] = args.config
log_dict['dataset'] = args.dataset
log_dict['batch_size'] = args.batch_size
log_dict['lr'] = args.lr

log_dict['train_loss'] = []
log_dict['train_acc'] = []
log_dict['val_acc'] = []
while not stop:
    epoch_train_loss, epoch_train_acc = train(model, optimizer, train_loader, device)

    # if i_epoch % 5 == 0:
    #     defreeze_no = i_epoch // 5
    #     if defreeze_no < len(model.fc):
    #         for param in model.fc[len(model.fc)-defreeze_no-1].parameters():
    #             param.requires_grad = True
    #     if defreeze_no < 4:
    #         conv_layers_retina = [0,3,6,8]
    #         for param in model.retina[conv_layers_retina[defreeze_no]].parameters():
    #             param.requires_grad = True
    
    # if i_epoch==20:
    #     for param in model.retina[0].parameters():
    #         param.requires_grad = False

    # Print training loss for this epoch
    print(
        f"Epoch {i_epoch} - Training Loss: {epoch_train_loss}, Avg. Accuracy: {epoch_train_acc}"
    )
    epoch_val_acc = validate(model, test_loader, device)
    print(f"Epoch {i_epoch} - Test Accuracy: {epoch_val_acc}%")


    if args.save_hist:
        model.save(os.path.join(filepath,"e{:02d}".format(i_epoch)))

    if epoch_val_acc > prev_best_acc:
        prev_best_acc = epoch_val_acc
        inc_count = 0
        if not args.save_hist:
            model.save(filepath)
    else:
        inc_count += 1
        if args.early_stop == 0:
            stop = i_epoch >= args.num_epochs-1
        elif inc_count >= args.early_stop:
            stop = True

    log_dict['train_loss'].append(epoch_train_loss)
    log_dict['train_acc'].append(epoch_train_acc)
    log_dict['val_acc'].append(epoch_val_acc)
    with open(log_file, 'w+') as f:
        yaml.dump(log_dict, f)
    i_epoch+=1