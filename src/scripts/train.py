import argparse
import os
import PIL

import matplotlib.pyplot as plt
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from receptive_fields.util.regularization import ActivationRegularization, WeightRegularization
from receptive_fields.util.dataset_transforms import RandomResize, ComposeImage
import torch_optimizer
from receptive_fields.optimizer.sgdw import SGDW

from receptive_fields.models.model_builder import load_model
from receptive_fields.util.training import train, validate
import torch

# Run from src directory!

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("--optim", type=str, default="rmsprop")
parser.add_argument("--momentum", type=float, default=0)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--weight_regularize", type=float, default=0)
parser.add_argument("--weight_norm", type=float, default=2)
parser.add_argument("--act_regularize", type=float, default=0)
parser.add_argument("--act_norm", type=float, default=2)
parser.add_argument("--save_hist", action="store_true", 
                    help="save the weights after each epoch") 
parser.add_argument("--num_epochs", type=int, default=20, 
                    help="stop after n_epochs epoch") 
parser.add_argument("--early_stop", type=int, default=0, 
                    help="stop after n epochs of not improving") 
parser.add_argument("--max_num_batches", type=int, default=None, 
                    help="use only n batches during training")
parser.add_argument("--enable_img_transforms", action="store_true", 
                    help="enable the image transforms that can be defined with add background and min/max resize")
parser.add_argument("--add_background", type=str, default="rl", 
                    help="add the rl or black background and place the image at a random position")
parser.add_argument("--min_resize", float, default=1,
                    help="the minimum factor the image is resized by before being composed onto the background") 
parser.add_argument("--max_resize", float, default=1,
                    help="the maximum factor the image is resized to before being composed onto the background") 

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
if args.optim == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == "sgdw":
    optimizer = torch_optimizer.SGDW(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == "msgdw":
    optimizer = SGDW(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, weight_norm=args.weight_norm)

act_regularizer = ActivationRegularization(model._activation_func, args.act_norm, args.act_regularize)
weight_regularizer = WeightRegularization(model, args.weight_norm, args.weight_regularize)

print("using", args.optim, "as optimizer")

transf = []
if args.enable_img_transforms:
    transf.append(RandomResize((args.min_resize, args.max_resize)))
    if args.add_background == "rl":
        bg = PIL.Image.open("../resources/empty-viewport.png")
    else:
        bg = torch.zeros((3, 120,160))
    transf.append(ComposeImage(bg))

transf.append(transforms.ToTensor())
if model.in_channels == 1:
    transf.append(transforms.Grayscale())

if args.dataset == "stl10":
    if model.img_size != 96 and not args.enable_img_transforms:
        transf.append(transforms.Resize(model.img_size, antialias=True))
    train_data = datasets.STL10(
        root="../data", split="train", download=True, transform=transforms.Compose(transf)
    )
    test_data = datasets.STL10(
        root="../data", split="test", download=True, transform=transforms.Compose(transf)
    )
else:
    if model.img_size != 32 and not args.enable_img_transforms:
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
log_dir = os.path.join("../models", _path_dir,"logs")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_file = os.path.join(log_dir, _file_name+".yaml")

log_dict =  args.__dict__

log_dict['train_loss'] = []
log_dict['train_acc'] = []
log_dict['val_acc'] = []
while not stop:
    epoch_train_loss, epoch_train_acc = train(model, optimizer, act_regularizer, weight_regularizer, train_loader, device, args.max_num_batches)

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