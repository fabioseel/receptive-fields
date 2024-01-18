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

from receptive_fields.util.experiment_setup import load_model, create_parser, load_dataset, load_log
from receptive_fields.util.training import train, validate
import torch

# Run from src directory!
parser = create_parser()
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

if args.save_hist:
    if not os.path.exists(filepath):
        os.mkdir(filepath)

train_data, test_data = load_dataset(args.dataset, True, True, args.enable_img_transforms, args.min_resize, args.max_resize, args.add_background, model.in_channels==1, model.img_size)

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
if os.path.exists(log_file):
    log_dict = load_log(log_file)
    prev_epochs = len(log_dict['train_loss'])
    log_dict['new_params_epoch_'+str(prev_epochs)]= args.__dict__
else:
    log_dict =  args.__dict__
    log_dict['train_loss'] = []
    log_dict['train_acc'] = []
    log_dict['val_acc'] = []

while not stop:
    epoch_train_loss, epoch_train_acc = train(model, optimizer, act_regularizer, weight_regularizer, train_loader, device, args.max_num_batches)

    # Print training loss for this epoch
    print(
        f"Epoch {i_epoch} - Training Loss: {epoch_train_loss}, Avg. Accuracy: {epoch_train_acc}"
    )
    epoch_val_acc = validate(model, test_loader, device)
    print(f"Epoch {i_epoch} - Test Accuracy: {epoch_val_acc}%")


    if args.save_hist:
        model.save(os.path.join(filepath,"e{:04d}".format(i_epoch)))

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