import torch
import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("out_dir", type=str)
parser.add_argument("--min_channels", default=1, type=int)
parser.add_argument("--max_channels", default=16, type=int)
parser.add_argument("--min_kernel", default=1, type=int)
parser.add_argument("--max_kernel", default=13, type=int)
parser.add_argument("--n_inp_channel", default=1, type=int)
parser.add_argument("--in_width", default=128, type=int)
parser.add_argument("--in_height", default=128, type=int)
parser.add_argument("--batch_size", default=128, type=int)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

channel_sizes = [c for c in range(args.min_channels,args.max_channels+1)]
kernel_sizes = [k for k in range(args.min_kernel,args.max_kernel+1)]

n_inp_channel=args.n_inp_channel

times = np.empty((len(kernel_sizes), len(channel_sizes)))
conv_times = np.empty((len(kernel_sizes), len(channel_sizes)))
conv_backprop_times = np.empty((len(kernel_sizes), len(channel_sizes)))
pool_times = np.empty((len(kernel_sizes), len(channel_sizes)))
sim_inputs = torch.rand(args.batch_size, n_inp_channel, args.in_width, args.in_height).to(device)
# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
for i, k in enumerate(kernel_sizes):
    for j, c in enumerate(channel_sizes):
        minimodel = torch.nn.Sequential(torch.nn.Conv2d(n_inp_channel, out_channels=c, kernel_size=k), torch.nn.Flatten())
        minimodel = minimodel.to(device)
        sim_out = torch.randint(minimodel(sim_inputs).shape[1], sim_inputs.shape[:1]).to(device)
        minimodel.train()  # Set the model to training mode

        optimizer = torch.optim.RMSprop(minimodel.parameters(), lr=0.005, weight_decay=1e-6)
        with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
            # Training loop
            running_loss = 0.0
            epoch_correct = 0
            pbar = tqdm(range(100))
            for _batch in pbar:
                optimizer.zero_grad()
                # Forward pass
                outputs = minimodel(sim_inputs)
                loss = criterion(outputs, sim_out)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                pbar.set_postfix({'Loss': loss.item()})
        times[i,j] = prof.total_average().self_cuda_time_total

        avgs = prof.key_averages()
        index_conv = np.argmax([avgs[i].key == "aten::conv2d" for i in range(len(avgs))])
        index_backprop = np.argmax([avgs[i].key == "aten::convolution_backward" for i in range(len(avgs))])
        index_pool = np.argmax([avgs[i].key == "aten::avg_pool2d_backward" for i in range(len(avgs))])
        conv_times[i,j] = avgs[index_conv].self_cuda_time_total
        conv_backprop_times[i,j] = avgs[index_backprop].self_cuda_time_total
        pool_times[i,j] = avgs[index_pool].self_cuda_time_total

times_dict = {
    "args": vars(args),
    "kernel_sizes": kernel_sizes,
    "channel_sizes": channel_sizes,
    "times": times,
    "conv_times": conv_times,
    "conv_backprop_times": conv_backprop_times,
    "pool_times": pool_times
}
save_to = args.out_dir
num_files = len(os.listdir(save_to))
with open(os.path.join(save_to, "run{:02d}.txt".format(num_files)), 'w+') as f:
        yaml.dump(times_dict, f)