import os
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn

from utils.util import find_max_epoch, print_size, training_loss, sampling, calc_diffusion_hyperparams

from imputer.SSSDS4Imputer import SSSDS4Imputer

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)

def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_logging,
          learning_rate,
         target_class):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model

    net = SSSDS4Imputer(**model_config).cuda()
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        
        
    target_class = target_class
    print(target_class)
    
    if target_class == 'uncond':
        
        training_data = np.load('../data/x_train.npy').transpose(0,2,1)
        target_indexes_train = np.arange(len(training_data))
        
        total_masks_train = []
        train_masks = np.load('../data/train_concepts.npy').transpose(0,2,1)
        for c in train_masks:
            total_masks_train.append(c[target_indexes_train])
        
        total_signals_train = []
        for i in range(len(train_masks)):
            for st in training_data:
                total_signals_train.append(st)
                
    
    
    elif target_class == 'class':
        
        training_data = np.load('../data/x_train.npy').transpose(0,2,1)
        y_train = np.load('../data/y_train.npy')
        target_indexes_train = np.where(y_train==1)[0]
        
        total_masks_train = []
        train_masks = np.load('../data/train_concepts.npy').transpose(0,2,1)
        for c in train_masks:
            total_masks_train.append(c[target_indexes_train])
        
        total_signals_train = []
        for i in range(len(train_masks)):
            for st in training_data:
                total_signals_train.append(st)
    
    
    
    elif target_class == 'norm':
        
        training_data = np.load('../data/x_train.npy').transpose(0,2,1)
        y_train = np.load('../data/y_train.npy')
        target_indexes_train = np.where(y_train==0)[0]
        
        total_masks_train = []
        train_masks = np.load('../data/train_concepts.npy').transpose(0,2,1)
        for c in train_masks:
            total_masks_train.append(c[target_indexes_train])
        
        total_signals_train = []
        for i in range(len(train_masks)):
            for st in training_data:
                total_signals_train.append(st)


    train_data = []
    for i in range(len(total_signals_train)):
        train_data.append([total_signals_train[i], total_masks_train[i]])

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=6, drop_last=True)
    

    
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for signal, mask in trainloader:

            signal = signal.float().cuda()
            mask = mask.float().cuda()
            loss_mask = ~mask.bool().cuda()
                        
            assert signal.size() == mask.size() == loss_mask.size()
            
            optimizer.zero_grad()
            
            X = signal, signal, mask, loss_mask
            
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams)
                        
            loss.backward()
            optimizer.step()
            
            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
                
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print(f'model at iteration {n_iter} is saved with loss of {round(best_loss,4)}')
                
            n_iter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config

    model_config = config['wavenet_config']

    train(**train_config)
