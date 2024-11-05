import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import copy
import os
import time
from time import perf_counter
import argparse
import random

def main():
    seed_number = 1234
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-nth', type=int, help='Input number')
    # parser.add_argument('-gpu_id', type=int, help='gpu id: 0 or 1')
    args = parser.parse_args() # args.nth
    nth = args.nth
    # gpu_id = args.gpu_id
    sec_len = 7
    ttstart = nth * sec_len
    ttend = min((nth+1)*sec_len, 52) 

    print(f"device is gpu: {torch.cuda.is_available()}")
    print(torch.cuda.device_count())
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    start_time = time.time()
    ## ------------------params to modify-------------------------- ##
    # -------------system----------------
    L = 4 # length of the lattice
    dim = 2 # dimension of the lattice
    
    T_min, T_max = 0.1, 5.1
    dT = 0.1
    nt = 1 + np.int64(np.round(np.round((T_max-T_min) / dT)))
    T = np.linspace(T_min, T_max, nt)
    T = np.round(T, 2)
    
    T_LBC_range = np.linspace(T_min-dT/2, T_max+dT/2, nt+1)
    T_LBC_range = np.round(T_LBC_range,2)
    
    eqSteps = 10**5 # mixing time
    mcSteps = 10**6 # MC steps = number of samples
    sampling_interval = 10 # sampling interval
    
    # -------------training params----------------
    lr = 1e-3

    # cross entropy loss
    def loss_func(outputs, targets):
        preds = F.softmax(outputs, dim=1)
        pred = preds[:, 0]
        target = targets[:,0]
        prereduce_loss = F.binary_cross_entropy(pred, target, reduction = 'none')
        losses = prereduce_loss.unsqueeze(1) * targets
        counts = torch.sum(targets, dim=0)
        factors = torch.nan_to_num(counts/counts/counts)
        loss = torch.mean(torch.sum(losses, dim=0) * factors)
        return loss

    epochs = 100
    # os.makedirs("./time/LBC", exist_ok=True)
    # for num_samples in [10**1, 3*10**1, 10**2, 3*10**2, 10**3, 3*10**3, 10**4, 3*10**4, 10**5, 10**6]:
    for num_samples in [10**6, 3*10**5]:
        batch_size = 64 if num_samples > 64 else num_samples

        epoch_checkpoint = int(epochs/10)
        print(f"-------------------num_samples={num_samples}-------------------")
        
        dataload_folder = f"../../../data/L={L}_Tmin={np.round(T_min,1)}_Tmax={np.round(T_max,1)}_eqSteps={eqSteps}_mcSteps={mcSteps}_interval={sampling_interval}/energy"
        energies_data = np.zeros((nt, num_samples), dtype=np.float32)
        T = np.round(T,1)
        for tt in tqdm(range(nt)):
            energies_data[tt,:] = np.load(f"{dataload_folder}/T={T[tt]}.npy")[:num_samples].astype(np.float32)
    
        data_mean, data_std = np.mean(energies_data), np.std(energies_data)
        data = (energies_data - data_mean) / data_std
        data = np.reshape(data, (-1))
        data = torch.tensor(data).unsqueeze(1)

        start = time.perf_counter()
        print(f"-------------------lr={lr}, epochs={epochs}-------------------")

        model = MLP_LBC(hidden_dims=16).to(device)
        num_params = get_n_params(model)
        print(f"Number of parameters: {num_params}")

        losses = torch.zeros((ttend-ttstart, epochs),dtype=torch.float32).to(device)

        for id_tt,tt in enumerate(tqdm(range(ttstart, ttend))):
            T_LBC_bp = T_LBC_range[tt]
            print(f"-------------------current bp Temperature: {T_LBC_bp:.2f}-------------------")
            
            n=5
            bl = np.maximum(tt-n, 0)
            br = np.minimum(nt, tt+n)

            targets = torch.zeros(num_samples*(br-bl), dtype=torch.float32)
            targets[:(tt-bl) * num_samples] = 1
            targets = targets.unsqueeze(1)
            targets = torch.cat((targets, 1 - targets), dim=1)

            dataset = TensorDataset(data[bl*num_samples:br*num_samples], targets)
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

            model = MLP_LBC(hidden_dims=16).to(device)
            optimizer = optim.Adam(model.parameters())
            scheduler = ExponentialLR(optimizer, gamma=0.9)

            model.train()
            # start = perf_counter()
            for epoch in range(epochs):
                running_loss = 0.0
                num_batches = 0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)

                    loss = loss_func(outputs, labels)
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    running_loss += loss.item()
                    num_batches += 1
                losses[id_tt, epoch] = running_loss / num_batches
                scheduler.step()
                
                if (epoch+1) % epoch_checkpoint == 0:
                    print(f'Epoch {epoch}: Average Loss = {losses[id_tt,epoch]}')
                os.makedirs(f"./trained/MLP_LBC_16/numsample={num_samples}_lr={lr}_epochs={epochs}/T_bp={T_LBC_range[tt]}", exist_ok=True)
                torch.save(model.state_dict(), f"./trained/MLP_LBC_16/numsample={num_samples}_lr={lr}_epochs={epochs}/T_bp={T_LBC_range[tt]}/epoch={epoch}.pt")
            
            # end = perf_counter()
            # np.save(f"./time/LBC/time_numsamples={num_samples}_epochs={epochs}_id={id_tt}.npy", np.array([end-start]))
            # print(f"eplased time = {(end-start)}s")
        print(f'\n---------------------***save file***------------------------\n')
        losses_cpu = losses.cpu().numpy()
        for id_tt,tt in enumerate(range(ttstart, ttend)):
            np.save(f"./trained/MLP_LBC_16/numsample={num_samples}_lr={lr}_epochs={epochs}/T_bp={T_LBC_range[tt]}/loss.npy", losses_cpu[id_tt,:])    
        end = time.perf_counter()
        print(f"eplased time = {(end-start)}s")
        
        # r = lbc(models, data, num_samples, device)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()