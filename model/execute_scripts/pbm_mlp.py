import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
import warnings

def main():
    seed_number = 1234
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)

    parser = argparse.ArgumentParser()
    parser.add_argument('-lrIndx', type=int, help='Input number')
    args = parser.parse_args() # args.lrIndx
    lrIndx = args.lrIndx
    lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
    lr = lrs[lrIndx]

    warnings.filterwarnings("ignore", category=UserWarning)
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
    
    eqSteps = 10**5 # mixing time
    mcSteps = 10**6 # MC steps = number of samples
    sampling_interval = 10 # sampling interval
    
    # -------------training params----------------
    epochs = 100
    # os.makedirs("./time/PBM", exist_ok=True)
    
    # mse loss
    def loss_func(outputs, target):
        losses = (outputs[:,0] - target[:,-1])**2 # shape (batch_size,)
        losses_filtered = target[:,:-1] * losses.unsqueeze(1) # (batch_size, nt=n_class) * (batch_size, 1)
        counts = torch.sum(target[:,:-1], dim=0)
        factors = torch.nan_to_num(counts/counts/counts)
        loss = torch.mean(torch.sum(losses_filtered, dim=0) * factors)
        return loss
    
    for num_samples in [10**1, 3*10**1, 10**2, 3*10**2, 10**3, 3*10**3, 10**4, 3*10**4, 10**5, 3*10**5, 10**6]:
        batch_size = 64 if num_samples > 100 else num_samples
        epoch_checkpoint = int(epochs/10)
            
        print(f"-------------------num_samples={num_samples}-------------------")

        dataload_folder = f"../../../data/L={L}_Tmin={np.round(T_min,1)}_Tmax={np.round(T_max,1)}_eqSteps={eqSteps}_mcSteps={mcSteps}_interval={sampling_interval}/energy"

        energies_data = np.zeros((nt, num_samples), dtype=np.float32)
        for tt in tqdm(range(nt)):
            energies_data[tt,:] = np.load(f"{dataload_folder}/T={T[tt]}.npy")[:num_samples].astype(np.float32)
        
        data_mean, data_std = np.mean(energies_data), np.std(energies_data)
        data = (energies_data - data_mean) / data_std # shape = (nt, N)
        data = np.reshape(data, (-1))
        data = torch.tensor(data).unsqueeze(1) # shape = (nt*N, 1)

        targets = torch.cat([F.one_hot(torch.arange(nt).repeat_interleave(num_samples), num_classes = nt), torch.tensor(T).repeat_interleave(num_samples).unsqueeze(1)], dim=1).to(torch.float32)

        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    
        start = time.perf_counter()
        print(f"-------------------lr={lr}, epochs={epochs}-------------------")

        model = MLP_PBM().to(device)
        num_params = get_n_params(model)
        print(f"Number of parameters: {num_params}")
    
        optimizer = optim.Adam(model.parameters(), lr=lr)

        losses = torch.zeros(epochs,dtype=torch.float32).to(device)

        # Training loop
        model.train()         
        # start = perf_counter()
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            num_batch = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = loss_func(outputs, labels)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batch += 1 
            losses[epoch] = epoch_loss / num_batch
            
            if (epoch+1) % epoch_checkpoint == 0:
                print(f'Epoch {epoch}: Average Loss = {losses[epoch]}')
            os.makedirs(f"./trained/MLP_PBM/numsample={num_samples}_lr={lr}_epochs={epochs}/", exist_ok=True)
            torch.save(model.state_dict(), f"./trained/MLP_PBM/numsample={num_samples}_lr={lr}_epochs={epochs}/epoch={epoch}.pt")
        
        # end = perf_counter()
        # np.save(f"./time/PBM/time_numsamples={num_samples}_epochs={epochs}.npy", np.array([end-start]))
        # print(f"eplased time = {(end-start)}s")
    
        print(f'\n---------------------***save file***------------------------\n')
        losses_cpu = losses.cpu().numpy()
        np.save(f"./trained/MLP_PBM/numsample={num_samples}_lr={lr}_epochs={epochs}/loss.npy", losses_cpu)   
        
        # r = pbm(model, data, num_samples, nt, dT, device)
        # os.makedirs(f"../../../results/disc/e/mlp/numsample={num_samples}_lr={lr}_epochs={epochs}/", exist_ok=True)
        # np.save(f"../../../results/disc/e/mlp/numsample={num_samples}_lr={lr}_epochs={epochs}/I_PBM_numsamples={num_samples}.npy", r)
        # np.save(f"../../../results/disc/e/mlp/numsample={num_samples}_lr={lr}_epochs={epochs}/T_PBM_numsamples={num_samples}.npy", T[1:-1])

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()