import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import PixelCNN
import numpy as np
from tqdm import tqdm
from utils import get_n_params
import matplotlib.pyplot as plt
import copy
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nth', type=int, help='Input number')
    # parser.add_argument('-gpu_id', type=int, help='gpu id: 0 or 1')
    args = parser.parse_args() # args.nth
    nth = args.nth
    # gpu_id = args.gpu_id
    sec_len = 7
    ttstart = nth * sec_len
    ttend = min((nth+1)*sec_len, 51) 
    
    print(f"device is gpu: {torch.cuda.is_available()}")
    print(torch.cuda.device_count())
    # device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Get the current default CUDA device index
    # current_device_index = torch.cuda.current_device()
    # print(f"Current CUDA device index: cuda:{current_device_index}")
    
    start_time = time.time()
    ## ------------------params to modify-------------------------- ##
    # -------------system----------------
    L = 4 # length of the lattice
    dim = 2 # dimension of the lattice

    T_min, T_max = 0.1, 5.1
    dT = 0.1
    nt = 1 + np.int64(np.round(np.round((T_max-T_min) / dT)))
    T = np.linspace(T_min, T_max, nt)

    eqSteps = 10**5 # mixing time
    mcSteps = 10**6 # MC steps = number of samples
    sampling_interval = 10 # sampling interval
    # -------------NN arch----------------
    depth = 3 # depth of the network
    width = 6 # width of the network
    kernel_size = 3 # kernel size
    bias = False # whether to force the 1st neuron to be 0 (for z2 symmetry)
    z2 = True # whether to use z2 symmetry (1. flip the input 2. use z2 distribution)
    res_block = True # whether to use residual block
    final_conv = True # whether to use final conv layer
    # -------------training params----------------
    lr = 1e-3
    # -------------other params----------------
    epsilon = 1e-8 # epsilon for numerical stability
    x_hat_clip = 0
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    ## -------------dataset size---------------- ##
    # os.makedirs("./time/FKL", exist_ok=True)
    for num_samples in [10**6, 10**5, 10**4, 10**3, 10**2, 10**1]:
        batch_size = 64 if num_samples > 100 else num_samples
        # epochs = 10
        epochs = int(10**7 / num_samples)
        epochs = epochs if epochs <= 10**5 else 10**5
        epoch_checkpoint = int(epochs/10)
        epoch_savepoint = 1 if epochs <= 1000 else int(epochs/1000)
        
        print(f'----------------------------------------------------------------')
        print(f'num_samples = {num_samples}, epochs = {epochs}, lr = {lr}, depth = {depth}, width = {width}, bias = {bias}, final_conv = {final_conv}')

        dataload_folder = f"../../data/L={L}_Tmin={np.round(T_min,1)}_Tmax={np.round(T_max,1)}_eqSteps={eqSteps}_mcSteps={mcSteps}_interval={sampling_interval}/state"
        spins_data = np.zeros((nt, num_samples, L, L), dtype=np.float32)

        T = np.round(T,1)
        for tt in range(nt):
            spins_data[tt,:,:,:] = np.load(f"{dataload_folder}/T={T[tt]}.npy")[:num_samples, :].astype(np.float32).reshape((num_samples, L, L))

        # ------------------load data---------------------------------
        T_tensor = torch.tensor(T).float()
        spins_data_tensor = torch.tensor(spins_data).float()

        # -----------------------------------------------------------
        net_params = {
            "L": L,
            "net_depth": depth,
            "net_width": width,
            "kernel_size": kernel_size,
            "bias": bias,
            "z2": z2,
            "res_block": res_block,
            "final_conv": final_conv,
            "x_hat_clip": x_hat_clip,
            "epsilon": epsilon,
            "device": device
        }
        # ------------------save all params as a text file-----------
        assert (kernel_size//2) * depth + 1 >= L, "kernel_size and depth are too small"

        model = PixelCNN(**net_params).to(device)
        print(f'# model parameters = {get_n_params(model)}')
        print(model)

        savemodel_folder = f"./f/width={width}/numsamples={num_samples}_lr={lr}_epochs={epochs}/net_depth={depth}_net_width={width}_kernel_size={kernel_size}_bias={bias}_z2={z2}_res_block={res_block}_final_conv={final_conv}"
        os.makedirs(savemodel_folder, exist_ok=True)

        start = time.perf_counter()

        losses = torch.zeros((ttend-ttstart, epochs),dtype=torch.float32).to(device)
        T_tensor = T_tensor.to(device)
        print(f"Using {device} device")

        for id_tt,tt in enumerate(tqdm(range(ttstart, ttend))):
            os.makedirs(f"{savemodel_folder}/T={T[tt]}", exist_ok=True)
        
            model = PixelCNN(**net_params).to(device)
            model.train()
            print('------------------------------------------------')
            print(f'current temperature = {T[tt]}')

            dataset = TensorDataset(spins_data_tensor[tt, :, :, :])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # start = time.perf_counter()
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                for batch in dataloader:
                    inputs = batch[0]
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        inputs = Variable(inputs.unsqueeze(1))
                        inputs = model.flip(inputs)
                    optimizer.zero_grad()
                    log_prob = model.log_prob(inputs)
                    loss = -log_prob.mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
                losses[id_tt, epoch] = total_loss / num_batches

                if (epoch+1) % epoch_checkpoint == 0:
                    print(f'Epoch {epoch}: Average Loss = {losses[id_tt,epoch]}')
                if (epoch+1) % epoch_savepoint == 0:
                    torch.save(model.state_dict(), f"{savemodel_folder}/T={T[tt]}/epoch={epoch}.pt")
            # end = time.perf_counter()
            # print(f"eplased time = {(end-start)}s")
            # np.save(f"./time/FKL/time_numsamples={num_samples}_epochs={epochs}_id={id_tt}.npy", np.array([end-start]))

        print(f'\n---------------------***save file***------------------------\n')
        losses_cpu = losses.cpu().numpy()
        for id_tt,tt in enumerate(range(ttstart, ttend)):
            np.save(f"{savemodel_folder}/T={T[tt]}/loss.npy", losses_cpu[id_tt,:])    
        end = time.perf_counter()
        print(f"eplased time = {(end-start)}s")    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()