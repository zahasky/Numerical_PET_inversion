import os
import math
import torch
import torch.nn as nn
import random
import numpy as np
import argparse

from random import shuffle
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='DataLoader for all the training and validation data')
# Geometry of the geologic core
parser.add_argument('--nlay', type=int, default=20, help='Number of layers (along the z-axis)')
parser.add_argument('--nrow', type=int, default=20, help='Number of rows (along the y-axis)')
parser.add_argument('--ncol', type=int, default=40, help='Number of columns (along the x-axis)')
# Data Loading
parser.add_argument('--idx-beg', type=int, default=26000, help='Index of the starting folder in the dataset')
parser.add_argument('--idx-end', type=int, default=26500, help='Index of the ending folder in the dataset')
parser.add_argument('--num-data', type=int, default=500, help='Number of the data in each folder')
parser.add_argument('--num-data-w', type=int, default=8, help='Number of the data wanted in each folder')

args = parser.parse_args()


#----------------
# Dataset Object
#----------------
class InputDataset(Dataset):
    def __init__(self, transform=None):
        # Directories for the folders that containing the input (arrival time) and labeling (permeability) data
        time_field_dir = os.path.join('.', 'o')
        perm_field_dir = os.path.join('.', 'p')

        # Containing names of the training/validation data folder
        index_1 = [x for x in range(args.idx_beg,args.idx_end,args.num_data)]
        numb = index_1.__getitem__(-1)                                    # number of training/validation data

        # List containing both the training (input_list) and labeling (label_list) data
        input_list = []
        input_list = [[] for p in range(numb)]

        q = 0                                                             # index indicating the ordering of the elements in the data list

        for i in index_1:
            # Indices for every .csv file
            index_2 = [y for y in range(args.num_data_w)]

            for j in index_2:
                model_lab_filename_sp = perm_field_dir + str(i) + '/corenm_k_3d_m2_' + str(i-j) + '.csv'
                model_inp_filename_sp = time_field_dir + str(i) +'/arrival_norm_diff_td' + str(i-j) + '_' + str(args.nlay) + '_' + str(args.nrow) + '_' \
                                        + str(args.ncol) +'.csv'

                # Try to catch and report the files containing the ValueError
                try:
                    # Loading the input (arrival time) and labeling (permeability) data
                    tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype= np.float32)
                    pdata_ex = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)

                    # Pre-processing the input (arrival time and porosity), labeling (permeability), and boundary condition (mean permeability) data
                    perm_m = tdata_ex[-1]/np.float32(9.8692e-16)           # conversion from m^2 to milidarcy
                    perm_m = np.log(np.abs(perm_m))
                    tdata_ex = tdata_ex[0:-1]
                    pdata_ex = pdata_ex[0:-3]/np.float32(9.8692e-16)

                    # Adding Gaussian distributed noise to the inputs
                    nstd = np.maximum(np.percentile(tdata_ex[tdata_ex != 0],99.5),np.abs(np.percentile(tdata_ex[tdata_ex != 0],0.5)))
                    gaussian_arr = np.random.normal(0,np.divide(nstd,70),len(tdata_ex))

                    # Taking the natural logrithm of the labeling permeability data
                    pdata_ex[pdata_ex == 0] = 1e-16                        # to prevent the zero padding from causing the log of zero error
                    pdata_ex = np.log(pdata_ex)
                    pdata_ex[pdata_ex < -20] = 0                           # change the original padding back

                    # Taking the natural logrithm of the input time data
                    tdata_ex[tdata_ex != 0] = tdata_ex[tdata_ex != 0] + gaussian_arr[tdata_ex != 0] # adding the noise
                    tdata_ex[tdata_ex == 0] = 1e-16                                                 # to prevent the zero padding from causing the log of zero error
                    tdata_ex = np.sign(tdata_ex)*np.log(np.abs(tdata_ex))
                    tdata_ex[tdata_ex < -20] = 0                                                    # change the original padding back

                    # Changing all the data to torch tensor
                    tdata_ex = torch.from_numpy(tdata_ex)
                    pdata_ex = torch.from_numpy(pdata_ex)
                    tdata_ex = tdata_ex.reshape(1,20,20,40)
                    pdata_ex = pdata_ex.reshape(1,20,20,40)

                    # Adding the boundary condition (mean permeability) to the input (arrival time and porosity) data
                    Pad = nn.ConstantPad3d((1,0,0,0,0,0), perm_m)
                    tdata_ex = Pad(tdata_ex)

                    # Loading all the processed data
                    input_list[q].append(tdata_ex)
                    input_list[q].append(pdata_ex)
                    q=q+1

                except ValueError:
                    print('Value error found in the file: ' + str(i-j))
                    continue

        # Shuffle the input dataset across different classes of data
        shuffle(input_list)
        # Remove all empty elements
        l = [ele for ele in input_list if ele != []]
        # print(input_list)

        self.input = l
        self.nrow = args.nrow
        self.ncol = args.ncol
        self.nlay = args.nlay
