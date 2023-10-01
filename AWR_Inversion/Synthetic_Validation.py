import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import EncDec3DNew

from EncDec3DNew import Encoder, Decoder
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import savetxt
from Accuracy3D import SSIM, R2, RMSE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------
#  Dataset Object
# ----------------
class InputDataset(Dataset):

    def __init__(self, transform=None):
        #initial conditions
        label_list = []
        input_list = []

        perm_field_dir = os.path.join('.', 'p')
        workdir = os.path.join('.', 'o')

        nlay = 20 # number of layers
        nrow = 20 # number of rows / grid cells
        ncol = 40 # number of columns (parallel to axis of core)

        # index_1 = [x for x in range(26000,26500,500)]
        index_1 = [x for x in range(23500,24000,500)]
        invalid = False

        numb = index_1.__getitem__(-1)
        input_list = [[] for p in range(numb)]

        q = 0   # index of the elements in the input list
        for i in index_1:
            #178 148
            index_2 = [y for y in range(111,112)] #indeces for every .csv file

            for j in index_2:
                model_lab_filename_sp = perm_field_dir + str(i) + '/corenm_k_3d_m2_' + str(i-j) + '.csv'
                model_inp_filename_sp = workdir + str(i) +'/arrival_norm_diff_td' + str(i-j) + '_' + str(nlay) + '_' + str(nrow) + '_' \
                                        + str(ncol) +'.csv'


                tdata_ex = np.loadtxt(model_inp_filename_sp, delimiter=',', dtype= np.float32)
                pdata_ex = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)

                # Pre-processing the input (arrival time and porosity), labeling (permeability), and boundary condition (mean permeability)$
                perm_m = tdata_ex[-1]/np.float32(9.8692e-16)               # Conversion from m^2 to milidarcy
                perm_m = np.log(np.abs(perm_m))
                tdata_ex = tdata_ex[0:-1]
                pdata_ex = pdata_ex[0:-3]/np.float32(9.8692e-16)

                # Adding Gaussian distributed noise to the inputs
                #nstd = 0
                nstd = np.maximum(np.percentile(tdata_ex[tdata_ex != 0],99.5),np.abs(np.percentile(tdata_ex[tdata_ex != 0],0.5)))
                gaussian_arr = np.random.normal(0,np.divide(nstd,70),len(tdata_ex))
                ntsd = np.divide(nstd,70)

                print(ntsd)

                # Taking the natural logrithm of the labeling permeability data
                pdata_ex[pdata_ex == 0] = 1e-16                        # to prevent the zero padding from causing the log of zero error
                pdata_ex = np.log(pdata_ex)
                pdata_ex[pdata_ex < -20] = 0                           # change the original padding back

                # Taking the natural logrithm of the input time data
                tdata_ex[tdata_ex != 0] = tdata_ex[tdata_ex != 0] + gaussian_arr[tdata_ex != 0] # adding the noise
                tdata_ex[tdata_ex == 0] = 1e-16                        # to prevent the zero padding from causing the log of zero error
                tdata_ex = np.sign(tdata_ex)*np.log(np.abs(tdata_ex))
                tdata_ex[tdata_ex < -20] = 0                           # change the original padding back

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

        self.input = input_list
        self.perm_m = perm_m
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.transform = transform

    def __getitem__(self, index):
        sample = self.input[index]

        return sample

    def __len__(self):
        return self.len

# --------------------------------
#  Initializing Training Datasets
# --------------------------------
# #initialize dataset object
dataset = InputDataset()
dataset_input = dataset.input[0:1]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=1, shuffle=True, num_workers=2)

nf, h, w = 1, 10, 20
Tensor = torch.FloatTensor

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load('./encoder_epoch_293_284.pth', map_location='cpu'))
encoder.eval()

decoder.load_state_dict(torch.load('./decoder_epoch_293_284.pth', map_location='cpu'))
decoder.eval()

for i, (imgs_inp) in enumerate(dataloader_input):
    for j, (image) in enumerate(imgs_inp):
        # Get inputs and targets
        if j == 0:
            imgs_inp_N = Variable(image.type(Tensor))
        else:
            target = Variable(image.type(Tensor))

    encoded_imgs = encoder(imgs_inp_N)
    decoded_imgs = decoder(encoded_imgs)

# SSIM accuracy calculation
ssim_accu = SSIM()
r2_accu = R2()
rmse_accu = RMSE()

accuracy1 = ssim_accu(decoded_imgs, target)
accuracy2 = r2_accu(decoded_imgs[target.nonzero(as_tuple=True)], target[target.nonzero(as_tuple=True)])
print('SSIM: ' + str(accuracy1.item()))
print('R2: ' + str(accuracy2.item()))

inp_img = target[0][0]
dec_img = decoded_imgs[0][0]
latent_img = encoded_imgs[0][0]

inp_img = inp_img.cpu().detach().numpy()
latent_img = latent_img.cpu().detach().numpy()
dec_img = dec_img.cpu().detach().numpy()

accuracy3 = rmse_accu(dec_img, inp_img)
print('RMSE: ' + str(accuracy3))

# Convert the natural log to the log base 10
inp_img = np.log10(np.exp(inp_img.flatten()))
dec_img = np.log10(np.exp(dec_img.flatten()))
inp_img = inp_img.reshape(20,20,40)
dec_img = dec_img.reshape(20,20,40)
dec_img[inp_img == 0] = 0
#inp_img = np.flip(inp_img, 0)
#dec_img = np.flip(dec_img, 0)


# =============================================================================
# PLOT DATA
# =============================================================================
# layer to plot
ilayer = 0
# fontsize
fs = 13
hfont = {'fontname':'Arial'}


# Grid cell size
grid_size = [0.25, 0.25, 0.25] # selected units [cm]

# Define grid
# Describe grid for results
Lx = dataset.ncol * grid_size[2]   # length of model in selected units
Ly = dataset.nrow * grid_size[1]   # length of model in selected units
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                 slice(0, Lx + grid_size[2], grid_size[2])]

# First figure with head and breakthrough time difference maps
fig1 = plt.figure(figsize=(18, 9))
ax2 = fig1.add_subplot(2, 2, 1, aspect='equal')
imp = plt.pcolor(x, y, inp_img[round(dataset.nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('log(k) [mD]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Permeability Layer 10', fontsize=fs+2, **hfont)
plt.clim(np.min(inp_img[round(dataset.nlay/2),:,:]), np.max(inp_img[round(dataset.nlay/2),:,:]))

ax2 = fig1.add_subplot(2, 2, 2, aspect='equal')
imp = plt.pcolor(x, y, dec_img[round(dataset.nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('log(k) [mD]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Decoded Permeability Layer 10', fontsize=fs+2, **hfont)
plt.clim(np.min(inp_img[round(dataset.nlay/2),:,:]), np.max(inp_img[round(dataset.nlay/2),:,:]))

ax2 = fig1.add_subplot(2, 2, 3, aspect='equal')
imp = plt.pcolor(x, y, inp_img[:,round(dataset.nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('log(k) [mD]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Permeability Row 10', fontsize=fs+2, **hfont)
plt.clim(np.min(inp_img[:,round(dataset.nrow/2),:]), np.max(inp_img[:,round(dataset.nrow/2),:]))

ax2 = fig1.add_subplot(2, 2, 4, aspect='equal')
imp = plt.pcolor(x, y, dec_img[:,round(dataset.nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('log(k) [mD]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Decoded Permeability Row 10', fontsize=fs+2, **hfont)
plt.clim(np.min(inp_img[:,round(dataset.nrow/2),:]), np.max(inp_img[:,round(dataset.nrow/2),:]))

plt.show()

inp_img = inp_img.flatten()
dec_img = dec_img.flatten()
dec_img[dec_img != 0] = (10**(dec_img[dec_img != 0]))*np.float32(9.8692e-16)
latent_img = latent_img.flatten()
