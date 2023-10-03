import os
import sys
import math
import time
import torch
import flopy
import sklearn
import itertools
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib as mpl
import mpl_scatter_density
import matplotlib.pyplot as plt

from numpy import savetxt
from torch.autograd import Variable
from sklearn.metrics import r2_score
from torch.nn import functional as F
from EncDec3DNew import Encoder, Decoder
from torch.utils.data import Dataset, DataLoader
from matplotlib.pylab import *
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from flopy_arrival_time_3d_functions import mt3d_pulse_injection_sim, flopy_arrival_map_function


def c_plot(fig, labeled, decoded):
    x = labeled[labeled != 0]
    y = decoded[labeled != 0]

    yellow_pink = LinearSegmentedColormap.from_list('yellow_pink', [
        (0, '#ffffff'),
        (1e-20, 'red'),
        (0.2, '#ff57bb'),
        (0.4, '#faaaae'),
        (0.6, '#fdf1d2'),
        (0.8, '#f8eaad'),
        (1, 'yellow'),
    ], N=256)

    bin = plt.hexbin(x, y, gridsize=100, cmap=yellow_pink)


    plt.title('R2 = {:.3f}'.format(r2_score(x, y)),**cfont)

    ymax = np.max(y)
    ymin = np.min(y)
    plt.plot([ymin,ymax],[ymin,ymax], color = 'dimgray')
    plt.xlabel('Experimental Arrival Time',**cfont)
    plt.ylabel('Validating Arrival Time',**cfont)

    return bin


# =============================================================================
#     EXPERIMENTAL DATA LOADING
# =============================================================================
class InputDataset(Dataset):

    def __init__(self, transform=None):
        input_list = []

        nlay = 20 # number of layers
        nrow = 20 # number of rows / grid cells
        ncol = 40 # number of columns (parallel to axis of core)

        exp_data_dir = os.path.join('.', 'Exp_Time')
        Normalized_BT_time = exp_data_dir + '/Edwards_2ml_2_3mm_at_norm.csv'
        #Normalized_BT_time = exp_data_dir + '/Navajo_2ml_2_2_3mm_at_norm.csv'
        #Normalized_BT_time = exp_data_dir + '/Berea_C1_2ml_2_3mm_at_norm.csv'
        #Normalized_BT_time = exp_data_dir + '/Bentheimer_2ml_2_3mm_at_norm.csv'
        #Normalized_BT_time = exp_data_dir + '/Indiana_2ml_2_3mm_at_norm.csv'
        #Normalized_BT_time = exp_data_dir + '/Estaillades_3ml_2_3mm_at_norm.csv'
        #Normalized_BT_time = exp_data_dir + '/Ketton_4ml_2_2_3mm_at_norm.csv'

        # Import core shape mask for permeability boundary processing
        # Hard coded mask for 20x20 grid cross-section, this creates the a cylindrical shape core
        mp = np.array([7, 5, 3, 2, 2, 1, 1])
        mask_corner = np.ones((10,10))
        for i in range(len(mp)):
            mask_corner[i, 0:mp[i]] = 0

        mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
        core_mask = np.concatenate((mask_top, np.flipud(mask_top)))
        ibound = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
        ibound = ibound.flatten()

        tdata_ex = np.loadtxt(Normalized_BT_time, delimiter=',', dtype= np.float32)

        # Get the geometric mean permeability of the core
        k_mean = tdata_ex[-1]/np.float32(9.8692e-16)
        k_mean = np.sign(k_mean)*np.log(np.abs(k_mean))

        # Apply the mask to the experimental arrival time data
        tdata_ex = tdata_ex[0:-1]
        tdata_ex[ibound == 0] = 0
        tdata_ex[ibound != 0] = np.sign(tdata_ex[ibound != 0])*np.log(np.abs(tdata_ex[ibound != 0]))

        tdata_ex = tdata_ex.reshape(1,20,20,40)
        tdata_ex = np.flip(tdata_ex, 1)
        tdata_ex = torch.from_numpy(tdata_ex.copy())
        Pad = nn.ConstantPad3d((1,0,0,0,0,0), k_mean)
        tdata_ex = Pad(tdata_ex)
        input_list.append(tdata_ex)

        self.input = input_list
        self.k_mean = k_mean
        self.exp_data_dir = exp_data_dir
        self.ibound = ibound


# =============================================================================
# IMAGE REGRESSION MODEL
# =============================================================================
# #initialize dataset object
dataset = InputDataset()
dataset_input = dataset.input[0:1]
dataloader_input = DataLoader(dataset=dataset_input, batch_size=1)

Tensor = torch.FloatTensor
encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load('./encoder_epoch_293_284.pth', map_location='cpu'))
encoder.eval()

decoder.load_state_dict(torch.load('./decoder_epoch_293_284.pth', map_location='cpu'))
decoder.eval()

for i, (imgs_inp) in enumerate(dataloader_input):
    imgs_inp_N = Variable(imgs_inp.type(Tensor))
    encoded_imgs = encoder(imgs_inp_N)
    decoded_imgs = decoder(encoded_imgs)

# Get the prediction
enc_time = imgs_inp_N[0][0]
dec_perm = decoded_imgs[0][0]
enc_time = enc_time.cpu().detach().numpy()
dec_perm = dec_perm.cpu().detach().numpy()

enc_time = np.delete(enc_time,0,axis=2)
enc_time = enc_time.flatten()
dec_perm = dec_perm.flatten()

# Set geometric information
nlay = 20 # number of layers / grid cells
nrow = 20 # number of rows / grid cells
ncol = 40 # number of columns (along to axis of core)

enc_time[dataset.ibound == 0] = 0
enc_time[dataset.ibound != 0] = -1*np.sign(enc_time[dataset.ibound != 0])*np.exp(-1*np.sign(enc_time[dataset.ibound != 0])*enc_time[dataset.ibound != 0])
dec_perm[dataset.ibound == 0] = 0
dec_perm[dataset.ibound != 0] = np.exp(dec_perm[dataset.ibound != 0])
dec_perm = dec_perm*np.float32(9.8692e-16)

# Get the mean and standard deviation of the predicted permeability distribution
core_avg_perm = dec_perm[np.nonzero(dec_perm)].mean()
core_std_perm = dec_perm[np.nonzero(dec_perm)].std()
print('The core average permeability is: ' + str(core_avg_perm))
print('The standard deviation of the core permeability is: ' + str(core_std_perm))


dec_perm = dec_perm.reshape(20,20,40)
enc_time = enc_time.reshape(20,20,40)

# =============================================================================
# FORWARD MODEL
# =============================================================================
# names of executable with path IF NOT IN CURRENT DIRECTORY
exe_name_mf = '/Users/zhuang296/Desktop/mac/mf2005'
exe_name_mt = '/Users/zhuang296/Desktop/mac/mt3dms'

# directory to save data
directory_name = '/Users/zhuang296/Desktop/FloPy1D'
workdir = os.path.join('.', directory_name)

# grid_size = [grid size in direction of Lx (layer thickness),
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = [0.23291, 0.23291, 0.25] # selected units [cm]
# Output control for MT3dms
# nprs (int):  the frequency of the output. If nprs > 0 results will be saved at
# the times as specified in timprs (evenly allocated between 0 and sim run length);
# if nprs = 0, results will not be saved except at the end of simulation; if NPRS < 0, simulation results will be
# saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
nprs = 150
# period length in selected units (for steady state flow it can be set to anything)
perlen_mf = [1., 90]
# Numerical method flag
mixelm = -1
# Porosity of the core
#enc_prsity = 0.167 # Indiana
#enc_prsity = 0.207 # Berea
#enc_prsity = 0.195 # Navajo
#enc_prsity = 0.251 # Bentheimer
enc_prsity = 0.413 # Edwards
#enc_prsity = 0.250 # Estaillades
#enc_prsity = 0.230 # Ketton

# Model workspace and new sub-directory
model_dirname = ('t_forward')
model_ws = os.path.join(workdir, model_dirname)

# Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
raw_hk = dec_perm*(1000*9.81*100*60/8.9E-4)
# print(np.average(raw_hk))
# print(np.max(raw_hk))
# print(np.min(raw_hk[raw_hk != 0]))

# Get the breakthrough time data
mf, mt, conc, timearray, km2_mean = mt3d_pulse_injection_sim(enc_prsity, model_dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt)
# calculate quantile arrival time map from MT3D simulation results
at_array, at_array_norm, at_diff_norm = flopy_arrival_map_function(conc, np.array(timearray), grid_size, 0.5, 0.1)

# Get the difference between the experimental arrival time data and simulated arrival time data
diff_img = at_diff_norm - enc_time

# =============================================================================
# PLOT DATA
# =============================================================================
# Layer to plot
ilayer = 0
# Font
fs = 10
hfont = {'fontname':'Arial','size':fs}
# Font for the cross-plots
fsc = 43
cfont = {'fontname':'Arial','size':fsc}

# Define grid
# Describe grid for results
Lx = ncol * grid_size[2]   # length of model in selected units
Ly = nrow * grid_size[1]   # length of model in selected units
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                 slice(0, Lx + grid_size[2], grid_size[2])]

max = np.max(np.percentile(enc_time.flatten(),[1,99]))
min = np.min(np.percentile(enc_time.flatten(),[1,99]))

# First figure with head and breakthrough time difference maps
fig1 = plt.figure(figsize=(18, 9))
ax2 = fig1.add_subplot(3, 2, 3, aspect='equal')
imp = plt.pcolor(x, y, enc_time[round(nlay/2),:,:], cmap='PiYG', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Real Arrival Time Layer 10', fontsize=fs, **hfont)
plt.clim(min, max)

ax2 = fig1.add_subplot(3, 2, 5, aspect='equal')
imp = plt.pcolor(x, y, at_diff_norm[round(nlay/2),:,:], cmap='PiYG', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Validation Arrival Time Layer 10', fontsize=fs, **hfont)
plt.clim(min, max)
#plt.clim(np.min(enc_time[:,:,:]), np.max(enc_time[:,:,:]))
#plt.clim(np.min(at_diff_norm[round(nlay/2),:,:]), np.max(at_diff_norm[round(nlay/2),:,:]))

ax2 = fig1.add_subplot(3, 2, 1, aspect='equal')
imp = plt.pcolor(x, y, dec_perm[round(nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Permeability [$m^{2}$]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Decoded Permeability Layer 10', fontsize=fs, **hfont)
plt.clim(np.min(dec_perm[round(nlay/2),:,:]), np.max(dec_perm[round(nlay/2),:,:]))

ax2 = fig1.add_subplot(3, 2, 2, aspect='equal')
imp = plt.pcolor(x, y, dec_perm[:,round(nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Permeability [$m^{2}$]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Decoded Permeability Row 10', fontsize=fs, **hfont)
plt.clim(np.min(dec_perm[:,round(nrow/2),:]), np.max(dec_perm[:,round(nrow/2),:]))

ax2 = fig1.add_subplot(3, 2, 4, aspect='equal')
imp = plt.pcolor(x, y, enc_time[:,round(nrow/2),:], cmap='PiYG', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Real Arrival Time Row 10', fontsize=fs, **hfont)
plt.clim(min, max)
#plt.clim(np.min(enc_time[:,:,:]), np.max(enc_time[:,:,:]))

ax2 = fig1.add_subplot(3, 2, 6, aspect='equal')
imp = plt.pcolor(x, y, at_diff_norm[:,round(nrow/2),:], cmap='PiYG', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Validation Arrival Time Row 10', fontsize=fs, **hfont)
plt.clim(min, max)
#plt.clim(np.min(enc_time[:,:,:]), np.max(enc_time[:,:,:]))
#plt.clim(np.min(at_diff_norm[:,round(nrow/2),:]), np.max(at_diff_norm[:,round(nrow/2),:]))

# #For plotting label-prediction difference map
# ax2 = fig1.add_subplot(3, 2, 1, aspect='equal')
# imp = plt.pcolor(x, y, diff_img[round(nlay/2),:,:], cmap='PiYG', edgecolors='k', linewidths=0.2)
# cbar = plt.colorbar()
# cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
# cbar.ax.tick_params(labelsize= (fs))
# ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
# ax2.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
# plt.title('Arrival Time Difference Layer 10', fontsize=fs, fontweight='bold', **hfont)
# plt.clim(-max, max)
#
# #For plotting label-prediction difference map
# ax2 = fig1.add_subplot(3, 2, 2, aspect='equal')
# imp = plt.pcolor(x, y, diff_img[:,round(nrow/2),:], cmap='PiYG', edgecolors='k', linewidths=0.2)
# cbar = plt.colorbar()
# cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
# cbar.ax.tick_params(labelsize= (fs))
# ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
# ax2.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
# plt.title('Arrival Time Difference Row 10', fontsize=fs, fontweight='bold', **hfont)
# plt.clim(-max, max)

plt.subplots_adjust( bottom=0.15, top=0.96, wspace=0.05, hspace=0.45)

dec_hk = dec_perm[:,:,:]*(1000*9.81*100*60/8.9E-4)

dec_hk = dec_hk.flatten()
enc_time = enc_time.flatten()
dec_perm = dec_perm.flatten()
at_diff_norm = at_diff_norm.flatten()

# Generate the cross-plot
fig2 = plt.figure(figsize=(14, 9))
ax3 = fig2.add_subplot(111)
y = at_diff_norm[enc_time != 0].flatten()
x = enc_time[enc_time != 0].flatten()
bin_plt = c_plot(fig2, x, y)
fig2.subplots_adjust(left=0.18,right=0.79,top=0.9,bottom=0.15)
# L B W H
cbar_ax = fig2.add_axes([0.82, 0.15, 0.03, 0.75])
cbar2 = fig2.colorbar(bin_plt, cax=cbar_ax)
cbar2.set_label('Count in Bin', fontsize=fsc, **cfont)
for t in cbar2.ax.get_yticklabels():
     t.set_fontsize(fsc)
for item in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(fsc)

# Generate histograms of log permeability
fig3 = plt.figure(figsize=(18, 9))
plt.hist(dec_perm[np.nonzero(dec_perm)], 40, facecolor='blue', alpha=0.5)
plt.xlabel('k [$m^{2}$]',**hfont)

plt.show()
