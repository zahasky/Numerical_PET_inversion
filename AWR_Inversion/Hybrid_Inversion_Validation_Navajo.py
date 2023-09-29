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
from torch.utils.data import Dataset, DataLoader
from matplotlib.pylab import *
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from Accuracy3D import SSIM
from flopy_arrival_time_3d_functions import mt3d_pulse_injection_sim, flopy_arrival_map_function


def c_plot(fig, labeled, decoded):
    x = labeled[labeled != 0]
    y = decoded[labeled != 0]

    blue_pink = LinearSegmentedColormap.from_list('blue_pink', [
        (0, '#ffffff'),
        (1e-20, '#0091AD'),
        (0.2, '#3fcdda'),
        (0.4, '#83f9f8'),
        (0.6, '#fdf1d2'),
        (0.8, '#faaaae'),
        (1, '#ff57bb'),
    ], N=256)

    bin = plt.hexbin(x, y, gridsize=100, cmap=blue_pink)


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
nlay = 20 # number of layers
nrow = 20 # number of rows / grid cells
ncol = 40 # number of columns (parallel to axis of core)

exp_data_dir = os.path.join('.', 'Exp_Time')
#Normalized_BT_time = exp_data_dir + '/Edwards_2ml_2_3mm_at_norm.csv'
Normalized_BT_time = exp_data_dir + '/Navajo_2ml_2_2_3mm_at_norm.csv'
#Normalized_BT_time = exp_data_dir + '/Berea_C1_2ml_2_3mm_at_norm.csv'
#Normalized_BT_time = exp_data_dir + '/Bentheimer_2ml_2_3mm_at_norm.csv'
#Normalized_BT_time = exp_data_dir + '/Indiana_2ml_2_3mm_at_norm.csv'
#Normalized_BT_time = exp_data_dir + '/Estaillades_3ml_2_3mm_at_norm.csv'
#Normalized_BT_time = exp_data_dir + '/Ketton_4ml_2_2_3mm_at_norm.csv'

#Normalized_BT_time = exp_data_dir + '/arrival_norm_diff_td25859_20_20_40.csv'
#Normalized_BT_time = exp_data_dir + '/arrival_norm_diff_td23389_20_20_40.csv'


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
enc_time = tdata_ex[:-1]
enc_time[ibound == 0] = 0

# Get the geometric mean permeability of the core
k_mean = tdata_ex[-1]/np.float32(9.8692e-16)
k_mean = np.sign(k_mean)*np.log(np.abs(k_mean))

enc_time = enc_time.reshape(nlay,nrow,ncol)
enc_time = np.flip(enc_time, 0)


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
enc_prsity = 0.195 # Navajo
#enc_prsity = 0.251 # Bentheimer
#enc_prsity = 0.413 # Edwards
#enc_prsity = 0.250 # Estaillades
#enc_prsity = 0.230 # Ketton

#enc_prsity = 0.220 #25859
#enc_prsity = 0.330 #23389

# Model workspace and new sub-directory
model_dirname = ('t_forward')
model_ws = os.path.join(workdir, model_dirname)

#raw_hk = np.loadtxt('./Kalman_testing_Berea/hk_inv1.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_Edwards/hk_inv1.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_Indiana/hk_inv1.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_Edwards/hk_inv2.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_Ketton/hk_inv4.csv', delimiter=',', dtype= np.float32)
raw_hk = np.loadtxt('./Kalman_testing_Navajo/hk_inv3.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_25859/hk_inv3.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_23389_Hyb/hk_inv2.csv', delimiter=',', dtype= np.float32)
#raw_hk = np.loadtxt('./Kalman_testing_25859_Hyb/hk_inv1.csv', delimiter=',', dtype= np.float32)

raw_hk = np.reshape(raw_hk, (nlay,nrow,ncol))
ENKF_perm = raw_hk/((1000*9.81*100*60)/(8.9E-4))

# Get the breakthrough time data
mf, mt, conc, timearray, km2_mean = mt3d_pulse_injection_sim(enc_prsity, model_dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt)
# calculate quantile arrival time map from MT3D simulation results
at_array, at_array_norm, at_diff_norm = flopy_arrival_map_function(conc, np.array(timearray), grid_size, 0.5, 0.1)

# Get the difference between the experimental arrival time data and simulated arrival time data
# diff_img = at_diff_norm - enc_time

ssim_accu = SSIM()
at_diff_norm = np.array(at_diff_norm)
en = np.exp(enc_time.astype(double))
va = np.exp(at_diff_norm.astype(double))
en = np.exp(en[en != 0])
va = np.exp(va[va != 0])
enc = torch.from_numpy(en)
val = torch.from_numpy(va)
enc = enc.reshape(1,1,20,20,40)
val = val.reshape(1,1,20,20,40)
print(ssim_accu(enc, val))

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
imp = plt.pcolor(x, y, ENKF_perm[round(nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Permeability [$m^{2}$]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Decoded Permeability Layer 10', fontsize=fs, **hfont)
plt.clim(np.min(ENKF_perm[round(nlay/2),:,:]), np.max(ENKF_perm[round(nlay/2),:,:]))

ax2 = fig1.add_subplot(3, 2, 2, aspect='equal')
imp = plt.pcolor(x, y, ENKF_perm[:,round(nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Permeability [$m^{2}$]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs))
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs-3, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs-3, **hfont)
plt.title('Decoded Permeability Row 10', fontsize=fs, **hfont)
plt.clim(np.min(ENKF_perm[:,round(nrow/2),:]), np.max(ENKF_perm[:,round(nrow/2),:]))

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

plt.subplots_adjust( bottom=0.15, top=0.96, wspace=0.05, hspace=0.45)

enc_time = enc_time.flatten()
at_diff_norm = at_diff_norm.flatten()

# Generate the cross-plot
fig2 = plt.figure(figsize=(14, 9))
ax3 = fig2.add_subplot(111)
y = at_diff_norm[enc_time != 0].flatten()
x = enc_time[enc_time != 0].flatten()
bin_plt = c_plot(fig2, x, y)
fig2.subplots_adjust(left=0.25,right=0.79,top=0.9,bottom=0.15)
# L B W H
cbar_ax = fig2.add_axes([0.82, 0.15, 0.03, 0.75])
cbar2 = fig2.colorbar(bin_plt, cax=cbar_ax)
cbar2.set_label('Count in Bin', fontsize=fsc, **cfont)
for t in cbar2.ax.get_yticklabels():
     t.set_fontsize(fsc-23)
for item in (ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(fsc-23)

plt.show()

#perm = raw_hk.flatten()/(1000*9.81*100*60/8.9E-4)
