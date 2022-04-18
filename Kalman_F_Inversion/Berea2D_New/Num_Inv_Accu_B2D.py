import numpy as np
import os
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
# Import custom plotting function
import plotting_functions as pfun
import torch
import torch.nn as nn

from numpy import savetxt
from Accuracy3D import SSIM, R2, RMSE


work_dir = os.path.join('.', 'Kalman_testing')
res_dir = os.path.join('.', 'results')
hk_dir = res_dir + '/dec_hk.csv'
hk = np.loadtxt(hk_dir, delimiter=',', dtype= np.float32)

# Accuracy calculation
rmse_accu = RMSE()
ssim_func = SSIM()
rmse = []
ssim = []

# =============================================================================
# Model geometry
# =============================================================================
# grid_size = [grid size in direction of Lx (layer thickness),
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = np.array([0.23291, 0.23291, 0.25]) # selected units [cm]
# grid_size = np.array([1, 1, 1]) # selected units [cm]
# grid dimensions
nlay = 20 # number of layers / grid cells
nrow = 1 # number of columns
ncol = 40 # number of slices (parallel to axis of core)

# Torch tensor for SSIM calculation
hk_tor = torch.from_numpy(hk)
hk_tor = hk_tor.reshape(1,1,nlay,nrow,ncol)

n_iter = 40
for i in range(n_iter):
    inv_dir = work_dir + '/hk_inv' + str(i+1) + '.csv'
    inv_hk = np.loadtxt(inv_dir, delimiter=',', dtype= np.float32)
    # For log hk inversion only
    inv_hk[inv_hk != 0] = np.exp(inv_hk[inv_hk != 0])

    # RMSE
    rmse_accu = np.sqrt(((hk - inv_hk) ** 2).mean())
    rmse.append(rmse_accu)

    # SSIM
    inv_hk_tor = torch.from_numpy(inv_hk)
    inv_hk_tor = inv_hk_tor.reshape(1,1,nlay,nrow,ncol)
    ssim_accu = ssim_func(hk_tor, inv_hk_tor)
    ssim.append(ssim_accu)

x_axis = np.arange(n_iter)+1

# Showing the SSIM inversion accuracy
plt.figure()
plt.plot(x_axis,ssim)
plt.title('Inversion Accuracy vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('SSIM')

# Showing the RMSE inversion accuracy
plt.figure()
plt.plot(x_axis,rmse)
plt.title('Inversion Accuracy vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('RMSE')

# Reshape the inverted hk field
hk = hk.reshape(nlay,nrow,ncol)
# Initial ensemble
#inv_dir_initial = work_dir + '/hk_initial.csv'
inv_dir_initial = work_dir + '/hk_initial.csv'
inv_hk_initial = np.loadtxt(inv_dir_initial, delimiter=',', dtype= np.float32)
inv_hk_initial[inv_hk_initial != 0] = np.exp(inv_hk_initial[inv_hk_initial != 0])
inv_hk_initial = inv_hk_initial.reshape(nlay,nrow,ncol)

# Chosen esemble
n_chosen = 3
inv_hk_dir = work_dir + '/hk_inv' + str(n_chosen) + '.csv'
inv_hk_chosen = np.loadtxt(inv_hk_dir, delimiter=',', dtype= np.float32)
inv_hk_chosen[inv_hk_chosen != 0] = np.exp(inv_hk_chosen[inv_hk_chosen != 0])
inv_hk_chosen = inv_hk_chosen.reshape(nlay,nrow,ncol)

# Qualitative accessment
# Initial ensemble
pfun.plot_2d(inv_hk_initial[:,0,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')
plt.title('Inital Random hk')
plt.clim(hk[:,0,:].min(), hk[:,0,:].max())

# Post-filter ensemble
# hk_mean = np.reshape(np.mean(np.exp(Z), axis=1), (nlay, nrow, ncol))
pfun.plot_2d(inv_hk_chosen[:,0,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')
plt.title('Iteration #' + str(n_chosen) +' w/ RMSE ' + str(rmse[n_chosen-1]) +' and w/ SSIM ' + str(ssim[n_chosen-1].item()))
# plt.clim(hk[:,0,:].min(), hk[:,0,:].max())

# True hk
pfun.plot_2d(hk[:,0,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')
plt.title('True hk')
plt.clim(hk[:,0,:].min(), hk[:,0,:].max())

plt.show()
