# -*- coding: utf-8 -*-
"""
This code is a simple small 3D test model that was used to generate the
5x5x10_small_3d_model_test results
Created on Tue Jun  8 10:21:17 2021
@author: zahas
"""
# Only packages called in this script need to be imported
import os
import sys
import time
import copy
import shutil
import numpy as np
# Packages for random field generation
import gstools as gs
# Import custom functions to run flopy stuff and analytical solutions
import model_functions as model
import matplotlib.pyplot as plt

from copy import deepcopy
from matplotlib import rc
from scipy import integrate
from random import randrange
from skopt.sampler import Lhs

rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 16



## LAPTOP PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
# exe_name_mt = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# # DELL 419 PATHS
# # names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
# exe_name_mt = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# # Terra Machine Path
# # names of executable with path IF NOT IN CURRENT DIRECTORY
exe_name_mf = 'C:\\Users\\zhuang296\\Documents\\NSKF\\mf2005'
exe_name_mt = 'C:\\Users\\zhuang296\\Documents\\NSKF\\mt3dms'

# # names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = '/Users/zhuang296/Desktop/mac/mf2005'
# exe_name_mt = '/Users/zhuang296/Desktop/mac/mt3dms'

# # directory to save data
# # directory_name = 'examples'
# # workdir = os.path.join('.', '3D_fields', directory_name)
# directory_name = 'Kalman_testing'
# workdir = os.path.join('D:\\Modflow_models', directory_name)

# directory to save data
directory_name = '/Users/zhuang296/Desktop/Kalman_testing'
workdir = os.path.join('.', directory_name)

# if the path exists then we will move on, if not then create a folder with the 'directory_name'
if os.path.isdir(workdir) is False:
    os.mkdir(workdir)
    print("Directory '% s' created" % workdir)

# Model workspace and new sub-directory
model_dirname = ('tracer')
model_ws = os.path.join(workdir, model_dirname)


# =============================================================================
# Model geometry
# =============================================================================
# grid_size = [grid size in direction of Lx (layer thickness),
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = np.array([0.23291, 0.23291, 0.25]) # selected units [cm]
# grid_size = np.array([1, 1, 1]) # selected units [cm]

# data load dimensions
dnlay = 20
dnrow = 20
dncol = 40
# grid dimensions
nlay = 20 # number of layers / grid cells
nrow = 1 # number of columns
ncol = 40 # number of slices (parallel to axis of core)


# =============================================================================
# Core shape and input/output control
# =============================================================================
# Hard coded mask for 20x20 grid cross-section, this creates the a cylindrical shape core. Ones are voxels with core and
# zeros are the area outside the core
# core_mask = generate_standard_20x20mask(het_hk)
# core_mask = np.ones([nlay, nrow, ncol])

# # Import core shape mask for permeability boundary processing
# # Hard coded mask for 5x5 grid cross-section, this creates the a cylindrical shape core
# mask_layer = np.ones((5,5))
# mask_layer[0,:] = [0,0,1,0,0]
# mask_layer[4,:] = [0,0,1,0,0]
# ibound = np.repeat(mask_layer[:, :, np.newaxis], ncol, axis=2)
# het_hk = het_hk*ibound

# Output control for MT3dms, this needs to align with the imaging information
timestep_length = 1 # minutes
scan_time = 120.0
pulse_duration = 2.0
equilibration_time = 1.0

# timprs (list of float): The total elapsed time at which the simulation
#     results are saved. The number of entries in timprs must equal nprs.
timprs = np.arange(timestep_length/2, scan_time + equilibration_time, timestep_length) # minutes
# period length in selected units (min)
# the first period is fluid pressure equilibration, the second period is tracer/bacteria
# injection, the third period is tracer displacement
perlen_mt = [equilibration_time, pulse_duration, scan_time+equilibration_time]
# Numerical method flag
mixelm = -1

# =============================================================================
# General model parameters
# =============================================================================
# porosity
prsity = 0.21
# dispersivity (cm)
al = 0.2
# injection rate (ml/min)
injection_rate = 2
# scale factor for 2D
sf = (3.1415*2.5**2*prsity)/(20*grid_size[1]*grid_size[0])
# advection velocity (cm/min)
v = injection_rate*sf/(3.1415*2.5**2*prsity)


# =============================================================================
# Heterogeneous permeability numerical models
# =============================================================================
# Import data
new_path = './Exp_Time'
if new_path not in sys.path:
    sys.path.append(new_path)

arrival_data = np.loadtxt(new_path + '/Berea_C1_2ml_2_3mm_at_norm.csv', delimiter=',')
# remove last value (average perm)
km2_obs = arrival_data[-1]
arrival_data = arrival_data[0:-1]
# Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
hk_mean_obs = km2_obs*(1000*9.81*100*60/8.9E-4)
# 3D case
at_obs = arrival_data.reshape(dnlay, dnrow, dncol)
at_obs = np.flip(at_obs, 0)
# 2D case
at_obs_slice = at_obs[:,9,:]
Y = at_obs_slice.flatten()


# =============================================================================
# Kalman Filter implementation
# This is largely implemented based on the description of the Standard EnKF
# Reference here: 10.1016/j.advwatres.2011.04.014 and
# Reference two: https://doi.org/10.1016/j.jhydrol.2018.07.073
#
# Quasi-Linear Kalman Ensemble Generator
# Reference: 10.1016/j.jhydrol.2016.11.064
# =============================================================================
# ensemble number (typically several hundred)
Ne = 800
# number of itereations
num_iterations = 30


# =============================================================================
# Starting the Normal Scored Samples Generation
# =============================================================================
# Number of bins for the normal score transformation
nb = 500
# Storing the x-axis of the original hk distribution
zbin = np.zeros((nb,Ne))
# Storing the x-axis of the normal scored hk distribution
zscore = np.zeros((nb,Ne))

num_obs = Y.size
num_parameters = nlay*nrow*ncol

# Parameter value preallocation (parameters are what we are estimating, ie k)
Z = np.zeros((num_parameters, Ne))
# State variable preallocation
Yj = np.zeros((num_obs, Ne))

# =============================================================================
# Ensemble permeability field generation
# =============================================================================
lhs = Lhs(lhs_type="classic", criterion=None)
Ps = lhs.generate([(-3., -2.0), (2.0, 5.0), (50.0, 150.0), (50.0, 150.0)], Ne + 10)

# example of how to efficiently save data to text file
np.savetxt('Berea_ensemble_2D_parameter_space', Ps , delimiter=',', fmt='%.3e')

kmD_exp_mean_obs = np.log10(hk_mean_obs/(9.869233E-13*9.81*100*60/8.9E-4))

for i in range(Ne):
    # initialization (random)
    # hk_initial = np.random.uniform(hk_mean_obs*0.1, hk_mean_obs*10, size=(nlay, nrow, ncol))
    p = Ps[i]
    # generate field with gstools toolbox (install info here: https://github.com/GeoStat-Framework/GSTools)
    gs_model = gs.Exponential(dim=3, var=10**p[0], len_scale=[p[1], p[2], p[3]], angles=[0, 0, 0])
    srf = gs.SRF(gs_model)
    # permeability field in millidarcy
    field = 10**(srf.structured([np.arange(nlay), np.arange(nrow), np.arange(ncol)]) + kmD_exp_mean_obs)
    # check for negative values
    field[field<0] = 1
    # convert from mD to hydraulic conductivity in cm/min
    hk_initial = field*(9.869233E-13*9.81*100*60/8.9E-4)
    # hk_initial = hk_initial*ibound

    ndummy_in = randrange(3)
    mf, mt, conc, times, hk_mean_obs = model.mt3d_conservative('tracer', hk_initial, prsity, grid_size, al,
                             v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, model_ws)

    at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

    zf = hk_initial.flatten()
    Z[zf != 0,i] = np.log(zf[zf != 0])
    yf = at_ensem.flatten()
    Yj[:,i] = yf

# Save the initial ensemble mean
Z_m = np.mean(Z, axis=1)
np.savetxt(workdir + '/hk_initial.csv', Z_m, delimiter=',')


# =============================================================================
# Inversion
# =============================================================================
# Count the number of times the value error is detected
error_count = 0

# Indexing the zeros in the original ensembles
non_zero_idx = Z_m != 0
zero_idx = Z_m == 0

# Error covariance matrix (based on measurement error of PET scanner)
scanner_error = 0.00001
R = np.identity(Z_m[non_zero_idx].size)*scanner_error
invR = np.linalg.inv(R)

def Kalman_generator(Z, Yj, Y, R, invR, hk_update, error_count):
    try:
        # Z from the previous iteration
        #Z_prev = deepcopy(Z)

        # determine parameters from input
        num_parameters, Ne = Z[non_zero_idx,:].shape
        num_obs = Y[non_zero_idx].size

        # calculate ensemble means for upcoming covariance calculations
        Zbar = np.mean(Z[non_zero_idx,:], axis=1)
        Ybar = np.mean(Yj[non_zero_idx,:], axis=1)

        # Calculate covariance and parameter cross-covariance
        Pzy = np.zeros((num_parameters, num_obs))
        Pyy = np.zeros((num_obs, num_obs))

        # Calculate parameter covariance
        for i in range(Ne):
            sig_z = np.array([Z[non_zero_idx,i]-Zbar])
            sig_y = np.array([Yj[non_zero_idx,i]-Ybar])
            Pyy += np.dot(sig_y.T,sig_y)
            Pzy += np.dot(sig_z.T,sig_y)

        Pzy = Pzy/Ne
        Pyy = Pyy/Ne

        # Kalman gain matrix G
        G = np.dot(Pzy, np.linalg.inv(Pyy+R))

        Li = np.zeros((Ne))

        # Updated parameters
        for i in range(Ne):
            # calculate residuals
            ri = np.array([Y[non_zero_idx]-Yj[non_zero_idx,i]])
            Li[i] = np.dot(np.dot(ri, invR), ri.T)
            Lif = Li[i]*2

            a = 0.99
            count = 0

            # Won't run when count is more than 11
            while Lif > Li[i] and count < 5:
                Z[non_zero_idx,i] = Z[non_zero_idx,i] + np.squeeze(a*np.dot(G, ri.T))

                # Get the updated hydraulic conductivity
                hk_update[non_zero_idx] = np.exp(Z[non_zero_idx, i])
                hk_update[zero_idx]= 0
                hk_update = np.reshape(hk_update, (nlay, nrow, ncol))

                ndummy_in = randrange(3)
                # Rerun model with new parameters
                mf, mt, conc, times, hk_mean_obs = model.mt3d_conservative('tracer', hk_update, prsity, grid_size, al,
                             v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, model_ws)

                at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

                # Forcasted state
                yf = at_ensem.flatten()
                # Updated parameters
                hk_update = hk_update.flatten()

                # calculate residuals
                ri = np.array([Y[non_zero_idx]-yf[non_zero_idx]])
                Lif = np.dot(np.dot(ri, invR), ri.T)
                Yj[non_zero_idx, i] = yf[non_zero_idx]
                count += 1
                a = a/2

                # Triggered when the Lif is not reduced to smaller than Li[i]
                if count == 4:
                    print('Min dampening factor reached on realization: ' + str(i+1))

    except Exception as e:
        error_count += 1
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('Error message: ' + str(e) + '; line: ' + str(exc_tb.tb_lineno))
        np.savetxt(workdir + '/hk_problem' + str(error_count) + '.csv', Z[:,i], delimiter=',')


    return Z, Yj, Li, error_count

#hk_initial_mean = np.reshape(np.mean(np.exp(Z), axis=1), (nlay, nrow, ncol))

en1_save = np.zeros((num_parameters, num_iterations))
Res_save = np.zeros((Ne, num_iterations))
# record start
start = time.time()

for iteration in range(0, num_iterations):
    print('Iteration #: ' +str(iteration+1))
    hk_update = np.zeros(num_obs)
    # iteration
    Z, Yj, Li, error_count = Kalman_generator(Z, Yj, Y, R, invR, hk_update, error_count)

    Res_save[:, iteration] = Li
    en1_save[:, iteration] = np.mean(Z, axis=1)

    # Save the inverted field at every iteration
    np.savetxt(workdir + '/hk_inv' + str(iteration+1) + '.csv', np.mean(Z, axis=1), delimiter=',')

# Print final run time
end_time = time.time() # end timer
# print('Model run time: ', end - start) # show run time
print(f"Total run time: {end_time - start:0.4f} seconds")
