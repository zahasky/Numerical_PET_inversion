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

from copy import deepcopy
from scipy import integrate
from random import randrange
from skopt.sampler import Lhs


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
exe_name_mf = 'C:\\Users\\zhuang296\\Documents\\NSKF_Ind\\mf2005'
exe_name_mt = 'C:\\Users\\zhuang296\\Documents\\NSKF_Ind\\mt3dms'

# # names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = '/Users/zhuang296/Desktop/mac/mf2005'
# exe_name_mt = '/Users/zhuang296/Desktop/mac/mt3dms'

# # directory to save data
# # directory_name = 'examples'
# # workdir = os.path.join('.', '3D_fields', directory_name)
# directory_name = 'Kalman_testing'
# workdir = os.path.join('D:\\Modflow_models', directory_name)

# directory to save data
directory_name = '/Users/zhuang296/Desktop/Kalman_testing_Indiana'
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


# =============================================================================
# Core shape and input/output control
# =============================================================================
# Import core shape mask for permeability boundary processing
# Hard coded mask for 20x20 grid cross-section, this creates the a cylindrical shape core
mp = np.array([7, 5, 3, 2, 2, 1, 1])
mask_corner = np.ones((10,10))
for i in range(len(mp)):
    mask_corner[i, 0:mp[i]] = 0

mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
core_mask = np.concatenate((mask_top, np.flipud(mask_top)))
ibound = np.repeat(core_mask[:, :, np.newaxis], dncol, axis=2)

# Import data
new_path = './Exp_Time'
if new_path not in sys.path:
    sys.path.append(new_path)

arrival_data = np.loadtxt(new_path + '/Indiana_2ml_2_3mm_at_norm.csv', delimiter=',')

# remove last value (average perm)
km2_obs = arrival_data[-1]
arrival_data = arrival_data[0:-1]

# Convert the Darcy mean permeabiltiy (in m^2) to hydraulic conductivity in cm/min
hk_obs = km2_obs*(1000*9.81*100*60/8.9E-4)
# 3D case
at_obs = arrival_data.reshape(dnlay, dnrow, dncol)
at_obs = np.flip(at_obs, 0)
arrival_data = at_obs*ibound

# Load the CNN predicted hydraulic conductivity
hk_cnn = np.loadtxt('./results/dec_hk_Ind.csv', delimiter=',')

zero_idx_hk = hk_cnn == 0
nonzero_idx_hk = hk_cnn != 0
# Account for the additional mean hk
zero_idx_tot = np.append(deepcopy(nonzero_idx_hk), False)
nonzero_idx_tot = np.append(deepcopy(nonzero_idx_hk), True)


# =============================================================================
# General model parameters
# =============================================================================
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

# porosity
prsity = 0.167 # Indiana

# dispersivity (cm)
al = 0.2
# injection rate (ml/min)
injection_rate = 4
# scale factor for 2D
sf = (3.1415*2.5**2*prsity)/(20*grid_size[1]*grid_size[0])
# advection velocity (cm/min)
v = injection_rate*sf/(3.1415*2.5**2*prsity)


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
Ne = 400
# number of itereations
num_iterations = 30

# =============================================================================
# Starting the Samples Generation
# =============================================================================
# Parameters for the ENKF model
Y = np.append(arrival_data.flatten(), np.log(hk_obs))
num_obs = Y[nonzero_idx_tot].size
num_parameters = num_obs

# Total parameters including the CNN predictions
total_parameters = dncol*dnrow*dnlay

# Parameter value preallocation (parameters are what we are estimating, ie k)
Z = np.zeros((num_parameters, Ne))
# State variable preallocation
Yj = np.zeros((num_obs, Ne))

# =============================================================================
# Ensemble permeability field generation
# =============================================================================
lhs = Lhs(lhs_type="classic", criterion=None)
Ps = lhs.generate([(-6., -1.), (1.0, 4.0), (1.0, 30.0), (1.0, 30.0), (0.0, 1.57), (0.0, 1.57), (0.0, 1.57)], Ne + 10)

# example of how to efficiently save data to text file
np.savetxt(workdir + '/Indiana_ensemble_3D_parameter_space', Ps , delimiter=',', fmt='%.3e')

# kmD_exp_mean_obs = np.log10(hk_mean_obs/(9.869233E-13*9.81*100*60/8.9E-4))

kmD_cnn = copy.deepcopy(hk_cnn)
kmD_cnn[nonzero_idx_hk] = hk_cnn[nonzero_idx_hk]/(9.869233E-13*9.81*100*60/8.9E-4)
kmD_cnn = np.reshape(kmD_cnn, (dnlay,dnrow,dncol))

for i in range(Ne):
    # initialization (random)
    p = Ps[i]
    # generate field with gstools toolbox (install info here: https://github.com/GeoStat-Framework/GSTools)
    gs_model = gs.Exponential(dim=3, var=10**p[0], len_scale=[p[1], p[2], p[3]], angles=[p[4], p[5], p[6]])
    srf = gs.SRF(gs_model)
    # permeability field in millidarcy
    field = 10**(srf.structured([np.arange(dnlay), np.arange(dnrow), np.arange(dncol)]))
    sample = kmD_cnn * np.array(field)

    # check for negative values
    sample[sample<0] = 1

    # Convert from mD to hydraulic conductivity in cm/min
    z_initial = sample*(9.869233E-13*9.81*100*60/8.9E-4)
    z_initial = z_initial.flatten()
    z_initial = np.reshape(z_initial, (dnlay, dnrow, dncol))

    ndummy_in = randrange(3)
    mf, mt, conc, times, hk_mean_obs = model.mt3d_conservative('tracer', z_initial, prsity, grid_size, al,
                                                               v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, model_ws)

    at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

    zf = np.append(z_initial.flatten(), hk_mean_obs)
    Z[:,i] = np.log(zf[nonzero_idx_tot])
    yf = np.append(at_ensem.flatten(), np.log(hk_mean_obs))
    Yj[:,i] = yf[nonzero_idx_tot]

# Save the initial ensemble mean
Z_m = np.mean(Z, axis=1)
np.savetxt(workdir + '/z_initial_3D.csv', Z_m, delimiter=',')


# =============================================================================
# ENKF Inversion
# =============================================================================
# record start
start = time.time()

# Error covariance matrix (based on measurement error of PET scanner)
# scanner_error = 0.00001
scanner_error = 0.001
R = np.identity(Z_m.size)*scanner_error
invR = np.linalg.inv(R)

def Kalman_generator(Z, Yj, Y, R, invR, hk_update):
    # determine parameters from input
    num_parameters, Ne = Z.shape
    num_obs = Y[nonzero_idx_tot].size

    # calculate ensemble means for upcoming covariance calculations
    Zbar = np.mean(Z, axis=1)
    Ybar = np.mean(Yj, axis=1)

    # Calculate covariance and parameter cross-covariance
    Pzy = np.zeros((num_parameters, num_obs))
    Pyy = np.zeros((num_obs, num_obs))

    # Calculate parameter covariance
    for i in range(Ne):
        sig_z = np.array([Z[:,i]-Zbar])
        sig_y = np.array([Yj[:,i]-Ybar])
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
        ri = np.array([Y[nonzero_idx_tot]-Yj[:,i]])
        Li[i] = np.dot(np.dot(ri, invR), ri.T)
        Lif = Li[i]*2

        a = 0.99
        count = 0

        # Won't run when count is more than 11
        while Lif > Li[i] and count < 5:
            Z[:,i] = Z[:,i] + np.squeeze(a*np.dot(G, ri.T))

            # Get the updated hydraulic conductivity
            hk_update[nonzero_idx_hk] = np.exp(Z[:-1, i])
            hk_update[zero_idx_hk]= 0
            hk_update = np.reshape(hk_update, (dnlay, dnrow, dncol))

            ndummy_in = randrange(3)
            # Rerun model with new parameters
            mf, mt, conc, times, hk_mean_obs = model.mt3d_conservative('tracer', hk_update, prsity, grid_size, al,
                                                                       v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, model_ws)

            at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

            # Forcasted state
            yf = np.append(at_ensem.flatten(), np.log(hk_mean_obs))
            # Updated parameters
            hk_update = hk_update.flatten()

            # calculate residuals
            ri = np.array([Y[nonzero_idx_tot]-yf[nonzero_idx_tot]])
            Lif = np.dot(np.dot(ri, invR), ri.T)
            Yj[:, i] = copy.deepcopy(yf[nonzero_idx_tot])
            count += 1
            # a = a/2

    return Z, Yj, Li


en1_save = np.zeros((num_parameters, num_iterations))
Res_save = np.zeros((Ne, num_iterations))

for iteration in range(0, num_iterations):
    print('Iteration #: ' +str(iteration+1))

    hk_update = np.zeros(total_parameters)
    hk_inverted = np.zeros(total_parameters)

    Z, Yj, Li = Kalman_generator(Z, Yj, Y, R, invR, hk_update)

    Res_save[:, iteration] = Li
    en1_save[:, iteration] = np.mean(Z, axis=1)

    hk_inverted[nonzero_idx_hk] = np.exp(np.mean(Z[:-1], axis=1))

    # Save the inverted field at every iteration
    np.savetxt(workdir + '/hk_inv' + str(iteration+1) + '.csv', hk_inverted, delimiter=',')

# Print final run time
end_time = time.time() # end timer
# print('Model run time: ', end - start) # show run time
print(f"Total run time: {end_time - start:0.4f} seconds")
