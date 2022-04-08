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
import NormalScoreTransformation

from copy import deepcopy
from matplotlib import rc
from scipy import integrate
from NormalScoreTransformation import Norm_Sc

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
# exe_name_mf = 'C:\\Users\\zhuang296\\Documents\\NSKF\\mf2005'
# exe_name_mt = 'C:\\Users\\zhuang296\\Documents\\NSKF\\mt3dms'

# # names of executable with path IF NOT IN CURRENT DIRECTORY
exe_name_mf = '/Users/zhuang296/Desktop/mac/mf2005'
exe_name_mt = '/Users/zhuang296/Desktop/mac/mt3dms'

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


# =============================================================================
# Model geometry
# =============================================================================
# grid_size = [grid size in direction of Lx (layer thickness),
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
# grid_size = np.array([0.23291, 0.23291, 0.25]) # selected units [cm]
grid_size = np.array([1, 1, 1]) # selected units [cm]
# grid dimensions
nlay = 5 # number of layers / grid cells
nrow = 5 # number of columns
ncol = 10 # number of slices (parallel to axis of core)


# =============================================================================
# Permeability field generation
# =============================================================================
# p = [log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z]
# it is good to play around with this to see what this looks like
p = [-1.0, 3.0, 1.0, 10.0, 10.0, 0.0, 0.0, 0.0]

# generate field with gstools toolbox (install info here: https://github.com/GeoStat-Framework/GSTools)
gs_model = gs.Exponential(dim=3, var=10**p[0], len_scale=[p[2], p[3], p[4]], angles=[p[5], p[6], p[7]])
srf = gs.SRF(gs_model, seed=20170518)

# permeability field in millidarcy
field = np.array(10**(srf.structured([np.arange(nlay), np.arange(nrow), np.arange(ncol)]) + p[1]))

# check for negative values
field[field<0] = 1
print('GS model mean permeability: '+ str(10**p[1]) + ' mD')

# convert from mD to m^2
field_km2 = field*(9.869233E-13/1000)
# save permeability field
save_data = np.append(field_km2.flatten('C'), [nlay, nrow, ncol])
np.savetxt(workdir + '/perm_field_1_m2.csv', save_data , delimiter=',', fmt='%.3e')
# Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
het_hk = field_km2*(1000*9.81*100*60/8.9E-4)

# Import core shape mask for permeability boundary processing
# Hard coded mask for 5x5 grid cross-section, this creates the a cylindrical shape core
mask_layer = np.ones((5,5))
#mask_layer[0,:] = [0,0,1,0,0]
#mask_layer[4,:] = [0,0,1,0,0]
ibound = np.repeat(mask_layer[:, :, np.newaxis], ncol, axis=2)
het_hk = het_hk*ibound


# =============================================================================
# Core shape and input/output control
# =============================================================================
# Hard coded mask for 20x20 grid cross-section, this creates the a cylindrical shape core. Ones are voxels with core and
# zeros are the area outside the core
# core_mask = generate_standard_20x20mask(het_hk)
# core_mask = np.ones([nlay, nrow, ncol])

# Output control for MT3dms, this needs to align with the imaging information
timestep_length = 2 # minutes
scan_time = 120.0
pulse_duration = 1.0
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
prsity = 0.35
# dispersivity (cm)
al = 0.2
# injection rate (ml/min)
injection_rate = 2
# advection velocity (cm/min)
v = injection_rate/(3.1415*2.5**2*prsity)


# =============================================================================
# Heterogeneous permeability numerical models
# =============================================================================
mf, mt, conc_true, btc_solute_heter, times_true, hk_mean_obs = model.mt3d_conservative('tracer', het_hk, prsity, al,
                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

at_obs = model.flopy_arrival_map_function(conc_true, times_true, grid_size, 0.5)
# Save the true hydraulic conductivity/permeability
np.savetxt(workdir + '/hk_field.csv', het_hk.flatten(), delimiter=',')


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
Ne = 3
# number of itereations
num_iterations = 2
# observed data
Y = at_obs.flatten()
het_hk = het_hk.flatten()


# =============================================================================
# Starting the Normal Scored Samples Generation
# =============================================================================
# Number of bins for the normal score transformation
nb = 250
# Storing the x-axis of the original hk distribution
zbin = np.zeros((nb,Ne))
# Storing the x-axis of the normal scored hk distribution
zscore = np.zeros((nb,Ne))

# # Normal transform the true Y
# ytruehist, ytruebins = np.histogram(Y, bins=nb)
# NS = Norm_Sc()
# # n_bins (x-axis of the new distribution)
# Y, ytrue_bins = NS.norm_score_trans(ytruebins[:-1], ytruehist, Y)

num_obs = Y.size
num_parameters = het_hk.size

# Parameter value preallocation (parameters are what we are estimating, ie k)
Z = np.zeros((num_parameters, Ne))
# State variable preallocation
Yj = np.zeros((num_obs, Ne))

kmD_exp_mean_obs = np.log10(hk_mean_obs/(9.869233E-13*9.81*100*60/8.9E-4))

for i in range(Ne):
    # initialization (random)
    # hk_initial = np.random.uniform(hk_mean_obs*0.1, hk_mean_obs*10, size=(nlay, nrow, ncol))
    # generate field with gstools toolbox (install info here: https://github.com/GeoStat-Framework/GSTools)
    gs_model = gs.Exponential(dim=3, var=10**p[0], len_scale=[p[2], p[3], p[4]], angles=[p[5], p[6], p[7]])
    srf = gs.SRF(gs_model)
    # permeability field in millidarcy
    field = 10**(srf.structured([np.arange(nlay), np.arange(nrow), np.arange(ncol)]) + kmD_exp_mean_obs)
    # check for negative values
    field[field<0] = 1
    # convert from mD to hydraulic conductivity in cm/min
    hk_initial = field*(9.869233E-13*9.81*100*60/8.9E-4)
    hk_initial = hk_initial*ibound

    mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_initial, prsity, al,
                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

    at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

    zf = hk_initial.flatten()
    Z[zf != 0, i] = np.log(zf[zf != 0])
    yf = at_ensem.flatten()
    Yj[:,i] = yf

# Save the initial ensemble mean
np.savetxt(workdir + '/hk_initial.csv', np.mean(Z, axis=1), delimiter=',')


# =============================================================================
# Inversion
# =============================================================================
# Count the number of times the value error is detected
error_count = 0

# Parameters for the reverse normal transformation
hk_mean = (10**p[1])*(9.869233E-13/1000)*(1000*9.81*100*60/8.9E-4)
hk_std = np.sqrt((10**p[0])*(9.869233E-13/1000)*(1000*9.81*100*60/8.9E-4))
min_da = np.log(hk_mean - 25*hk_std)
max_da = np.log(hk_mean + 55*hk_std)
power = 1

# Indexing the zeros in the original ensembles
non_zero_idx = het_hk != 0
zero_idx = het_hk == 0

# Error covariance matrix (based on measurement error of PET scanner)
scanner_error = 0.00001
R = np.identity(het_hk[non_zero_idx].size)*scanner_error
invR = np.linalg.inv(R)

def Kalman_generator(Z, Yj, Y, R, invR, hk_update, error_count, min_da, max_da, power):
    try:
        # Z from the previous iteration
        #Z_prev = deepcopy(Z)

        # determine parameters from input
        num_parameters, Ne = Z[non_zero_idx,:].shape
        num_obs = Y[non_zero_idx].size

        # Normal score transformation
        for i in range(Ne):
            # For Z
            zhist, zbins = np.histogram(deepcopy(Z[non_zero_idx,i]),bins=nb)
            NS = Norm_Sc()
            # zscore (x-axis of the new distribution)
            Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(deepcopy(zbins[:-1]), deepcopy(zhist), deepcopy(Z[non_zero_idx,i]))
            # The original bins
            zbin[:,i] = deepcopy(zbins[:-1])

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

            a = 1
            count = 0

            #Won't run when count is more than 11
            while Lif > Li[i] and count < 11:
                Z[non_zero_idx,i] = Z[non_zero_idx,i] + np.squeeze(a*np.dot(G, ri.T))

                # Reverse normal score transformation of z
                nor, b = np.histogram(deepcopy(Z[non_zero_idx,i]),bins=nb)
                NS = Norm_Sc()
                Z[non_zero_idx,i] = NS.reverse_norm_score_trans(deepcopy(b[:-1]), deepcopy(nor), deepcopy(zscore[:,i]), deepcopy(zbin[:,i]), deepcopy(Z[non_zero_idx,i]), min_da, max_da, power)

                #hk_update[non_zero_idx] = np.exp(deepcopy(Z_rev[non_zero_idx, i]))
                hk_update[non_zero_idx] = np.exp(deepcopy(Z[non_zero_idx, i]))
                hk_update[zero_idx]= 0
                hk_update = np.reshape(hk_update, (nlay, nrow, ncol))

                # Rerun model with new parameters
                mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_update, prsity, al,
                                                                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

                at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

                # Forcasted state
                yf = at_ensem.flatten()
                # Updated parameters
                hk_update = hk_update.flatten()

                # calculate residuals
                ri = np.array([Y[non_zero_idx]-yf[non_zero_idx]])
                Lif = np.dot(np.dot(ri, invR), ri.T)
                Yj[non_zero_idx, i] = deepcopy(yf[non_zero_idx])
                count += 1
                #a = a/2

                # Triggered when the Lif is not reduced to smaller than Li[i]
                if count == 10:
                    print('Min dampening factor reached on realization: ' + str(i))

                    # If no acceptance could be reached, ensemble member i is resampled from the full ensemble of last iteration
                    #rand1 = np.random.choice([x for x in range(Ne-1)])
                    #rand2 = np.random.choice([x for x in range(Ne-1)])
                    #Z[non_zero_idx, i] = (deepcopy(Z_prev[non_zero_idx, rand1]) + deepcopy(Z_prev[non_zero_idx, rand2])) / 2

                    # Normal score transformation of z for the next update
                    zhist, zbins = np.histogram(deepcopy(Z[non_zero_idx,i]),bins=nb)
                    NS = Norm_Sc()
                    # zscore (x-axis of the new distribution)
                    Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(deepcopy(zbins[:-1]), deepcopy(zhist), deepcopy(Z[non_zero_idx,i]))
                    # The original bins
                    zbin[:,i] = deepcopy(zbins[:-1])
                else:
                    # Normal score transformation of z for the next update
                    zhist, zbins = np.histogram(deepcopy(Z[non_zero_idx,i]),bins=nb)
                    NS = Norm_Sc()
                    # zscore (x-axis of the new distribution)
                    Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(deepcopy(zbins[:-1]), deepcopy(zhist), deepcopy(Z[non_zero_idx,i]))
                    # The original bins
                    zbin[:,i] = deepcopy(zbins[:-1])

            # Reverse normal score transformation of z
            nor, b = np.histogram(deepcopy(Z[non_zero_idx,i]),bins=nb)
            NS = Norm_Sc()
            Z[non_zero_idx,i] = NS.reverse_norm_score_trans(deepcopy(b[:-1]), deepcopy(nor), deepcopy(zscore[:,i]), deepcopy(zbin[:,i]), deepcopy(Z[non_zero_idx,i]), min_da, max_da, power)

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
    Z, Yj, Li, error_count = Kalman_generator(Z, Yj, Y, R, invR, hk_update, error_count, min_da, max_da, power)

    Res_save[:, iteration] = Li
    en1_save[:, iteration] = np.mean(Z, axis=1)

    # Save the inverted field at every iteration
    np.savetxt(workdir + '/hk_inv' + str(iteration+1) + '.csv', np.mean(Z, axis=1), delimiter=',')

# Print final run time
end_time = time.time() # end timer
# print('Model run time: ', end - start) # show run time
print(f"Total run time: {end_time - start:0.4f} seconds")
