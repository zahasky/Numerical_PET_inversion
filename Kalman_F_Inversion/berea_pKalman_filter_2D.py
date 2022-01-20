# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:21:17 2021

@author: zahas
"""
# Only packages called in this script need to be imported
import os
import shutil
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import time
import copy

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 16

# Packages for random field generation
import gstools as gs
from skopt.sampler import Lhs

# Import custom functions to run flopy stuff and analytical solutions
import model_functions as model

import sys
new_path = 'C:\\Users\\czahasky\\Documents\\kalman_filter_berea_2d'
if new_path not in sys.path:
    sys.path.append(new_path)
    
# Import custom plotting function
import plotting_functions as pfun

## LAPTOP PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
# exe_name_mt = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# DELL 419 PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
#exe_name_mf = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
#exe_name_mt = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# Terra PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
exe_name_mf = 'C:\\Users\\czahasky\\Documents\\kalman_filter_berea_2d\\mf2005'
exe_name_mt = 'C:\\Users\\czahasky\\Documents\\kalman_filter_berea_2d\\mt3dms'

# directory to save data
# directory_name = 'examples'
# workdir = os.path.join('.', '3D_fields', directory_name)
directory_name = 'Berea_kalman'
workdir = os.path.join('C:\\Users\\czahasky\\Documents', directory_name)
# if the path exists then we will move on, if not then create a folder with the 'directory_name'
if os.path.isdir(workdir) is False:
    os.mkdir(workdir) 
    print("Directory '% s' created" % workdir)
    
# =============================================================================
# Load Experimental Data
# =============================================================================  
# grid_size = [grid size in direction of Lx (layer thickness), 
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = np.array([0.2329, 0.2329, 0.2388]) # selected units [cm]
# data load dimensions
dnlay = 20
dnrow = 20
dncol = 40
# model dimensions (different because we are just going to use a 2D slice of 3D data)
nlay = 20
nrow = 1
ncol = 40
# Import data
arrival_data = np.loadtxt(new_path + '\\Berea_C1_2ml_2_3mm_at_norm.csv', delimiter=',')
# remove last value (average perm)
km2_obs = arrival_data[-1]
arrival_data = arrival_data[0:-1]
# Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
hk_mean_obs = km2_obs*(1000*9.81*100*60/8.9E-4)
at_obs = arrival_data.reshape(dnlay, dnrow, dncol)
at_obs = np.flip(at_obs, 0)
at_obs_slice = at_obs[:,9,:]

# change some values to zero to see how that messes things up
at_obs_slice[0:2,:]=0
at_obs_slice[-1,:]=0

# Hard coded mask for 20x20 grid cross-section, this creates the a cylindrical shape core. Ones are voxels with core and 
# zeros are the area outside the core
#core_mask = model.generate_standard_20x20mask(at_obs)
#at_obs = model.apply_core_mask(at_obs, core_mask)

# pfun.plot_2d(at_obs[:,:,0], grid_size[0], grid_size[1], 'arrival time', cmap='bwr')
pfun.plot_2d(at_obs_slice, grid_size[0], grid_size[2], 'Measured arrival time', cmap='bwr')
plt.clim([-0.2, 0.2])

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
# Kalman Filter implementation
# This is largely implemented based on the description of the Standard EnKF
# Reference here: 10.1016/j.advwatres.2011.04.014 and
# Reference two: https://doi.org/10.1016/j.jhydrol.2018.07.073

# Quasi-Linear Kalman Ensemble Generator
# Reference: 10.1016/j.jhydrol.2016.11.064
# =============================================================================
# ensemble number (typically several hundred)
Ne = 800
# number of itereations
num_iterations = 12
# observed data
Y = np.append(at_obs_slice.flatten(), hk_mean_obs)

num_obs = Y.size
num_parameters = nlay*nrow*ncol
# parameter value preallocation (parameters are what we are estimating, ie k)
Z = np.zeros((num_parameters, Ne))
# state variable preallocation
Yj = np.zeros((num_obs, Ne))

# error covariance matrix (based on measurement error of PET scanner)
scanner_error = 0.0001
# scanner_error = 0.006 # calculated from bentheimer data
R = np.identity(num_obs)*scanner_error
invR = np.linalg.inv(R)

# =============================================================================
# Ensemble permeability field generation 
# =============================================================================
lhs = Lhs(lhs_type="classic", criterion=None)
    
# log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z, number of dummy slices, por-per a, por-perm b
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
    # hk_initial = model.apply_core_mask(hk_initial, core_mask)
    
    ## Plot permeability field
    # plot_2d(np.log10(field[:,10,:]), grid_size[0], grid_size[2], 'log10(mD)', cmap='gray')   
    # pfun.plot_2d(hk_initial[:,10,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')  
    # plt.show()

    mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_initial, prsity, al, 
                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

    at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

    zf = np.array(hk_initial.flatten())
    zfl = np.log(zf)
    zfl[zf==0] =0
    Z[:,i] = np.log(zf)
    # Z[:,i] = zf
    # forcasted state 
    # yf = np.array(at_array_norm.flatten())
    yf = np.append(at_ensem.flatten(), hk_mean)
    Yj[:,i] = yf
    

def Kalman_generator(Z, Yj, Y, R, invR):
    # determine parameters from input
    num_parameters, Ne = Z.shape
    num_obs = Y.size
    
    # calculate ensemble means for upcoming covariance calculations
    Zbar = np.mean(Z, axis=1)
    Ybar = np.mean(Yj, axis=1)
    
    # preallocate covariance
    Pyy = np.zeros((num_obs, num_obs))
    # Calculate covarance matrix of the predicted observations 
    for i in range(Ne):
        sig = np.array([Yj[:,i]-Ybar])
        Pyy += np.dot(sig.T,sig)
    Pyy = Pyy/Ne 
    
    # Calculate parameter cross-covariance
    # preallocate ensemble covariance
    Pzy = np.zeros((num_parameters, num_obs))
    # Calculate parameter covariance
    for i in range(Ne):
        sig_z = np.array([Z[:,i]-Zbar])
        sig_y = np.array([Yj[:,i]-Ybar])
        Pzy += np.dot(sig_z.T,sig_y)
    Pzy = Pzy/Ne
    
    # Kalman gain matrix G
    G = np.dot(Pzy, np.linalg.inv(Pyy+R))
    
    Li = np.zeros((Ne))
    # updated parameters
    for i in range(Ne):
        a = 0.99
        # calculate residuals
        ri = np.array([Y-Yj[:,i]])
        Li[i] = np.dot(np.dot(ri, invR), ri.T)
        Lif = Li[i]*2
    
        count = 1
        while Lif > Li[i] and count < 3:
            Z[:,i] = Z[:,i] + np.squeeze(a*np.dot(G, ri.T))
            zfl = np.exp(Z[:,i])
            zfl[zf==0] =0
            hk_update = np.reshape(zfl, (nlay, nrow, ncol))
            # hk_update = np.reshape(Z[:,i], (nlay, nrow, ncol))
            # print(hk_update)
            # rerun model with new parameters
            mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_update, prsity, al, 
                                    grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)
        
            at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)
            # forcasted state 
            yf = np.append(at_ensem.flatten(), hk_mean)
            
            # calculate residuals
            ri = np.array([Y-yf])
            Lif = np.dot(np.dot(ri, invR), ri.T)
            a = a/2
            count += 1
            if count == 3:
                # print('Min dampening factor reached on realization: ' + str(i))
                print('Dampening factor cut on realization: ' + str(i))
            
            Yj[:,i] = yf
        
    return Z, Yj, Li


hk_initial_mean = np.reshape(np.mean(np.exp(Z), axis=1), (nlay, nrow, ncol))

en1_save = np.zeros((num_parameters, num_iterations))
Res_save = np.zeros((Ne, num_iterations))
# record start
start = time.time() 

for iteration in range(0, num_iterations):
    print('Iteration #: ' +str(iteration))
    # iteration 
    Z, Yj, Li = Kalman_generator(Z, Yj, Y, R, invR)
        
    Res_save[:, iteration] = Li
    en1_save[:, iteration] = np.mean(Z, axis=1)
    # Print final run time
    end_time = time.time() # end timer
    # print('Model run time: ', end - start) # show run time
    print(f"Current run time: {end_time - start:0.4f} seconds")

# Print final run time
end_time = time.time() # end timer
# print('Model run time: ', end - start) # show run time
print(f"Total run time: {end_time - start:0.4f} seconds")
    
    
# Initial ensemble mean
pfun.plot_2d(hk_initial_mean[:,0,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('Inital Random K')
# plt.clim(het_hk[:,10,:].min(), het_hk[:,10,:].max())

# Post-filter ensemble mean
hk_mean = np.reshape(np.mean(np.exp(Z), axis=1), (nlay, nrow, ncol))
pfun.plot_2d(hk_mean[:,0,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('Iteration #' + str(iteration) +' mean K')
# plt.clim(het_hk[:,1,:].min(), het_hk[:,1,:].max())
# Ensemble
hk_std = np.reshape(np.std(np.exp(Z), axis=1), (nlay, nrow, ncol))
pfun.plot_2d(hk_std[:,0,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('Iteration #' + str(iteration) +' STD K')
# True hk
# pfun.plot_2d(het_hk[:,1,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
# plt.title('True K')
# plt.clim(het_hk[:,1,:].min(), het_hk[:,1,:].max())



# Plot Change in values of ensemble 1
# fig, axis = plt.subplots(1,1, figsize=(6,5), dpi=300)
# # plt.plot(het_hk.flatten(), np.exp(hk_init.flatten()), '.', color='black')
# acolors = plt.cm.Reds(np.linspace(0,1,num_iterations+3))
# for iteration in range(0, num_iterations):
#     plt.plot(het_hk.flatten(), np.exp(en1_save[:,iteration]), '.', color=acolors[iteration+2])
# plt.title('Ne = ' + str(Ne))
# plt.ylabel('Ensemble mean Kh [cm/min]')
# plt.xlabel('True Kh [cm/min]')
# plt.xlim([0, 0.1])
# plt.ylim([0, 0.1])

# Plot resduals
fig, axis = plt.subplots(1,1, figsize=(6,5), dpi=300)
bcolors = plt.cm.Blues(np.linspace(0,1,Ne+2))
xi = np.arange(1, num_iterations, 1)
for i in range(0, Ne):
    plt.plot(xi, Res_save[i,1:], '.', color=bcolors[i+2])
plt.title('Change in ensemble residuals')
plt.yscale('log')
# plot subset of data
x = np.arange(8, num_iterations, 1)
y = np.mean(Res_save[:,8:], axis=0)
plt.plot(x,y)
# Fit linear equation to data
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, z[1]+z[0]*x, color='red')
print('Residual slope: ' + str(z[0]) + ', Ne = ' + str(Ne) + ', iterations = '
      + str(num_iterations) + ', error = ' + str(scanner_error))



# =============================================================================
# Run mean perm
# =============================================================================
mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_mean, prsity, al, 
                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

pfun.plot_2d(at_ensem[:,0,:], grid_size[0], grid_size[2], 'Predicted arrival time', cmap='bwr')
plt.clim([-0.2, 0.2])

# =============================================================================
# SAVE DATA  (OPTIONAL)
# =============================================================================
# save some data 
save_filename_hk = new_path + '\\' + 'hk_mean_'  + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
save_data = hk_mean.flatten('C')
np.savetxt(save_filename_hk, save_data, delimiter=',', fmt='%.3e')

save_filename_hk = new_path + '\\' + 'hk_std_'  + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
save_data = hk_std.flatten('C')
np.savetxt(save_filename_hk, save_data, delimiter=',', fmt='%.3e')

# residual information
save_filename_res = new_path + '\\' + 'residual_data.csv'
np.savetxt(save_filename_res, Res_save, delimiter=',', fmt='%.3e')


# =============================================================================
# Speed testing
# =============================================================================
# start = time.time() 
# for i in range(100000):
#     Psf2 += np.outer(sig, sig)
# # Print final run time
# end_time = time.time() # end timer
# # print('Model run time: ', end - start) # show run time
# print(f"Run time: {end_time - start:0.4f} seconds")

# start = time.time() 
# for i in range(100000):
#     Psf += np.dot(sig.T,sig)
# # Print final run time
# end_time = time.time() # end timer
# # print('Model run time: ', end - start) # show run time
# print(f"Run time: {end_time - start:0.4f} seconds")
# pfun.plot_2d(Psf, 1, 1, 'Psf', cmap='viridis')
#     # plt.title('Initial Random K')  
# pfun.plot_2d(Psf2, 1, 1, 'Psf2', cmap='viridis')
#     # plt.title('Initial Random K') 