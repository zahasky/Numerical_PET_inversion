# -*- coding: utf-8 -*-
"""
This code is a simple small 3D test model that was used to generate the 
5x5x10_small_3d_model_test results
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

# Import custom functions to run flopy stuff and analytical solutions
import model_functions as model

import sys
new_path = 'D:\\Dropbox\\Codes'
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
exe_name_mf = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
exe_name_mt = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# directory to save data
# directory_name = 'examples'
# workdir = os.path.join('.', '3D_fields', directory_name)
directory_name = 'Kalman_testing'
workdir = os.path.join('D:\\Modflow_models', directory_name)
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
field = 10**(srf.structured([np.arange(nlay), np.arange(nrow), np.arange(ncol)]) + p[1])
# check for negative values
field[field<0] = 1
print('GS model mean permeability: '+ str(10**p[1]) + ' mD')
# convert from mD to m^2
field_km2 = field*(9.869233E-13/1000) 
# save permeability field
save_data = np.append(field_km2.flatten('C'), [nlay, nrow, ncol])
np.savetxt(workdir + '\\perm_field_1_m2.csv', save_data , delimiter=',', fmt='%.3e')  
# To reimport a saved field uncomment the following
# field_km2 = np.loadtxt(perm_field_dir + '\\core_k_3d_m2_' + str(td) +'.csv', delimiter=',')
# # crop off last three values that give model dimensions
# field_km2 = field_km2[:-3]
# field_km2 = field_km2.reshape(nlay, nrow, ncol) 

# Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
het_hk = field_km2*(1000*9.81*100*60/8.9E-4)

## Plot permeability field
# plot_2d(np.log10(field[:,10,:]), grid_size[0], grid_size[2], 'log10(mD)', cmap='gray')   
pfun.plot_2d(het_hk[:,1,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')     
# plt.title('True K')
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
# =============================================================================
# PLOT DATA 
# =============================================================================
### Cross-sectional plots (using the 'plot_2d' function in the 'plotting_functions.py' script)
# Uncomment the line below to plot a 'center slice' plot along the long axis of the core
pfun.plot_2d(conc_true[10,:,1,:], grid_size[0], grid_size[2], 'concentration', cmap='OrRd')
plt.clim(0, 0.1) 
pfun.plot_2d(conc_true[10,:,:,1], grid_size[0], grid_size[2], 'concentration', cmap='OrRd')
plt.clim(0, 0.1) 
pfun.plot_2d(conc_true[10,1,:,:], grid_size[0], grid_size[2], 'concentration', cmap='OrRd')
plt.clim(0, 0.1) 

pfun.plot_2d(at_obs[:,1,:], grid_size[0], grid_size[2], 'arrival time', cmap='viridis')
# pfun.plot_2d(conc[3,:,1,:], grid_size[0], grid_size[2], 'concentration', cmap='OrRd')
# This is how you would plot a slice of the core perpendicular to the axis rather than parallel
# plot_2d(conc[10,:,:,10], grid_size[0], grid_size[1], 'concentration', cmap='OrRd')

# Check linearity of model parameters and state variables
fig, axis = plt.subplots(1,1, figsize=(6,5), dpi=300)
plt.plot(np.log(het_hk.flatten()), at_obs.flatten(), 'o', color='black')
plt.title('Logged Kh vs arrival time')

fig, axis = plt.subplots(1,1, figsize=(6,5), dpi=300)
_ = plt.hist(at_obs.flatten(), bins='auto')
plt.title('Arrival time difference')

fig, axis = plt.subplots(1,1, figsize=(6,5), dpi=300)
_ = plt.hist(np.log(het_hk.flatten()), bins='auto')
plt.title('True log(Kh)')

# pore volume calculation
PV = v*times_true/(ncol*grid_size[2])
# PV = 2*times/(core_mask.sum()*np.prod(grid_size)*ncol*prsity)

## PLOT
n = 5
# Define colormap for tracer
tcolors = plt.cm.Blues(np.linspace(0,1,n))
acolors = plt.cm.Reds(np.linspace(0,1,n))

# Breakthrough curves
fig, axis = plt.subplots(1,1, figsize=(8,5), dpi=300)
plt.plot(PV, btc_solute_heter, label='conservative', color='black')
plt.title('Heterogeneous permeability')
plt.ylabel('$C/C_0$ [-]')
plt.xlabel('PV [-]')
plt.show()


#### Concentration/Depositional profiles
# extract grid cell centers
# x = mf.dis.sr.xcenter[:-2]
# # plot profiles at some fraction of PV injected
# plot_pv = 0.5
# timestep = (np.abs(PV - plot_pv)).argmin()

# fig, axis = plt.subplots(1,2, figsize=(15,5), dpi=300)
# C_profile = np.sum(np.sum(conc_kf[timestep, :, :, :], axis=0), axis=0)/core_mask.sum()
# axis[0].plot(x, C_profile, label='single kf =' +str(kf1), color=tcolors[1])
# axis[0].set_title('Concentration Profile [PV=0.5]')
# axis[0].set(ylabel='$C/C_0$ [-]',xlabel ='Distance from inlet [cm]')
# axis[0].set_ylim([0, 0.1])
# # axis[0].set(ylabel='$C/C_0$ [-]',xlabel ='Time [minutes]', yscale='log')
# # axis[0].set_ylim([1E-4, 1])
# axis[0].legend(loc ='upper right')
# plt.tight_layout()

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
num_iterations = 50
# observed data
Y = np.append(at_obs.flatten(), hk_mean_obs)

num_obs = Y.size
num_parameters = nlay*nrow*ncol
# parameter value preallocation (parameters are what we are estimating, ie k)
Z = np.zeros((num_parameters, Ne))
# state variable preallocation
Yj = np.zeros((num_obs, Ne))

# error covariance matrix (based on measurement error of PET scanner)
scanner_error = 0.00001
R = np.identity(num_obs)*scanner_error
invR = np.linalg.inv(R)

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

    mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_initial, prsity, al, 
                            grid_size, v, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

    at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)

    zf = np.array(hk_initial.flatten())
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
    
        count = 0
        while Lif > Li[i] and count < 11:
            Z[:,i] = Z[:,i] + np.squeeze(a*np.dot(G, ri.T))
        
            hk_update = np.reshape(np.exp(Z[:,i]), (nlay, nrow, ncol))
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
            if count == 10:
                print('Min dampening factor reached on realization: ' + str(i))
            
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
print(f"Total run time: {end_time - start:0.4f} seconds")
    
    
# Initial ensemble mean
pfun.plot_2d(hk_initial_mean[:,1,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('Inital Random K')
plt.clim(het_hk[:,1,:].min(), het_hk[:,1,:].max())

# Post-filter ensemble mean
hk_mean = np.reshape(np.mean(np.exp(Z), axis=1), (nlay, nrow, ncol))
pfun.plot_2d(hk_mean[:,1,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('Iteration #' + str(iteration) +' mean K')
plt.clim(het_hk[:,1,:].min(), het_hk[:,1,:].max())
# Ensemble
hk_std = np.reshape(np.std(np.exp(Z), axis=1), (nlay, nrow, ncol))
pfun.plot_2d(hk_std[:,1,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('Iteration #' + str(iteration) +' STD K')
# True hk
pfun.plot_2d(het_hk[:,1,:], grid_size[0], grid_size[2], '[cm/min]', cmap='gray')    
plt.title('True K')
plt.clim(het_hk[:,1,:].min(), het_hk[:,1,:].max())



# Plot Change in values of ensemble 1
fig, axis = plt.subplots(1,1, figsize=(6,5), dpi=300)
# plt.plot(het_hk.flatten(), np.exp(hk_init.flatten()), '.', color='black')
acolors = plt.cm.Reds(np.linspace(0,1,num_iterations+3))
for iteration in range(0, num_iterations):
    plt.plot(het_hk.flatten(), np.exp(en1_save[:,iteration]), '.', color=acolors[iteration+2])
plt.title('Ne = ' + str(Ne))
plt.ylabel('Ensemble mean Kh [cm/min]')
plt.xlabel('True Kh [cm/min]')
plt.xlim([0, 0.1])
plt.ylim([0, 0.1])

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
# SAVE DATA  (OPTIONAL)
# =============================================================================
# save some data 
# save_filename_btdn = heterogeneous_arrival_dir + '\\' + 'arrival_norm_diff_' + model_dirname + '_'  + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
# save_data = np.append(at_diff_norm.flatten('C'), [km2_mean])
# np.savetxt(save_filename_btdn, save_data, delimiter=',', fmt='%.3e')


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