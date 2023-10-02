# -*- coding: utf-8 -*-
"""
flopy_arrival_time_3d_diff_generation.py
Created on Fri Jun  5 08:23:38 2020

@author: Christopher Zahasky

This script is used to generate 3D arrival time maps using synthetically 
generated 3D permeability fields. This script calls functions from the python
script titled 'flopy_bt_3d_functions.py'
"""

# Only packages called in this script need to be imported
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# Packages for random field generation
import gstools as gs
from skopt.sampler import Lhs
from scipy.special import erfinv as erfinv

# Import custom functions to run flopy stuff, the calls should be structured as:
# from file import function
from flopy_arrival_time_3d_functions import mt3d_pulse_injection_sim, flopy_arrival_map_function, plot_2d

# lognormal cumulative density function sampling for vug generation
def sample_lognormal_cdf(zero2one, sig_ag, mu_ag):
    r = np.exp((2**(1/2)*sig_ag)*erfinv(2*zero2one-1) + np.log(mu_ag))
    return r

# apply vugs in random locations
def vug_map_generation_function(field, vugs, vug_perm):
    nlay, nrow, ncol = np.shape(field)
    # Determine the number of vugs (rows in vug array)
    nvug, s = np.shape(vugs)
    # Iterate through vuggs
    for n in range(nvug):
        
        lay_range = [round(vugs[n,0] - vugs[n,3]), round(vugs[n,0] + vugs[n,3])]
        if lay_range[0]  < 0: lay_range[0] = 0
        if lay_range[1]  > nlay: lay_range[1] = nlay

        row_range = [round(vugs[n,1] - vugs[n,3]), round(vugs[n,1] + vugs[n,3])]
        if row_range[0]  < 0: row_range[0] = 0
        if row_range[1]  > nrow: row_range[1] = nrow
        
        col_range = [round(vugs[n,2] - vugs[n,3]), round(vugs[n,2] + vugs[n,3])]
        if col_range[0]  < 0: col_range[0] = 0
        if col_range[1]  > ncol: col_range[1] = ncol
        
        # loop through x values
        for i in range(lay_range[0], lay_range[1]):
            # loop through y values
            for j in range(row_range[0], row_range[1]):
                
                for k in range(col_range[0], col_range[1]):
                    # find linear distance from grid center to microzone center
                    rc = (((i+0.5) - vugs[n,0])**2 + 
                          ((j+0.5) - vugs[n,1])**2 +
                          ((k+0.5) - vugs[n,2])**2)**(1/2)
    
                    if rc<= vugs[n,3]:
                        field[i, j, k] = vug_perm                    
    return field

# apply planes in random locations
def planar_feature_map_generation_function(field, X, Y, Z, nplanes, orientation, plane_perm):
    
    n = 0
    # Iterate through the number of planes
    while n < nplanes:
        
        if n == 0 or orientation == 0:
            #randomly generate orientation
            A = (np.random.rand(4)*2)-1
            A[3] = abs(A[3])
        else:
            A[3] = np.random.rand(1)*10
        
        # nice test plot
        # Pl = (A[3] - A[0]*X - A[1]*Y) / A[2]
        # fig = plt.figure(figsize=(15, 3), dpi=200)
        # ax = fig.add_subplot(projection='3d')
        # sc = ax.scatter(X, Y, Pl, s=1, cmap='Greys')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
        
        # points on plane at x and y = 0 use whatever coefficient is larger
        if abs(A[2]) > abs(A[0]):
            Q = Z + (A[3]/A[2])
            # calculate the distance of every grid cell center
            Dist = abs(A[0]*X + A[1]*Y + A[2]*Q)/ ((A[0]**2 + A[1]**2 + A[2]**2)**(1/2))
        else:
            Q = X + (A[3]/A[0])
            Dist = abs(A[0]*Q + A[1]*Y + A[2]*Z)/ ((A[0]**2 + A[1]**2 + A[2]**2)**(1/2))
            
        
        # Nice plotting functions
        # fig = plt.figure(figsize=(15, 3), dpi=200)
        # ax = fig.add_subplot(projection='3d')
        # sc = ax.scatter(X, Y, Z, s=1, c=Dist, cmap='viridis')
        # plt.colorbar(sc)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
        
        # plane = np.zeros([nlay, nrow, ncol])
        # plane[Dist<0.177] = 1
        # fig = plt.figure(figsize=(15, 3), dpi=200)
        # ax = fig.add_subplot(projection='3d')
        # sc = ax.scatter(X, Y, Z, s=plane, c=plane, cmap='Greys')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
        
        if np.min(Dist) < 0.177:
            # then set all of the values less than half a gridcell equal to the plane permeability
            field[Dist<0.177] = plane_perm
            # and update the n counter
            n +=1
        
                    
    return field

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
directory_name = 'models'
workdir = os.path.join('D:\\Training_data_generation_3D\\tdata_planar_2_5k_r2', directory_name)
# directory for homogeneous models
homogeneous_arrival_dir = os.path.join('D:\\Training_data_generation_3D\\tdata_planar_2_5k_r2\\homogeneous_porosity_at')
# directory for heterogeneous models
# heterogeneous_arrival_dir = os.path.join('D:\\Training_data_generation_3D\\tdata_python26k\\June_21_new_mask\\heterogeneous_porosity_new_mask')

# Set path to perm maps
# perm_field_dir = os.path.join('.')
# perm_field_dir = os.path.join('.', '3D_fields', directory_name)
new_perm_field_dir = os.path.join('D:\\Training_data_generation_3D\\tdata_planar_2_5k_r2\\syn_core_perm_maps')

# =============================================================================
# PARAMETER SPACE DEFINITION
# =============================================================================
# Generation options
# generate new sample space and permeability fields (True or False)
generate_sample_space = True
generate_fields = True
parameter_space_csv_name = 'parameter_space_2_5k_planar_r2.csv'

# number of training datasets desired
ntraining_data = 2501
# number of extra datasets to generate in case of sim or at calc failure
extra_sets = 100

if generate_sample_space == True:
    lhs = Lhs(lhs_type="classic", criterion=None)
    
    # log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z, number of dummy slices,
    # Ps = lhs.generate([(-4., -0.5), (1.0, 4.3), (1.0, 50.), (1.0, 50.0), (1.0, 50.0), \
    #                   (0.0, 1.57), (0.0, 1.57), (0.0, 1.57), (1,3)], ntraining_data + extra_sets)
    # Sample space for porosity
    # log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z, number of dummy slices, por-per a, por-perm b
    # Ps = lhs.generate([(-4., -0.5), (1.0, 4.3), (1.0, 50.), (1.0, 50.0), (1.0, 50.0), \
    #                   (0.0, 1.57), (0.0, 1.57), (0.0, 1.57), (0,2), (0.25, 1.0), (5, 20)], ntraining_data + extra_sets)
    # Sample space for vuggs
    # log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z, number of dummy slices, vugg_perm, n_vuggs, log_distribution_vug_size in vox
    # Ps = lhs.generate([(-4., -0.5), (1.0, 4.3), (1.0, 50.), (1.0, 50.0), (1.0, 50.0),\
    #                    (0.0, 1.57), (0.0, 1.57), (0.0, 1.57), (0,3), (-1.5, 1.5), (5, 50), (-0.5, 0.2)], ntraining_data + extra_sets)
        
    # Sample space for planes
    # log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z, number of dummy slices, plane_perm, n_planes, random or parallel
    Ps = lhs.generate([(-4., -0.5), (1.0, 4.3), (1.0, 50.), (1.0, 50.0), (1.0, 50.0),\
                       (0.0, 1.57), (0.0, 1.57), (0.0, 1.57), (0,3), (-1.5, 1.5), (1, 6), (0, 1)], ntraining_data + extra_sets)

    # example of how to efficiently save data to text file    
    np.savetxt(parameter_space_csv_name, Ps , delimiter=',', fmt='%.3e')
else:
    Ps = np.loadtxt(parameter_space_csv_name, delimiter=',')
    
# =============================================================================
# VARIABLES THAT DON'T CHANGE
# =============================================================================
# Hard coded mask for 20x20 grid cross-section
mp = np.array([7, 5, 3, 2, 2, 1, 1])
mask_corner = np.ones((10,10))
for i in range(len(mp)):
    mask_corner[i, 0:mp[i]] = 0
    
mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
core_mask = np.concatenate((mask_top, np.flipud(mask_top)))

# grid_size = [grid size in direction of Lx (layer thickness), 
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = [0.23291, 0.23291, 0.25] # selected units [cm]
# grid dimensions
nlay = 20 # number of layers / grid cells
nrow = 20 # number of columns 
ncol = 40 # number of slices (parallel to axis of core)


x_ = np.linspace(grid_size[0]/2, grid_size[0]*nlay, nlay)
y_ = np.linspace(grid_size[1]/2, grid_size[1]*nrow, nrow)
z_ = np.linspace(grid_size[2]/2, grid_size[2]*ncol, ncol)

X, Y, Z = np.meshgrid(x_, y_, z_, indexing='xy')

# Output control for MT3dms
# nprs (int):  the frequency of the output. If nprs > 0 results will be saved at 
# the times as specified in timprs (evenly allocated between 0 and sim run length); 
# if nprs = 0, results will not be saved except at the end of simulation; if NPRS < 0, simulation results will be 
# saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
nprs = 150 
# period length in selected units (minutes)
# the first period is fluid pressure equilibration, the second period is tracer
# injection, the third period is tracer displacement
perlen_mt = [2., 1., 90]
# Numerical method flag
mixelm = -1

# Call function and time it
start_td = time.time() # start a timer

# value of back up realizations to draw from
replacement_counter = ntraining_data + 1

td = 0
while td < ntraining_data:
    
    print('TRAINING DATASET: ' + str(td))
       
    # ========================================================================
    # PERMEABILITY FIELD GENERATION
    # ========================================================================
    p = Ps[td-1]
    if generate_fields == True:
        model = gs.Exponential(dim=3, var=10**p[0], len_scale=[p[2], p[3], p[4]], angles=[p[5], p[6], p[7]])
        srf = gs.SRF(model, seed=20170519)
        # permeability field in millidarcy
        field = 10**(srf.structured([np.arange(nlay), np.arange(nrow), np.arange(ncol)]) + p[1])
        
        ##### ADD planes
        nplanes = p[10]
        orientation = p[11]
        plane_perm = (10**p[1])* (10**p[9])
        if plane_perm > 20000:
            plane_perm = 20000
        
        # call function that generates random planes
        planar_feature_map_generation_function(field, X, Y, Z, nplanes, orientation, plane_perm)
        
        # check for negative values
        field[field<0] = 1
        print('GS model mean perm: '+ str(10**p[1]) + ' mD')
        # convert from mD to km^2
        field_km2 = field*(9.869233E-13/1000)
        # set permeability values outside of core to zero with core mask
        for col in range(ncol):
            field_km2[:,:,col] = np.multiply(field_km2[:,:,col], core_mask)
        
        # Save data in small format
        save_data = np.append(field_km2.flatten('C'), [nlay, nrow, ncol])
        np.savetxt(new_perm_field_dir + '\\coreplane_k_3d_m2_' + str(td+2500)+'.csv', save_data , delimiter=',', fmt='%.3e')
    else:
        # Import permeability map
        field_km2 = np.loadtxt(new_perm_field_dir + '\\coreplane_k_3d_m2_' + str(td+2500) +'.csv', delimiter=',')
        # crop off last three values that give model dimensions
        field_km2 = field_km2[:-3]
        field_km2 = field_km2.reshape(nlay, nrow, ncol)
        # plot_2d(field_km2[:,:,1], grid_size[0], grid_size[1], 'perm', cmap='gray')
        
        # # set permeability values outside of core to zero with core mask
        # for col in range(ncol):
        #     field_km2[:,:,col] = np.multiply(field_km2[:,:,col], core_mask)
        # # plot_2d(field_km2[:,:,1], grid_size[0], grid_size[1], 'perm', cmap='gray')
        
        # save_data = np.append(field_km2.flatten('C'), [nlay, nrow, ncol])
        # np.savetxt(new_perm_field_dir + '\\corenm_k_3d_m2_' + str(td)+'.csv', save_data , delimiter=',', fmt='%.3e')
        
        
    
    # number of dummy slices at inlet boundary
    ndummy_in = int(p[8])  
    # Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
    raw_hk = field_km2*(1000*9.81*100*60/8.9E-4)
    
    # # Calculatie heterogeneous porosity field
    # prsity_field = ((np.log(field_km2/(9.869233E-13/1000))/p[9]) + p[10])/100
    # # replace infinity values with zeros
    # prsity_field[field_km2<1e-25]=0
    # # set upper limit of 80 percent porosity
    # prsity_field[prsity_field >0.8]=0.8

    # hom_prsity_field = copy.deepcopy(field_km2)
    # # replace with homogeneous values
    # hom_prsity_field[field_km2<1e-25]=0
    # hom_prsity_field[field_km2>1e-25]=0.20
    hom_prsity_field = 0.25 # NOT USED INSIDE FUNCTION
    
    # Describe grid for results    
    Lx = (ncol) * grid_size[2]   # length of model in selected units 
    Ly = (nrow) * grid_size[1]   # length of model in selected units 
    # number of dummy slices
    ndummy_in = p[8]
    # Model workspace and new sub-directory
    model_dirname = ('td'+ str(td+2500))
    model_ws = os.path.join(workdir, model_dirname)
    
    mf, mt, conc, timearray, km2_mean = mt3d_pulse_injection_sim(model_dirname, 
                model_ws, raw_hk, hom_prsity_field, grid_size, ndummy_in,  perlen_mt, nprs, 
                mixelm, exe_name_mf, exe_name_mt)
    # print('Core average perm: '+ str(km2_mean) + ' m^2')
    
    # Option to plot and calculate geometric mean to double check that core average perm in close
    # raw_km2_array = field_km2.flatten()
    # index = np.argwhere(raw_km2_array > 0) 
    # geo_mean = np.exp(np.sum(np.log(raw_km2_array[index]))/index.size)
    # print('Geometric mean perm: ' + str(geo_mean) + ' m^2')
    
    # calculate quantile arrival time map from MT3D simulation results
    at_array, at_array_norm, at_diff_norm = flopy_arrival_map_function(conc, np.array(timearray), grid_size, 0.5)
    
    # In some cases the models may fail to run or there are issues with calculating arrival times
    # When this happens the realization is replaced with a random realization 
    # from a second sampling. This occurs less than 50 times in the entire 
    # 20,000 data initially generated
    if isinstance(at_diff_norm, int):
        # replace parameters
        Ps[td-1] = Ps[replacement_counter]
        # resave parameter space information  
        np.savetxt(parameter_space_csv_name, Ps , delimiter=',', fmt='%.3e')
        print('Training dataset: ' + str(td) + ', replaced with realization: ' + str(replacement_counter))
        # update replacement counter
        replacement_counter += 1
        continue
        
    # =============================================================================
    # SAVE DATA 
    # =============================================================================
    # save normalized arrival time difference data
    save_filename_btdn = homogeneous_arrival_dir + '\\' + 'arrival_norm_diff_' + model_dirname + '_'  + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(at_diff_norm.flatten('C'), [km2_mean])
    np.savetxt(save_filename_btdn, save_data, delimiter=',', fmt='%.3e')
    
    # save unnormalized breakthrough data
    save_filename_btd = homogeneous_arrival_dir + '\\' + 'arrival_norm_' + model_dirname + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(at_array_norm.flatten('C'), [km2_mean])
    np.savetxt(save_filename_btd, save_data, delimiter=',', fmt='%.3e')
    
    
    #### MODEL WITH HETEROGENEOUS POROSITY FIELD
    # mf, mt, conc, timearray, km2_mean = mt3d_pulse_injection_sim(model_dirname, 
    #             model_ws, raw_hk, prsity_field, grid_size, ndummy_in,  perlen_mt, nprs, 
    #             mixelm, exe_name_mf, exe_name_mt)
    
    # # calculate quantile arrival time map from MT3D simulation results
    # at_array, at_array_norm, at_diff_norm = flopy_arrival_map_function(conc, np.array(timearray), grid_size, 0.5)
    
    # # In some cases the models may fail to run or there are issues with calculating arrival times
    # # When this happens the realization is replaced with a random realization 
    # # from a second sampling. This occurs less than 50 times in the entire 
    # # 20,000 data initially generated
    # if isinstance(at_diff_norm, int):
    #     # replace parameters
    #     Ps[td-1] = Ps[replacement_counter]
    #     # resave parameter space information  
    #     np.savetxt(parameter_space_csv_name, Ps , delimiter=',', fmt='%.3e')
    #     print('Training dataset: ' + str(td) + ', replaced with realization: ' + str(replacement_counter))
    #     # update replacement counter
    #     replacement_counter += 1
    #     continue
        
    # # =============================================================================
    # # SAVE DATA 
    # # =============================================================================
    # # save normalized arrival time difference data
    # save_filename_btdn = heterogeneous_arrival_dir + '\\' + 'arrival_norm_diff_' + model_dirname + '_'  + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    # save_data = np.append(at_diff_norm.flatten('C'), [km2_mean])
    # np.savetxt(save_filename_btdn, save_data, delimiter=',', fmt='%.3e')
    
    # # save unnormalized breakthrough data
    # save_filename_btd = heterogeneous_arrival_dir + '\\' + 'arrival_norm_' + model_dirname + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    # save_data = np.append(at_array_norm.flatten('C'), [km2_mean])
    # np.savetxt(save_filename_btd, save_data, delimiter=',', fmt='%.3e')
    
    # Try to delete the previous folder of MODFLOW and MT3D files
    if td > 1:
        old_model_ws = os.path.join(workdir, ('td'+ str(td-1)))
        # Try to remove tree; if failed show an error using try...except on screen
        try:
            shutil.rmtree(old_model_ws)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    
    # Update td counter        
    td += 1
    
# Print final run time
end_td = time.time() # end timer
print('Minutes to run ' + str(ntraining_data) + ' training simulations: ', (end_td - start_td)/60) # show run time
    
# =============================================================================
# PLOT DATA 
# =============================================================================
# Define grid    
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                  slice(0, Lx + grid_size[2], grid_size[2])]
# layer to plot
ilayer = 10
# # fontsize
fs = 24

hfont = {'fontname':'Arial'}


n = 21
colors = np.flipud(plt.cm.viridis(np.linspace(0,1,n)))
r, c, s = np.shape(at_array_norm)
x_coord = np.linspace(0, grid_size[2]*s, s)

print(np.std(at_array_norm[:,:,0]))

fig0, (ax01, ax02) =  plt.subplots(1, 2, figsize=(14, 4), dpi=400)
for i in range(0,19):
    ax01.plot(x_coord, at_array_norm[i,11,:], '.-', color=colors[i])
    ax02.plot(x_coord, at_array_norm[11,i,:], '.-', color=colors[i]) 
# Load breakthrough time data
tdata_norm = np.loadtxt(save_filename_btdn, delimiter=',')
load_km2 = tdata_norm[-1]
tdata_norm = tdata_norm[0:-1]
tdata_norm = tdata_norm.reshape(nlay, nrow, ncol)
    
plot_km2 = field_km2[ilayer,:,:]/9.869233E-13*1000
plot_2d(plot_km2, grid_size[1], grid_size[2], 'Permeability', cmap='Greys')

plot_km2 = field_km2[:,10,:]/9.869233E-13*1000
plot_2d(plot_km2, grid_size[1], grid_size[2], 'Permeability', cmap='Greys')
# plot_2d(prsity_field[ilayer,:,:], grid_size[1], grid_size[2], 'Porosity', cmap='Greys')
# clim_pv = np.max(np.abs([np.max(plot_km2), np.min(plot_km2)]))
# plt.clim(-clim_pv, clim_pv)

# First figure with concentration data
pv = (np.array(timearray)-2)*(2/(3.1415*(2.5**2)*10*0.25))
fig1 = plt.figure(figsize=(10, 15))
ax0 = fig1.add_subplot(3, 1, 1, aspect='equal')
imp = plt.pcolor(x, y, conc[12,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,0.4) 
cbar.set_label('Solute concentration', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f PV' %pv[12], fontsize=fs+2, **hfont)

ax1 = fig1.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, conc[20,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,0.4) 
cbar.set_label('Solute concentration', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f PV' %pv[20], fontsize=fs+2, **hfont)

ax2 = fig1.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, conc[45,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,0.4) 
cbar.set_label('Solute concentration', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f PV' %pv[45], fontsize=fs+2, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)



# Second figure with head and breakthrough time difference maps
fig2 = plt.figure(figsize=(10, 15))
ax0 = fig2.add_subplot(3, 1, 1, aspect='equal')
dp_pressures = at_array[ilayer,:,:]
imp = plt.pcolor(x, y, dp_pressures, cmap='Purples', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0, np.max(dp_pressures)) 
cbar.set_label('Time [min]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('0.50 Quantile Arrival Time [min]', fontsize=fs+2, **hfont)

ax1 = fig2.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, at_array_norm[ilayer,:,:], cmap='Blues', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('PV [-]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('0.50 Quantile Arrival Time [-]', fontsize=fs+2, **hfont)

ax2 = fig2.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, tdata_norm[ilayer,:,:], cmap='PiYG', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Quantile Arrival Time Difference', fontsize=fs+2, **hfont)
clim_pv = np.max(np.abs([np.max(tdata_norm), np.min(tdata_norm)]))
plt.clim(-clim_pv, clim_pv)

gaussian_arr = np.random.normal(0,0.006, tdata_norm[:,:,0].shape)
syn_data_inlet = tdata_norm[:,:,0]+gaussian_arr

plot_2d(syn_data_inlet, grid_size[0], grid_size[1], 'arrival time', cmap='bwr')
clim_pv = np.max(np.abs([np.max(syn_data_inlet), np.min(syn_data_inlet)]))
plt.clim(-clim_pv, clim_pv)

plot_2d(tdata_norm[:,:,0], grid_size[0], grid_size[1], 'arrival time', cmap='bwr')
clim_pv = np.max(np.abs([np.max(syn_data_inlet), np.min(syn_data_inlet)]))
plt.clim(-clim_pv, clim_pv)