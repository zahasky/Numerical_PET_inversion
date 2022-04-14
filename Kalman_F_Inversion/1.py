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
from random import randrange
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
ndummy_in = randrange(3)
ndummy_in = 0

mf, mt, conc_true, btc_solute_heter, times_true, hk_mean_obs = model.mt3d_conservative('tracer', het_hk, prsity, al,
                            grid_size, v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

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

    ndummy_in = randrange(3)
    ndummy_in = 0
    print(ndummy_in)
    mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_initial, prsity, al,
                            grid_size, v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

    at_ensem = model.flopy_arrival_map_function(conc, times, grid_size, 0.5)
    print(ndummy_in)

    zf = hk_initial.flatten()
    Z[zf != 0, i] = np.log(zf[zf != 0])
    yf = at_ensem.flatten()
    Yj[:,i] = deepcopy(yf)

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
            zhist, zbins = np.histogram(Z[non_zero_idx,i],bins=nb)
            NS = Norm_Sc()
            # zscore (x-axis of the new distribution)
            Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(zbins[:-1], zhist, Z[non_zero_idx,i])
            # The original bins
            zbin[:, i] = zbins[:-1]

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
            while Lif > Li[i] and count < 11:
                Z[non_zero_idx,i] = Z[non_zero_idx,i] + np.squeeze(a*np.dot(G, ri.T))

                # Reverse normal score transformation of z
                nor, b = np.histogram(Z[non_zero_idx,i],bins=nb)
                NS = Norm_Sc()
                Z[non_zero_idx,i] = NS.reverse_norm_score_trans(b[:-1], nor, zscore[:,i], zbin[:,i], Z[non_zero_idx,i], min_da, max_da, power)

                # Get the updated hydraulic conductivity
                hk_update[non_zero_idx] = np.exp(Z[non_zero_idx, i])
                hk_update[zero_idx]= 0
                hk_update = np.reshape(hk_update, (nlay, nrow, ncol))

                # Rerun model with new parameters
                ndummy_in = randrange(3)
                mf, mt, conc, btc, times, hk_mean = model.mt3d_conservative('tracer', hk_update, prsity, al,
                            grid_size, v, ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir)

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
                if count == 10:
                    print('Min dampening factor reached on realization: ' + str(i+1))

                    # If no acceptance could be reached, ensemble member i is resampled from the full ensemble of last iteration
                    #rand1 = np.random.choice([x for x in range(Ne-1)])
                    #rand2 = np.random.choice([x for x in range(Ne-1)])
                    #Z[non_zero_idx, i] = (Z_prev[non_zero_idx, rand1] + Z_prev[non_zero_idx, rand2]) / 2

                    # Normal score transformation of z for the next update
                    zhist, zbins = np.histogram(Z[non_zero_idx,i],bins=nb)
                    NS = Norm_Sc()
                    # zscore (x-axis of the new distribution)
                    Z[non_zero_idx,i], zscore[:,i] = NS.norm_score_trans(zbins[:-1], zhist, Z[non_zero_idx,i])
                    # The original bins
                    zbin[:,i] = zbins[:-1]
                else:
                    # Normal score transformation of z for the next update
                    zhist, zbins = np.histogram(Z[non_zero_idx, i], bins=nb)
                    NS = Norm_Sc()
                    # zscore (x-axis of the new distribution)
                    Z[non_zero_idx, i], zscore[:, i] = NS.norm_score_trans(zbins[:-1], zhist, Z[non_zero_idx, i])
                    # The original bins
                    zbin[:, i] = zbins[:-1]

            # Reverse normal score transformation of z
            nor, b = np.histogram(Z[non_zero_idx,i],bins=nb)
            NS = Norm_Sc()
            Z[non_zero_idx,i] = NS.reverse_norm_score_trans(b[:-1], nor, zscore[:,i], zbin[:,i], Z[non_zero_idx,i], min_da, max_da, power)

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

















# -*- coding: utf-8 -*-
"""
Created on:
@author:Christopher Zahasky
This script contains the numerical and analytical model functions, ploting
functions, and  ....
"""

# All packages called by functions should be imported
import sys
import os
import numpy as np
from scipy import integrate

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy

# This function creates a cyclindrical mask that is hardcoded to be identical for all cores that are 20x20.
def generate_standard_20x20mask(hk):
    hk_size = hk.shape
    if hk_size[0] == 20 and hk_size[1] == 20:
        mp = np.array([7, 5, 3, 2, 2, 1, 1])
        mask_corner = np.ones((10,10))
        for i in range(len(mp)):
            mask_corner[i, 0:mp[i]] = 0

        mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
        core_mask = np.concatenate((mask_top, np.flipud(mask_top)))
    else:
        print('Dimensions not 20x20, no core mask generated')
        core_mask = np.ones([hk_size[0], hk_size[1]])

    return core_mask

def apply_core_mask(hom_hk, core_mask):
    hk_size = hom_hk.shape
    # set permeability values outside of core to zero with core mask by multiplying mask of zeros and ones to every slice
    for col in range(hk_size[2]):
        hom_hk[:,:,col] = np.multiply(hom_hk[:,:,col], core_mask)
    return hom_hk

# FloPy Model function
def mt3d_conservative(dirname, raw_hk, prsity, al, grid_size, v,
                             ndummy_in, perlen_mt, timprs, mixelm, exe_name_mf, exe_name_mt, workdir):
    # Model workspace and new sub-directory
    model_ws = os.path.join(workdir, dirname)
# =============================================================================
#     UNIT INFORMATION
# =============================================================================
    # units must be set for both MODFLOW and MT3D, they have different variable names for each
    # time units (itmuni in MODFLOW discretization package)
    # 1 = seconds, 2 = minutes, 3 = hours, 4 = days, 5 = years
    itmuni = 2 # MODFLOW length units
    mt_tunit = 'M' # MT3D units
    # length units (lenuniint in MODFLOW discretization package)
    # 0 = undefined, 1 = feet, 2 = meters, 3 = centimeters
    lenuni = 3 # MODFLOW units
    mt_lunit = 'CM' # MT3D units

# =============================================================================
#     STRESS PERIOD INFO
# =============================================================================
    perlen_mf = [np.sum(perlen_mt)]
    # number of stress periods (MF input), calculated from period length input
    nper_mf = len(perlen_mf)
    # number of stress periods (MT input), calculated from period length input
    nper = len(perlen_mt)

# =============================================================================
#     MODEL DIMENSION AND MATERIAL PROPERTY INFORMATION
# =============================================================================
    # Make model dimensions the same size as the hydraulic conductivity field input
    # NOTE: that the there are two additional columns added as dummy slices (representing coreholder faces)
    hk_size = raw_hk.shape
    # determine dummy slice perm based on maximum hydraulic conductivity
    dummy_slice_hk = raw_hk.max()*10
    # define area with hk values above zero
    core_mask = np.ones((hk_size[0], hk_size[1]))
    core_mask = np.multiply(core_mask, raw_hk[:,:,0])
    core_mask[np.nonzero(core_mask)] = 1
    # define hk in cells with nonzero hk to be equal to 10x the max hk
    # This represents the 'coreholder' slices
    dummy_ch = core_mask[:,:, np.newaxis]*dummy_slice_hk
    # option to uncomment to model porosity field
    # dummy_ch_por = core_mask[:,:, np.newaxis]*0.15

    # # Additional dummy inlet to replicate imperfect boundary
    # dummy_slice_in =raw_hk[:,:,0]*core_mask
    # dummy_slice_in = dummy_slice_in.reshape(hk_size[0], hk_size[1], 1)
    # # concantenate dummy slice on hydraulic conductivity array
    # ndummy_in = int(ndummy_in)
    # if ndummy_in > 0:
    #     dummy_block = np.repeat(dummy_slice_in, ndummy_in, axis=2)
    #     hk = np.concatenate((dummy_ch, dummy_block, raw_hk, dummy_ch), axis=2)
    #
    #     # # do the same with porosity slices
    #     # dummy_slice_in =prsity_field[:,:,0]*core_mask
    #     # dummy_slice_in = dummy_slice_in.reshape(hk_size[0], hk_size[1], 1)
    #     # dummy_block = np.repeat(dummy_slice_in, ndummy_in, axis=2)
    #     # prsity = np.concatenate((dummy_ch_por, dummy_block, prsity_field, dummy_ch_por), axis=2)
    #
    # else:
    #     hk = np.concatenate((dummy_ch, raw_hk, dummy_ch), axis=2)
    #     # prsity = np.concatenate((dummy_ch_por, prsity_field, dummy_ch_por), axis=2)

    hk = np.concatenate((dummy_ch, raw_hk, dummy_ch), axis=2)
    # hk = raw_hk
    # Model information (true in all models called by 'p01')
    nlay = int(hk_size[0]) # number of layers / grid cells
    nrow = int(hk_size[1]) # number of rows / grid cells
    # ncol = hk_size[2]+2+ndummy_in # number of columns (along to axis of core)
    ncol = hk_size[2]+2
    delv = grid_size[0] # grid size in direction of Lx (nlay)
    delr = grid_size[2] # grid size in direction of Ly (nrow)
    delc = grid_size[1] # grid size in direction of Lz (ncol)

    laytyp = 0
    # cell elevations are specified in variable BOTM. A flat layer is indicated
    # by simply specifying the same value for the bottom elevation of all cells in the layer
    botm = [-delv * k for k in range(1, nlay + 1)]

    # ADDITIONAL MATERIAL PROPERTIES
    # prsity = 0.25 # porosity. float or array of floats (nlay, nrow, ncol)
    prsity = prsity
    al = al # longitudental dispersivity
    trpt = 0.3 # ratio of horizontal transverse dispersivity to longitudenal dispersivity
    trpv = 0.3 # ratio of vertical transverse dispersivity to longitudenal dispersivity

# =============================================================================
#     BOUNDARY AND INTIAL CONDITIONS
# =============================================================================
    # backpressure, give this in kPa for conversion
    bp_kpa = 70

     # Initial concentration (MT input)
    c0 = 0.
    # Stress period 2 concentration
    c1 = 1.0

    # Core radius
    core_radius = 2.54 # [cm]
    # Calculation of core area
    core_area = 3.1415*core_radius**2
    # Calculation of mask area
    mask_area = np.sum(core_mask)*grid_size[0]*grid_size[1]
    # total specific discharge or injection flux (rate/area)
    # q = injection_rate*(mask_area/core_area)/np.sum(core_mask)
    # scale injection rate locally by inlet permeability
    q_total = v*mask_area*prsity
    # print(q_total)
    # q_total = injection_rate/core_area
    q = q_total/np.sum(dummy_ch)

    # MODFLOW head boundary conditions, <0 = specified head, 0 = no flow, >0 variable head
    # ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    ibound = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)

    # inlet conditions (currently set with well so inlet is zero)
    # ibound[:, :, 0] = ibound[:, :, -1]*-1
    # outlet conditions
    # ibound[5:15, 5:15, -1] = -1
    ibound[:, :, -1] = ibound[:, :, -1]*-1

    # MODFLOW constant initial head conditions
    strt = np.zeros((nlay, nrow, ncol), dtype=float)
    # Lx = (hk_size[2]*delc)
    # Q = injection_rate/(core_area)
    # geo_mean_k = np.exp(np.sum(np.log(hk[hk>0]))/len(hk[hk>0]))
    # h1 = Q * Lx/geo_mean_k
    # print(h1)

    # convert backpressure to head units
    if lenuni == 3: # centimeters
        hout = bp_kpa*1000/(1000*9.81)*100
    else: # elseif meters
        if lenuni == 2:
            hout = bp_kpa*1000/(1000*9.81)
    # assign outlet pressure as head converted from 'bp_kpa' input variable
    # index the inlet cell
    # strt[:, :, 0] = h1+hout
    strt[:, :, -1] = core_mask*hout
    # strt[:, :, -1] = hout

    # Stress period well data for MODFLOW. Each well is defined through defintition
    # of layer (int), row (int), column (int), flux (float). The first number corresponds to the stress period
    # Example for 1 stress period: spd_mf = {0:[[0, 0, 1, q],[0, 5, 1, q]]}
    well_info = np.zeros((int(np.sum(core_mask)), 4), dtype=float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0

    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            # index_n +=1
            # print(index_n)
            if core_mask[layer, row] > 0:
                q_dummy = q*dummy_ch[layer,row]
                well_info[index_n] = [layer, row, 0, q_dummy[0]]
                index_n +=1


    # Now insert well information into stress period data
    # Generalize this for multiple stress periods (see oscillatory flow scripts)
    # This has the form: spd_mf = {0:[[0, 0, 0, q],[0, 5, 1, q]], 1:[[0, 1, 1, q]]}
    spd_mf={0:well_info}

    # MT3D stress period data, note that the indices between 'spd_mt' must exist in 'spd_mf'
    # This is used as input for the source and sink mixing package
    # Itype is an integer indicating the type of point source, 2=well, 3=drain, -1=constant concentration
    itype = 2
    cwell_info = np.zeros((int(np.sum(core_mask)), 5), dtype=float)
    # cwell_info = np.zeros((nrow*nlay, 5), dtype=np.float)
    # Nested loop to define every inlet face grid cell as a well
    index_n = 0
    for layer in range(0, nlay):
        for row in range(0, nrow):
            # index_n = layer*nrow + row
            if core_mask[layer, row] > 0:
                cwell_info[index_n] = [layer, row, 0, c0, itype]
                index_n +=1
            # cwell_info[index_n] = [layer, row, 0, c0, itype]

    # Second stress period
    cwell_info2 = cwell_info.copy()
    cwell_info2[:,3] = c1
    # Second stress period
    cwell_info3 = cwell_info.copy()
    cwell_info3[:,3] = c0
    # Now apply stress period info
    spd_mt = {0:cwell_info, 1:cwell_info2, 2:cwell_info3}

    # Concentration boundary conditions, this is neccessary to indicate
    # inactive concentration cells outside of the more
    # If icbund = 0, the cell is an inactive concentration cell;
    # If icbund < 0, the cell is a constant-concentration cell;
    # If icbund > 0, the cell is an active concentration cell where the
    # concentration value will be calculated. (default is 1).
    icbund = np.repeat(core_mask[:, :, np.newaxis], ncol, axis=2)
    # icbund[0, 0, 0] = -1
    # Initial concentration conditions, currently set to zero everywhere
    # sconc = np.zeros((nlay, nrow, ncol), dtype=np.float)
    # sconc[0, 0, 0] = c0

# =============================================================================
# MT3D OUTPUT CONTROL
# =============================================================================
    # nprs (int): A flag indicating (i) the frequency of the output and (ii) whether
    #     the output frequency is specified in terms of total elapsed simulation
    #     time or the transport step number. If nprs > 0 results will be saved at
    #     the times as specified in timprs; if nprs = 0, results will not be saved
    #     except at the end of simulation; if NPRS < 0, simulation results will be
    #     saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
    nprs = len(timprs)

    # timprs (list of float): The total elapsed time at which the simulation
    #     results are saved. The number of entries in timprs must equal nprs. (default is None).
    timprs = timprs
    # obs (array of int): An array with the cell indices (layer, row, column)
    #     for which the concentration is to be printed at every transport step.
    #     (default is None). obs indices must be entered as zero-based numbers as
    #     a 1 is added to them before writing to the btn file.
    # nprobs (int): An integer indicating how frequently the concentration at
    #     the specified observation points should be saved. (default is 1).

# =============================================================================
# START CALLING MODFLOW PACKAGES AND RUN MODEL
# =============================================================================
    # Start callingwriting files
    modelname_mf = dirname + '_mf'
    # same as 1D model
    mf = flopy.modflow.Modflow(modelname=modelname_mf, model_ws=model_ws, exe_name=exe_name_mf)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper_mf,
                                   delr=delr, delc=delc, top=0., botm=botm,
                                   perlen=perlen_mf, itmuni=itmuni, lenuni=lenuni)

    # MODFLOW basic package class
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    # MODFLOW layer properties flow package class
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, laytyp=laytyp)
    # MODFLOW well package class
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd_mf)
    # MODFLOW preconditioned conjugate-gradient package class
    pcg = flopy.modflow.ModflowPcg(mf, mxiter=100, rclose=1e-5, relax=0.97)
    # MODFLOW Link-MT3DMS Package Class (this is the package for solute transport)
    lmt = flopy.modflow.ModflowLmt(mf)
    # # MODFLOW output control package
    oc = flopy.modflow.ModflowOc(mf)

    mf.write_input()
    # RUN MODFLOW MODEL, set to silent=False to see output in terminal
    mf.run_model(silent=True)

# =============================================================================
# START CALLING MT3D PACKAGES AND RUN MODEL
# =============================================================================
    # RUN MT3dms solute tranport
    modelname_mt = dirname + '_mt'

    # MT3DMS Model Class
    # Input: modelname = 'string', namefile_ext = 'string' (Extension for the namefile (the default is 'nam'))
    # modflowmodelflopy.modflow.mf.Modflow = This is a flopy Modflow model object upon which this Mt3dms model is based. (the default is None)
    mt = flopy.mt3d.Mt3dms(modelname=modelname_mt, model_ws=model_ws,
                           exe_name=exe_name_mt, modflowmodel=mf)

    # Basic transport package class
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=icbund, prsity=prsity, sconc=0,
                             tunit=mt_tunit, lunit=mt_lunit, nper=nper, perlen=perlen_mt,
                             nprs=nprs, timprs=timprs)

    # mixelm is an integer flag for the advection solution option,
    # mixelm = 0 is the standard finite difference method with upstream or
    # central in space weighting.
    # mixelm = 1 is the forward tracking method of characteristics, this seems to result in minimal numerical dispersion.
    # mixelm = 2 is the backward tracking
    # mixelm = 3 is the hybrid method
    # mixelm = -1 is the third-ord TVD scheme (ULTIMATE)
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm)

    dsp = flopy.mt3d.Mt3dDsp(mt, al=al, trpt=trpt)

    # =============================================================================
    ## Note this additional line to call the  (package MT3D react)

    # set check if rc1 is a single value
    # if type(rc1)==np.ndarray: # if prsity is an array
    #     # if rc1 is an array add the dummy slices
    #     rc1_dummy_slice = np.zeros((hk_size[0], hk_size[1], 1))
    #     # concantenate dummy slice on rc1 array
    #     rc1 = np.concatenate((rc1_dummy_slice, rc1, rc1_dummy_slice), axis=2)

    # rct = flopy.mt3d.Mt3dRct(mt, isothm=0, ireact=1, igetsc=0, rc1=rc1)
    #if want to test for conservative tracer, input rc1 = 0.
    # =============================================================================
    # source and sink mixing package
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=spd_mt)
    gcg = flopy.mt3d.Mt3dGcg(mt)

    mt.write_input()
    fname = os.path.join(model_ws, 'MT3D001.UCN')
    # if os.path.isfile(fname):
    #     os.remove(fname)
    mt.run_model(silent=True)





















    # Extract concentration information
    # fname = os.path.join(model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.UcnFile(fname)
    timearray = np.array(ucnobj.get_times()) # convert to min
    # print(times)
    conc = ucnobj.get_alldata()

    # Extract head information
    fname = os.path.join(model_ws, modelname_mf+'.hds')
    hdobj = flopy.utils.HeadFile(fname)
    heads = hdobj.get_data()

    # set inactive cell pressures to zero, by default inactive cells have a pressure of -999
    # heads[heads < -990] = 0

    # convert heads to pascals
    if lenuni == 3: # centimeters
        pressures = heads/100*(1000*9.81)
    else: # elseif meters
        if lenuni == 2:
            pressures = heads*(1000*9.81)

    # crop off extra concentration slices
    conc = conc[:,:,:,1:-1]
    # MT3D sets the values of all concentrations in cells outside of the model
    # to 1E30, this sets them to 0
    conc[conc>2]=0
    # extract breakthrough curve data
    c_btc = np.transpose(np.sum(np.sum(conc[:, :, :, -1], axis=1), axis=1)/core_mask.sum())

    # calculate pressure drop
    p_inlet = pressures[:, :, 1]*core_mask
    p_inlet = np.mean(p_inlet[p_inlet>1])
    # print(p_inlet)
    p_outlet = pressures[:, :, -1]*core_mask
    p_outlet = np.mean(p_outlet[p_outlet>1])
    dp = p_inlet-p_outlet
    # crop off extra pressure slices
    pressures = pressures[:,:,1:-1]
    # print('Pressure drop: '+ str(dp/1000) + ' kPa')

    # calculate mean permeability from pressure drop
    # water viscosity
    mu_water = 0.00089 # Pa.s
    L = hk_size[2]*delc
    km2_mean = (q_total/mask_area)*L*mu_water/dp /(60*100**2)
    kD_mean = km2_mean/9.869233E-13 # Darcy
    hk_mean = km2_mean*(1000*9.81*100*60/8.9E-4) # cm/min
    # print('Core average perm: '+ str(km2_mean/9.869233E-13*1000) + ' mD')

    # Option to plot and calculate geometric mean to double check that core average perm in close
    geo_mean_K = np.exp(np.sum(np.log(raw_hk[raw_hk>0]))/len(raw_hk[raw_hk>0]))
    geo_mean_km2 = geo_mean_K/(1000*9.81*100*60/8.9E-4)
    # print('Geometric mean perm: ' + str(geo_mean_km2/9.869233E-13*1000) + ' mD')

    # Possible output: mf, mt, conc, timearray, km2_mean, pressures
    return mf, mt, conc, c_btc, timearray, hk_mean


# Much faster quantile calculation
def quantile_calc(btc_1d, timearray, quantile):
    # calculate cumulative amount of solute passing by location
    M0i = integrate.cumtrapz(btc_1d, timearray)
    # normalize by total to get CDF
    quant = M0i/M0i[-1]
    # calculate midtimes
    mid_time = (timearray[1:] + timearray[:-1]) / 2.0

    # now linearly interpolate to find quantile
    gind = np.argmax(quant > quantile)
    m = (quant[gind] - quant[gind-1])/(mid_time[gind] - mid_time[gind-1])
    b = quant[gind-1] - m*mid_time[gind-1]

    tau = (quantile-b)/m

    # plot check
    # xp = [mid_time[gind-1], mid_time[gind]]
    # plt.plot(mid_time, quant, '-o')
    # plt.plot(xp, m*xp+b, '-r')
    # plt.plot(tau, quantile, 'ok')
    return tau


# Function to calculate the quantile arrival time map
def flopy_arrival_map_function(conc, timearray, grid_size, quantile):
    # determine the size of the data
    conc_size = conc.shape

    # define area with hk values above zero
    core_mask = np.copy(conc[0,:,:,0])
    core_mask[core_mask<2] = 1
    core_mask[core_mask>2] = 0

    # MT3D sets the values of all concentrations in cells outside of the model
    # to 1E30, this sets them to 0
    conc[conc>2]=0

    # sum of slice concentrations for calculating inlet and outlet breakthrough
    oned = np.nansum(np.nansum(conc, 1), 1)

    # arrival time calculation in inlet slice
    tau_in = quantile_calc(oned[:,0], timearray, quantile)

    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[:,-1], timearray, quantile)

    # core length
    model_length = grid_size[2]*conc_size[3]
    # array of grid cell centers before interpolation
    z_coord_model = np.arange(grid_size[2]/2, model_length, grid_size[2])

    # Preallocate arrival time array
    at_array = np.zeros((conc_size[1], conc_size[2], conc_size[3]), dtype=float)

    for layer in range(0, conc_size[1]):
        for row in range(0, conc_size[2]):
            for col in range(0, conc_size[3]):
                # Check if outside core
                if core_mask[layer, row] > 0:
                    cell_btc = conc[:, layer, row, col]
                    # check to make sure tracer is in grid cell
                    if cell_btc.sum() > 0:
                        # call function to find quantile of interest
                        tau_vox = quantile_calc(cell_btc, timearray, quantile)
                        if tau_vox > 0:
                            at_array[layer, row, col] = tau_vox
                        else:
                            break # if tau can't be calculated then break the nested for loop and run a different model
            else:
                continue
            break
        else:
            continue
        break

    if tau_vox == 0: # these set a flag that is used to regenerate training realization
        at_array = 0
        at_array_norm = 0
        at_diff_norm = 0

    else:
        # v = (model_length-grid_size[2])/(tau_out - tau_in)
        # print('advection velocity: ' + str(v))

        # Normalize arrival times
        at_array_norm = (at_array-tau_in)/(tau_out - tau_in)

        # vector of ideal mean arrival time based average v
        at_ideal = z_coord_model/z_coord_model[-1]

        # Turn this vector into a matrix so that it can simply be subtracted from
        at_ideal_array = np.tile(at_ideal, (conc_size[1], conc_size[2], 1))

        # Arrival time difference map
        at_diff_norm = (at_ideal_array - at_array_norm)

        # Replace values outside model with zeros
        for col in range(0, conc_size[3]):
            at_array[:,:,col] = np.multiply(at_array[:,:,col], core_mask)
            at_array_norm[:,:,col] = np.multiply(at_array_norm[:,:,col], core_mask)
            at_diff_norm[:,:,col] = np.multiply(at_diff_norm[:,:,col], core_mask)

    # output options: at_array, at_array_norm, at_diff_norm
    return at_diff_norm
