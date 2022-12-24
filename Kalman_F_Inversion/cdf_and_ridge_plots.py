# -*- coding: utf-8 -*-
# import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt


from matplotlib import rc

rc('font',**{'family':'serif','serif':['Arial']})
fs = 9
plt.rcParams['font.size'] = fs

#number of ensemble members
ensemble_n = 400

#### plot breakthrough curves of voxels in fracture (as determined by binary matrix)
fig, axs = plt.subplots(2, 2, dpi=200, figsize=(8.5, 4.5))

sig = np.random.uniform(0.1,1,ensemble_n)
mu = 0 # mean
Hyb_Ens_dir_ini = os.path.join('.', 'Initial_Ens_Hybrid')
Hyb_Ens_dir_fin = os.path.join('.', 'Final_Ens_Hybrid')

# Berea Ensemble Loading
ini_B = []
fin_B = []

for i in range(ensemble_n):
    dir_ini = Hyb_Ens_dir_ini + '/Initial_Berea_Ens/Ens' + str(i) + '.csv'
    dir_fin = Hyb_Ens_dir_fin + '/Final_Berea_Ens/Ens' + str(i) + '.csv'

    ini_B.append(np.loadtxt(dir_ini , delimiter=',', dtype= np.float32))
    fin_B.append(np.loadtxt(dir_fin , delimiter=',', dtype= np.float32))

# Indiana Ensemble Loading
ini_I = []
fin_I = []

for i in range(ensemble_n):
    dir_ini = Hyb_Ens_dir_ini + '/Initial_Indiana_Ens/Ens' + str(i) + '.csv'
    dir_fin = Hyb_Ens_dir_fin + '/Final_Indiana_Ens/Ens' + str(i) + '.csv'

    ini_I.append(np.loadtxt(dir_ini , delimiter=',', dtype= np.float32))
    fin_I.append(np.loadtxt(dir_fin , delimiter=',', dtype= np.float32))

# Navajo Ensemble Loading
ini_N = []
fin_N = []

for i in range(ensemble_n):
    dir_ini = Hyb_Ens_dir_ini + '/Initial_Navajo_Ens/Ens' + str(i) + '.csv'
    dir_fin = Hyb_Ens_dir_fin + '/Final_Navajo_Ens/Ens' + str(i) + '.csv'

    ini_N.append(np.loadtxt(dir_ini , delimiter=',', dtype= np.float32))
    fin_N.append(np.loadtxt(dir_fin , delimiter=',', dtype= np.float32))

# Ketton Ensemble Loading
ini_K = []
fin_K = []

for i in range(ensemble_n):
    dir_ini = Hyb_Ens_dir_ini + '/Initial_Ketton_Ens/Ens' + str(i) + '.csv'
    dir_fin = Hyb_Ens_dir_fin + '/Final_Ketton_Ens/Ens' + str(i) + '.csv'

    ini_K.append(np.loadtxt(dir_ini , delimiter=',', dtype= np.float32))
    fin_K.append(np.loadtxt(dir_fin , delimiter=',', dtype= np.float32))

# Edwards Ensemble Loading
ini_E = []
fin_E = []

for i in range(ensemble_n):
    dir_ini = Hyb_Ens_dir_ini + '/Initial_Edwards_Ens/Ens' + str(i) + '.csv'
    dir_fin = Hyb_Ens_dir_fin + '/Final_Edwards_Ens/Ens' + str(i) + '.csv'

    ini_E.append(np.loadtxt(dir_ini , delimiter=',', dtype= np.float32))
    fin_E.append(np.loadtxt(dir_fin , delimiter=',', dtype= np.float32))

for i in range(0,ensemble_n):
    s = ini_B[i]
    s_fin = fin_B[i]

    # sort the data:
    s_sorted = np.sort(s)
    s_sorted_fin = np.sort(s_fin)
    data_sorted = s_sorted[0:12640]
    data_sorted_fin = s_sorted_fin[0:12640]

    # data_sorted_new = np.sort(snew)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)
    # plot data

    if i < ensemble_n-1:
        axs[0,0].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2)
        axs[0,0].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2)
    else:
        axs[0,0].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2, label='Initial Ensemble')
        axs[0,0].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2, label='Final Ensemble')
        legend = axs[0,0].legend(loc="upper left")
        legend.get_title().set_fontsize('3')

for tick in axs[0,0].xaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

for tick in axs[0,0].yaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

fig.suptitle('Permeability Perturbation [mD] (EnKF-CNN)')
# plt.xlabel('Permeability Perturbation')
axs[0,0].set_ylabel('CDF')
axs[0,0].set_title('Berea')

for i in range(0,ensemble_n):
    s = ini_K[i]
    s_fin = fin_K[i]

    # sort the data:
    s_sorted = np.sort(s)
    s_sorted_fin = np.sort(s_fin)
    data_sorted = s_sorted[0:12640]
    data_sorted_fin = s_sorted_fin[0:12640]

    # data_sorted_new = np.sort(snew)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)
    # plot data
    if i < ensemble_n-1:
        axs[0,1].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2)
        axs[0,1].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2)
    else:
        axs[0,1].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2, label='Initial Ensemble')
        axs[0,1].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2, label='Final Ensemble')
        legend = axs[0,1].legend(loc="upper left")
        legend.get_title().set_fontsize('3')


for tick in axs[0,1].xaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

for tick in axs[0,1].yaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

axs[0,1].set_title('Carbonate')

for i in range(0,ensemble_n):
    s = ini_N[i]
    s_fin = fin_N[i]

    # sort the data:
    s_sorted = np.sort(s)
    s_sorted_fin = np.sort(s_fin)
    data_sorted = s_sorted[0:12640]
    data_sorted_fin = s_sorted_fin[0:12640]

    # data_sorted_new = np.sort(snew)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)
    # plot data
    if i < ensemble_n-1:
        axs[1,0].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2)
        axs[1,0].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2)
    else:
        axs[1,0].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2, label='Initial Ensemble')
        axs[1,0].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2, label='Final Ensemble')
        legend = axs[1,0].legend(loc="upper left")
        legend.get_title().set_fontsize('3')

for tick in axs[1,0].xaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

for tick in axs[1,0].yaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

axs[1,0].set_title('Navajo')
axs[1,0].set_ylabel('CDF')

for i in range(0,ensemble_n):
    s = ini_I[i]
    s_fin = fin_I[i]

    # sort the data:
    s_sorted = np.sort(s)
    s_sorted_fin = np.sort(s_fin)
    data_sorted = s_sorted[0:12640]
    data_sorted_fin = s_sorted_fin[0:12640]

    # data_sorted_new = np.sort(snew)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data_sorted)) / (len(data_sorted) - 1)
    # plot data
    if i < ensemble_n-1:
        axs[1,1].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2)
        axs[1,1].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2)
    else:
        axs[1,1].plot(data_sorted, p, color='b', linewidth=0.5, alpha = 0.2, label='Initial Ensemble')
        axs[1,1].plot(data_sorted_fin, p, color='r', linewidth=0.5, alpha = 0.2, label='Final Ensemble')
        legend = axs[1,1].legend(loc="upper left")
        legend.get_title().set_fontsize('3')

for tick in axs[1,1].xaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

for tick in axs[1,1].yaxis.get_major_ticks():
    tick.label.set_fontsize(fs-2)

axs[1,1].set_title('Indiana')

fig.subplots_adjust(wspace=0.25, hspace=0.25)
plt.tight_layout()
fig.savefig('AWR_CDF.jpg', format='jpg', dpi=1200)




######################## RIDGE PLOTS ##################
import matplotlib.cm as cm

N = 5
nbin = 80
colors=cm.RdPu(np.linspace(0.3, 1, N))

Hyb_Ens_dir_fin = os.path.join('.', 'Final_Ens_Hybrid')
Ens_dir_fin = os.path.join('.', 'Final_Ens')


# Berea Ensemble Loading
hyb_Ber_f = []
Ber_f = []

for i in range(ensemble_n):
    hyb_fin = Hyb_Ens_dir_fin + '/Final_Berea_Ens/Ens' + str(i) + '.csv'
    fin = Ens_dir_fin + '/Final_Berea_Ens/Ens' + str(i) + '.csv'

    hyb_Ber_f.append(np.loadtxt(hyb_fin , delimiter=',', dtype= np.float32))
    Ber_f.append(np.loadtxt(fin , delimiter=',', dtype= np.float32))

# Indiana Ensemble Loading
hyb_Ind_f = []
Ind_f = []

for i in range(ensemble_n):
    hyb_fin = Hyb_Ens_dir_fin + '/Final_Indiana_Ens/Ens' + str(i) + '.csv'
    fin = Ens_dir_fin + '/Final_Indiana_Ens/Ens' + str(i) + '.csv'

    hyb_Ind_f.append(np.loadtxt(hyb_fin , delimiter=',', dtype= np.float32))
    Ind_f.append(np.loadtxt(fin , delimiter=',', dtype= np.float32))

# Navajo Ensemble Loading
hyb_Nav_f = []
Nav_f = []

for i in range(ensemble_n):
    hyb_fin = Hyb_Ens_dir_fin + '/Final_Navajo_Ens/Ens' + str(i) + '.csv'
    fin = Ens_dir_fin + '/Final_Navajo_Ens/Ens' + str(i) + '.csv'

    hyb_Nav_f.append(np.loadtxt(hyb_fin , delimiter=',', dtype= np.float32))
    Nav_f.append(np.loadtxt(fin , delimiter=',', dtype= np.float32))

# Ketton Ensemble Loading
hyb_Ket_f = []
Ket_f = []

for i in range(ensemble_n):
    hyb_fin = Hyb_Ens_dir_fin + '/Final_Ketton_Ens/Ens' + str(i) + '.csv'
    fin = Ens_dir_fin + '/Final_Ketton_Ens/Ens' + str(i) + '.csv'

    hyb_Ket_f.append(np.loadtxt(hyb_fin , delimiter=',', dtype= np.float32))
    Ket_f.append(np.loadtxt(fin , delimiter=',', dtype= np.float32))

# Edwards Ensemble Loading
hyb_Edw_f = []
Edw_f = []

for i in range(ensemble_n):
    hyb_fin = Hyb_Ens_dir_fin + '/Final_Edwards_Ens/Ens' + str(i) + '.csv'
    fin = Ens_dir_fin + '/Final_Edwards_Ens/Ens' + str(i) + '.csv'

    hyb_Edw_f.append(np.loadtxt(hyb_fin , delimiter=',', dtype= np.float32))
    Edw_f.append(np.loadtxt(fin , delimiter=',', dtype= np.float32))


fig2, axs2 = plt.subplots(N, figsize=(9, 9), constrained_layout=True)

Hyb_Ber_std = np.std(hyb_Ber_f, axis=0)
Ber_std = np.std(Ber_f, axis=0)
hist, be = np.histogram(Hyb_Ber_std, bins=nbin, density=False)
hist2, be2 = np.histogram(Ber_std, bins=nbin, density=False)
axs2[0].fill_between(be[:-1]+(be[1]- be[0])/2, hist, color=colors[2], alpha=0.5, label='Hybrid Final')
axs2[0].fill_between(be2[:-1]+(be2[1]- be2[0])/2, hist2, color=colors[4], alpha=0.5, label='EnKF Final')
axs2[0].set_title('Berea')
axs2[0].legend()
np.savetxt('./Hyb_Ber_std.csv', Hyb_Ber_std, delimiter=',')
np.savetxt('./Ber_std.csv', Ber_std, delimiter=',')

Hyb_Ket_std = np.std(hyb_Ket_f, axis=0)
Ket_std = np.std(Ket_f, axis=0)
hist, be = np.histogram(Hyb_Ket_std, bins=nbin, density=False)
hist2, be2 = np.histogram(Ket_std, bins=nbin, density=False)
axs2[1].fill_between(be[:-1]+(be[1]- be[0])/2, hist, color=colors[2], alpha=0.5, label='Hybrid Final')
axs2[1].fill_between(be2[:-1]+(be2[1]- be2[0])/2, hist2, color=colors[4], alpha=0.5, label='EnKF Final')
axs2[1].set_title('Ketton')
axs2[1].legend()
np.savetxt('./Hyb_Ket_std.csv', Hyb_Ket_std, delimiter=',')
np.savetxt('./Ket_std.csv', Ket_std, delimiter=',')

Hyb_Edw_std = np.std(hyb_Edw_f, axis=0)
Edw_std = np.std(Edw_f, axis=0)
hist, be = np.histogram(Hyb_Edw_std, bins=nbin, density=False)
hist2, be2 = np.histogram(Edw_std, bins=nbin, density=False)
axs2[2].fill_between(be[:-1]+(be[1]- be[0])/2, hist, color=colors[2], alpha=0.5, label='Hybrid Final')
axs2[2].fill_between(be2[:-1]+(be2[1]- be2[0])/2, hist2, color=colors[4], alpha=0.5, label='EnKF Final')
axs2[2].set_title('Edwards Brown')
axs2[2].legend()
np.savetxt('./Hyb_Edw_std.csv', Hyb_Edw_std, delimiter=',')
np.savetxt('./Edw_std.csv', Edw_std, delimiter=',')

Hyb_Nav_std = np.std(hyb_Nav_f, axis=0)
Nav_std = np.std(Nav_f, axis=0)
hist, be = np.histogram(Hyb_Nav_std, bins=nbin, density=False)
hist2, be2 = np.histogram(Nav_std, bins=nbin, density=False)
axs2[3].fill_between(be[:-1]+(be[1]- be[0])/2, hist, color=colors[2], alpha=0.5, label='Hybrid Final')
axs2[3].fill_between(be2[:-1]+(be2[1]- be2[0])/2, hist2, color=colors[4], alpha=0.5, label='EnKF Final')
axs2[3].set_title('Navajo')
axs2[3].legend()
np.savetxt('./Hyb_Nav_std.csv', Hyb_Nav_std, delimiter=',')
np.savetxt('./Nav_std.csv', Nav_std, delimiter=',')

Hyb_Ind_std = np.std(hyb_Ind_f, axis=0)
Ind_std = np.std(Ind_f, axis=0)
hist, be = np.histogram(Hyb_Ind_std, bins=nbin, density=False)
hist2, be2 = np.histogram(Ind_std, bins=nbin, density=False)
axs2[4].fill_between(be[:-1]+(be[1]- be[0])/2, hist, color=colors[2], alpha=0.5, label='Hybrid Final')
axs2[4].fill_between(be2[:-1]+(be2[1]- be2[0])/2, hist2, color=colors[4], alpha=0.5, label='EnKF Final')
axs2[4].set_title('Indiana')
axs2[4].legend()
np.savetxt('./Hyb_Ind_std.csv', Hyb_Ind_std, delimiter=',')
np.savetxt('./Ind_std.csv', Ind_std, delimiter=',')

axs2[4].set_xlabel('STD')
fig2.savefig('AWR_RIDGE.jpg', format='jpg', dpi=1200)

plt.show()
