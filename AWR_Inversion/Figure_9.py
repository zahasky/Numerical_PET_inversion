import os
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_scatter_density

from matplotlib import rcParams
from matplotlib import gridspec
from matplotlib import cm
from sklearn.metrics import r2_score
from matplotlib.colors import LinearSegmentedColormap


def c_plot(fig, labeled, decoded, core_type, top_title, left_ax, bot_ax):
    x = labeled[labeled != 0]
    y = decoded[labeled != 0]

    # yellow_pink = LinearSegmentedColormap.from_list('yellow_pink', [
    #     (0, '#ffffff'),
    #     (1e-20, 'red'),
    #     (0.2, '#ff57bb'),
    #     (0.4, '#faaaae'),
    #     (0.6, '#fdf1d2'),
    #     (0.8, '#f8eaad'),
    #     (1, 'yellow'),
    # ], N=256)

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


    if top_title == True:
        plt.title(core_type + ', R2 = {:.3f}'.format(r2_score(x, y)),**font, fontweight="bold")
    else:
        plt.title('R2 = {:.3f}'.format(r2_score(x, y)),**font, fontweight="bold")

    ymax = np.max(y)
    ymin = np.min(y)
    plt.plot([ymin,ymax],[ymin,ymax], linewidth=0.5, color = 'black')

    if bot_ax == True:
        plt.xlabel('Experimental Arrival Time',**font)
    if left_ax == True:
        plt.ylabel('Modeled Arrival Time',**font)

    return bin


# Load 8 pairs of labeling and decoded permeability from the test set
num_plot = 15
res_dir = os.path.join('.', 'Ber_Cross')

# Labeling Permeability (lab_perm), Decoded Permeability (dec_perm)
lab_perm = []
dec_perm = []
core_names = ['Berea','Navajo','Indiana','Edwards','Ketton']

# Loading the data for plotting
for i in range(num_plot):
    lab_dir = res_dir + '/inp' + str(i) + '.csv'
    dec_dir = res_dir + '/dec' + str(i) + '.csv'

    lab = np.loadtxt(lab_dir , delimiter=',', dtype= np.float32)
    dec = np.loadtxt(dec_dir , delimiter=',', dtype= np.float32)
    lab_perm.append(lab)
    dec_perm.append(dec)

# Determine whether to highlight and adjust axis of certain graph
left_ax = False
bot_ax = False
top_title = False
core_type = ''

# Generate 15 cross-plots
font = {'family':'sans-serif', 'size':13}
plt.rc('font',**font)
fig1 = plt.figure(figsize=(18, 9))
gs=gridspec.GridSpec(3,5,left=0.06,bottom=0.06,right=0.92,top=0.965,wspace=0.275,hspace=0.28)
# gs=gridspec.GridSpec(4,5,left=0.06,bottom=0.25,right=0.92,top=0.75,wspace=0.25,hspace=0.25)

for j in range(num_plot):
    if j == 0 or j%5 == 0:
        left_ax = True
    if j > 9:
        bot_ax = True
    if j < 5:
        top_title = True
        core_type = core_names[j]

    ax = fig1.add_subplot(gs[j],projection='scatter_density')
    bin = c_plot(fig1, lab_perm[j], dec_perm[j], core_type, top_title, left_ax, bot_ax)

    left_ax = False
    bot_ax = False
    top_title = False

# LEFT BOTTOM WIDTH HEIGHT
cbar_ax = fig1.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = fig1.colorbar(bin, cax=cbar_ax)
cbar.set_label('Count in Bin', **font,fontsize=17)

plt.savefig('PC.png', dpi=500)
plt.show()
