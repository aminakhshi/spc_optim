import os
import datetime
import pickle
import random
import numpy as np
import pandas as pd
from scipy import spatial
from scipy import stats
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skspatial.objects import Points
from skspatial.objects import Sphere
from skspatial.plotting import plot_3d

from common.visualizations import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontprops = matplotlib.font_manager.FontProperties(size=10)
cm = 1/2.54



class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def read_files(folder_path, file_type : str = '.mat'):
    """
    This function is designed to accept a folder path and return a list of all data files within that folder
    :param folder_path: path to a folder
    :return: list of all desired files with full path
    """
    data_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_type):
                data_list.append(os.path.join(root, file))
    return data_list

data_path = os.path.join(os.getcwd(),'Recording Data') 
date = datetime.datetime.now().strftime('%Y%m%d')
results_path = os.path.join(os.getcwd(),'result',f'{date}')
os.makedirs(results_path, exist_ok=True)
with cd(data_path):
    files_list = read_files(data_path, file_type='.pkl')
    for fname in files_list:
        file_id = os.path.split(fname)[-1]
        if file_id.startswith('spiketimes'):
            spiketimes = pickle.load(open(fname, 'rb'))
        elif file_id.startswith('block_cell_id'):
            cb_ids = pickle.load(open(fname, 'rb'))
        elif file_id.startswith('serotonin_cell_id'):
            cs_ids = pickle.load(open(fname, 'rb'))
        else:
            raise TypeError(f"{file_id} is not a data file")
            

def get_isi(spiketimes, bins= None, return_firing=True, duration=None, dtype= None):
    if bins is None:
        bins=np.arange(0, 1, 0.002)
    if return_firing == True:
        if not duration:
            duration = 100
    if not dtype:
        dtype = type(spiketimes)
    if dtype == dict:
        isi_data = {'block':[],
                    'control':[],
                    'serotonin': []}
        if return_firing == True:
            mean_firing = {'block':[],
                        'control':[],
                        'serotonin': []}
        for key, val in spiketimes.items():
            dtype_val = type(val)
            if dtype_val == list:
                for f_val in val:
                    hist, edge = np.histogram(np.diff(f_val), bins=bins, density=True)
                    isi_data[key].append(hist)
                    if return_firing == True:
                        mean_firing[key].append(len(f_val)/duration)
    elif dtype == list:
        isi_data = []
        if return_firing == True:
            mean_firing = []
        for f_val in spiketimes:
            hist, edge = np.histogram(np.diff(f_val), bins=bins, density=True)
            isi_data.append(hist)
            if return_firing == True:
                mean_firing.append(len(f_val)/duration)
    elif dtype == numpy.ndarray:
        isi_data, edge = np.histogram(np.diff(f_val), bins=bins, density=True)
        if return_firing == True:
            mean_firing = len(isi_data)/duration
    
    if return_firing == True:
        return isi_data, mean_firing
    else:
        return isi_data



bins = np.arange(0, 1, 0.002)

isi_data, mean_firing = get_isi(spiketimes, bins, return_firing=True)  
                
control = np.array(isi_data['control'])
control[np.where(control==0)]=1e-12

block = np.array(isi_data['block'])
block[np.where(block==0)]=1e-12

serotonin = np.array(isi_data['serotonin'])
serotonin[np.where(serotonin==0)]=1e-12

pca_c = PCA(n_components=20).fit(control)
pca_control_b = pca_c.transform(control[cb_ids,:])
pca_control_s = pca_c.transform(control[cs_ids,:])
pca_block = pca_c.transform(block)
pca_serotonin = pca_c.transform(serotonin)


# =============================================================================
# 
# =============================================================================
selected_cells = [68, 104, 14, 26, 18, 47, 54, 84, 137, 50, 66, 76, 83, 36, 70, 12, 58, 65, 39, 46]
neuralData = [spiketimes['control'][cell_id] for cell_id in selected_cells]
sample_fr = [mean_firing['control'][cell_id] for cell_id in selected_cells]
fr_args = np.argsort(sample_fr)
sorted_samples = [neuralData[cell_id] for cell_id in fr_args]
lineoffsets = [i/8 for i in range(len(neuralData))]
fig, ax = plt.subplots(figsize=(12*cm, 8*cm))
cs = 0
for i in range(len(sorted_samples)):
    ss = ax.eventplot(sorted_samples[i], linelengths = 0.05, lineoffsets = lineoffsets[i],
                       linestyles='solid',  colors='#08306b', linewidths=1) 
    for k in ss:
        k._capstyle = 'round'
    cs += 1
    ax.set_xlim(18,23)
xtick = ax.get_xticks()
ax.set_xticklabels([str(xi) for xi in range(len(xtick))])  

scalebar = AnchoredSizeBar(ax.transData,
                            0.5, '500 ms', 'lower left', 
                            pad=-1.0,
                            color='black',
                            frameon=False,
                            size_vertical=0.01,
                            fontproperties=fontprops)
ax.add_artist(scalebar)
for key, spine in ax.spines.items():
    spine.set_visible(False)
ax.axes.xaxis.set_ticklabels([])
plt.axis('off')
plt.savefig(os.path.join(results_path,f'raster_data.pdf'), dpi=300, facecolor='w', bbox_inches='tight')
plt.close()

# =============================================================================
# 
# =============================================================================
def bootsrap(data, bins = bins, nwin=60, step=10, tdur=100, mode='hist', kind='time'):
    if mode=='hist':
        new_hist = []
        last = int(tdur+step-nwin)
        for ns in range(0,last,step):
            temp = data[(ns<=data) & (data<ns+nwin)]
            hist, edge = np.histogram(np.diff(temp), bins=bins)
            unity_hist = hist / hist.sum()
            new_hist.append(unity_hist)
        new_data = new_hist
    return np.array(new_data)


def histogram_plot(arr1, ax, cutoff = 199, mode=None, cell_id=None):
    arr1 = arr1[:, :cutoff]
    mean_arr1 = np.mean(arr1, axis=0)*100
    std_arr1 = np.std(arr1, axis=0)*100
    isi = bins[:cutoff]
    ax.plot(isi, mean_arr1, color='black', lw=0.8, label='iv')
    if mode == 'b':
        color = 'orange'
    elif mode == 'c':
        color = '#08306b'
    elif mode == 's':
        color = 'red'
    ax.fill_between(isi, mean_arr1-std_arr1, mean_arr1+std_arr1, color=color)
    ax.set_xscale('log')
    ax.set_ylim(0, 25)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend(loc='upper right', fontsize = 10)


fig, axs = plt.subplots(nrows=1, ncols=4, constrained_layout=True, figsize=(17*cm,4*cm))
for cell_id, ax in zip([104,47,50,83], axs.flat):
    isi_boot_data = bootsrap(spiketimes['control'][cell_id])
    isi_boot_data[np.where(isi_boot_data==0)]=1e-12
    histogram_plot(isi_boot_data, ax, cutoff=199,  mode='c', cell_id= cell_id)
plt.savefig(os.path.join(results_path,f"data_hits.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 
# =============================================================================
def get_sphere(data):
    x_data = data[:,:3]
    xdata = []
    ydata = []
    zdata = []
    for item in range(len(x_data)):
        xdata.append(x_data[item, 0])
        ydata.append(x_data[item, 1])
        zdata.append(x_data[item, 2])
    points = Points(np.column_stack((xdata, ydata, zdata)))
    sphere_data = Sphere.best_fit(points)
    return sphere_data


fig = plt.figure(figsize=(11.5*cm, 8*cm))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter3D(pca_control_b[:, 0] , pca_control_b[:, 1] , pca_control_b[:, 2],
                s=50, marker = 'o', color = '#08306b', label = f'control')
sphere_bc = get_sphere(pca_control_b)
# sphere_bc.plot_3d(ax, alpha=0.2, color='dodgerblue')
ax.set_ylim(-60,40)
ax.set_xlim(-30,175)
ax.set_zlim(-40,80)
# ax.azim = -49
# # ax.dist = 10
# ax.elev = 13
ax.set_xlabel('PC 1 (%{:.2f})'.format(pca_c.explained_variance_ratio_[0]*100), fontsize=12)
ax.set_ylabel('PC 2 (%{:.2f})'.format(pca_c.explained_variance_ratio_[1]*100), fontsize=12)
ax.set_zlabel('PC 3 (%{:.2f})'.format(pca_c.explained_variance_ratio_[2]*100), fontsize=12)
# ax.grid(False)


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(pca_block[:, 0] , pca_block[:, 1] , pca_block[:, 2],
                s=50, marker = '^', color = 'orange', label = f'block')
# sphere_block = get_sphere(pca_block)
# sphere_block.plot_3d(ax, alpha=0.2, color='orange')
ax.set_ylim(-60,40)
ax.set_xlim(-30,175)
ax.set_zlim(-40,80)
# ax.azim = -49
# # ax.dist = 10
# ax.elev = 13
# ax.set_xlabel('PC 1 (%{:.2f})'.format(pca_c.explained_variance_ratio_[0]*100), fontsize=12)
# ax.set_ylabel('PC 2 (%{:.2f})'.format(pca_c.explained_variance_ratio_[1]*100), fontsize=12)
# ax.set_zlabel('PC 3 (%{:.2f})'.format(pca_c.explained_variance_ratio_[2]*100), fontsize=12)
# ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(results_path,"scatter3d_control_block.pdf"), bbox_inches='tight')
plt.close()

# =============================================================================
# 
# =============================================================================
fig = plt.figure(figsize=(11.5*cm, 8*cm))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter3D(pca_control_s[:, 0] , pca_control_s[:, 1] , pca_control_s[:, 2],
                s=50, marker = 'o', color = '#08306b', label = f'control')
# sphere_bc = get_sphere(pca_control_s)
# sphere_bc.plot_3d(ax, alpha=0.2, color='#08306b')
ax.set_xlim(-40,120)
ax.set_ylim(-120,20)
ax.set_zlim(-40,80)
ax.azim = -49
# ax.dist = 10
ax.elev = 13
# ax.set_xlabel('PC 1 (%{:.2f})'.format(pca_c.explained_variance_ratio_[0]*100), fontsize=12)
# ax.set_ylabel('PC 2 (%{:.2f})'.format(pca_c.explained_variance_ratio_[1]*100), fontsize=12)
# ax.set_zlabel('PC 3 (%{:.2f})'.format(pca_c.explained_variance_ratio_[2]*100), fontsize=12)
# ax.grid(False)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(pca_serotonin[:, 0] , pca_serotonin[:, 1] , pca_serotonin[:, 2],
                s=50, marker = '^', color = 'red', label = f'serotonin')
# sphere_block = get_sphere(pca_serotonin)
# sphere_block.plot_3d(ax, alpha=0.2, color='red')
ax.set_xlim(-40,120)
ax.set_ylim(-120,20)
ax.set_zlim(-40,80)
ax.azim = -49
# ax.dist = 10
ax.elev = 13
# ax.set_xlabel('PC 1 (%{:.2f})'.format(pca_c.explained_variance_ratio_[0]*100), fontsize=12)
# ax.set_ylabel('PC 2 (%{:.2f})'.format(pca_c.explained_variance_ratio_[1]*100), fontsize=12)
# ax.set_zlabel('PC 3 (%{:.2f})'.format(pca_c.explained_variance_ratio_[2]*100), fontsize=12)
# ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(results_path,"scatter3d_control_serotonin.pdf"), bbox_inches='tight')
plt.close()
# =============================================================================
# 
# =============================================================================

def cluster_distances(arr1, arr2 = None, k = None):
    if not k:
        k=1
    if not arr2:
        dist = spatial.distance.cdist(arr1, arr1)
        updist = dist[np.triu_indices_from(dist, k=k)]
    else:
        dist = spatial.distance.cdist(arr1, arr2)
        updist = dist[np.triu_indices_from(dist, k=k)]
    return np.mean(updist)

vol_dist_stats_block = []
vol_dist_stats_serotonin = []
for i in range(10):
    random.seed(i)
    sample_selections_block = random.sample(range(len(block)), 30)
    bids = [cb_ids[x] for x in sample_selections_block]
    sample_selections_serotonin = random.sample(range(len(serotonin)), 20)
    sids = [cs_ids[x] for x in sample_selections_serotonin]
    pc_cb = pca_c.transform(control[bids,:])
    vol_cb = spatial.ConvexHull(pc_cb[:, :3])
    pc_b = pca_c.transform(block[sample_selections_block,:])
    vol_b = spatial.ConvexHull(pc_b[:, :3])
    pc_cs = pca_c.transform(control[sids,:])
    vol_cs = spatial.ConvexHull(pc_cs[:, :3])
    pc_s = pca_c.transform(serotonin[sample_selections_serotonin,:])
    vol_s = spatial.ConvexHull(pc_s[:, :3])
    volume_cbl = vol_b.volume/vol_cb.volume
    volume_csl = vol_s.volume/vol_cs.volume
    dist_cbl = cluster_distances(pc_b[:, :3])/cluster_distances(pc_cb[:, :3])
    dist_csl = cluster_distances(pc_s[:, :3])/cluster_distances(pc_cs[:, :3])
    vol_dist_stats_block.append([vol_cb.volume, vol_b.volume, cluster_distances(pc_cb[:, :3]), cluster_distances(pc_b[:, :3])])
    vol_dist_stats_serotonin.append([vol_cs.volume, vol_s.volume, cluster_distances(pc_cs[:, :3]), cluster_distances(pc_s[:, :3])])

vol_stats_block = np.array(vol_dist_stats_block)[:,[0,1]]
dist_stats_block = np.array(vol_dist_stats_block)[:,[2,3]]
vol_stats_serotonin = np.array(vol_dist_stats_serotonin)[:,[0,1]]
dist_stats_serotonin = np.array(vol_dist_stats_serotonin)[:,[2,3]]

block_cond = []
block_cond.append(vol_stats_block)
block_cond.append(dist_stats_block)

serotonin_cond = []
serotonin_cond.append(vol_stats_serotonin)
serotonin_cond.append(dist_stats_serotonin)


fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(4.5*cm, 8*cm))
for sts, ax, kind in zip(block_cond, axs.flat, ['volume %', 'distance %']):
    stat, p_val = stats.ttest_ind(sts[:,0], sts[:,1])
    sts[:,1] = sts[:,1]/np.median(sts[:,0])
    sts[:,0] = sts[:,0]/np.median(sts[:,0])
    ax.boxplot(sts)
    ax.set_ylabel(kind, fontsize=12)
    ax.set_yticks([0,0.4,0.8,1.2], [0,40,80,120],
                      fontsize=10)
    if kind == 'volume %':
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        print(f"volume stat test: *p = {p_val}")
    if kind == 'distance %':
        ax.set_xticks([1, 2], ['control', 'block'], rotation = 20,
                      fontsize=12)
        print(f"distance stat test: **p = {p_val}")
plt.savefig(os.path.join(results_path,f"ratios_block.pdf"), dpi=300, bbox_inches='tight')


fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(4.5*cm, 8*cm))
for sts, ax, kind in zip(serotonin_cond, axs.flat, ['% volume', '% distance']):
    stat, p_val = stats.ttest_ind(sts[:,0], sts[:,1])
    sts[:,1] = sts[:,1]/np.median(sts[:,0])
    sts[:,0] = sts[:,0]/np.median(sts[:,0])
    ax.boxplot(sts)
    ax.set_ylabel(kind, fontsize=12)
    if kind == '% volume':
        ax.set_yticks([0,1,2,3,4,5], [0,100,200,300,400,500],
                          fontsize=10)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        print(f"volume stat test: *p = {p_val}")

    if kind == '% distance':
        ax.set_xticks([1, 2], ['control', 'serotonin'], rotation = 20,
                      fontsize=12)
        ax.set_yticks([0,0.5,1,1.5,2,2.5], [0,50,100,150,200,250],
                          fontsize=10)
        print(f"distance stat test: **p = {p_val}")
plt.savefig(os.path.join(results_path,f"ratios_serotonin.pdf"), dpi=300, bbox_inches='tight')
