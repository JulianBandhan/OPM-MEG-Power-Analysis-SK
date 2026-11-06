# -*- coding: utf-8 -*-
"""
Plot the MEG38 language atlas on the cortical surface: all regions in distinct
colours, then individual regions highlighted one at a time.

Created on Tue Mar 18 14:26:51 2025

@author: Julian Bandhan
"""
from functions.common_imports import np, plt, random, stats, scipy, pg, sns, pd, loadmat, Axes3D, image, Line2D
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2
import numpy as np

atlasimg = r"path/to/meg38_lang.nii.gz"            # atlas volume
label_file = r'path/to/meg38_lang.txt'             # region labels, one per line
fs_dir = r'path/to/freesurfer_subjects'            # FreeSurfer subjects directory
out_dir = r'path/to/output/nummed_meg38lang'       # where single-region plots are saved

# %% 1 - Qualitative colour map with 38 distinct colours

from matplotlib.colors import ListedColormap

values = np.linspace(0,1,38)  # one evenly spaced value per region (0 to 1)

# tab20b gives 20 colours; the remaining 18 come from tab20c
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in range(20)] + [plt.get_cmap('tab20c')(i) for i in range(18)]
custom_cmap = ListedColormap(colors)


# %% 2 - All regions, one colour each

reg_name = pd.read_csv(label_file, header=None, delimiter='\t')[0]

# Shift by one step (1/37) so each region falls in its own colour bin
data = values-0.02702703

fig1 = surface_brain_plotv3(
    make_atlas_nifti(image.load_img(atlasimg), data),
    fs_dir,
    surf = 'pial',
    cmap=custom_cmap,
    symmetric=False,
    cbar_label=f'')

# %% 3 - Single regions highlighted, one figure each

reg_name = pd.read_csv(label_file, header=None, delimiter='\t')[0]

for i in range(38,42):
    num = i+1

    # 1 for the current region, 0 everywhere else
    values = np.zeros(len(reg_name))
    values[i] = 1

    fig = surface_brain_plotv2(make_atlas_nifti(image.load_img(atlasimg), values),
                              fs_dir,
                              cmap = "Blues",
                              symmetric=False,
                              cbar_label=fr"{reg_name[i]}", datmin = 0, datmax = 1
                              )

    filepath = fr'{out_dir}/{i}_{reg_name[i]}.png'
    # Uncomment to save and close each figure
    #fig.savefig(filepath)
    #plt.close('all')