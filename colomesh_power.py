# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:09:55 2025

@author: Julian Bandhan
"""
import importlib
from functions.common_imports import np, plt, random, stats, scipy, pg, sns, pd, loadmat, Axes3D, image, Line2D, MixedLM, sm
from functions.psd_plots import plot_stuff_vs_stuff, plot_stuff_vs_stuffv2, extract_band_psd
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2, make_4d_atlas_nifti
from functions.lme_stuff import lme_stuff, run_lme
import warnings
from statsmodels.stats.multitest import multipletests
import os


# Atlas images (MEG 38-region atlas)
atlasimg = r"path/to/MEG_atlas_38_regions_4D.nii.gz"
atlasimg3d = r"path/to/MEG_atlas_38_regions_3D.nii.gz"

plt.rcParams['font.family'] = 'Arial'

# Output directory for figures
base_dir = r'path/to/output/powerspec'
os.makedirs(base_dir, exist_ok=True)

######################################################################
#           Spectral parameterization: aperiodic slope and offset
######################################################################

# %% 1 - Load data
prime_data = loadmat(r'path/to/163ptp_powers_meg38_zscore.mat')['data']
subject_data = pd.read_excel(r'path/to/opm_rest_babyeinstein_statv6_1.xlsx')


# MATLAB 1x1 struct loads as a nested array; pull out the struct itself
if isinstance(prime_data, np.ndarray):
    prime_data = prime_data[0, 0]


# Indices of participants to exclude (empty = keep everyone)
drop_idx =  []
subject_data.drop(drop_idx)


# Subject IDs from the MATLAB struct (format depends on how the PSDs were exported)
subjects = prime_data['subject']
subjects = np.delete(subjects,drop_idx,axis = 0)

if isinstance(subjects, np.ndarray):
    if subjects.dtype.type is np.object_:  # MATLAB cell array -> list of strings
        subjects = [str(s[0]) for s in subjects]

# Strip leftover brackets/quotes from the IDs
subjects = [s.strip("[]'") for s in subjects]

# Build "subject-sess-XX-run-XX" visit IDs from the demographics sheet
subject_visitId = [
    fr"{subj.strip('[]\'')}-sess-{int(sess):02}-run-{int(run):02}"
    for subj, sess, run in zip(subject_data['subject'], subject_data['session'], subject_data['run'])
]

prime_sex = subject_data['sex']

# Frequency axis
prime_freq = prime_data['freq'][:,0]


# PSD array: (n_subjects, n_freqs, n_regions)
# n_freqs = 513 (1025 for 1 s windows); n_regions = 38 for the MEG38 atlas
prime_psd = prime_data['psd']
prime_psd = np.delete(prime_psd, drop_idx, axis = 0)

ages = prime_data['age'][:,0]
ages  = np.delete(ages, drop_idx, axis = 0)

# Age bins: ages <= 1 go in bin 1, otherwise floor(age)
age_bin = np.array(ages)
age_bin = np.where(age_bin <= 1, 1, np.floor(age_bin))



# Region labels (short and full names)
reg_name = pd.read_csv(r'path/to/meg38_labels.txt', header=None, delimiter='\t')[0]
reg_name_full = pd.read_csv(r'path/to/meg38_labels_full.txt', header=None, delimiter='\t')[0]

num_sub = len(subjects)
num_reg = len(reg_name)

# %% 2 - Spectral parameterization (FOOOF / specparam)
################################################
# Fit over the range set in fooof_settings['freq_range'].
# Gamma bands are kept outside the fit and use the raw PSD.
################################################

# Reloaded so edits to fooof_func are picked up without restarting the kernel
import importlib
import functions.fooof_func as fooof_func

importlib.reload(fooof_func)

from functions.fooof_func import run_specparam_batch, SpectralModel,run_fooof_batch
fooof_settings = {
    'freq_range': [3, 40],
    'peak_width_limits': [2, 4],
    'max_n_peaks': np.inf,
    'peak_threshold': 1,
    'min_peak_height': 0,
    'aperiodic_mode': 'fixed',
    'space': 'linear'
}

fooof_results = run_specparam_batch(prime_psd, prime_freq, num_sub, num_reg, fooof_settings)

fooof_periodic_psd = fooof_results['periodic']
fooof_aperiodic_psd = fooof_results['aperiodic']
fooof_spectrum_psd = fooof_results['spectrum']
# Aperiodic params: index 0 = offset, last index = exponent (slope).
# In 'knee' mode, index 1 is the knee.
fooof_params = fooof_results['params']
fooof_params[:,:,0] = 10**fooof_params[:,:,0]  # offset from log10 back to linear
fooof_rmse = fooof_results['r_squared']  # note: this holds R^2, not RMSE
fooof_models = fooof_results['models']
n_bins = fooof_results['n_bins']
freq_bins_low = fooof_results['freqs_used']  # frequencies within freq_range

# Subject-level averages across regions
offset_mean = np.mean(fooof_params[:,:,0],axis = 1)
slope_mean = np.mean(fooof_params[:,:,-1],axis = 1)


# %% 3 - Build dataframes


# Periodic (aperiodic-removed) PSD, low-frequency bins only
periodic_adj = fooof_periodic_psd[:, 0:46, :38]

# Frequency axis for the raw PSD
freq_bins = prime_freq[6:138]
num_freqs= len(freq_bins)

# Matching low-frequency bins from the parameterized range
freq_bins_low2 = freq_bins_low[0:46]
num_freqs_low = len(freq_bins_low2)


# Raw PSD, averaged across regions
df_psd_all = pd.DataFrame({
    'Subject': np.repeat(subjects, num_freqs),
    'Age': np.repeat(ages, num_freqs),
    'Age_bin':np.repeat(age_bin, num_freqs),
    'Frequency': np.tile(freq_bins,num_sub),
    'PSD': np.mean(prime_psd[:,6:138,:], axis = 2).flatten()
    })

# Low-frequency PSD: raw, periodic, aperiodic and full model fit (averaged across regions)
df_psd_low = pd.DataFrame({
    'Subject':np.repeat(subjects,num_freqs_low),
    'Age':np.repeat(ages,num_freqs_low),
    'Age_bin':np.repeat(age_bin,num_freqs_low),
    'Frequency':np.tile(freq_bins_low2,num_sub),
    'PSD_raw':np.mean(prime_psd[:,6:52,:], axis = 2).flatten(),
    'PSD':np.mean(periodic_adj,axis = 2).flatten(),
    'ap_PSD': np.mean(fooof_aperiodic_psd[:,0:46,:],axis = 2).flatten(),
    'Spectrum':np.mean(fooof_spectrum_psd[:,0:46,:],axis = 2).flatten()
    })


# Slope and offset per subject per region (long format, region-major order)
df_fooof_params_reg = pd.DataFrame({
    'Subject': np.tile(subjects,num_reg),
    'Subject_visit_ID':np.tile(subject_visitId,num_reg),
    'Sex': np.tile(prime_sex,num_reg),
    'Age': np.tile(ages,num_reg),
    'Region': np.repeat(reg_name, num_sub),
    'Slope': fooof_params[:,:,-1].T.flatten(),
    'Offset': fooof_params[:,:,0].T.flatten()
    })





# Flattened subject x region columns, reused by the regional dataframes below
subjects_flat = np.tile(subjects, num_reg)              # subject list repeated once per region
subject_ids_flat = np.tile(subject_visitId, num_reg)
sex_flat = np.tile(prime_sex, num_reg)
ages_flat = np.tile(ages, num_reg)
regions_flat = np.repeat(reg_name, num_sub)             # each region repeated once per subject

# Slope and offset per subject, averaged across regions
df_fooof_params_avg = pd.DataFrame({
    'Subject': subjects,
    'Age': ages,
    'Sex': prime_sex,
    'Age_bin':age_bin,
    'Slope': slope_mean,
    'Offset': offset_mean
})


###############################################################
# Band power: SUM powers for normalized timeseries data,
# MEAN powers for z-scored timeseries data.
###############################################################




# Mean band power per subject per region (periodic PSD for theta/alpha/beta, raw PSD for gamma)
theta_psd = np.mean(fooof_periodic_psd[:,0:5,:], axis = 1)
theta_psd_flat = theta_psd.T.flatten()
alpha_psd = np.mean(fooof_periodic_psd[:,5:17,:], axis = 1)
alpha_psd_flat = alpha_psd.T.flatten()
beta_psd = np.mean(fooof_periodic_psd[:,17:46,:], axis = 1)
beta_psd_flat = beta_psd.T.flatten()
lowgamma_psd = np.mean(prime_psd[:,52:94,:], axis = 1)
lowgamma_psd_flat = lowgamma_psd.T.flatten()
highgamma_psd = np.mean(prime_psd[:,111:137,:], axis = 1)
highgamma_psd_flat = highgamma_psd.T.flatten()

df_psd_reg_band = pd.DataFrame({
    'Subject': subjects_flat,
    'Subject_visit_ID': subject_ids_flat,
    'Region': regions_flat,
    'Age': ages_flat,
    'Sex':sex_flat,
    'Theta': theta_psd_flat,
    'Alpha': alpha_psd_flat,
    'Beta': beta_psd_flat,
    'LowGamma':lowgamma_psd_flat,
    'HighGamma':highgamma_psd_flat
    })


# Mean band power per subject per region, all from the raw PSD
theta_raw_psd = np.mean(prime_psd[:, 6:11, :], axis = 1)
theta_raw_psd_flat = theta_raw_psd.T.flatten()
alpha_raw_psd = np.mean(prime_psd[:, 11:23, :], axis = 1)
alpha_raw_psd_flat = alpha_raw_psd.T.flatten()
beta_raw_psd = np.mean(prime_psd[:, 21:52, :], axis = 1)
beta_raw_psd_flat = beta_raw_psd.T.flatten()
lowgamma_psd = np.mean(prime_psd[:,52:94,:], axis = 1)
lowgamma_psd_flat = lowgamma_psd.T.flatten()
highgamma_psd = np.mean(prime_psd[:,111:137,:], axis = 1)
highgamma_psd_flat = highgamma_psd.T.flatten()

df_raw_psd_reg_band = pd.DataFrame({
    'Subject': subjects_flat,
    'Subject_visit_ID': subject_ids_flat,
    'Region': regions_flat,
    'Age': ages_flat,
    'Sex': sex_flat,
    'Theta': theta_raw_psd_flat,
    'Alpha': alpha_raw_psd_flat,
    'Beta': beta_raw_psd_flat,
    'LowGamma':lowgamma_psd_flat,
    'HighGamma':highgamma_psd_flat
    })


# Long-format dataframe of every frequency bin within a band, per subject per region
def build_band_df(psd_band, freqs, band_label):
    n_freqs = len(freqs)

    return pd.DataFrame({
        'Subject': np.tile(np.repeat(subjects, n_freqs), num_reg),
        'Subject_visit_ID': np.tile(np.repeat(subject_visitId, n_freqs), num_reg),
        'Age': np.tile(np.repeat(ages, n_freqs), num_reg),
        'Region': np.repeat(reg_name, num_sub * n_freqs),
        'Frequency': np.tile(freqs, num_sub * num_reg),
        'PSD': psd_band.T.flatten(),
        'Band': band_label
    })

# Periodic PSD, per band: (n_subjects, n_band_freqs, n_regions)
theta_psd_ind = fooof_periodic_psd[:, 0:5, :]
alpha_psd_ind = fooof_periodic_psd[:, 5:17, :]
beta_psd_ind  = fooof_periodic_psd[:, 17:46, :]

theta_df = build_band_df(theta_psd_ind, freq_bins_low[0:5], 'Theta')
alpha_df = build_band_df(alpha_psd_ind, freq_bins_low[5:17], 'Alpha')
beta_df  = build_band_df(beta_psd_ind,  freq_bins_low[17:46], 'Beta')

# Raw PSD, per band: (n_subjects, n_band_freqs, n_regions)
theta_psd_raw_ind = prime_psd[:, 6:11, :]
alpha_psd_raw_ind = prime_psd[:, 11:23, :]
beta_psd_raw_ind  = prime_psd[:, 23:52, :]

theta_raw_df = build_band_df(theta_psd_raw_ind, prime_freq[6:11], 'Theta')
alpha_raw_df = build_band_df(alpha_psd_raw_ind, prime_freq[11:23], 'Alpha')
beta_raw_df  = build_band_df(beta_psd_raw_ind,  prime_freq[23:52], 'Beta')

# %% 4 - Age x frequency power surface (linear interpolation)

from scipy.interpolate import griddata
import numpy as np
from matplotlib.colors import Normalize

# Flatten to scattered (age, frequency, power) samples
n_subj, n_freq = num_sub, num_freqs_low
Ages_scatter  = np.repeat(ages, n_freq)
Freqs_scatter = np.tile(freq_bins_low2, n_subj)
Power_scatter = np.mean(periodic_adj, axis=2).reshape(-1)

# Regular grid (overwritten by the 256x256 grid below)
Age_grid  = np.linspace(Ages_scatter.min(), Ages_scatter.max(), 400)
Freq_grid = np.linspace(Freqs_scatter.min(), Freqs_scatter.max(), 400)


# float32 to halve memory use
pts  = np.column_stack([Ages_scatter.astype(np.float32),
                        Freqs_scatter.astype(np.float32)])
vals = Power_scatter.astype(np.float32)

# 256x256 interpolation grid
Age_grid  = np.linspace(ages.min(), ages.max(), 256).astype(np.float32)
Freq_grid = np.linspace(freq_bins_low2.min(), freq_bins_low2.max(), 256).astype(np.float32)
A, F = np.meshgrid(Age_grid, Freq_grid)

Z = griddata(pts, vals, (A, F), method='linear')
# Light smoothing
Z = gaussian_filter(Z, sigma=(0.8, 0.8)).astype(np.float32)

norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))

plt.figure(figsize=(9, 4.8))
pc = plt.pcolormesh(A, F, Z, shading='gouraud',
                    cmap='turbo', norm=norm)
plt.colorbar(pc, pad=0.02, label='Power (Hz$^{-1}$)')
plt.xlabel('Age (months)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()


# %% 5 - LOWESS helper

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from numpy.random import default_rng

def lowess_fit_at_x(x, y, frac=0.25, it=1):
    """
    Return LOWESS-fitted values at the original x points.
    statsmodels returns values sorted by x, so they are mapped back to the input order.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        # Too few points to fit; return NaNs of the same length
        yhat = np.full_like(y, np.nan, dtype=float)
        return yhat

    lo = lowess(y[m], x[m], frac=frac, it=it, return_sorted=True)  # (k, 2), sorted by x
    xs, ys = lo[:, 0], lo[:, 1]

    # Interpolate back onto the original x (NaN positions stay NaN)
    yhat = np.full_like(y, np.nan, dtype=float)
    yhat[m] = np.interp(x[m], xs, ys, left=np.nan, right=np.nan)
    return yhat



# %% 6 - LOWESS surface and cluster permutation functions
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter, label
from matplotlib.colors import Normalize
from numpy.random import default_rng

def lowess_surface(ages, psd, Age_grid, Freq_grid, frac=0.01, it=1, blur_sigma=(0.8,0.2)):
    """
    LOWESS fit of power vs age, separately for each frequency, evaluated on Age_grid.
    Returns an (n_freqs, n_ages) surface, optionally Gaussian-smoothed.
    """
    nF = len(Freq_grid)
    Z = np.empty((nF, len(Age_grid)), dtype=np.float32)
    for i in range(nF):
        y = psd[:, i]
        m = np.isfinite(ages) & np.isfinite(y)
        if m.sum() < 5:
            Z[i,:] = np.nan
            continue
        lo = lowess(y[m], ages[m], frac=frac, it=it, return_sorted=True)
        xs, ys = lo[:,0], lo[:,1]
        Z[i,:] = np.interp(Age_grid, xs, ys, left=np.nan, right=np.nan)
    if blur_sigma is not None:
        Z = gaussian_filter(Z, sigma=blur_sigma)
    return Z

def cluster_perm_lowess(ages, psd, Age_grid, Freq_grid, n_perm=1000, frac=0.25, it=1,
                        blur_sigma=(0.8,0.2), alpha=0.05, random_state=123, verbose_every=100):
    """
    Two-sided cluster-mass permutation test on the LOWESS surface (ages shuffled).
    Returns the observed surface, a boolean mask of significant clusters, and the pixel threshold.
    Not used in the current analysis.
    """
    rng = default_rng(random_state)
    # Observed surface
    Z_obs = lowess_surface(ages, psd, Age_grid, Freq_grid, frac, it, blur_sigma)

    nF, nA = Z_obs.shape
    idx = np.arange(len(ages))

    # Pass 1: null mean and SD per pixel, used to z-score
    mean_acc = np.zeros_like(Z_obs, dtype=np.float64)
    sq_acc   = np.zeros_like(Z_obs, dtype=np.float64)
    for b in range(n_perm):
        rng.shuffle(idx)
        Zb = lowess_surface(ages[idx], psd, Age_grid, Freq_grid, frac, it, blur_sigma)
        np.nan_to_num(Zb, copy=False)  # NaNs treated as 0 in the moments
        mean_acc += Zb
        sq_acc   += Zb*Zb
    null_mean = mean_acc / n_perm
    null_std  = np.sqrt(np.maximum(sq_acc / n_perm - null_mean**2, 1e-12))

    # z-score the observed surface
    Zz_obs = (Z_obs - null_mean) / null_std

    # Pass 2: null distribution of max |z| per permutation -> FWE pixel threshold
    maxabs = np.empty(n_perm, dtype=np.float32)
    for b in range(n_perm):
        rng.shuffle(idx)
        Zb = lowess_surface(ages[idx], psd, Age_grid, Freq_grid, frac, it, blur_sigma)
        Zz = (Zb - null_mean) / null_std
        maxabs[b] = np.nanmax(np.abs(Zz))
        if verbose_every and ((b+1)%verbose_every==0): print(f"[perm {b+1}/{n_perm}]")
    thr = np.quantile(maxabs, 1 - alpha)

    # Two-sided supra-threshold mask
    supra = np.abs(Zz_obs) >= thr

    # Label observed clusters (4-connectivity); cluster mass = sum of |z| within the cluster
    lab, nlab = label(supra)
    masses_obs = []
    for k in range(1, nlab+1):
        masses_obs.append(np.nansum(np.abs(Zz_obs)[lab==k]))
    masses_obs = np.array(masses_obs, dtype=np.float64)

    # Pass 3: null distribution of max cluster mass
    max_mass_null = np.zeros(n_perm, dtype=np.float64)
    for b in range(n_perm):
        rng.shuffle(idx)
        Zb = lowess_surface(ages[idx], psd, Age_grid, Freq_grid, frac, it, blur_sigma)
        Zz = (Zb - null_mean) / null_std
        supra_b = np.abs(Zz) >= thr
        lab_b, nlab_b = label(supra_b)
        if nlab_b == 0:
            max_mass_null[b] = 0.0
        else:
            max_mass_null[b] = np.max([np.nansum(np.abs(Zz)[lab_b==k]) for k in range(1, nlab_b+1)])

    # Keep observed clusters with cluster-wise p < alpha (FWER controlled)
    sig_mask = np.zeros_like(supra, dtype=bool)
    if masses_obs.size > 0:
        for k in range(1, nlab+1):
            mass_k = np.nansum(np.abs(Zz_obs)[lab==k])
            p_k = (1 + np.sum(max_mass_null >= mass_k)) / (1 + n_perm)
            if p_k < alpha:
                sig_mask |= (lab == k)

    return Z_obs, sig_mask, thr

# %% 7 - LOWESS-smoothed age x frequency power surface (raw PSD)

from matplotlib.colors import Normalize

n_subj, n_freq = num_sub, num_freqs_low
psd_vals = np.mean(prime_psd[:,6:69,:], axis = 2)         # (n_subjects, n_freqs), averaged across regions
#psd_vals = np.mean(periodic_adj,axis = 2)               # alternative: periodic PSD

psd = psd_vals

Age_grid  = np.linspace(ages.min(), ages.max(), 400)
Freq_grid = freq_bins_low2.copy()

Z_obs = lowess_surface(ages, psd_vals, Age_grid, Freq_grid, frac = 0.25)

A, F = np.meshgrid(Age_grid, Freq_grid)
norm = Normalize(vmin=np.nanmin(Z_obs), vmax=np.nanmax(Z_obs))

plt.figure(figsize=(18,11))
pc = plt.pcolormesh(A, F, Z_obs, shading='gouraud',
                    cmap='turbo', norm=norm)

cbar = plt.colorbar(pc, pad=0.02)
cbar.set_label('Power (Hz$^{-1}$)', fontsize=54)
cbar.ax.tick_params(labelsize=48)

plt.xlabel('Age (years)', fontsize=54)
plt.ylabel('Frequency (Hz)', fontsize=54)

plt.tick_params(axis='x', labelsize=48)
plt.tick_params(axis='y', labelsize=48)

plt.xticks([1,2,3,4,5])
plt.yticks([5,10,15,20,25])

plt.tight_layout()


filepath = os.path.join(base_dir,fr'power_colormesh_smooth.png')
plt.savefig(filepath)
plt.show()