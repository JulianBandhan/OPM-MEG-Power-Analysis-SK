# -*- coding: utf-8 -*-
"""
Trial counts, bad channels and head motion: descriptive stats and age effects.

Created on Wed Dec 18 14:41:35 2024 (trial plots)
Updated on Tue Sep 23 16:03:39 2025 (summary stats)
@author: Julian Bandhan

Sections
    1 - Descriptive summary (counts, mean ± SD/SE)
    2 - Trials / HM vs age with Pearson correlation (scan-level, with QC thresholds)
    3 - Trials / HM vs age with MixedLM: y ~ Age + (1|Subject) (no QC thresholds)

Expected columns in the demographics sheet:
    subject, session, sex, age, age_bin, trials, original_trials, num_bad_channels, HM
'trials' = trials retained after preprocessing; 'original_trials' = trials before rejection.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


# --------------------------------------------------------------------------
# Configuration
demo_path = r'path/to/opm_rest_babyeinstein_v7stat.xlsx'
use_std = True        # True = mean ± SD, False = mean ± SE
# --------------------------------------------------------------------------

# %% 0 - Load data

ptp = pd.read_excel(demo_path, sheet_name='Sheet1').copy()

# Force key columns to numeric (blank or text cells become NaN)
for col in ['age', 'trials', 'original_trials', 'num_bad_channels', 'HM', 'age_bin']:
    if col in ptp.columns:
        ptp[col] = pd.to_numeric(ptp[col], errors='coerce')


# %% 1 - Descriptive summary

df = ptp.copy()

# One row per subject, for per-subject attributes such as sex
subj_info = df.drop_duplicates('subject')

# --- Counts ---
total_scans = len(df)                           # each row is one scan
total_unique_subjects = subj_info['subject'].nunique()
num_males = subj_info.loc[subj_info['sex'] == 'M', 'subject'].nunique()
num_females = subj_info.loc[subj_info['sex'] == 'F', 'subject'].nunique()

def spread(series: pd.Series) -> float:
    """Return SD or SE depending on the use_std flag."""
    s = series.std(ddof=1)
    if use_std:
        return s
    else:
        n = series.count()
        return s / np.sqrt(n)

# --- Scan-level means ± spread ---
avg_age = df['age'].mean()
spread_age = spread(df['age'])

avg_trials = df['trials'].mean()
spread_trials = spread(df['trials'])

avg_bad_channels = df['num_bad_channels'].mean()
spread_bad_channels = spread(df['num_bad_channels'])

avg_head_motion = df['HM'].mean()
spread_head_motion = spread(df['HM'])

# --- Trials per age bin ---
if 'age_bin' in df and df['age_bin'].notna().any():
    def agg_func(x):
        return pd.Series({
            'avg_trials': x.mean(),
            'spread_trials': spread(x),
            'count': x.count()
        })
    avg_trials_by_age_bin = (
        df.groupby('age_bin', dropna=True)['trials']
          .apply(agg_func)
          .sort_index()
    )
else:
    avg_trials_by_age_bin = pd.DataFrame(columns=['avg_trials', 'spread_trials', 'count'])

# --- Trials per participant (mean across each participant's scans) ---
trials_by_subject = df.groupby('subject')['trials'].mean()
avg_trials_per_ptp = trials_by_subject.mean()
spread_trials_per_ptp = spread(trials_by_subject)

# --- Print summary ---
label = "std" if use_std else "SE"
summary = {
    'total_scans': total_scans,
    'unique_subjects': total_unique_subjects,
    'unique_males': num_males,
    'unique_females': num_females,
    f'avg_age_yrs (± {label})': f"{avg_age:.2f} ± {spread_age:.2f}",
    f'avg_trials_per_scan (± {label})': f"{avg_trials:.2f} ± {spread_trials:.2f}",
    f'avg_trials_per_subject (± {label})': f"{avg_trials_per_ptp:.2f} ± {spread_trials_per_ptp:.2f}",
    f'avg_bad_channels (± {label})': f"{avg_bad_channels:.2f} ± {spread_bad_channels:.2f}",
    f'avg_head_motion (± {label})': f"{avg_head_motion:.2f} ± {spread_head_motion:.2f}",
}

print(pd.Series(summary))
print("\nAverage trials per age group:\n", avg_trials_by_age_bin)

# --- Sessions per subject ---
sessions_per_subject = df.groupby('subject')['session'].nunique()
session_summary = sessions_per_subject.value_counts().sort_index()
print("\nSessions per subject:\n", session_summary)


# %% 2.0 - Pearson section: per-age-bin subsets with QC thresholds

# Scans per age bin, keeping 30 < trials < 200 and < 15 bad channels
trial_nums = lambda x: ptp.loc[(ptp['age_bin']==x) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'trials']]
trial_nums_orig = lambda x: ptp.loc[(ptp['age_bin']==x) & (ptp['original_trials'] > 30)& (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'original_trials']]

HM_nums = lambda x: ptp.loc[(ptp['age_bin']==x) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'HM']]


# %% 2.1 - Retained trials vs age (Pearson)

fig = plt.figure()
plt.xlabel('Age')
plt.ylabel('Number of Trials - QC')

# Scatter per age bin; legend shows the bin mean
for i in range(1,6):
    avg_trials = np.sum(trial_nums(i).trials)/len(trial_nums(i))
    plt.scatter(trial_nums(i).age, trial_nums(i).trials, label = f'{i}YR: {round(avg_trials)} /sub')


# Linear fit and Pearson p-value across all bins
x = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'age']
y = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 40)& (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'trials']
corr_coeff, p_value = pearsonr(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label = f'p-value = {p_value:.2e}')
plt.legend()

# %% 2.2 - Head motion vs age (Pearson)

fig = plt.figure()
plt.xlabel('Age')
plt.ylabel('HM - QC')

# Scatter per age bin; legend shows the bin mean
for i in range(1,6):
    avg_HM = np.sum(HM_nums(i).HM)/len(HM_nums(i))
    plt.scatter(HM_nums(i).age, HM_nums(i).HM, label = f'{i}YR: {round(avg_HM)} /sub')


# Linear fit and Pearson p-value across all bins (NaN HM values excluded from the correlation)
x = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'age']
y = ptp.loc[(ptp['age_bin']!=0) & (ptp['trials'] > 30) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels']<21), 'HM']

valid = x.notna() & y.notna()
corr_coeff, p_value = pearsonr(x[valid], y[valid])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label = f'p-value = {p_value:.2e}')
plt.legend()

# %% 2.3 - Original trials vs age (Pearson)

fig = plt.figure()
plt.xlabel('Age')
plt.ylabel('Number of Trials')

# Scatter per age bin; legend shows the bin mean
for i in range(1,6):
    avg_trials = np.sum(trial_nums_orig(i).original_trials)/len(trial_nums_orig(i))
    plt.scatter(trial_nums_orig(i).age, trial_nums_orig(i).original_trials, label = f'{i}YR: {round(avg_trials)} /sub')


# Linear fit and Pearson p-value across all bins
x = ptp.loc[(ptp['age_bin']!=0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels']<20), 'age']
y = ptp.loc[(ptp['age_bin']!=0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 )& (ptp['num_bad_channels']<20), 'original_trials']

corr_coeff, p_value = pearsonr(x, y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label = f'p-value = {p_value:.2e}')

plt.legend()

# %% 2.4 - Retained and original trials vs age, stacked (Pearson)

# Per-age-bin subsets for this figure: 40 < trials < 200 and < 15 bad channels
trial_nums = lambda x: ptp.loc[(ptp['age_bin'] == x) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'trials']]
trial_nums_orig = lambda x: ptp.loc[(ptp['age_bin'] == x) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 15), ['age', 'original_trials']]

fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# (a) Retained trials
axs[0].set_title('Trials vs. Age')
axs[0].set_xlabel('Age')
axs[0].set_ylabel('Number of Trials')

for i in range(1, 6):
    avg_trials = np.sum(trial_nums(i).trials) / len(trial_nums(i))
    axs[0].scatter(trial_nums(i).age, trial_nums(i).trials, label=f'{i}YR: {round(avg_trials)} /sub')

x = ptp.loc[(ptp['age_bin'] != 0) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 21), 'age']
y = ptp.loc[(ptp['age_bin'] != 0) & (ptp['trials'] > 40) & (ptp['trials'] < 200 ) & (ptp['num_bad_channels'] < 21), 'trials']

corr_coeff, p_value = pearsonr(x, y)
axs[0].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label=f'p-value = {p_value:.2e}')
axs[0].legend()

# (b) Original trials
axs[1].set_title('Original Trials vs. Age')
axs[1].set_xlabel('Age')
axs[1].set_ylabel('Number of Original Trials')

for i in range(1, 6):
    avg_trials = np.sum(trial_nums_orig(i).original_trials) / len(trial_nums_orig(i))
    axs[1].scatter(trial_nums_orig(i).age, trial_nums_orig(i).original_trials, label=f'{i}YR: {round(avg_trials)} /sub')

x_orig = ptp.loc[(ptp['age_bin'] != 0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 20), 'age']
y_orig = ptp.loc[(ptp['age_bin'] != 0) & (ptp['original_trials'] > 40) & (ptp['original_trials'] < 200 ) & (ptp['num_bad_channels'] < 20), 'original_trials']

corr_coeff_orig, p_value_orig = pearsonr(x_orig, y_orig)
axs[1].plot(np.unique(x_orig), np.poly1d(np.polyfit(x_orig, y_orig, 1))(np.unique(x_orig)), label=f'p-value = {p_value_orig:.2e}')
axs[1].legend()

plt.tight_layout()
# plt.savefig(r'path/to/output/trial_numsv2.png')
plt.show()


# %% 3.0 - MixedLM section: helpers and per-age-bin subsets (no QC thresholds)
"""
All models in this section use MixedLM with a random intercept per subject:
  Trials ~ Age + (1|Subject)
  HM ~ Age + (1|Subject)
  OriginalTrials ~ Age + (1|Subject)
"""

trial_nums      = lambda b: ptp.loc[(ptp['age_bin'] == b), ['age', 'trials']].dropna()
trial_nums_orig = lambda b: ptp.loc[(ptp['age_bin'] == b), ['age', 'original_trials']].dropna()
HM_nums         = lambda b: ptp.loc[(ptp['age_bin'] == b), ['age', 'HM']].dropna()

def fit_lme(ycol, df, xcol='age', group='subject', reml=True):
    """
    Fit MixedLM: ycol ~ xcol + (1|group).
    Drops rows with NaNs in ycol, xcol or group.
    Returns (result, data used), or (None, None) if there is too little data to fit.
    """
    use = df[[ycol, xcol, group]].dropna().copy()
    if use.empty or use[xcol].nunique() < 2:
        return None, None
    X = sm.add_constant(use[xcol].to_numpy())
    y = use[ycol].to_numpy()
    g = use[group].astype(str).to_numpy()
    mdl = MixedLM(endog=y, exog=X, groups=g)
    res = mdl.fit(reml=reml)
    return res, use

def add_lme_line(ax, res, xvals, label_prefix=''):
    """
    Plot the population-level fit line (fixed effects only).
    Works whether res.params is a pandas Series or a numpy array.
    """
    if hasattr(res.params, 'index'):
        const = res.params['const']
        slope_name = [k for k in res.params.index if k != 'const'][0]
        beta = res.params[slope_name]
        p = res.pvalues[slope_name]
        se = res.bse[slope_name]
    else:
        const = res.params[0]
        beta = res.params[1]
        p = res.pvalues[1]
        se = res.bse[1]

    yfit = const + beta * xvals
    ax.plot(
        xvals, yfit, linewidth=2,
        label=f"{label_prefix}coef={beta:.3f}, p={p:.2e}"
    )


def print_lme_summary(title, res):
    print(f"\n===== {title} =====")
    print(res.summary())
    print("Fixed effects:\n", res.params)
    print("P-values:\n", res.pvalues)

# %% 3.1 - Trials ~ Age + (1|Subject)

plt.figure(figsize=(8,5))
plt.title('Trials vs Age (MixedLM)')
plt.xlabel('Age (years)')
plt.ylabel('Trials')

for i in range(1, 6):
    dat = trial_nums(i)
    if len(dat) == 0:
        continue
    avg_trials = dat['trials'].mean()
    plt.scatter(dat['age'], dat['trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials)} /sub')

res_trials, df_trials = fit_lme('trials', ptp, xcol='age', group='subject', reml=True)
if res_trials is not None:
    xgrid = np.linspace(df_trials['age'].min(), df_trials['age'].max(), 200)
    add_lme_line(plt.gca(), res_trials, xgrid, label_prefix='')
    print_lme_summary("LME: Trials ~ Age + (1|Subject)", res_trials)

plt.legend()
plt.tight_layout()
plt.show()

# %% 3.2 - HM ~ Age + (1|Subject)

plt.figure(figsize=(8,5))
plt.title('Head Motion vs Age (MixedLM)')
plt.xlabel('Age (years)')
plt.ylabel('HM')

for i in range(1, 6):
    dat = HM_nums(i)
    if len(dat) == 0:
        continue
    plt.scatter(dat['age'], dat['HM'], s=18, alpha=0.8, label=f'{i}YR')

res_hm, df_hm = fit_lme('HM', ptp, xcol='age', group='subject', reml=True)
if res_hm is not None:
    xgrid = np.linspace(df_hm['age'].min(), df_hm['age'].max(), 200)
    add_lme_line(plt.gca(), res_hm, xgrid, label_prefix='')
    print_lme_summary("LME: HM ~ Age + (1|Subject)", res_hm)

plt.legend()
plt.tight_layout()
plt.show()

# %% 3.3 - OriginalTrials ~ Age + (1|Subject)

plt.figure(figsize=(8,5))
plt.title('Original Trials vs Age (MixedLM)')
plt.xlabel('Age (years)')
plt.ylabel('Original Trials')

for i in range(1, 6):
    dat = trial_nums_orig(i)
    if len(dat) == 0:
        continue
    avg_trials_o = dat['original_trials'].mean()
    plt.scatter(dat['age'], dat['original_trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials_o)} /sub')

res_orig, df_orig = fit_lme('original_trials', ptp, xcol='age', group='subject', reml=True)
if res_orig is not None:
    xgrid = np.linspace(df_orig['age'].min(), df_orig['age'].max(), 200)
    add_lme_line(plt.gca(), res_orig, xgrid, label_prefix='')
    print_lme_summary("LME: OriginalTrials ~ Age + (1|Subject)", res_orig)

plt.legend()
plt.tight_layout()
plt.show()

# %% 3.4 - Retained and original trials vs age, stacked (MixedLM fit lines)

fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# (a) Retained trials
axs[0].set_title('Trials vs Age (MixedLM)')
axs[0].set_ylabel('Trials')
for i in range(1, 6):
    dat = trial_nums(i)
    if len(dat) == 0:
        continue
    avg_trials = dat['trials'].mean()
    axs[0].scatter(dat['age'], dat['trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials)} /sub')
if res_trials is not None:
    xgrid = np.linspace(df_trials['age'].min(), df_trials['age'].max(), 200)
    add_lme_line(axs[0], res_trials, xgrid, label_prefix='')
axs[0].legend()

# (b) Original trials
axs[1].set_title('Original Trials vs Age (MixedLM)')
axs[1].set_xlabel('Age (years)')
axs[1].set_ylabel('Original Trials')
for i in range(1, 6):
    dat = trial_nums_orig(i)
    if len(dat) == 0:
        continue
    avg_trials_o = dat['original_trials'].mean()
    axs[1].scatter(dat['age'], dat['original_trials'], s=18, alpha=0.8, label=f'{i}YR: {round(avg_trials_o)} /sub')
if res_orig is not None:
    xgrid = np.linspace(df_orig['age'].min(), df_orig['age'].max(), 200)
    add_lme_line(axs[1], res_orig, xgrid, label_prefix='')
axs[1].legend()

plt.tight_layout()
plt.show()