"""
SpecParam batch fitting (specparam 2.0.0rc3)

Fits SpecParam models to power spectra for every subject and region and returns
the periodic, aperiodic, and full model fits, aperiodic parameters, R² values,
frequencies used, and fitted models.

psd_data must be in linear units, with shape [num_sub, num_freqs, num_reg].
Only supports 'fixed' aperiodic mode; 'knee' requires modifying params.

@Author: Julian Bandhan
Updated: 2026-11-03
"""

import numpy as np
from specparam import SpectralModel
#from specparam.analysis import compute_r_squared

def r_squared(y, y_pred):
    """Coefficient of determination (R²) for two 1‑D arrays."""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan



def run_specparam_batch(psd_data, freqs, num_sub, num_reg, settings):
    """
    Run SpecParam batch fitting over all subjects and regions.

    Parameters:
        psd_data   : ndarray, shape [num_sub, num_freqs, num_reg]
        freqs      : ndarray, shape [num_freqs]
        num_sub    : int
        num_reg    : int
        settings   : dict of SpecParam fitting parameters

    Returns:
        dict with keys:
            'periodic', 'aperiodic', 'spectrum', 'params', 'r_squared',
            'n_bins', 'freqs_used', 'models'
    """
    # ----- Unpack settings -----
    freq_range        = settings.get('freq_range', [3, 40])
    peak_width_limits = settings.get('peak_width_limits', [2, 5])
    max_n_peaks       = settings.get('max_n_peaks', np.inf)
    peak_threshold    = settings.get('peak_threshold', 0)
    min_peak_height   = settings.get('min_peak_height', 0)
    aperiodic_mode    = settings.get('aperiodic_mode', 'fixed')
    space = settings.get('space', 'log')

    # ----- Preallocate after first model tells us n_bins -----
    # ---- boolean mask & bins ----
    mask       = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    n_bins     = mask.sum()
    freqs_used = freqs[mask]

    # ---- pre‑allocate ----
    periodic   = np.zeros((num_sub, n_bins, num_reg))
    aperiodic  = np.zeros((num_sub, n_bins, num_reg))
    observed   = np.zeros((num_sub, n_bins, num_reg))
    params     = np.zeros((num_sub, num_reg, 2))   # offset, slope
    r2_scores  = np.zeros((num_sub, num_reg))
    models     = [[None for _ in range(num_reg)] for _ in range(num_sub)]

    # ----- Initialize arrays -----
    periodic   = np.zeros((num_sub, n_bins, num_reg))
    aperiodic  = np.zeros((num_sub, n_bins, num_reg))
    observed   = np.zeros((num_sub, n_bins, num_reg))
    params     = np.zeros((num_sub, num_reg, 2))  # offset, slope
    r2_scores  = np.zeros((num_sub, num_reg))
    models     = [[None for _ in range(num_reg)] for _ in range(num_sub)]

    # ----- Main fitting loop -----
    for i in range(num_sub):
        for j in range(num_reg):

            raw_psd = psd_data[i, :, j]

            sp = SpectralModel(peak_width_limits=peak_width_limits,
                               max_n_peaks=max_n_peaks,
                               peak_threshold=peak_threshold,
                               min_peak_height=min_peak_height,
                               aperiodic_mode=aperiodic_mode)
            sp.fit(freqs, raw_psd, freq_range)
            
            
            # Recalculate peak model
            
            pe_fit = sp.get_model(component='peak', space=space)
            
            ap_fit = sp.get_model(component='aperiodic', space=space)
            
            tot_fit = sp.get_model(component='full', space=space)
            
            
            # # model components (log space)
            # ap_fit  = sp.get_model(component='aperiodic', space='log')
            # pe_fit  = sp.get_model(component='peak', space='log')
            #tot_fit = 10**ap_fit + 10**pe_fit

            # save
            periodic[i, :, j]   = pe_fit
            aperiodic[i, :, j]  = ap_fit
            observed[i, :, j]   =  tot_fit
            params[i, j, :]     = sp.aperiodic_params_
            log_raw_psd = np.log10(raw_psd[mask])
            log_raw_psd = np.log10(raw_psd[mask])
	    log_tot_fit = sp.get_model(component='full', space='log')
	    r2_scores[i, j] = r_squared(log_raw_psd, log_tot_fit)
            models[i][j]        = sp

    return {
        'periodic':   periodic,
        'aperiodic':  aperiodic,
        'spectrum':   observed,
        'params':     params,
        'r_squared':  r2_scores,
        'n_bins':     n_bins,
        'freqs_used': freqs_used,
        'models':     models
    }
