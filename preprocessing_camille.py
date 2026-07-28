#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:28:40 2026

@author: camille
"""

# Packages ####################################################################

import os
import mne
import pyprep
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
# %matplotlib qt

# Parameters ##################################################################

idx_part = 2

# Channel names, types and montage
ch_name = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
           'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
           'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz',
           'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6',
           'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6',
           'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10',
           'PO8', 'PO4', 'O2', 'left_ear', 'right_ear', 'EOG_up', 'EOG_down',
           'EOG_left', 'EOG_right', 'hand_1', 'hand_2']
ch_type = {'left_ear': 'misc', 'right_ear': 'misc',
           'EOG_up': 'eog', 'EOG_down': 'eog', 'EOG_left': 'eog',
           'EOG_right': 'eog', 'hand_1': 'emg', 'hand_2': 'emg'}
montage = 'biosemi64'

# Events
event_dict = {'stimulus_go': 1,
              'stimulus_no_go': 2,
              'response': 4,
              'start_probe': 8,
              'end_probe': 16}

# Spectral analysis
param_psd = {'method': 'welch', 'fmin': 1., 'fmax': 100.}
bands = {'Delta (1-4 Hz)': (1, 4),
         'Theta (4-8 Hz)': (4, 8),
         'Alpha (8-12 Hz)': (8, 12),
         'Beta (12-30 Hz)': (12, 30),
         'Gamma (30-45 Hz)': (30, 45)}

# Bad channels description
bad_desc_dict = {
    'bad_all': 'every bad channels',
    'bad_by_deviation': 'channels with abnormally high or low overall amplitudes',
    'bad_by_hf_noise': 'channels with abnormally high amounts of high-frequency noise',
    'bad_by_flat': 'channels that have near-flat signals',
    'bad_by_nan': 'channels that contain NaN values',
    'bad_by_correlation': 'channels that sometimes don’t correlate with any other channels',
    'bad_by_SNR': 'channels that have a low signal-to-noise ratio'
}

# Filters and references
hpf = 0.2  # (Hz)
lpf = 30  # (Hz)
notch = None  # (Hz)
ref = ['left_ear', 'right_ear']

# ICA Parameters
ica_l_freq = 1  # ICA HPF (Hz)
n_components = 0.99  # Set number of components explaining x% of variance
random_state = 42  # Set ICA seed for reproductibility
ica_method = 'fastica'
decim = 4
ica_measure = 'correlation'
ica_thresh = 0.7

# Data reading ################################################################

mapping_part = {
    1: "T19",
    2: "T28",
    3: "T40",
    4: "T52",
    5: "T54",
}
part_name = mapping_part[idx_part]
dir_path = Path(os.getcwd(),"Data",part_name,"EEG", part_name + "_SART_prep.fif")
raw = mne.io.read_raw_fif(dir_path, preload=True)
# Map channel names and types
ch_dict = dict(zip(raw.ch_names, ch_name))
mne.rename_channels(raw.info, ch_dict)
raw.set_channel_types(ch_type)
raw.set_montage(montage, on_missing='ignore')

# Compute epochs ##############################################################

events = mne.find_events(raw, stim_channel="Status")
events_fig = mne.viz.plot_events(
    events,
    event_id=event_dict,
    sfreq=raw.info['sfreq'],
    first_samp=raw.first_samp,
    on_missing='ignore'
)

# Processing ##################################################################

# BAD CHANNELS DETECTION
raw.plot(
    duration=10,
    n_channels=20,
    decim=4,
    events=events,
    event_id=event_dict,
    event_color=None,
    theme='dark'
)

# Spectrum analysis: plot broad-range spectra (lin+log scales) + topomaps
# (Welch's method for computation efficiency)
spectrum = raw.compute_psd(
    **param_psd,
    n_fft=round(raw.info['sfreq']) * 4  # length of Welch window (pts)
)
spec_fig, axs = plt.subplots(ncols=2, figsize=(16, 7), tight_layout=True, clear=True)
axs[0] = spectrum.plot(axes=axs[0])
axs[1] = spectrum.plot(xscale='log', axes=axs[1])
spectrum.plot_topomap(bands=bands, cmap='Spectral_r', size=10.0)
plt.show(block=True)

# PyPREP : find bad channels following specific criteria (correlation, deviation...)
noisy_chan = pyprep.NoisyChannels(raw)
print('Finding bad by correlation...')
noisy_chan.find_bad_by_correlation(correlation_secs=1.0,
                                   correlation_threshold=0.4,
                                   frac_bad=0.01)
print('Finding bad by deviation...')
noisy_chan.find_bad_by_deviation(deviation_threshold=5.0)
print('Finding bad by high-frequency noise...')
noisy_chan.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
print('Finding bad by NaN or flat...')
noisy_chan.find_bad_by_nan_flat()
print('Finding bad by signal-to-noise ratio...')
noisy_chan.find_bad_by_SNR()

bad_chan_dict = noisy_chan.get_bads(as_dict=True)
for k in bad_chan_dict:
    for idx, val in enumerate(bad_chan_dict.get(k)):
        bad_chan_dict.get(k)[idx] = str(val)
raw.info['bads'].extend(bad_chan_dict.get('bad_all'))

# Replot for verification + selection if needed
raw.plot(
    duration=10,
    n_channels=20,
    decim=4,
    events=events,
    event_id=event_dict,
    event_color=None,
    theme='dark',
    block=True
)
if raw.info['bads'] != []:
    bad_chan_fig = (raw.copy()
                    .pick(raw.info['bads'])
                    .plot_sensors(show_names=True))
    
# Filters
raw_filt = raw.copy().filter(hpf, lpf, picks=raw.ch_names[0:66])
raw_filt.notch_filter(notch) if notch is not None else None



# Rereference
raw_reref = raw_filt.copy().set_eeg_reference(ref, ch_type='eeg')
raw_reref = raw_filt.copy().set_eeg_reference("average", projection=False, verbose=False)

raw_reref.plot(
    duration=10,
    n_channels=20,
    decim=4,
    events=events,
    event_id=event_dict,
    event_color=None,
    theme='dark',
    block=False
)
# %%% OCULAR CORRECTION WITH ICA
# Perform ICA
raw_preica = raw_reref.copy().filter(ica_l_freq, h_freq=None)
epochs_preica = mne.make_fixed_length_epochs(raw_preica, duration=10)
picks = mne.pick_types(
    epochs_preica.info,
    eeg=True,
    eog=False,
    misc=False,
    exclude='bads',
)
ica = mne.preprocessing.ICA(
    n_components=n_components,
    random_state=random_state,
    method=ica_method,
    max_iter='auto',
)
ica.fit(epochs_preica, picks=picks, decim=decim, reject_by_annotation=False)

# Use EOG channels for selecting components reflecting ocular artifacts (blinks +
# saccades)
ica.exclude = []
eog_indices, eog_scores = ica.find_bads_eog(
    raw_preica,
    measure=ica_measure,
    threshold=ica_thresh,
)
ica.exclude = eog_indices

# Plot properties of components
ica_scores_fig = ica.plot_scores(eog_scores)
ica.plot_sources(raw_preica, show_scrollbars=False)
ica_fig = ica.plot_components()
plt.show(block=True)

ocu_dict = {}
for ocu_components in ica.exclude:  # avoid to reprocess ICA for final report
    ocu_dict.update(
        {ocu_components: ica.plot_properties(raw_preica, picks=ocu_components, show=False)})

# Apply ICA to signal
raw_postica = raw_reref.copy()
ica.apply(raw_postica)

# Interpolation (if bad channels detected)
raw_postica.interpolate_bads(reset_bads=True) if raw_postica.info['bads'] else None

