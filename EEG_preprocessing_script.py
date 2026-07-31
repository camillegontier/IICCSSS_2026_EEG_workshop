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
import csv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
# %matplotlib qt

# Data reading ################################################################

dir_path = Path(os.getcwd(),"Data","sub-001_task-CBOFF_run-1_eeg.set")
raw = mne.io.read_raw_eeglab(dir_path, preload=True)
raw.plot(
    duration=10,
    n_channels=20,
    decim=4,
    theme='dark'
     )

# Channel types ################################################################

# Print channel names
print(raw.info)
# Channel names, types and montage
ch_type = {'ECG': 'ecg'}
montage = 'easycap-M10'
print(raw.info.ch_names)
raw.get_channel_types()

# Map channel names and types
raw.set_channel_types(ch_type)
print(raw.info)
raw.get_channel_types()
# Check number of channels and sampling rate
# Montage #####################################################################

montage = mne.channels.make_standard_montage("easycap-M1")
montage.plot(sphere="eeglab")


# raw.set_montage(montage, on_missing='ignore')
print(raw.info)

# Compute epochs ##############################################################
events = mne.events_from_annotations(raw)
idx_trial_start = np.where(events[:,2]==2)[0]

# read behavior data
# with open("DATA\sub-001_task-CBOFF_run-1_beh.tsv") as fd:
    
#     rd = csv.reader(fd, delimiter="\t")
#     next(rd)
#     row_idx = 0
#     for row in rd:
#         if int(row[9]) != row_idx:
#             print(row_idx)
#             tasktype = row[3]
#             stimulus = row[10]
#             RT = float(row[12])
#             if tasktype == stimulus:
#                 events[idx_trial_start[row_idx],2] = 10
#             else:
#                 events[idx_trial_start[row_idx],2] = 11
#             if RT > 0:
#                 events = np.vstack([events, [events[idx_trial_start[row_idx],0]+RT*raw.info['sfreq']/1000,0,12]])
#             row_idx = row_idx + 1
# # Events
event_dict = {'stimulus_go': 10,
              'stimulus_no_go': 11,
              'response': 12}

# events_fig = mne.viz.plot_events(
#     events,
#     event_id=event_dict,
#     sfreq=raw.info['sfreq'],
#     first_samp=raw.first_samp,
#     on_missing='ignore'
# )

events_fig = mne.viz.plot_events(
    events,
    sfreq=raw.info['sfreq'],
    first_samp=raw.first_samp,
    on_missing='ignore'
)
# raw.plot(
#     duration=10,
#     n_channels=20,
#     decim=4,
#     events=events,
#     event_id=event_dict,
#     event_color=None,
#     theme='dark'
    
# )
# check number of trials
# Filtering ###################################################################

# Filters and references
hpf = 0.2  # (Hz) # to remove drifts, before epochs for edge artifacts
lpf = 100  # (Hz)
notch = None  # (Hz)
# Filters
raw_filt = raw.copy().filter(hpf, lpf)
raw_filt.notch_filter(notch) if notch is not None else None



# Referencing #################################################################
# ref = ['left_ear', 'right_ear']
raw.set_eeg_reference("average")
# Usually average AFTER the bad channels
# # Rereference
# raw_reref = raw_filt.copy().set_eeg_reference(ref, ch_type='eeg')
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


# BAD CHANNELS DETECTION ##################################################################
# raw.plot(
#     duration=10,
#     n_channels=20,
#     decim=4,
#     events=events,
#     event_id=event_dict,
#     event_color=None,
#     theme='dark'
    
# )

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
    theme='dark'
)
if raw.info['bads'] != []:
    bad_chan_fig = (raw.copy()
                    .pick(raw.info['bads'])
                    .plot_sensors(show_names=True))
    



# %%% OCULAR CORRECTION WITH ICA

# ICA Parameters
ica_l_freq = 1  # ICA HPF (Hz)
n_components = 0.99  # Set number of components explaining x% of variance
random_state = 42  # Set ICA seed for reproductibility
ica_method = 'fastica'
decim = 4
ica_measure = 'correlation'
ica_thresh = 0.7

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

# Balistocardiogram

# Spectral analysis ############################################################

# Epoch = add a dimension to the data


hpf = 1 # (Hz)
lpf = 40 # (Hz)
notch = None # (Hz)
# Filters
raw.filter(hpf, lpf)
raw.notch_filter(50)

raw.pick(['Oz','O1','O2','POz'])

# psd = raw.compute_psd(
# method="welch",
# fmin=1,
# fmax=30,
# tmin=60, 
# tmax=73,
# n_fft=8192
# )

# psd = raw.compute_psd(
# method="multitaper",
# fmin=5,
# fmax=25,
# bandwidth=0.1,
# )
# psd.plot()





epochs = mne.Epochs(
raw,
events,
event_id=1,
tmin=0,
tmax=15, # or whatever the stimulus duration is
baseline=None,
preload=True,
)

psd = epochs.compute_psd(
method="multitaper",
fmin=1,
fmax=30,bandwidth=0.1,
)

psd.plot()
 
33333333333333333333333333333333333333333333

raw.drop_channels(["ECG"])

spectrum = raw.compute_psd(method='welch', fmin=0, 
fmax=100, 
tmin=165, 
tmax=180, 
)
spec_fig, axs = plt.subplots(ncols=2, figsize=(16, 7), tight_layout=True, clear=True)
axs[0] = spectrum.plot(axes=axs[0],dB=True)
axs[1] = spectrum.plot(xscale='log', axes=axs[1])
plt.show()





# # Spectral analysis
param_psd = {'method': 'welch', 'fmin': 1., 'fmax': 100.}
bands = {'Delta (1-4 Hz)': (1, 4),
'Theta (4-8 Hz)': (4, 8),
'Alpha (8-12 Hz)': (8, 12),
'Beta (12-30 Hz)': (12, 30),
'Gamma (30-45 Hz)': (30, 45)}

# Spectrum analysis: plot broad-range spectra (lin+log scales) + topomaps
# (Welch's method for computation efficiency)
spectrum = raw.compute_psd(
**param_psd,
n_fft=round(raw.info['sfreq']) * 1 # length of Welch window (pts)
)
spec_fig, axs = plt.subplots(ncols=2, figsize=(16, 7), tight_layout=True, clear=True)
axs[0] = spectrum.plot(axes=axs[0],dB=True)
axs[1] = spectrum.plot(xscale='log', axes=axs[1])
spectrum.plot_topomap(bands=bands, cmap='Spectral_r', size=10.0)
plt.show(block=True)




