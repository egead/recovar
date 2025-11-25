import seisbench
import numpy as np
import torch
import obspy
import matplotlib.pyplot as plt
import seisbench.models as sbm
from pathlib import Path 
from matplotlib.dates import DateFormatter
import pandas as pd
from io import StringIO
import matplotlib.dates as mdates

def phasenet_eqt_picks(stream, arrivals_csv_path, recovar_instance_dir, recovar_merged_dir, 
                       arrival_tz='Europe/Istanbul', crop_start=None, crop_end=None):
    model_phasenet = sbm.PhaseNet.from_pretrained("geofon")
    annotations_phasenet = model_phasenet.annotate(stream)

    model_eqt = sbm.EQTransformer.from_pretrained("stead")
    annotations_eqt = model_eqt.annotate(stream)

    fig = plt.figure(figsize=(15, 16))
    axs = fig.subplots(5, 1, sharex=True, gridspec_kw={"hspace": 0})

    #CATALOG
    station_name = stream[0].stats.station
    arrivals_df = pd.read_csv(arrivals_csv_path)
    arrivals_df['arrival_time'] = pd.to_datetime(arrivals_df['arrival_time'])
    arrivals_df['arrival_time'] = arrivals_df['arrival_time'].dt.tz_localize(arrival_tz).dt.tz_localize(None)
    
    station_arrivals = arrivals_df[arrivals_df['station'] == station_name]
    
    p_arrivals = station_arrivals[station_arrivals['phase'] == 'P']
    s_arrivals = station_arrivals[station_arrivals['phase'] == 'S']
    
    p_times = mdates.date2num(p_arrivals['arrival_time'].dt.to_pydatetime())
    s_times = mdates.date2num(s_arrivals['arrival_time'].dt.to_pydatetime())
    
    # WAVEFORM
    waveform_color = 'black'
    for i in range(3):
        times = stream[i].times("matplotlib")
        axs[0].plot(times, stream[i].data, label=stream[i].stats.channel, color=waveform_color, linewidth=0.8, alpha=0.7)
    
    for idx, p_time in enumerate(p_times):
        axs[0].axvline(p_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='P arrival' if idx == 0 else None)
    for idx, s_time in enumerate(s_times):
        axs[0].axvline(s_time, color='blue', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='S arrival' if idx == 0 else None)
    axs[0].legend()

    # PHASENET
    phasenet_colors = {'P': 'orange', 'S': 'steelblue'} 
    for i in range(2):
        if annotations_phasenet[i].stats.channel[-1] != "N":
            times_phasenet = annotations_phasenet[i].times("matplotlib")
            channel = annotations_phasenet[i].stats.channel
            color = phasenet_colors.get(channel[-1], 'gray')
            axs[1].plot(times_phasenet, annotations_phasenet[i].data, label=channel, color=color, linewidth=1.5)
    
    for idx, p_time in enumerate(p_times):
        axs[1].axvline(p_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='P arrival' if idx == 0 else None)
    for idx, s_time in enumerate(s_times):
        axs[1].axvline(s_time, color='blue', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='S arrival' if idx == 0 else None)
    axs[1].legend()

    # EQT
    eqt_colors = ['orange', 'steelblue', 'green'] 
    for i in range(3):
        times_eqt = annotations_eqt[i].times("matplotlib")
        axs[2].plot(times_eqt, annotations_eqt[i].data, label=annotations_eqt[i].stats.channel, 
                   color=eqt_colors[i], linewidth=1.5)
    
    for idx, p_time in enumerate(p_times):
        axs[2].axvline(p_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='P arrival' if idx == 0 else None)
    for idx, s_time in enumerate(s_times):
        axs[2].axvline(s_time, color='blue', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='S arrival' if idx == 0 else None)
    axs[2].legend()
    
    # RECOVAR INSTANCE
    csv_path_instance = Path(recovar_instance_dir) / f"KO_{station_name}_predictions.csv"
    
    df_instance = pd.read_csv(csv_path_instance, parse_dates=["timestamp"])
    df_instance['timestamp'] = df_instance['timestamp'].dt.tz_convert(arrival_tz).dt.tz_localize(None)
    
    predictions_instance = pd.Series(data=df_instance['eq_probability'].values, index=df_instance['timestamp'])
    predictions_instance = predictions_instance.sort_index()
    
    if predictions_instance.index.tz is not None:
        predictions_instance.index = predictions_instance.index.tz_localize(None)
    
    pred_times_instance = mdates.date2num(predictions_instance.index.to_pydatetime())
    
    axs[3].fill_between(pred_times_instance, 0, predictions_instance.values, alpha=0.3, color='green', label=f'{station_name} (INSTANCE prob)')
    axs[3].plot(pred_times_instance, predictions_instance.values, color='green', linewidth=1.5)
    
    for idx, p_time in enumerate(p_times):
        axs[3].axvline(p_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='P arrival' if idx == 0 else None)
    for idx, s_time in enumerate(s_times):
        axs[3].axvline(s_time, color='blue', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='S arrival' if idx == 0 else None)
    
    axs[3].set_ylim(0, 1)
    axs[3].set_ylabel('INSTANCE Event Score')
    axs[3].legend(loc='upper left')
    
    # RECOVAR MERGED
    csv_path_merged = Path(recovar_merged_dir) / f"KO_{station_name}_predictions.csv"
    
    df_merged = pd.read_csv(csv_path_merged, parse_dates=["timestamp"])
    df_merged['timestamp'] = df_merged['timestamp'].dt.tz_convert(arrival_tz).dt.tz_localize(None)
    
    predictions_merged = pd.Series(data=df_merged['eq_probability'].values, index=df_merged['timestamp'])
    predictions_merged = predictions_merged.sort_index()
    
    if predictions_merged.index.tz is not None:
        predictions_merged.index = predictions_merged.index.tz_localize(None)
    
    pred_times_merged = mdates.date2num(predictions_merged.index.to_pydatetime())
    
    axs[4].fill_between(pred_times_merged, 0, predictions_merged.values, alpha=0.3, color='steelblue', label=f'{station_name} (MERGED prob)')
    axs[4].plot(pred_times_merged, predictions_merged.values, color='steelblue', linewidth=1.5)
    
    for idx, p_time in enumerate(p_times):
        axs[4].axvline(p_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='P arrival' if idx == 0 else None)
    for idx, s_time in enumerate(s_times):
        axs[4].axvline(s_time, color='blue', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='S arrival' if idx == 0 else None)
    
    axs[4].set_ylim(0, 1)
    axs[4].set_ylabel('MERGED Event Score')
    axs[4].legend(loc='upper left')
    #CROPPING
    if crop_start is not None or crop_end is not None:
        xlims = list(axs[0].get_xlim())
        if crop_start is not None:
            xlims[0] = mdates.date2num(pd.to_datetime(crop_start))
        if crop_end is not None:
            xlims[1] = mdates.date2num(pd.to_datetime(crop_end))
        for ax in axs:  # Fix: add missing colon here
            ax.set_xlim(xlims)
        
        #Rescale waveform y axis amplitdue 
        y_min, y_max = float('inf'), float('-inf')
        for i in range(3):
            times = stream[i].times("matplotlib")
            mask = (times >= xlims[0]) & (times <= xlims[1])
            if mask.any():
                y_min = min(y_min, stream[i].data[mask].min())
                y_max = max(y_max, stream[i].data[mask].max())
        if y_min != float('inf'):
            margin = (y_max - y_min) * 0.1
            axs[0].set_ylim(y_min - margin, y_max + margin)   
    
    axs[4].xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    axs[4].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.suptitle(f'{station_name} - {DATA_NAME}', fontsize=14)
    fig.autofmt_xdate()
    
    if crop_start is not None or crop_end is not None:
        plt.savefig(f"{DATA_NAME}_phasenet_eqt_picks_crop_{crop_start}-{crop_end}.png")
    else: 
        plt.savefig(f"{DATA_NAME}_phasenet_eqt_picks.png")


def preprocess(stream):
    for tr in stream:
        tr.detrend(type='constant')     
        tr.detrend(type='linear')
        tr.filter('bandpass', freqmin=1.0,freqmax=20.0)

def find_mseed_files(root: str):
    root_path = Path(root)
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mseed":
            yield p

ARRIVALS_CSV = '2OCT_14-30.csv' 
RECOVAR_INSTANCE_DIR = '/home/ege/recovar/reproducibility/phasenet_eqt/2OCT_14-30_INSTANCE_1SEC'
RECOVAR_MERGED_DIR = '/home/ege/recovar/reproducibility/phasenet_eqt/2OCT_14-30_MERGED_1SEC'
PATH = '/home/ege/10NOV_18-19/'

for m in find_mseed_files(PATH):
    DATA_NAME = m.stem                     
    stream = obspy.read(m)                
    preprocess(stream)   
    phasenet_eqt_picks(stream, ARRIVALS_CSV, RECOVAR_INSTANCE_DIR, RECOVAR_MERGED_DIR, 
                       arrival_tz='Europe/Istanbul',
                       crop_start='2025-11-10 18:38:40', 
                       crop_end='2025-11-10 18:50:40')   