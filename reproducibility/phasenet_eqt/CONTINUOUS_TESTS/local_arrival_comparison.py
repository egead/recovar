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

def phasenet_eqt_picks(stream, arrivals_csv_path):
    model_phasenet = sbm.PhaseNet.from_pretrained("geofon")
    annotations_phasenet = model_phasenet.annotate(stream)

    model_eqt = sbm.EQTransformer.from_pretrained("stead")
    annotations_eqt = model_eqt.annotate(stream)

    fig = plt.figure(figsize=(15, 13))
    axs = fig.subplots(4, 1, sharex=True, gridspec_kw={"hspace": 0})

    #CATALOG
    station_name = stream[0].stats.station
    arrivals_df = pd.read_csv(arrivals_csv_path)
    arrivals_df['arrival_time'] = pd.to_datetime(arrivals_df['arrival_time'])
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
    
    # RECOVAR
    csv_path = Path(f"/home/ege/recovar/reproducibility/phasenet_eqt/10NOV_18-19_INSTANCE_1SEC/KO_{station_name}_predictions.csv")
    
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    predictions = pd.Series(data=df['eq_probability'].values, index=df['timestamp'])
    predictions = predictions.sort_index()
    
    if predictions.index.tz is not None:
        predictions.index = predictions.index.tz_localize(None)
    
    pred_times = mdates.date2num(predictions.index.to_pydatetime())
    
    axs[3].fill_between(pred_times, 0, predictions.values, alpha=0.3, color='steelblue', label=f'{station_name} (prob)')
    axs[3].plot(pred_times, predictions.values, color='steelblue', linewidth=1.5)
    
    for idx, p_time in enumerate(p_times):
        axs[3].axvline(p_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='P arrival' if idx == 0 else None)
    for idx, s_time in enumerate(s_times):
        axs[3].axvline(s_time, color='blue', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='S arrival' if idx == 0 else None)
    
    axs[3].set_ylim(0, 1)
    axs[3].set_ylabel('Event Score')
    axs[3].legend(loc='upper left')
    
    axs[0].legend()
    axs[1].legend()
    axs[1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    axs[2].legend()
    axs[2].xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
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

ARRIVALS_CSV = '10NOV_2-3.csv' 
PATH='/home/ege/10NOV_2-3/'

for m in find_mseed_files(PATH):
    DATA_NAME = m.stem                     
    stream = obspy.read(m)                
    preprocess(stream)   
    phasenet_eqt_picks(stream, ARRIVALS_CSV)