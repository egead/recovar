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

def phasenet_eqt_picks(stream):
    model_phasenet = sbm.PhaseNet.from_pretrained("geofon")
    annotations_phasenet = model_phasenet.annotate(stream)

    model_eqt = sbm.EQTransformer.from_pretrained("stead")
    annotations_eqt = model_eqt.annotate(stream)

    fig = plt.figure(figsize=(15, 13))
    axs = fig.subplots(4, 1, sharex=True, gridspec_kw={"hspace": 0})

    CATALOG_DATA =  """10/11/2025 03:09:26,2.9,Sındırgı (Balıkesir)
10/11/2025 02:57:20,2.8,Sındırgı (Balıkesir)
10/11/2025 02:52:53,3.7,Sındırgı (Balıkesir)
10/11/2025 02:48:57,4.8,Sındırgı (Balıkesir)
10/11/2025 02:26:54,2.6,Sındırgı (Balıkesir)
10/11/2025 02:07:29,2.4,Sındırgı (Balıkesir)
10/11/2025 01:56:47,2.7,Sındırgı (Balıkesir))""".strip()

    catalog_df = pd.read_csv(StringIO(CATALOG_DATA), header=None, names=['datetime','magnitude','location'])
    catalog_df['datetime'] = pd.to_datetime(catalog_df['datetime'], format='%d/%m/%Y %H:%M:%S')
    catalog_df['datetime'] = catalog_df['datetime'].dt.tz_localize('Europe/Istanbul').dt.tz_convert('UTC').dt.tz_localize(None)
    catalog_times = mdates.date2num(catalog_df['datetime'].dt.to_pydatetime())


    # WAVEFORM
    waveform_color = 'black'
    for i in range(3):
        times = stream[i].times("matplotlib")
        axs[0].plot(times, stream[i].data, label=stream[i].stats.channel, color=waveform_color, linewidth=0.8, alpha=0.7)
    
    for idx, eq_time in enumerate(catalog_times):
        axs[0].axvline(eq_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='Catalog EQ' if idx == 0 else None)

    # PHASENET
    phasenet_colors = {'P': 'orange', 'S': 'steelblue'} 
    for i in range(2):
        if annotations_phasenet[i].stats.channel[-1] != "N":
            times_phasenet = annotations_phasenet[i].times("matplotlib")
            channel = annotations_phasenet[i].stats.channel
            color = phasenet_colors.get(channel[-1], 'gray')
            axs[1].plot(times_phasenet, annotations_phasenet[i].data, label=channel, color=color, linewidth=1.5)
    
    for idx, eq_time in enumerate(catalog_times):
        axs[1].axvline(eq_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='Catalog EQ' if idx == 0 else None)

    # EQT
    eqt_colors = ['orange', 'steelblue', 'green'] 
    for i in range(3):
        times_eqt = annotations_eqt[i].times("matplotlib")
        axs[2].plot(times_eqt, annotations_eqt[i].data, label=annotations_eqt[i].stats.channel, 
                   color=eqt_colors[i], linewidth=1.5)
    
    for idx, eq_time in enumerate(catalog_times):
        axs[2].axvline(eq_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='Catalog EQ' if idx == 0 else None)
    
    # RECOVAR
    station_name = stream[0].stats.station
    csv_path = Path(f"/home/ege/recovar/reproducibility/phasenet_eqt/10NOV_2-3_INSTANCE/KO_{station_name}_predictions.csv")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    predictions = pd.Series(data=df['eq_probability'].values, index=df['timestamp'])
    predictions = predictions.sort_index()
    
    # Handle datetime
    if predictions.index.tz is not None:
        predictions.index = predictions.index.tz_localize(None)
    
    pred_times = mdates.date2num(predictions.index.to_pydatetime())
    
    axs[3].fill_between(pred_times, 0, predictions.values, alpha=0.3, color='steelblue', label=f'{station_name} (prob)')
    axs[3].plot(pred_times, predictions.values, color='steelblue', linewidth=1.5)
    
    for idx, eq_time in enumerate(catalog_times):
        axs[3].axvline(eq_time, color='red', linestyle='--', alpha=0.6, linewidth=1.2,
                       label='Catalog EQ' if idx == 0 else None)
    
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

PATH='/home/ege/10NOV_2-3/'

for m in find_mseed_files(PATH):
    DATA_NAME = m.stem                     
    stream = obspy.read(m)                
    preprocess(stream)   
    phasenet_eqt_picks(stream)