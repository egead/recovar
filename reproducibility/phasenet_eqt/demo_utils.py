"""
Utility functions for YAZEL RECOVAR demo notebook.
"""

import obspy
import pandas as pd
from pathlib import Path


def load_example_picks(phasenet_pick_dir, catalog, max_files=50, min_tp=3, min_fp=3):
    """
    Load and categorize PhaseNet picks into TRUE and FALSE examples.

    Parameters
    ----------
    phasenet_pick_dir : str
        Directory containing PhaseNet pick files
    catalog : pd.DataFrame
        Catalog DataFrame with p_arrival_time column
    max_files : int, optional
        Maximum number of files to check (default: 50)
    min_tp : int, optional
        Minimum number of TRUE PICK examples to find (default: 3)
    min_fp : int, optional
        Minimum number of FALSE PICK examples to find (default: 3)

    Returns
    -------
    tuple
        (tp_examples, fp_examples) - Lists of example dictionaries
    """
    files = sorted(Path(phasenet_pick_dir).glob('*.mseed'))
    phasenet_picks = pd.read_csv(f"{phasenet_pick_dir}/metadata.csv")

    tp_examples = []  # TRUE PICK examples (matches catalog)
    fp_examples = []  # FALSE PICK examples (no catalog match)

    for file in files[:max_files]:
        pick_row = phasenet_picks[phasenet_picks['filename'] == file.name]
        if pick_row.empty:
            continue

        try:
            stream = obspy.read(file)
            stream.merge()

            # Separate waveform traces from annotation traces (P, S, N probabilities)
            waveform_traces = [tr for tr in stream if not tr.stats.channel.endswith(('P', 'S', 'N'))]

            # Extract metadata from waveform
            station = waveform_traces[0].stats.station
            window_start = waveform_traces[0].stats.starttime.datetime
            window_end = waveform_traces[0].stats.endtime.datetime

            # Get PhaseNet pick time from metadata
            phasenet_pick = pd.to_datetime(pick_row['pick_time'].values[0], format='mixed')

            # Check if there's a catalog pick in this window
            catalog_picks = catalog[
                (catalog['p_arrival_time'] >= window_start) &
                (catalog['p_arrival_time'] <= window_end)
            ]

            example = {
                'file': file,
                'stream': stream,  # Keep full stream (waveforms + annotations)
                'station': station,
                'phasenet_pick': phasenet_pick,
                'window_start': window_start,
                'window_end': window_end,
                'catalog_pick': catalog_picks.iloc[0]['p_arrival_time'] if not catalog_picks.empty else None
            }

            if not catalog_picks.empty:
                tp_examples.append(example)
            else:
                fp_examples.append(example)

        except Exception as e:
            continue

        # Stop when we have enough examples
        if len(tp_examples) >= min_tp and len(fp_examples) >= min_fp:
            break

    return tp_examples, fp_examples
