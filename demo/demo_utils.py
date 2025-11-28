import obspy
import pandas as pd
from pathlib import Path


def load_example_picks(phasenet_pick_dir, catalog, max_files=None, min_tp=3, min_fp=3):
    """
    Load and categorize PhaseNet picks into TRUE and FALSE examples.

    Parameters
    ----------
    phasenet_pick_dir : str
        Directory containing PhaseNet pick files
    catalog : pd.DataFrame
        Catalog DataFrame with p_arrival_time column
    max_files : int or None, optional
        Maximum number of files to check. If None, process all files (default: None)
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

    # Process all files if max_files is None, otherwise limit
    files_to_process = files if max_files is None else files[:max_files]

    for file in files_to_process:
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

            phasenet_pick = pd.to_datetime(pick_row['pick_time'].values[0], format='mixed')

            # Check if there's a catalog pick in this window
            catalog_picks = catalog[
                (catalog['p_arrival_time'] >= window_start) &
                (catalog['p_arrival_time'] <= window_end)
            ]

            example = {
                'file': file,
                'stream': stream, 
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

        # Stop when we have enough examples (only if max_files is not None)
        if max_files is not None and len(tp_examples) >= min_tp and len(fp_examples) >= min_fp:
            break

    return tp_examples, fp_examples


def load_preprocessed_data(preprocessed_dir, precompute_phasenet=True):
    """
    Load preprocessed demo data from truepicks/falsepicks directories.

    Parameters
    ----------
    preprocessed_dir : str
        Directory containing preprocessed truepicks/ and falsepicks/ subdirectories
    precompute_phasenet : bool, optional
        If True, precompute PhaseNet probabilities for all examples (default: True)

    Returns
    -------
    tuple
        (tp_examples, fp_examples, tp_metadata, fp_metadata)
        - tp_examples: list of dicts with stream, file, station, picks, etc.
        - fp_examples: list of dicts with stream, file, station, picks, etc.
        - tp_metadata: DataFrame with RECOVAR scores for true picks
        - fp_metadata: DataFrame with RECOVAR scores for false picks
    """
    from demo_plotting import get_phasenet_probabilities
    import numpy as np
    import json

    preprocessed_path = Path(preprocessed_dir)
    truepicks_dir = preprocessed_path / 'truepicks'
    falsepicks_dir = preprocessed_path / 'falsepicks'

    tp_metadata = pd.read_csv(preprocessed_path / 'truepicks_metadata.csv')
    fp_metadata = pd.read_csv(preprocessed_path / 'falsepicks_metadata.csv')

    with open(preprocessed_path / 'truepicks_scores.json', 'r') as f:
        tp_scores_data = json.load(f)
    with open(preprocessed_path / 'falsepicks_scores.json', 'r') as f:
        fp_scores_data = json.load(f)

    # Create filename to scores_array mapping
    tp_scores_map = {item['filename']: np.array(item['scores_array']) for item in tp_scores_data}
    fp_scores_map = {item['filename']: np.array(item['scores_array']) for item in fp_scores_data}

    # Convert datetime columns
    tp_metadata['phasenet_pick'] = pd.to_datetime(tp_metadata['phasenet_pick'])
    tp_metadata['catalog_pick'] = pd.to_datetime(tp_metadata['catalog_pick'])
    tp_metadata['window_start'] = pd.to_datetime(tp_metadata['window_start'])
    tp_metadata['window_end'] = pd.to_datetime(tp_metadata['window_end'])

    fp_metadata['phasenet_pick'] = pd.to_datetime(fp_metadata['phasenet_pick'])
    fp_metadata['window_start'] = pd.to_datetime(fp_metadata['window_start'])
    fp_metadata['window_end'] = pd.to_datetime(fp_metadata['window_end'])

    # Load true picks
    tp_examples = []
    for idx, row in tp_metadata.iterrows():
        file_path = truepicks_dir / row['filename']
        stream = obspy.read(str(file_path))
        stream.merge()

        example = {
            'file': file_path,
            'stream': stream,
            'station': row['station'],
            'phasenet_pick': row['phasenet_pick'],
            'catalog_pick': row['catalog_pick'],
            'window_start': row['window_start'],
            'window_end': row['window_end'],
            'max_score': row['max_score'],
            'mean_score': row['mean_score']
        }

        # Precompute PhaseNet probabilities if requested
        if precompute_phasenet:
            example['phasenet_result'] = get_phasenet_probabilities(stream)

        # Load actual RECOVAR scores array
        scores_array = tp_scores_map[row['filename']]
        example['recovar_result'] = {
            'max_score': row['max_score'],
            'mean_score': row['mean_score'],
            'scores_array': scores_array
        }

        tp_examples.append(example)

    # Load false picks
    fp_examples = []
    for idx, row in fp_metadata.iterrows():
        file_path = falsepicks_dir / row['filename']
        stream = obspy.read(str(file_path))
        stream.merge()

        example = {
            'file': file_path,
            'stream': stream,
            'station': row['station'],
            'phasenet_pick': row['phasenet_pick'],
            'catalog_pick': None,
            'window_start': row['window_start'],
            'window_end': row['window_end'],
            'max_score': row['max_score'],
            'mean_score': row['mean_score']
        }

        # Precompute PhaseNet probabilities if requested
        if precompute_phasenet:
            example['phasenet_result'] = get_phasenet_probabilities(stream)

        # Load actual RECOVAR scores array
        scores_array = fp_scores_map[row['filename']]
        example['recovar_result'] = {
            'max_score': row['max_score'],
            'mean_score': row['mean_score'],
            'scores_array': scores_array
        }

        fp_examples.append(example)

    return tp_examples, fp_examples, tp_metadata, fp_metadata


def print_confusion_matrix(tp_kept, tp_filtered, fp_kept, fp_filtered, threshold):
    TP = len(tp_kept)     
    FN = len(tp_filtered)  
    TN = len(fp_filtered) 
    FP = len(fp_kept)      

    print(f"Performance at threshold = {threshold}:")
    print(f"  Catalog Events RECOVAR kept (TP):   {TP:3d} ")
    print(f"  Catalog Events Recovar lost (FN):  {FN:3d} ")
    print(f"  False Phasenet Pick Recovar FILTERED (TN):   {TN:3d}")
    print(f"  False Phasenet Pick Recovar KEPT (FP):  {FP:3d} ")
