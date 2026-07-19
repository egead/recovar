from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import seisbench.models as sbm
from obspy import UTCDateTime


ROOT = Path("/mnt/data_a/ege")
RAW = Path("/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/prepared_waveforms/day_by_day")
RESULT = ROOT / "recovar_results" / "exp_instance" / "representation_learning_autoencoder_ensemble" / "representation_cross_covariances" / "training_instance" / "testing_SILIVRI2019" / "split0" / "meta.csv"
OUTPUT = Path(__file__).resolve().parent.parent / "phasenet_instance_silivri_scores.npz"
WINDOW_SECONDS = 30.0
CONTEXT_SECONDS = 60.0
JOIN_GAP_SECONDS = 120.0


def parse_mixed_datetime(values):
    try:
        return pd.to_datetime(values, format="mixed")
    except (TypeError, ValueError):
        return values.map(lambda value: pd.to_datetime(value) if pd.notna(value) else pd.NaT)


def station_directories():
    directories = {}
    for directory in RAW.iterdir():
        if directory.is_dir():
            directories[directory.name.upper()] = directory
    return directories


def waveform_index(directory, station):
    entries = []
    paths = sorted(set(directory.rglob("*.mseed")) | set(directory.rglob("*.miniseed")))
    if not paths:
        raise FileNotFoundError(f"no MiniSEED files found in {directory}")
    for path in paths:
        try:
            stream = obspy.read(str(path), headonly=True)
        except Exception:
            continue
        matching = [trace for trace in stream if trace.stats.station.upper() == station.upper()]
        if matching:
            entries.append((path, min(trace.stats.starttime for trace in matching), max(trace.stats.endtime for trace in matching)))
    if not entries:
        raise RuntimeError(f"no readable MiniSEED headers found for station {station} in {directory}")
    return entries


def join_ranges(rows):
    groups = []
    for row in rows.sort_values("window_start").itertuples():
        start = UTCDateTime(row.window_start.to_pydatetime())
        end = start + WINDOW_SECONDS
        if groups and start <= groups[-1][1] + JOIN_GAP_SECONDS:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append((row.Index, start, end))
        else:
            groups.append([start, end, [(row.Index, start, end)]])
    return groups


def read_raw_stream(entries, station, start, end):
    paths = [path for path, file_start, file_end in entries if file_end >= start and file_start <= end]
    if not paths:
        raise RuntimeError(f"no raw waveform overlaps {station} {start} to {end}")
    stream = obspy.Stream()
    for path in paths:
        stream += obspy.read(str(path), starttime=start, endtime=end)
    stream = obspy.Stream([trace for trace in stream if trace.stats.station.upper() == station.upper()])
    if not stream:
        raise RuntimeError(f"no traces for {station} {start} to {end}")
    stream.merge(method=1, fill_value=0)
    stream.trim(start, end)
    return stream


def p_probability_traces(model, stream):
    annotations = model.annotate(stream)
    return [trace for trace in annotations if trace.stats.channel.upper().endswith("P")]


def window_score(traces, start, end):
    maxima = []
    for trace in traces:
        overlap_start = max(start, trace.stats.starttime)
        overlap_end = min(end, trace.stats.endtime)
        if overlap_end < overlap_start:
            continue
        data = trace.slice(overlap_start, overlap_end).data
        if len(data):
            maxima.append(float(np.nanmax(data)))
    if not maxima:
        raise RuntimeError(f"PhaseNet produced no P probabilities for {start} to {end}")
    return max(maxima)


def main():
    if not RESULT.exists():
        raise FileNotFoundError(f"test metadata not found: {RESULT}")
    if not RAW.exists():
        raise FileNotFoundError(f"original waveform directory not found: {RAW}")
    metadata = pd.read_csv(RESULT).drop(columns=["Unnamed: 0", "index"], errors="ignore")
    required = {"trace_name", "trace_start_time", "station_name", "crop_offset"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"missing test metadata columns: {sorted(missing)}")
    metadata["window_start"] = parse_mixed_datetime(metadata["trace_start_time"]) + pd.to_timedelta(
        pd.to_numeric(metadata["crop_offset"], errors="raise") / 100.0,
        unit="s",
    )
    if metadata["window_start"].isna().any():
        raise ValueError("test metadata contains invalid window times")
    directories = station_directories()
    model = sbm.PhaseNet.from_pretrained("instance")
    model.eval()
    scores = np.full(len(metadata), np.nan, dtype=np.float32)
    completed = 0
    total = len(metadata)
    for station, rows in metadata.groupby("station_name", sort=True):
        station_key = str(station).upper()
        if station_key not in directories:
            candidates = [path for key, path in directories.items() if station_key in key]
            if len(candidates) != 1:
                raise FileNotFoundError(f"could not identify raw waveform directory for station {station}")
            directory = candidates[0]
        else:
            directory = directories[station_key]
        entries = waveform_index(directory, station_key)
        for first, last, windows in join_ranges(rows):
            raw_start = first - CONTEXT_SECONDS
            raw_end = last + CONTEXT_SECONDS
            stream = read_raw_stream(entries, station_key, raw_start, raw_end)
            traces = p_probability_traces(model, stream)
            if not traces:
                raise RuntimeError(f"PhaseNet produced no P channel for {station} {raw_start} to {raw_end}")
            for index, start, end in windows:
                scores[index] = window_score(traces, start, end)
                completed += 1
            print(f"PhaseNet completed:{completed}/{total}")
    if not np.isfinite(scores).all():
        missing_indices = np.flatnonzero(~np.isfinite(scores))
        raise RuntimeError(f"missing PhaseNet scores for metadata rows {missing_indices[:20].tolist()}")
    np.savez_compressed(OUTPUT, scores=scores, trace_names=metadata["trace_name"].astype(str).to_numpy())
    print(OUTPUT)


if __name__ == "__main__":
    main()
