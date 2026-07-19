import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kfold_tester import KFoldTester
from recovar import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder


ROOT = Path("/mnt/data_a/ege")
PICKS = Path("/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv")
CATALOG = Path(__file__).resolve().parent.parent / "silivri_durand_catalog.txt"
CACHE = Path(__file__).resolve().parent.parent / "silivri_mag_cache.npz"
PHASENET_CACHE = Path(__file__).resolve().parent.parent / "phasenet_instance_silivri_scores.npz"
RESULTS = ROOT / "recovar_results"
OUTPUT = ROOT / "RECOVAR_SILIVRI2019" / "magnitude_analysis"
EXPERIMENTS = {
    "No dilation": ("SILIVRI2019_NODILATION_20EP", 18, "SILIVRI2019"),
    "Dilation": ("SILIVRI2019_DYNAMIC_64", 6, "SILIVRI2019"),
    "INSTANCE-trained": ("exp_instance", 11, "instance"),
}
MATCH_TOLERANCE_SECONDS = 0.05
COINCIDENCE_WINDOW_SECONDS = 20.0
COINCIDENCE_LEVELS = [1]
HISTOGRAM_EDGES = np.arange(0.5, 5.76, 0.25)
PERCENTILE_COLORS = ["#dbe9f6", "#a9cce8", "#74a9cf", "#3f7fba", "#175a94"]


def result_dir(experiment, train_dataset):
    return RESULTS / experiment / "representation_learning_autoencoder_ensemble" / "representation_cross_covariances" / f"training_{train_dataset}" / "testing_SILIVRI2019" / "split0"


def column(frame, names):
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"none of {names} found; available columns: {list(frame.columns)}")


def parse_mixed_datetime(values):
    try:
        return pd.to_datetime(values, format="mixed")
    except (TypeError, ValueError):
        return values.map(lambda value: pd.to_datetime(value) if pd.notna(value) else pd.NaT)


def prepare_picks(picks):
    picks = picks.copy()
    if not any(name in picks.columns for name in ["event_id", "source_id", "event", "origin_id"]):
        picks["event_id"] = parse_mixed_datetime(picks["orgtime"]).astype(str)
    if any(name in picks.columns for name in ["magnitude", "mag", "source_magnitude", "event_magnitude"]):
        return picks
    for path in PICKS.parent.glob("*.csv"):
        if path == PICKS:
            continue
        candidate = pd.read_csv(path)
        origin_names = [name for name in ["orgtime", "origin_time", "time", "datetime"] if name in candidate.columns]
        magnitude_names = [name for name in ["magnitude", "mag", "source_magnitude", "event_magnitude"] if name in candidate.columns]
        if not origin_names or not magnitude_names:
            continue
        origin = origin_names[0]
        magnitude = magnitude_names[0]
        event_magnitudes = candidate[[origin, magnitude]].dropna().copy()
        event_magnitudes[origin] = parse_mixed_datetime(event_magnitudes[origin])
        event_magnitudes = event_magnitudes.sort_values(origin)
        picks["_orgtime"] = parse_mixed_datetime(picks["orgtime"])
        picks = pd.merge_asof(
            picks.sort_values("_orgtime"),
            event_magnitudes,
            left_on="_orgtime",
            right_on=origin,
            direction="nearest",
            tolerance=pd.Timedelta(seconds=0.1),
        )
        picks["magnitude"] = picks[magnitude]
        if picks["magnitude"].notna().any():
            return picks
    catalog = pd.read_csv(
        CATALOG,
        sep=r"\s+",
        skiprows=1,
        header=None,
        usecols=[1, 2, 3, 4, 5, 6, 11],
        names=["year", "month", "day", "hour", "minute", "second", "magnitude"],
    )
    catalog["catalog_orgtime"] = pd.to_datetime(
        catalog[["year", "month", "day", "hour", "minute"]]
    ) + pd.to_timedelta(catalog["second"], unit="s")
    picks["_orgtime"] = parse_mixed_datetime(picks["orgtime"])
    picks = pd.merge_asof(
        picks.sort_values("_orgtime"),
        catalog[["catalog_orgtime", "magnitude"]].sort_values("catalog_orgtime"),
        left_on="_orgtime",
        right_on="catalog_orgtime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    )
    if not picks["magnitude"].notna().any():
        raise RuntimeError(f"no events in {CATALOG} matched the phase-pick origin times")
    return picks


def load_model_result(experiment, epoch, train_dataset):
    directory = result_dir(experiment, train_dataset)
    if not (directory / "meta.csv").exists() or not (directory / f"scores{epoch}.csv").exists():
        tester = KFoldTester(
            exp_name=experiment,
            representation_learning_model_class=RepresentationLearningMultipleAutoencoder,
            classifier_model_class=ClassifierMultipleAutoencoder,
            train_dataset=train_dataset,
            test_dataset="SILIVRI2019",
            split=0,
            epochs=[epoch],
            apply_resampling=False,
            resample_while_keeping_total_waveforms_fixed=False,
            resample_eq_ratio=0.5,
            method_params={},
        )
        tester.test()
    metadata = pd.read_csv(directory / "meta.csv").drop(columns=["Unnamed: 0", "index"], errors="ignore")
    scores = pd.read_csv(directory / f"scores{epoch}.csv")["eq_probabilities"].to_numpy()
    if len(metadata) != len(scores):
        raise ValueError(f"metadata and scores differ in length for {experiment}")
    metadata["score"] = scores
    return metadata


def load_phasenet_instance_result(reference_metadata):
    if not PHASENET_CACHE.exists():
        raise FileNotFoundError(
            f"{PHASENET_CACHE} is missing; run python silivri_phasenet_cache.py in the PyTorch/SeisBench environment"
        )
    scores = np.load(PHASENET_CACHE)["scores"]
    if len(reference_metadata) != len(scores):
        raise ValueError("PhaseNet scores and Silivri test metadata differ in length")
    result = reference_metadata.copy()
    result["score"] = scores
    return result


def attach_events(metadata, picks):
    station = column(picks, ["station", "station_name", "station_code"])
    arrival = column(picks, ["p_arrival_time", "p_time", "p_pick", "trace_P_arrival_time"])
    event_id = column(picks, ["event_id", "source_id", "event", "origin_id"])
    magnitude = column(picks, ["magnitude", "mag", "source_magnitude", "event_magnitude"])
    metadata = metadata.copy()
    metadata["p_time"] = parse_mixed_datetime(metadata["trace_start_time"]) + pd.to_timedelta(
        pd.to_numeric(metadata["p_arrival_sample"], errors="coerce") / 100.0,
        unit="s",
    )
    metadata["station_key"] = metadata["station_name"].astype(str)
    picks = picks.copy()
    picks["station_key"] = picks[station].astype(str)
    picks[arrival] = parse_mixed_datetime(picks[arrival])
    events = metadata.loc[
        metadata["label"].eq("eq") & metadata["p_time"].notna() & metadata["station_key"].notna()
    ].sort_values("p_time")
    picks = picks.loc[picks[arrival].notna() & picks["station_key"].notna()].sort_values(arrival)
    matched = pd.merge_asof(
        events,
        picks[["station_key", arrival, event_id, magnitude]],
        left_on="p_time",
        right_on=arrival,
        by="station_key",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_SECONDS),
    )
    if matched[event_id].isna().any():
        raise RuntimeError(f"{matched[event_id].isna().sum()} event windows did not match the pick catalog")
    return matched, event_id, magnitude


def nth_station_score(frame, group_column, min_stations):
    station_scores = frame.groupby([group_column, "station_name"], as_index=False)["score"].max()
    return station_scores.groupby(group_column)["score"].apply(
        lambda values: np.sort(values.to_numpy())[-min_stations] if len(values) >= min_stations else -np.inf
    )


def event_scores_by_coincidence(matched, event_id, magnitude, min_stations):
    scores = nth_station_score(matched, event_id, min_stations).rename("score")
    magnitudes = matched.groupby(event_id)[magnitude].first()
    return pd.concat([magnitudes, scores], axis=1).reset_index()


def noise_coincidence_scores(metadata, min_stations):
    noise = metadata.loc[metadata["label"].eq("no")].copy()
    crop_values = noise["crop_offset"] if "crop_offset" in noise.columns else pd.Series(0, index=noise.index)
    crop_offset = pd.to_numeric(crop_values, errors="coerce").fillna(0)
    noise["detection_time"] = parse_mixed_datetime(noise["trace_start_time"]) + pd.to_timedelta(
        crop_offset / 100.0 + 15.0,
        unit="s",
    )
    noise = noise.dropna(subset=["detection_time", "score", "station_name"]).sort_values("detection_time")
    times = noise["detection_time"].to_numpy(dtype="datetime64[ns]")
    stations = noise["station_name"].astype(str).to_numpy()
    scores = noise["score"].to_numpy()
    window = np.timedelta64(int(COINCIDENCE_WINDOW_SECONDS * 1e9), "ns")
    coincidence_scores = []
    i = 0
    while i < len(noise):
        j = int(np.searchsorted(times, times[i] + window, side="right"))
        station_maxima = {}
        for station, score in zip(stations[i:j], scores[i:j]):
            station_maxima[station] = max(score, station_maxima.get(station, -np.inf))
        if len(station_maxima) >= min_stations:
            coincidence_scores.append(np.sort(np.fromiter(station_maxima.values(), dtype=float))[-min_stations])
        i = j
    return np.sort(np.asarray(coincidence_scores, dtype=float))


def magnitude_auc_rows(model_events, model_noise, magnitude, magnitude_edges):
    rows = []
    bins = pd.IntervalIndex.from_breaks(magnitude_edges, closed="left")
    for model_name in model_events:
        for min_stations in COINCIDENCE_LEVELS:
            events = model_events[model_name][min_stations].copy()
            noise = model_noise[model_name][min_stations]
            finite = np.concatenate([events["score"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(), noise])
            floor = np.min(finite) - max(np.ptp(finite), 1.0) * 1e-6
            noise = np.where(np.isfinite(noise), noise, floor)
            events["score"] = events["score"].where(np.isfinite(events["score"]), floor)
            events["magnitude_bin"] = pd.cut(events[magnitude], magnitude_edges, right=False)
            for interval in bins:
                positive = events.loc[events["magnitude_bin"].eq(interval), "score"].to_numpy()
                auc = np.nan
                if len(positive) and len(noise):
                    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(noise))])
                    scores = np.concatenate([positive, noise])
                    auc = roc_auc_score(labels, scores)
                rows.append(
                    {
                        "model": model_name,
                        "min_stations": min_stations,
                        "magnitude_bin": str(interval),
                        "magnitude_left": interval.left,
                        "magnitude_right": interval.right,
                        "n_events": len(positive),
                        "n_noise_associations": len(noise),
                        "roc_auc": auc,
                    }
                )
    return pd.DataFrame(rows)


def save_magnitude_histogram(magnitudes, title, stem):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.hist(magnitudes, bins=HISTOGRAM_EDGES, color="0.45", edgecolor="white", linewidth=0.7)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of events")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def save_test_histogram(magnitudes, magnitude_edges):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.hist(magnitudes, bins=HISTOGRAM_EDGES, color="0.45", edgecolor="white", linewidth=0.7)
    for percentile, value, color in zip(
        [20, 40, 60, 80], magnitude_edges[1:-1], PERCENTILE_COLORS[1:]
    ):
        ax.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{percentile}th percentile ($M={value:.2f}$)",
        )
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of events")
    ax.set_title("Matched test-set magnitude distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "silivri_test_magnitude_histogram.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "silivri_test_magnitude_histogram.png", dpi=300, bbox_inches="tight")


def main():
    picks = prepare_picks(pd.read_csv(PICKS))
    model_events = {}
    model_noise = {}
    event_id = None
    magnitude = None
    reference_metadata = None
    for label, (experiment, epoch, train_dataset) in EXPERIMENTS.items():
        metadata = load_model_result(experiment, epoch, train_dataset)
        reference_metadata = metadata
        matched, event_id, magnitude = attach_events(metadata, picks)
        model_events[label] = {}
        model_noise[label] = {}
        for min_stations in COINCIDENCE_LEVELS:
            events = event_scores_by_coincidence(matched, event_id, magnitude, min_stations)
            noise_scores = noise_coincidence_scores(metadata, min_stations)
            model_events[label][min_stations] = events
            model_noise[label][min_stations] = noise_scores
    phasenet_metadata = load_phasenet_instance_result(reference_metadata)
    matched, event_id, magnitude = attach_events(phasenet_metadata, picks)
    model_events["PhaseNet-INSTANCE"] = {}
    model_noise["PhaseNet-INSTANCE"] = {}
    for min_stations in COINCIDENCE_LEVELS:
        model_events["PhaseNet-INSTANCE"][min_stations] = event_scores_by_coincidence(
            matched, event_id, magnitude, min_stations
        )
        model_noise["PhaseNet-INSTANCE"][min_stations] = noise_coincidence_scores(
            phasenet_metadata, min_stations
        )
    cache = {}
    for model_name in model_events:
        key = model_name.lower().replace(" ", "_")
        for min_stations in COINCIDENCE_LEVELS:
            events = model_events[model_name][min_stations]
            prefix = f"{key}_{min_stations}station"
            cache[f"{prefix}_event_id"] = events[event_id].astype(str).to_numpy()
            cache[f"{prefix}_magnitude"] = events[magnitude].to_numpy(dtype=float)
            cache[f"{prefix}_event_score"] = events["score"].to_numpy(dtype=float)
            cache[f"{prefix}_noise_score"] = model_noise[model_name][min_stations]
    np.savez_compressed(CACHE, **cache)
    reference_events = model_events[next(iter(model_events))][1]
    magnitude_edges = np.unique(np.quantile(reference_events[magnitude].dropna(), np.linspace(0.0, 1.0, 6)))
    magnitude_edges[0] = np.nextafter(magnitude_edges[0], -np.inf)
    magnitude_edges[-1] = np.nextafter(magnitude_edges[-1], np.inf)
    auc_summary = magnitude_auc_rows(model_events, model_noise, magnitude, magnitude_edges)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    auc_summary.to_csv(OUTPUT / "silivri_auc_by_magnitude.csv", index=False)
    model_names = list(model_events)
    colors = {"No dilation": "#3b6ea8", "Dilation": "#d95f45", "INSTANCE-trained": "#4c956c", "PhaseNet-INSTANCE": "#8b5ea7"}
    auc_fig, ax = plt.subplots(figsize=(9.0, 5.2))
    labels = [
        f"{left_pct}–{right_pct}%\n[{left:.2f}, {right:.2f})"
        for left_pct, right_pct, left, right in zip(
            range(0, 100, 20), range(20, 101, 20), magnitude_edges[:-1], magnitude_edges[1:]
        )
    ]
    x = np.arange(len(labels))
    width = 0.8 / len(model_names)
    truth = pd.read_csv(CATALOG, sep=r"\s+", skiprows=1, header=None, usecols=[11], names=["magnitude"])
    save_magnitude_histogram(
        truth["magnitude"].dropna().to_numpy(),
        "Durand catalog magnitude distribution",
        "silivri_catalog_magnitude_histogram",
    )
    save_test_histogram(reference_events[magnitude].dropna().to_numpy(), magnitude_edges)
    for col, model_name in enumerate(model_names):
        values = auc_summary.loc[
            auc_summary["model"].eq(model_name) & auc_summary["min_stations"].eq(1)
        ]
        offset = (col - (len(model_names) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values["roc_auc"].to_numpy(),
            width=width,
            color=colors[model_name],
            edgecolor="0.2",
            linewidth=0.5,
            label=model_name,
        )
        counts = values["n_events"].to_numpy()
        for bar, count in zip(bars, counts):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"n={count}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("1-station detection")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_xlabel("Magnitude interval")
    auc_fig.tight_layout()
    auc_fig.savefig(OUTPUT / "silivri_auc_by_magnitude.pdf", bbox_inches="tight")
    auc_fig.savefig(OUTPUT / "silivri_auc_by_magnitude.png", dpi=300, bbox_inches="tight")
    print(OUTPUT / "silivri_auc_by_magnitude.pdf")


if __name__ == "__main__":
    main()
