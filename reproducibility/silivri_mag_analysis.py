import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kfold_tester import KFoldTester
from recovar import ClassifierMultipleAutoencoder, RepresentationLearningMultipleAutoencoder


ROOT = Path("/mnt/data_a/ege")
PICKS = Path("/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30/processed_catalogs/kara74a_phase_picks.csv")
CATALOG = Path(__file__).resolve().parent.parent / "silivri_durand_catalog.txt"
RESULTS = ROOT / "recovar_results"
OUTPUT = ROOT / "RECOVAR_SILIVRI2019" / "magnitude_analysis"
EXPERIMENTS = {
    "No dilation": ("SILIVRI2019_NODILATION_20EP", 18),
    "Dilation": ("SILIVRI2019_DYNAMIC_64", 6),
}
TARGET_FPR = 0.01
MATCH_TOLERANCE_SECONDS = 0.05
BIN_EDGES = np.arange(0.5, 6.6, 0.5)


def result_dir(experiment):
    return RESULTS / experiment / "representation_learning_autoencoder_ensemble" / "representation_cross_covariances" / "training_SILIVRI2019" / "testing_SILIVRI2019" / "split0"


def column(frame, names):
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"none of {names} found; available columns: {list(frame.columns)}")


def prepare_picks(picks):
    picks = picks.copy()
    if not any(name in picks.columns for name in ["event_id", "source_id", "event", "origin_id"]):
        picks["event_id"] = pd.to_datetime(picks["orgtime"]).astype(str)
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
        event_magnitudes[origin] = pd.to_datetime(event_magnitudes[origin])
        event_magnitudes = event_magnitudes.sort_values(origin)
        picks["_orgtime"] = pd.to_datetime(picks["orgtime"])
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
    picks["_orgtime"] = pd.to_datetime(picks["orgtime"])
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


def load_model_result(experiment, epoch):
    directory = result_dir(experiment)
    if not (directory / "meta.csv").exists() or not (directory / f"scores{epoch}.csv").exists():
        tester = KFoldTester(
            exp_name=experiment,
            representation_learning_model_class=RepresentationLearningMultipleAutoencoder,
            classifier_model_class=ClassifierMultipleAutoencoder,
            train_dataset="SILIVRI2019",
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


def attach_events(metadata, picks):
    station = column(picks, ["station", "station_name", "station_code"])
    arrival = column(picks, ["p_arrival_time", "p_time", "p_pick", "trace_P_arrival_time"])
    event_id = column(picks, ["event_id", "source_id", "event", "origin_id"])
    magnitude = column(picks, ["magnitude", "mag", "source_magnitude", "event_magnitude"])
    metadata = metadata.copy()
    metadata["p_time"] = pd.to_datetime(metadata["trace_start_time"]) + pd.to_timedelta(
        pd.to_numeric(metadata["p_arrival_sample"], errors="coerce") / 100.0,
        unit="s",
    )
    metadata["station_key"] = metadata["station_name"].astype(str)
    picks = picks.copy()
    picks["station_key"] = picks[station].astype(str)
    picks[arrival] = pd.to_datetime(picks[arrival])
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


def threshold_at_fpr(metadata):
    noise = metadata.loc[metadata["label"].eq("no"), "score"].dropna().to_numpy()
    if not len(noise):
        raise RuntimeError("no noise windows available for threshold selection")
    return float(np.quantile(noise, 1.0 - TARGET_FPR))


def main():
    picks = prepare_picks(pd.read_csv(PICKS))
    model_events = {}
    thresholds = {}
    event_id = None
    magnitude = None
    for label, (experiment, epoch) in EXPERIMENTS.items():
        metadata = load_model_result(experiment, epoch)
        thresholds[label] = threshold_at_fpr(metadata)
        matched, event_id, magnitude = attach_events(metadata, picks)
        model_events[label] = matched.groupby(event_id, as_index=False).agg({magnitude: "first", "score": "max"})
    catalog_events = picks[[event_id, magnitude]].dropna().drop_duplicates(event_id)
    summaries = []
    for label, events in model_events.items():
        events = events.copy()
        events["detected"] = events["score"] >= thresholds[label]
        events["magnitude_bin"] = pd.cut(events[magnitude], BIN_EDGES, right=False)
        summary = events.groupby("magnitude_bin", observed=False)["detected"].agg(["sum", "count"]).reset_index()
        summary["recall"] = summary["sum"] / summary["count"].replace(0, np.nan)
        summary["model"] = label
        summary["threshold"] = thresholds[label]
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "silivri_recall_by_magnitude.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    axes[0].hist(catalog_events[magnitude], bins=BIN_EDGES, color="0.45", edgecolor="white")
    axes[0].set_ylabel("Catalog events")
    centers = BIN_EDGES[:-1] + np.diff(BIN_EDGES) / 2
    for label in EXPERIMENTS:
        recall = summary.loc[summary["model"].eq(label), "recall"].to_numpy()
        axes[1].plot(centers, recall, marker="o", label=label)
    axes[1].set_xlabel("Magnitude")
    axes[1].set_ylabel("Event recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "silivri_magnitude_analysis.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "silivri_magnitude_analysis.png", dpi=300, bbox_inches="tight")
    print(OUTPUT / "silivri_magnitude_analysis.pdf")


if __name__ == "__main__":
    main()
