import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker


COLOR_DILATION   = "#d1495b"
COLOR_NODILATION = "#1a1a1a"
PALETTE_FALLBACK = ["#1f6fb4", "#2a9d3a", "#7a4ea8", "#b15928"]


def style_for(exp_name):
    import re
    name = exp_name.upper()
    if re.search(r"(^|[_\W])NO[_]?DILATION([_\W]|$)", name):
        return COLOR_NODILATION, (0, (5, 2))
    if re.search(r"(^|[_\W])(DILATION|DYNAMIC)([_\W0-9]|$)", name):
        return COLOR_DILATION, "-"
    return None, "-"


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#1a1a1a",
        "axes.titlecolor": "#1a1a1a",
        "text.color": "#1a1a1a",
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderaxespad": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_roc(results_dir, exp_name, rep_model, classifier, train_dataset,
             test_dataset, split, epoch):
    base = (Path(results_dir) / exp_name / rep_model / classifier
            / f"training_{train_dataset}" / f"testing_{test_dataset}"
            / f"split{split}")
    meta_path = base / "meta.csv"
    score_path = base / f"scores{epoch}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not score_path.exists():
        raise FileNotFoundError(score_path)

    df_meta = pd.read_csv(meta_path)
    df_score = pd.read_csv(score_path)
    n = min(len(df_meta), len(df_score))
    df_meta = df_meta.iloc[:n]
    df_score = df_score.iloc[:n]

    y_true = (df_meta["label"] == "eq").astype(int).values
    y_score = df_score["eq_probabilities"].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)


def parse_curve(spec):
    parts = spec.split(":")
    if len(parts) == 3:
        exp, train, epoch = parts
        label = None
    elif len(parts) == 4:
        exp, train, epoch, label = parts
    else:
        raise ValueError(f"--curve must be exp:train:epoch[:label], got {spec!r}")
    return {"exp": exp, "train": train, "epoch": int(epoch),
            "label": label or exp}


def save_both(fig, out_path):
    out = Path(out_path)
    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    return png, pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--test-dataset", required=True)
    p.add_argument("--rep-model", default="autoencoder")
    p.add_argument("--classifier", default="autocovariance")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--curve", action="append", required=True,
                   help="exp_name:train_dataset:epoch[:legend_label]")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--log-fpr", action="store_true")
    p.add_argument("--log-tpr", action="store_true")
    p.add_argument("--inset", action="store_true")
    p.add_argument("--inset-xmax", type=float, default=0.2)
    p.add_argument("--inset-ymin", type=float, default=0.8)
    p.add_argument("--xmin-log", type=float, default=1e-4)
    p.add_argument("--ymin-log", type=float, default=1e-2)
    p.add_argument("--figsize", nargs=2, type=float, default=(7.0, 5.4))
    p.add_argument("--no-gap-band", action="store_true")
    args = p.parse_args()

    set_style()
    curves = [parse_curve(c) for c in args.curve]

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    plotted = []
    for i, c in enumerate(curves):
        fpr, tpr, roc_auc = load_roc(
            args.results_dir, c["exp"], args.rep_model, args.classifier,
            c["train"], args.test_dataset, args.split, c["epoch"])
        mapped_color, ls = style_for(c["exp"])
        color = mapped_color or PALETTE_FALLBACK[i % len(PALETTE_FALLBACK)]
        lw = 2.4 if ls == "-" else 1.6
        ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                label=f"{c['label']} (AUC = {roc_auc:.3f})",
                solid_capstyle="round", zorder=4 + i)
        plotted.append((fpr, tpr, color, ls))

    if len(plotted) == 2 and not args.no_gap_band:
        (fpr_a, tpr_a, _, _), (fpr_b, tpr_b, _, _) = plotted
        fpr_grid = np.unique(np.concatenate([fpr_a, fpr_b]))
        tpr_a_i = np.interp(fpr_grid, fpr_a, tpr_a)
        tpr_b_i = np.interp(fpr_grid, fpr_b, tpr_b)
        upper = np.maximum(tpr_a_i, tpr_b_i)
        lower = np.minimum(tpr_a_i, tpr_b_i)
        ax.fill_between(fpr_grid, lower, upper,
                        color="0.55", alpha=0.12, lw=0, zorder=2)

    if not (args.log_fpr or args.log_tpr):
        ax.plot([0, 1], [0, 1], color="0.55", lw=0.7,
                linestyle=(0, (3, 3)), zorder=1)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    if args.title and not args.no_title:
        ax.set_title(args.title)
    elif not args.no_title:
        ax.set_title(f"ROC — test on {args.test_dataset}")

    if args.log_fpr:
        ax.set_xscale("log")
        ax.set_xlim(args.xmin_log, 1.0)
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(
            base=10.0, subs=tuple(range(2, 10)), numticks=12))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    else:
        ax.set_xlim(0, 1)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

    if args.log_tpr:
        ax.set_yscale("log")
        ax.set_ylim(args.ymin_log, 1.05)
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(
            base=10.0, subs=tuple(range(2, 10)), numticks=12))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    else:
        ax.set_ylim(0, 1.005)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

    ax.grid(True, which="major", color="0.7", alpha=0.45, lw=0.5)
    ax.grid(True, which="minor", color="0.7", alpha=0.18, lw=0.4)
    ax.set_axisbelow(True)

    legend_loc = "lower right"
    if args.log_tpr and not args.log_fpr:
        legend_loc = "lower right"
    ax.legend(loc=legend_loc, handlelength=2.6, borderaxespad=0.6)

    if args.inset and not args.log_fpr and not args.log_tpr:
        axins = ax.inset_axes([0.12, 0.55, 0.42, 0.40])
        for fpr, tpr, color, ls in plotted:
            axins.plot(fpr, tpr, color=color, lw=1.6, linestyle=ls)
        axins.set_xlim(0, args.inset_xmax)
        axins.set_ylim(args.inset_ymin, 1.002)
        axins.tick_params(labelsize=8)
        axins.grid(True, alpha=0.2, lw=0.4)
        axins.set_axisbelow(True)
        for s in ("top", "right"):
            axins.spines[s].set_visible(False)
        ax.indicate_inset_zoom(axins, edgecolor="0.5", lw=0.7, alpha=0.8)

    fig.tight_layout()
    png, pdf = save_both(fig, args.out)
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
