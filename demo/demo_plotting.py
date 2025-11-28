"""
Plotting utilities for YAZEL RECOVAR demo notebook.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_phasenet_probabilities(stream):
    p_trace = stream.select(channel='Pha')[0]
    sampling_rate = p_trace.stats.sampling_rate
    times = np.arange(len(p_trace.data)) / sampling_rate

    return {
        'p_prob': 1 - p_trace.data,
        'times': times,
        'sampling_rate': sampling_rate
    }


def plot_side_by_side_comparison(example1, result1, phasenet1, example2, result2, phasenet2,
                                  threshold=0.07, score_type='max'):
    """
    Plot two examples side by side for comparison.
    """
    fig, axes = plt.subplots(5, 2, figsize=(18, 12), sharex='col')

    for col, (example, recovar_result, phasenet_result) in enumerate([(example1, result1, phasenet1),
                                                                        (example2, result2, phasenet2)]):
        stream = example['stream']

        # Separate waveforms from annotations
        waveform_traces = [tr for tr in stream if not tr.stats.channel.endswith(('P', 'S', 'N')) or len(tr.stats.channel) > 1]
        z_trace = [tr for tr in waveform_traces if 'Z' in tr.stats.channel][0]
        n_trace = [tr for tr in waveform_traces if 'N' in tr.stats.channel or 'Y' in tr.stats.channel][0]
        e_trace = [tr for tr in waveform_traces if 'E' in tr.stats.channel or 'X' in tr.stats.channel][0]

        # Time arrays
        sampling_rate = z_trace.stats.sampling_rate
        times = np.arange(len(z_trace.data)) / sampling_rate

        # Calculate pick positions
        window_start = example['window_start']
        phasenet_pick = example['phasenet_pick']
        catalog_pick = example['catalog_pick']

        phasenet_pick_sec = (phasenet_pick - window_start).total_seconds()
        if catalog_pick:
            catalog_pick_sec = (catalog_pick - window_start).total_seconds()

        # Determine filtering decision
        score_value = recovar_result['max_score'] if score_type == 'max' else recovar_result['mean_score']
        is_kept = score_value >= threshold
        is_true_pick = catalog_pick is not None

        # Sliding window parameters
        window_size = 3000
        stride = 100
        trim_samples = 500

        # Calculate window centers
        n_windows = len(recovar_result['scores_array'])
        window_centers = [(i * stride + trim_samples + window_size // 2) / sampling_rate
                         for i in range(n_windows)]
        scores = recovar_result['scores_array']

        # Plot waveforms
        for row, (trace, comp) in enumerate(zip([e_trace, n_trace, z_trace], ['E', 'N', 'Z'])):
            ax = axes[row, col]
            ax.plot(times, trace.data, 'k-', linewidth=0.5, alpha=0.7)
            ax.set_ylabel(f'{comp}', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Mark picks
            ax.axvline(phasenet_pick_sec, color='blue', linestyle='--',
                      linewidth=2, alpha=0.7)
            if catalog_pick:
                ax.axvline(catalog_pick_sec, color='green', linestyle='--',
                          linewidth=2, alpha=0.7)

        # Plot PhaseNet probabilities
        ax_phasenet = axes[3, col]
        pn_times = phasenet_result['times']
        pn_probs = phasenet_result['p_prob']

        ax_phasenet.plot(pn_times, pn_probs, 'b-', linewidth=1.5, alpha=0.8)
        ax_phasenet.fill_between(pn_times, 0, pn_probs, color='blue', alpha=0.2)
        ax_phasenet.axhline(0.3, color='red', linestyle=':', linewidth=2.5, alpha=0.8)
        ax_phasenet.axvline(phasenet_pick_sec, color='blue', linestyle='--', linewidth=2, alpha=0.5)

        ax_phasenet.set_ylabel('PhaseNet\nP-prob', fontsize=10)
        ax_phasenet.set_ylim(-0.05, 1.05)
        ax_phasenet.grid(True, alpha=0.3)

        # Plot RECOVAR scores
        ax_score = axes[4, col]
        colors = plt.cm.RdYlGn(scores)

        for i in range(len(window_centers)):
            ax_score.scatter(window_centers[i], scores[i], c=[colors[i]],
                           s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

        ax_score.plot(window_centers, scores, 'k-', linewidth=1, alpha=0.3)
        ax_score.axhline(threshold, color='purple', linestyle='-.', linewidth=2,
                        label=f'Threshold: {threshold:.2f}', alpha=0.8)
        ax_score.axhline(score_value, color='red', linestyle='--', linewidth=2,
                        label=f'{score_type.title()}: {score_value:.3f}', alpha=0.7)

        ax_score.set_ylabel('RECOVAR\nScore', fontsize=10)
        ax_score.set_xlabel('Time (seconds)', fontsize=10)
        ax_score.set_ylim(-0.05, 1.05)
        ax_score.grid(True, alpha=0.3)
        ax_score.legend(loc='upper right', fontsize=8)

        # Column title
        pick_type = "TRUE PICK" if is_true_pick else "FALSE PICK"
        decision = "KEPT" if is_kept else "FILTERED"
        decision_color = 'green' if (is_true_pick and is_kept) or (not is_true_pick and not is_kept) else 'red'

        axes[0, col].set_title(f"{pick_type} - {decision}\nStation: {example['station']}",
                              fontsize=12, fontweight='bold', color=decision_color, pad=10)

    plt.tight_layout()
    plt.show()


def plot_example(example, recovar_result, phasenet_result, threshold=0.07, score_type='max'):
    """
    Plot a single example showing waveforms, PhaseNet probabilities, and RECOVAR scores.
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    stream = example['stream']

    # Separate waveforms from annotations
    waveform_traces = [tr for tr in stream if not tr.stats.channel.endswith(('P', 'S', 'N')) or len(tr.stats.channel) > 1]
    z_trace = [tr for tr in waveform_traces if 'Z' in tr.stats.channel][0]
    n_trace = [tr for tr in waveform_traces if 'N' in tr.stats.channel or 'Y' in tr.stats.channel][0]
    e_trace = [tr for tr in waveform_traces if 'E' in tr.stats.channel or 'X' in tr.stats.channel][0]

    # Time arrays
    sampling_rate = z_trace.stats.sampling_rate
    times = np.arange(len(z_trace.data)) / sampling_rate

    # Calculate pick positions
    window_start = example['window_start']
    phasenet_pick = example['phasenet_pick']
    catalog_pick = example['catalog_pick']

    phasenet_pick_sec = (phasenet_pick - window_start).total_seconds()
    if catalog_pick:
        catalog_pick_sec = (catalog_pick - window_start).total_seconds()

    # Determine filtering decision
    score_value = recovar_result['max_score'] if score_type == 'max' else recovar_result['mean_score']
    is_kept = score_value >= threshold
    is_true_pick = catalog_pick is not None

    # Sliding window parameters
    window_size = 3000
    stride = 100
    trim_samples = 500

    # Calculate window centers
    n_windows = len(recovar_result['scores_array'])
    window_centers = [(i * stride + trim_samples + window_size // 2) / sampling_rate
                     for i in range(n_windows)]
    scores = recovar_result['scores_array']

    # Plot waveforms
    for row, (trace, comp) in enumerate(zip([e_trace, n_trace, z_trace], ['E', 'N', 'Z'])):
        ax = axes[row]
        ax.plot(times, trace.data, 'k-', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(f'{comp}', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark picks
        ax.axvline(phasenet_pick_sec, color='blue', linestyle='--',
                  linewidth=2, alpha=0.7, label='PhaseNet pick' if row == 0 else '')
        if catalog_pick:
            ax.axvline(catalog_pick_sec, color='green', linestyle='--',
                      linewidth=2, alpha=0.7, label='Catalog pick' if row == 0 else '')

        if row == 0 and catalog_pick:
            ax.legend(loc='upper right', fontsize=8)

    # Plot PhaseNet probabilities
    ax_phasenet = axes[3]
    pn_times = phasenet_result['times']
    pn_probs = phasenet_result['p_prob']

    ax_phasenet.plot(pn_times, pn_probs, 'b-', linewidth=1.5, alpha=0.8)
    ax_phasenet.fill_between(pn_times, 0, pn_probs, color='blue', alpha=0.2)
    ax_phasenet.axhline(0.3, color='red', linestyle=':', linewidth=2.5, alpha=0.8, label='Threshold')
    ax_phasenet.axvline(phasenet_pick_sec, color='blue', linestyle='--', linewidth=2, alpha=0.5)

    ax_phasenet.set_ylabel('PhaseNet\nP-prob', fontsize=10)
    ax_phasenet.set_ylim(-0.05, 1.05)
    ax_phasenet.grid(True, alpha=0.3)

    # Plot RECOVAR scores
    ax_score = axes[4]
    colors = plt.cm.RdYlGn(scores)

    for i in range(len(window_centers)):
        ax_score.scatter(window_centers[i], scores[i], c=[colors[i]],
                       s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

    ax_score.plot(window_centers, scores, 'k-', linewidth=1, alpha=0.3)
    ax_score.axhline(threshold, color='purple', linestyle='-.', linewidth=2,
                    label=f'Threshold: {threshold:.2f}', alpha=0.8)
    ax_score.axhline(score_value, color='red', linestyle='--', linewidth=2,
                    label=f'{score_type.title()}: {score_value:.3f}', alpha=0.7)

    ax_score.set_ylabel('RECOVAR\nScore', fontsize=10)
    ax_score.set_xlabel('Time (seconds)', fontsize=10)
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.grid(True, alpha=0.3)
    ax_score.legend(loc='upper right', fontsize=8)

    # Title
    pick_type = "TRUE PICK" if is_true_pick else "FALSE PICK"
    decision = "KEPT" if is_kept else "FILTERED"
    decision_color = 'green' if (is_true_pick and is_kept) or (not is_true_pick and not is_kept) else 'red'

    fig.suptitle(f"{pick_type} - {decision}\nStation: {example['station']}",
                 fontsize=14, fontweight='bold', color=decision_color, y=0.995)

    plt.tight_layout()
    plt.show()


def plot_threshold_tradeoff(tp_max_scores, fp_max_scores, recommended_threshold=0.07):
    thresholds = np.linspace(0, 0.3, 100)

    missed_tps = []  # True positives that would be missed (filtered)
    filtered_fps = []  # False positives that would be filtered

    for thresh in thresholds:
        # Count TPs with score < threshold (these would be missed)
        missed = sum(1 for score in tp_max_scores if score < thresh)
        missed_tps.append(missed)

        # Count FPs with score < threshold (these would be filtered correctly)
        filtered = sum(1 for score in fp_max_scores if score < thresh)
        filtered_fps.append(filtered)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot missed TPs
    color1 = 'tab:red'
    ax1.set_xlabel('RECOVAR Threshold', fontsize=12)
    ax1.set_ylabel('Missed Catalog (True) Picks', color=color1, fontsize=12)
    line1 = ax1.plot(thresholds, missed_tps, color=color1, linewidth=2, label='Missed TPs')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for filtered FPs
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Filtered False Picks', color=color2, fontsize=12)
    line2 = ax2.plot(thresholds, filtered_fps, color=color2, linewidth=2, label='Filtered FPs')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add vertical line at recommended threshold
    ax1.axvline(x=recommended_threshold, color='black', linestyle='--', linewidth=1.5,
                label=f'Recommended threshold ({recommended_threshold})', alpha=0.8)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines] + [f'Recommended threshold ({recommended_threshold})']
    ax1.legend(lines + [ax1.axvline(x=recommended_threshold, color='black', linestyle='--')],
               labels, loc='center left', fontsize=10)

    plt.title('Threshold Trade-off: Missed TPs vs Filtered FPs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print stats at recommended threshold
    idx = np.argmin(np.abs(thresholds - recommended_threshold))
    print(f"\nAt recommended threshold {recommended_threshold}:")
    print(f"  Missed TPs: {missed_tps[idx]}/{len(tp_max_scores)}")
    print(f"  Filtered FPs: {filtered_fps[idx]}/{len(fp_max_scores)}")

    return missed_tps[idx], filtered_fps[idx]
