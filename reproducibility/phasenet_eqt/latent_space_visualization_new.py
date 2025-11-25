import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

WAVEFORM_COLORS = ['blue', 'green', 'red']

def _plot_waveform_channel(ax, timesteps, waveform, channel_idx, ylim_min=None, ylim_max=None, color='blue', show_xticks=True):
    """
    Plots a single waveform channel on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        timesteps (np.ndarray): Array of timesteps
        waveform (np.ndarray): Waveform data for one channel
        channel_idx (int): Channel index (0-based)
        color (str): Color for the plot
        show_xticks (bool): Whether to show x-axis tick labels and label
    """
    channels =['E', 'N', 'Z']
    ax.plot(timesteps, waveform, color=color, linewidth=1)
    if channel_idx == 0:
        ax.set_title("Waveform", fontsize=14, pad=5)
        
    if ylim_min != None and ylim_max != None:
        ax.set_ylim(ymin=ylim_min, ymax=ylim_max)
    
    ax.tick_params(axis='y', labelsize=10)
    if show_xticks:
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax.grid(True)

def _plot_heatmap(ax, heatmap, vmin=None, vmax=None, title=None):
    """
    Plots the heatmap on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        heatmap (np.ndarray): Shape (94, 64)
    """
    if vmin != None and vmax != None:
        cax = ax.imshow(heatmap, aspect='auto', cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
    else:
        cax = ax.imshow(heatmap, aspect='auto', cmap='magma', origin='lower')
        
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

def plot_latent_samples(waveform, feature_maps, path):
    fig = plt.figure(figsize=(20, 12))
    main_gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    waveform_gs = main_gs[0, 0].subgridspec(3, 1, hspace=0.3)

    #WAVEFORM
    timesteps_eq = np.arange(waveform.shape[0])

    for channel in range(waveform.shape[1]):
        ax = fig.add_subplot(waveform_gs[channel, 0])
        show_xticks = (channel == waveform.shape[1] - 1)
        _plot_waveform_channel(ax, timesteps_eq, waveform[:, channel], channel,
                              color=WAVEFORM_COLORS[channel % len(WAVEFORM_COLORS)],
                              show_xticks=show_xticks)

    #FEATURE MAPS 
    feature_gs = main_gs[0, 1].subgridspec(5, 1, hspace=0.6)

    for map_idx, fmap in enumerate(feature_maps):
        ax_heatmap = fig.add_subplot(feature_gs[map_idx, 0])
        fmap_np = fmap.numpy() if hasattr(fmap, 'numpy') else fmap
        _plot_heatmap(ax_heatmap, fmap_np.T, title=f"Feature Map {map_idx + 1}")
                    
    fig.text(0.30, 0.935, 'Sample', ha='center', va='center', fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(path)
    plt.close(fig) 