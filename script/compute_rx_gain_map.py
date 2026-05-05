import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# --- Environment Setup ---
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.scene.domain import Scene, Box, Obstacle, Transmitter, Receiver
from src.core.cache import _load_registry, load_scene

# --- Paths ---
CACHE_DIR = _ROOT / "cache"
REGISTRY_PATH = CACHE_DIR / "field_registry.json"
FIELDS_DIR = CACHE_DIR / "precomputed_static_fields"


def get_latest_registry_entry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    if not registry:
        raise ValueError("Registry is empty.")
    # Return the most recent entry (index 0 if appended to top, or sort by created_at)
    return sorted(registry, key=lambda x: x.get('created_at', ''), reverse=True)[0]


def compute_and_plot_2d_gain_map():
    # 1. Load Data
    entry = get_latest_registry_entry()
    npz_path = FIELDS_DIR / entry["filename"]
    scene_path = FIELDS_DIR / entry["scene_filename"]

    print(f"Loading scene: {scene_path.name}")
    scene = load_scene(scene_path)
    
    print(f"Loading field data: {npz_path.name}")
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    
    pos_cpu = data['pos_cpu']       # (N_STEPS, N_RAYS, 3)
    n_pts_cpu = data['n_pts_cpu']   # (N_RAYS,)
    sp_cpu = data['step_powers']    # (N_STEPS, N_RAYS)
    
    N_STEPS, N_RAYS, _ = pos_cpu.shape
    print(f"Data loaded in {time.time() - t0:.2f}s. Rays: {N_RAYS:,}, Max steps: {N_STEPS}")

    # 2. Grid Configuration
    print("Computing vectorized 2D power map...")
    t1 = time.time()
    GRID_RES = 300
    SUB_STEPS = 5  # Sub-segments for continuous rendering

    bmin = scene.box.box_min.astype(np.float32)
    bmax = scene.box.box_max.astype(np.float32)
    span_x = float(bmax[0] - bmin[0])
    span_y = float(bmax[1] - bmin[1])

    # 3. Vectorized Sub-segment Interpolation & FSPL
    alphas = np.linspace(0, 1, SUB_STEPS)[:, None, None, None]
    A = pos_cpu[:-1][None, ...]  # Start points: (1, N_STEPS-1, N_RAYS, 3)
    B = pos_cpu[1:][None, ...]   # End points: (1, N_STEPS-1, N_RAYS, 3)
    
    interp_pts = A + alphas * (B - A)  # (SUB_STEPS, N_STEPS-1, N_RAYS, 3)

    pos_orig = pos_cpu[0][None, None, :, :]  # Ray origins: (1, 1, N_RAYS, 3)
    
    # Distances for FSPL calculation
    dist_to_P = np.linalg.norm(interp_pts - pos_orig, axis=3)
    dist_to_P = np.maximum(dist_to_P, 0.5)

    dist_start_seg = np.linalg.norm(A - pos_orig, axis=3)
    dist_start_seg = np.maximum(dist_start_seg, 0.5)

    pwr_start_seg = sp_cpu[:-1][None, ...]
    
    # Calculate FSPL dynamic power: P = P_start - 20 * log10(d_current / d_start)
    pwr_interp = pwr_start_seg - 20.0 * np.log10(dist_to_P / dist_start_seg)

    # 4. Valid Segment Masking
    step_idx = np.arange(N_STEPS - 1)[:, None]
    valid_mask = (step_idx < (n_pts_cpu[None, :] - 1))[None, ...]
    valid_mask = np.repeat(valid_mask, SUB_STEPS, axis=0)

    pts_v = interp_pts[valid_mask].reshape(-1, 3)
    pwr_v = pwr_interp[valid_mask].ravel()

    # Filter out values below noise floor to avoid processing useless data
    noise_floor = float(scene.noise_floor_dbm) if hasattr(scene, 'noise_floor_dbm') else -120.0
    valid_pwr_mask = pwr_v > noise_floor
    pts_v = pts_v[valid_pwr_mask]
    pwr_v = pwr_v[valid_pwr_mask]

    # 5. Grid Accumulation (Max Power per Cell)
    cx = np.clip(((pts_v[:, 0] - bmin[0]) / span_x * GRID_RES).astype(np.int32), 0, GRID_RES - 1)
    cy = np.clip(((pts_v[:, 1] - bmin[1]) / span_y * GRID_RES).astype(np.int32), 0, GRID_RES - 1)

    grid_lin = np.zeros((GRID_RES, GRID_RES), dtype=np.float32)
    pwr_lin = np.power(10.0, pwr_v / 10.0)
    np.maximum.at(grid_lin, (cx, cy), pwr_lin)

    with np.errstate(divide='ignore'):
        power_map = np.where(grid_lin > 0, 10.0 * np.log10(grid_lin), np.nan)

    # 6. Post-processing (Shadow preservation & Smoothing)
    # Using a 1% percentile to ensure shadow regions are preserved without dropping to pure numerical noise
    p_min_vis = np.nanpercentile(power_map, 1)
    p_max_vis = np.nanmax(power_map)

    # Fill empty space with the minimum visual value before Gaussian filtering to preserve sharp shadow edges
    power_map_filled = np.where(np.isnan(power_map), p_min_vis - 5.0, power_map)
    power_map_smoothed = gaussian_filter(power_map_filled, sigma=0.8)
    
    # Final hard clip to scale limits
    power_map_final = np.clip(power_map_smoothed, p_min_vis, p_max_vis)

    print(f"Map computed in {time.time() - t1:.2f}s. Range: [{p_min_vis:.1f}, {p_max_vis:.1f}] dBm")

    # 7. Visualization
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0a0a12')
    ax.set_facecolor('#0a0a12')

    extent = [bmin[0], bmax[0], bmin[1], bmax[1]]
    im = ax.imshow(power_map_final.T, origin='lower', extent=extent,
                   cmap='inferno', vmin=p_min_vis, vmax=p_max_vis, interpolation='nearest')

    # Draw Obstacles (Black face, Cyan edges for contrast)
    for obs in scene.obstacles:
        w = obs.box_max[0] - obs.box_min[0]
        l = obs.box_max[1] - obs.box_min[1]
        ax.add_patch(plt.Rectangle((obs.box_min[0], obs.box_min[1]), w, l,
                                   fc='#050505', ec='#00ffff', lw=1.2, alpha=0.9, zorder=5))

    # Mark TX and RX
    for tx in scene.transmitters:
        ax.plot(tx.position[0], tx.position[1], '^', color='#ff3333', markersize=10, 
                markeredgecolor='white', markeredgewidth=1, label='TX', zorder=10)


    # Aesthetics
    ax.set_title(f"2D Signal Coverage Map | Hash: {entry['hash']}", color='white', pad=15, fontsize=14)
    ax.set_xlabel("X (m)", color='gray', fontsize=12)
    ax.set_ylabel("Y (m)", color='gray', fontsize=12)
    ax.tick_params(colors='gray')
    ax.grid(color='#222222', linestyle='--', linewidth=0.5, zorder=0)

    # Custom Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Received Power (dBm)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='gray')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#111111', 
              edgecolor='#333333', labelcolor='white')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compute_and_plot_2d_gain_map()