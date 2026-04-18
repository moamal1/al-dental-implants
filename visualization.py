"""
Visualization utilities for dental implant planning.

Includes:
- Basic 3-view planning figure (pipeline output)
- Advanced multi-view viewer with bone density analysis
- Nerve mask post-processing
- Bone density classification
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import (
    label,
    binary_opening,
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
)

import config

NERVE_OVERLAY_COLOR = "darkorange"
IMPLANT_CENTER_COLOR = "deepskyblue"

# ── Density overlay colormap ────────────────────────────────────────────────
_DENSITY_COLORS = [
    (0.2, 0.2, 0.2, 0.0),
    (0.8, 0.6, 0.2, 0.3),
    (1.0, 1.0, 1.0, 0.5),
]
DENSITY_CMAP = LinearSegmentedColormap.from_list("density_cmap", _DENSITY_COLORS)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def norm_slice(arr2d):
    """Normalize a 2D slice to [0, 1] using 1st–99th percentile clipping."""
    arr = arr2d.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 - p1 < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - p1) / (p99 - p1), 0, 1)


def compute_implant_voxel_dims(diameter_mm, length_mm, spacing):
    """Convert implant physical dimensions to voxel units."""
    spacing_mean = float(np.mean(spacing))
    radius_vox = max(2.0, (diameter_mm / spacing_mean) / 2.0)
    half_len_vox = max(3, int(round((length_mm / spacing_mean) / 2.0)))
    return radius_vox, half_len_vox


# ═══════════════════════════════════════════════════════════════════════════
# Nerve mask post-processing
# ═══════════════════════════════════════════════════════════════════════════

def _foreground_crop_slices(mask, margin=8):
    """Return a padded foreground bounding box for sparse 3-D masks."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return (slice(0, mask.shape[0]),
                slice(0, mask.shape[1]),
                slice(0, mask.shape[2]))

    mins = np.maximum(coords.min(axis=0) - int(margin), 0)
    maxs = np.minimum(coords.max(axis=0) + int(margin) + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))

def keep_largest_components(mask, num_keep=2, min_size=200):
    """Keep only the N largest connected components above a minimum size."""
    struct = generate_binary_structure(3, 2)
    labeled, num = label(mask, structure=struct)

    if num == 0:
        return mask.astype(np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    if len(component_sizes) <= 1:
        return mask.astype(np.uint8)

    candidate_ids = np.flatnonzero(component_sizes >= int(min_size))
    candidate_ids = candidate_ids[candidate_ids != 0]

    if len(candidate_ids) == 0:
        return mask.astype(np.uint8)

    ranked = candidate_ids[np.argsort(component_sizes[candidate_ids])[::-1]]
    keep_ids = ranked[:num_keep]
    return np.isin(labeled, keep_ids).astype(np.uint8)


def postprocess_nerve_mask(nerve_mask_raw):
    """Clean the predicted nerve mask with morphology inside a tight ROI."""
    raw_mask = nerve_mask_raw.astype(bool)
    if not np.any(raw_mask):
        return np.zeros_like(nerve_mask_raw, dtype=np.uint8)

    crop_slices = _foreground_crop_slices(
        raw_mask,
        margin=config.NERVE_MASK_CROP_MARGIN,
    )
    mask = raw_mask[crop_slices]

    struct = generate_binary_structure(3, 1)
    mask = binary_opening(mask, structure=struct, iterations=1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    mask = binary_fill_holes(mask)
    mask = keep_largest_components(
        mask,
        num_keep=config.NERVE_MASK_KEEP_COMPONENTS,
        min_size=config.NERVE_MASK_MIN_COMPONENT_SIZE,
    )

    cleaned = np.zeros_like(raw_mask, dtype=np.uint8)
    cleaned[crop_slices] = mask.astype(np.uint8)
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
# Bone density analysis
# ═══════════════════════════════════════════════════════════════════════════

def classify_bone_density(mean_hu):
    """Classify bone density based on mean Hounsfield Units (Misch scale)."""
    if mean_hu > 1250:
        return "D1 (Dense cortical)"
    elif mean_hu > 850:
        return "D2 (Thick cortical/porous)"
    elif mean_hu > 350:
        return "D3 (Thin porous cortical)"
    else:
        return "D4 (Fine trabecular)"


def calculate_bone_density(image, center_vox, radius_vox, half_len_vox):
    """Calculate bone density statistics around the implant site."""
    cx, cy, cz = center_vox

    x_indices = np.arange(
        max(0, cx - half_len_vox), min(image.shape[0], cx + half_len_vox + 1)
    )
    y_indices = np.arange(
        max(0, cy - int(radius_vox)), min(image.shape[1], cy + int(radius_vox) + 1)
    )
    z_indices = np.arange(
        max(0, cz - half_len_vox), min(image.shape[2], cz + half_len_vox + 1)
    )

    if len(x_indices) == 0 or len(y_indices) == 0 or len(z_indices) == 0:
        return 0.0, 0.0, "Unknown"

    implant_region = image[
        x_indices[:, None, None],
        y_indices[None, :, None],
        z_indices[None, None, :],
    ]

    mean_hu = float(np.mean(implant_region))
    std_hu = float(np.std(implant_region))
    density_class = classify_bone_density(mean_hu)

    return mean_hu, std_hu, density_class


# ═══════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════════════════════

def add_implant_axial(ax, cx, cy, r):
    """Draw implant cross-section on an axial view."""
    ax.add_patch(Circle((cx, cy), r, fill=False, color="deepskyblue", linewidth=2.8))
    ax.scatter(cx, cy, c=IMPLANT_CENTER_COLOR, s=40, zorder=5)


def add_implant_coronal(ax, zc, xc, r, x_start, x_end):
    """Draw implant body on a coronal view."""
    body = Rectangle(
        (zc - r, x_start), 2 * r, x_end - x_start,
        fill=False, edgecolor="deepskyblue", linewidth=2.8,
    )
    ax.add_patch(body)
    ax.scatter(zc, xc, c=IMPLANT_CENTER_COLOR, s=40, zorder=5)


def add_implant_sagittal(ax, zc, yc, r):
    """Draw implant cross-section on a sagittal view."""
    ax.add_patch(Circle((zc, yc), r, fill=False, color="deepskyblue", linewidth=2.8))
    ax.scatter(zc, yc, c=IMPLANT_CENTER_COLOR, s=40, zorder=5)


def add_density_overlay(ax, image_slice, center_point, radius):
    """Overlay bone density visualization around the implant site (vectorized)."""
    h, w = image_slice.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - center_point[0]) ** 2 + (xx - center_point[1]) ** 2)
    mask = dist < radius * 2

    density_norm = np.clip((image_slice - 0.3) / 0.7, 0, 1)
    density_overlay = np.zeros((h, w, 4))
    density_overlay[mask] = DENSITY_CMAP(density_norm[mask])

    ax.imshow(density_overlay, origin="lower")


def draw_cross_section(ax, image, nerve_mask, x_idx, center_y, center_z,
                       radius_vox, half_len_vox, title):
    """Draw a cross-sectional view at a given x index."""
    sec_img = norm_slice(image[x_idx, :, :])
    sec_mask = nerve_mask[x_idx, :, :]

    ax.imshow(sec_img, cmap="gray", origin="lower")
    if np.any(sec_mask):
        ax.contour(sec_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.4)

    rect = Rectangle(
        (center_z - radius_vox, center_y - half_len_vox),
        2 * radius_vox, 2 * half_len_vox,
        fill=False, edgecolor="deepskyblue", linewidth=2.2,
    )
    ax.add_patch(rect)

    ax.plot(
        [center_z, center_z],
        [center_y - half_len_vox, center_y + half_len_vox],
        color="deepskyblue", linewidth=2.8,
    )
    ax.scatter(center_z, center_y, c="yellow", s=24, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruction helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_panoramic_like(image, nerve_mask, x_center, width=50):
    """Build a pseudo-panoramic projection by averaging axial slices."""
    x_start = max(0, x_center - width // 2)
    x_end = min(image.shape[0], x_center + width // 2)

    pano = norm_slice(np.mean(image[x_start:x_end, :, :], axis=0))
    pano_mask = (np.mean(nerve_mask[x_start:x_end, :, :], axis=0) > 0.15).astype(np.uint8)
    return pano, pano_mask


def build_tangential_view(image, nerve_mask, bx, by, bz, width=50):
    """Build a tangential view by averaging along the y-axis."""
    y_start = max(0, by - width // 2)
    y_end = min(image.shape[1], by + width // 2)

    tang = norm_slice(np.mean(image[:, y_start:y_end, :], axis=1))
    tang_mask = (np.mean(nerve_mask[:, y_start:y_end, :], axis=1) > 0.15).astype(np.uint8)
    return tang, tang_mask


# ═══════════════════════════════════════════════════════════════════════════
# Interactive point selector (mouse-click on CT)
# ═══════════════════════════════════════════════════════════════════════════

def interactive_point_selector(image, nerve_mask):
    """Show an interactive 3-panel viewer for the user to click the implant
    target point directly on the CT image.

    Controls:
        Scroll wheel      — browse slices
        Left-click         — select / move the target point
        Enter              — confirm selection
        Escape / close     — cancel

    Returns (bx, by, bz) voxel coordinates, or None if cancelled.
    """
    X, Y, Z = image.shape

    state = {
        "z": Z // 2,       # current axial slice index
        "y": Y // 2,       # current coronal slice index
        "x": X // 2,       # current sagittal slice index
        "selected": None,  # (bx, by, bz) when user clicks
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Click on the image to select the implant target location\n"
        "Scroll = browse slices  |  Enter = confirm  |  Escape = cancel",
        fontsize=12,
    )

    def _redraw():
        for ax in axes:
            ax.clear()

        z, y, x = state["z"], state["y"], state["x"]

        # ── Axial (X-Y plane at z) ──
        axes[0].imshow(norm_slice(image[:, :, z]), cmap="gray", origin="lower")
        axes[0].axhline(y=x, color="yellow", lw=0.5, alpha=0.4)
        axes[0].axvline(x=y, color="yellow", lw=0.5, alpha=0.4)
        axes[0].set_title(f"Axial  z={z}", fontsize=11)
        axes[0].axis("off")

        # ── Coronal (X-Z plane at y) ──
        axes[1].imshow(norm_slice(image[:, y, :]), cmap="gray", origin="lower")
        axes[1].axhline(y=x, color="yellow", lw=0.5, alpha=0.4)
        axes[1].axvline(x=z, color="yellow", lw=0.5, alpha=0.4)
        axes[1].set_title(f"Coronal  y={y}", fontsize=11)
        axes[1].axis("off")

        # ── Sagittal (Y-Z plane at x) ──
        axes[2].imshow(norm_slice(image[x, :, :]), cmap="gray", origin="lower")
        axes[2].axhline(y=y, color="yellow", lw=0.5, alpha=0.4)
        axes[2].axvline(x=z, color="yellow", lw=0.5, alpha=0.4)
        axes[2].set_title(f"Sagittal  x={x}", fontsize=11)
        axes[2].axis("off")

        # ── Draw selected point as green + on every panel ──
        if state["selected"] is not None:
            bx, by, bz = state["selected"]
            hs = config.CLICK_ROI_HALF_SIZE
            # Axial
            axes[0].scatter(by, bx, c="lime", s=90, marker="+", linewidths=2, zorder=10)
            axes[0].add_patch(Rectangle(
                (by - hs, bx - hs), 2 * hs, 2 * hs,
                fill=False, edgecolor="lime", lw=1.4, linestyle="--",
            ))
            # Coronal
            axes[1].scatter(bz, bx, c="lime", s=90, marker="+", linewidths=2, zorder=10)
            axes[1].add_patch(Rectangle(
                (bz - hs, bx - hs), 2 * hs, 2 * hs,
                fill=False, edgecolor="lime", lw=1.4, linestyle="--",
            ))
            # Sagittal
            axes[2].scatter(bz, by, c="lime", s=90, marker="+", linewidths=2, zorder=10)
            axes[2].add_patch(Rectangle(
                (bz - hs, by - hs), 2 * hs, 2 * hs,
                fill=False, edgecolor="lime", lw=1.4, linestyle="--",
            ))

        fig.canvas.draw_idle()

    def _on_scroll(event):
        if event.inaxes == axes[0]:
            state["z"] = int(np.clip(
                state["z"] + (1 if event.button == "up" else -1), 0, Z - 1))
        elif event.inaxes == axes[1]:
            state["y"] = int(np.clip(
                state["y"] + (1 if event.button == "up" else -1), 0, Y - 1))
        elif event.inaxes == axes[2]:
            state["x"] = int(np.clip(
                state["x"] + (1 if event.button == "up" else -1), 0, X - 1))
        else:
            return
        _redraw()

    def _on_click(event):
        if event.inaxes is None or event.button != 1:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        if event.inaxes == axes[0]:          # Axial: row→X, col→Y
            bx, by, bz = row, col, state["z"]
        elif event.inaxes == axes[1]:        # Coronal: row→X, col→Z
            bx, by, bz = row, state["y"], col
        elif event.inaxes == axes[2]:        # Sagittal: row→Y, col→Z
            bx, by, bz = state["x"], row, col
        else:
            return

        bx = int(np.clip(bx, 0, X - 1))
        by = int(np.clip(by, 0, Y - 1))
        bz = int(np.clip(bz, 0, Z - 1))

        state["selected"] = (bx, by, bz)
        state["x"], state["y"], state["z"] = bx, by, bz
        fig.suptitle(
            f"Selected: [{bx}, {by}, {bz}]  —  "
            "Press Enter to confirm  |  Click to change  |  Escape to cancel",
            fontsize=12, color="green",
        )
        _redraw()

    def _on_key(event):
        if event.key == "escape":
            state["selected"] = None
            plt.close(fig)
        elif event.key == "enter" and state["selected"] is not None:
            plt.close(fig)

    fig.canvas.mpl_connect("scroll_event", _on_scroll)
    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    _redraw()
    plt.tight_layout()
    plt.show()  # blocks until user closes the window

    return state["selected"]


# ═══════════════════════════════════════════════════════════════════════════
# Basic 3-view planning figure (pipeline output)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_axis_on_view(ax, center_2d, axis_proj_2d, half_len_vox):
    """Draw the implant axis arrow on a 2-D view."""
    c0, c1 = center_2d
    d0, d1 = axis_proj_2d
    ax.annotate(
        "", xy=(c1 + d1 * half_len_vox, c0 + d0 * half_len_vox),
        xytext=(c1 - d1 * half_len_vox, c0 - d0 * half_len_vox),
        arrowprops=dict(arrowstyle="->", color="lime", lw=2.2),
    )


def _draw_target_side_overlay(ax, image_shape_2d, side, z_slice_idx, view):
    """Draw a translucent overlay showing the selected jaw side.

    Parameters:
        ax:              matplotlib axes
        image_shape_2d:  (rows, cols) of the displayed slice
        side:            'left' or 'right'
        z_slice_idx:     the z-index of the displayed slice (for views that contain Z)
        view:            'axial' | 'coronal' | 'sagittal'
    """
    rows, cols = image_shape_2d
    mid_z = cols // 2 if view in ("axial", "coronal") else None

    if mid_z is None:
        return  # sagittal is a single Z slice — no boundary to show

    if side == "left":
        rect = Rectangle((0, 0), mid_z, rows, color="cyan", alpha=0.08)
    else:
        rect = Rectangle((mid_z, 0), cols - mid_z, rows, color="cyan", alpha=0.08)
    ax.add_patch(rect)
    ax.axvline(x=mid_z, color="cyan", linewidth=1.2, linestyle="--", alpha=0.6)


def _draw_roi_overlay(ax, target_region, view):
    """Draw the user-selected ROI box and click-point on a 2-D view.

    Parameters:
        ax:             matplotlib axes
        target_region:  dict with type='roi', voxel=[x,y,z], half_size=N
        view:           'axial' | 'coronal' | 'sagittal'
    """
    vx, vy, vz = target_region["voxel"]
    hs = target_region.get("half_size", config.CLICK_ROI_HALF_SIZE)

    if view == "axial":
        center_h, center_v = vy, vx          # imshow cols=Y, rows=X
    elif view == "coronal":
        center_h, center_v = vz, vx          # imshow cols=Z, rows=X
    elif view == "sagittal":
        center_h, center_v = vz, vy          # imshow cols=Z, rows=Y
    else:
        return

    ax.scatter(center_h, center_v, c="lime", s=90, marker="+",
               linewidths=2, zorder=10, label="User click")
    ax.add_patch(Rectangle(
        (center_h - hs, center_v - hs), 2 * hs, 2 * hs,
        fill=False, edgecolor="lime", linewidth=1.4, linestyle="--",
    ))


def _draw_target_region(ax, target_region, view, slice_shape):
    """Dispatch target-region overlay drawing based on type."""
    if target_region is None:
        return
    rtype = target_region.get("type", "")
    if rtype == "side":
        _draw_target_side_overlay(
            ax, slice_shape, target_region["value"], 0, view)
    elif rtype in ("roi", "point"):
        _draw_roi_overlay(ax, target_region, view)


def generate_basic_planning_figure(image, nerve_mask, planning_result, spacing,
                                   output_path):
    """Generate the basic 3-panel (axial/coronal/sagittal) planning figure."""
    bx, by, bz = planning_result["implant_center"]
    length_mm = planning_result["implant_length"]
    diameter_mm = planning_result["implant_diameter"]
    angle_deg = planning_result["implant_angle"]
    nerve_distance = planning_result["nerve_distance"]
    bone_class = planning_result.get("bone_density_class", "")
    bone_hu = planning_result.get("bone_density_hu", None)
    axis_vec = planning_result.get("implant_axis_vector", [1, 0, 0])
    target_region = planning_result.get("target_region", None)

    radius_vox, half_len_vox = compute_implant_voxel_dims(
        diameter_mm, length_mm, spacing
    )
    X = image.shape[0]
    x1 = max(0, bx - half_len_vox)
    x2 = min(X - 1, bx + half_len_vox)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial (x-y plane at z=bz)
    axes[0].imshow(image[:, :, bz], cmap="gray")
    axes[0].imshow(nerve_mask[:, :, bz], alpha=0.25)
    axes[0].scatter(by, bx, c=IMPLANT_CENTER_COLOR, s=70, label="Implant center")
    axes[0].add_patch(
        Circle((by, bx), radius_vox, fill=False, color="yellow", linewidth=2)
    )
    _draw_axis_on_view(axes[0], (bx, by), (axis_vec[0], axis_vec[1]), half_len_vox)
    axes[0].set_title(f"Axial z={bz}")
    axes[0].axis("off")
    axes[0].legend(loc="lower right", fontsize=8)
    _draw_target_region(axes[0], target_region, "axial", image[:, :, bz].shape)

    # Coronal (x-z plane at y=by)
    axes[1].imshow(image[:, by, :], cmap="gray")
    axes[1].imshow(nerve_mask[:, by, :], alpha=0.25)
    # Bone boundary contour
    coronal_bone = (image[:, by, :] >= config.BONE_HU_MIN) & (image[:, by, :] <= config.BONE_HU_MAX)
    if np.any(coronal_bone):
        axes[1].contour(coronal_bone.astype(np.float32), levels=[0.5],
                        colors="orange", linewidths=1.0, linestyles="dashed")
    axes[1].scatter(bz, bx, c=IMPLANT_CENTER_COLOR, s=70)
    axes[1].plot([bz, bz], [x1, x2], color="yellow", linewidth=4)
    axes[1].plot(
        [bz - radius_vox, bz + radius_vox], [bx, bx],
        color="yellow", linewidth=3,
    )
    _draw_axis_on_view(axes[1], (bx, bz), (axis_vec[0], axis_vec[2]), half_len_vox)
    axes[1].set_title(f"Coronal y={by}")
    axes[1].axis("off")
    _draw_target_region(axes[1], target_region, "coronal", image[:, by, :].shape)

    # Sagittal (y-z plane at x=bx)
    axes[2].imshow(image[bx, :, :], cmap="gray")
    axes[2].imshow(nerve_mask[bx, :, :], alpha=0.25)
    axes[2].scatter(bz, by, c=IMPLANT_CENTER_COLOR, s=70)
    axes[2].add_patch(
        Circle((bz, by), radius_vox, fill=False, color="yellow", linewidth=2)
    )
    _draw_axis_on_view(axes[2], (by, bz), (axis_vec[1], axis_vec[2]), half_len_vox)
    axes[2].set_title(f"Sagittal x={bx}")
    axes[2].axis("off")

    density_str = f"   Bone: {bone_hu:.0f} HU [{bone_class}]" if bone_hu is not None else ""

    target_str = ""
    if target_region:
        rtype = target_region.get("type", "")
        if rtype == "side":
            target_str = f"   Target: {target_region['value']} side"
        elif rtype == "roi":
            target_str = f"   Target: click {target_region['voxel']}"
        elif rtype == "point":
            target_str = f"   Target: point {target_region['voxel']}"
        elif rtype == "bbox":
            target_str = f"   Target: bbox"

    summary = (
        f"Center: [{bx}, {by}, {bz}]   "
        f"Length: {length_mm:.2f} mm   "
        f"Diameter: {diameter_mm:.2f} mm   "
        f"Angle: {angle_deg:.1f}°   "
        f"Nerve distance: {nerve_distance:.2f} mm"
        f"{density_str}{target_str}"
    )
    fig.suptitle(summary, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved planning figure to: {output_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Advanced multi-view viewer
# ═══════════════════════════════════════════════════════════════════════════

def generate_detailed_viewer(image, nerve_mask, planning_meta, spacing,
                             output_path):
    """Generate the full 10-panel viewer with bone density analysis.

    Parameters:
        image: 3D numpy array (CT volume)
        nerve_mask: 3D numpy array (post-processed nerve mask)
        planning_meta: dict from planning_result.json
        spacing: tuple (sx, sy, sz) in mm
        output_path: path to save the output PNG
    """
    # Support both old and new key names for backward compatibility
    bx, by, bz = planning_meta.get("implant_center",
                                    planning_meta.get("implant_center_voxel", [0, 0, 0]))
    length_mm = float(planning_meta.get("implant_length",
                                        planning_meta.get("suggested_implant_length_mm", 10.0)))
    diameter_mm = float(planning_meta.get("implant_diameter",
                                          planning_meta.get("suggested_implant_diameter_mm", 4.0)))
    angle_deg = float(planning_meta.get("implant_angle",
                                        planning_meta.get("suggested_implant_angle_deg", 90.0)))
    distance_to_nerve_mm = float(planning_meta.get("nerve_distance",
                                                    planning_meta.get("distance_to_nerve_mm", 0.0)))
    bone_hu_meta = planning_meta.get("bone_density_hu", None)
    bone_cls_meta = planning_meta.get("bone_density_class", "")
    axis_vec = planning_meta.get("implant_axis_vector", [1, 0, 0])
    bone_thickness_mm = planning_meta.get("bone_thickness_mm", None)
    wall_distance_mm = planning_meta.get("wall_distance_mm", None)
    target_region = planning_meta.get("target_region", None)
    sx, sy, sz = [float(v) for v in spacing]

    radius_vox, half_len_vox = compute_implant_voxel_dims(
        diameter_mm, length_mm, (sx, sy, sz)
    )
    X = image.shape[0]
    x1 = max(0, bx - half_len_vox)
    x2 = min(X - 1, bx + half_len_vox)

    # Bone density — use pre-computed value from planning if available
    mean_hu, std_hu, bone_class = calculate_bone_density(
        image, [bx, by, bz], radius_vox, half_len_vox
    )
    if bone_hu_meta is not None:
        mean_hu = bone_hu_meta
        bone_class = bone_cls_meta or classify_bone_density(mean_hu)
    print(f"Bone Density: {mean_hu:.1f} +/- {std_hu:.1f} HU  [{bone_class}]")

    # ── Prepare slices ───────────────────────────────────────────────────
    axial_img = norm_slice(image[:, :, bz])
    axial_mask = nerve_mask[:, :, bz]
    coronal_img = norm_slice(image[:, by, :])
    coronal_mask = nerve_mask[:, by, :]
    sagittal_img = norm_slice(image[bx, :, :])
    sagittal_mask = nerve_mask[bx, :, :]

    # Zoomed axial
    zr = config.ZOOM_RADIUS
    x_min = max(0, bx - zr)
    x_max = min(image.shape[0], bx + zr)
    y_min = max(0, by - zr)
    y_max = min(image.shape[1], by + zr)
    zoom_axial_img = norm_slice(image[x_min:x_max, y_min:y_max, bz])
    zoom_axial_mask = nerve_mask[x_min:x_max, y_min:y_max, bz]

    # Cross sections
    cross_indices = [
        max(0, min(X - 1, bx + d))
        for d in config.CROSS_SECTION_OFFSETS
    ]

    # Panoramic & tangential
    pano_img, pano_mask = build_panoramic_like(
        image, nerve_mask, bx, width=config.PANORAMIC_WIDTH
    )
    tang_img, tang_mask = build_tangential_view(
        image, nerve_mask, bx, by, bz, width=config.TANGENTIAL_WIDTH
    )

    # ── Build figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[1.0, 1.0, 0.9],
        width_ratios=[1.15, 1.15, 1.15, 1.0],
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    ax_pano = fig.add_subplot(gs[2, 0:3])
    ax_summary = fig.add_subplot(gs[2, 3])

    # Row 1: Axial / Coronal / Sagittal / Zoom
    ax = axes[0]
    ax.imshow(axial_img, cmap="gray", origin="lower")
    add_density_overlay(ax, axial_img, [by, bx], radius_vox)
    if np.any(axial_mask):
        ax.contour(axial_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.6)
    add_implant_axial(ax, by, bx, radius_vox)
    ax.set_title(f"Axial View (z={bz})", fontsize=13)
    ax.axis("off")
    _draw_target_region(ax, target_region, "axial", axial_img.shape)

    ax = axes[1]
    ax.imshow(coronal_img, cmap="gray", origin="lower")
    add_density_overlay(ax, coronal_img, [bz, bx], radius_vox)
    # Bone boundary contour on coronal slice
    coronal_bone = (image[:, by, :] >= config.BONE_HU_MIN) & (image[:, by, :] <= config.BONE_HU_MAX)
    if np.any(coronal_bone):
        ax.contour(coronal_bone.astype(np.float32), levels=[0.5],
                   colors="orange", linewidths=1.0, linestyles="dashed")
    if np.any(coronal_mask):
        ax.contour(coronal_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.6)
    add_implant_coronal(ax, bz, bx, radius_vox, x1, x2)
    ax.set_title(f"Coronal View (y={by})", fontsize=13)
    ax.axis("off")
    _draw_target_region(ax, target_region, "coronal", coronal_img.shape)

    ax = axes[2]
    ax.imshow(sagittal_img, cmap="gray", origin="lower")
    add_density_overlay(ax, sagittal_img, [bz, by], radius_vox)
    if np.any(sagittal_mask):
        ax.contour(sagittal_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.6)
    add_implant_sagittal(ax, bz, by, radius_vox)
    ax.set_title(f"Sagittal View (x={bx})", fontsize=13)
    ax.axis("off")

    ax = axes[3]
    local_cx = by - y_min
    local_cy = bx - x_min
    ax.imshow(zoom_axial_img, cmap="gray", origin="lower")
    add_density_overlay(ax, zoom_axial_img, [local_cx, local_cy], radius_vox)
    if np.any(zoom_axial_mask):
        ax.contour(zoom_axial_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.4)
    add_implant_axial(ax, local_cx, local_cy, radius_vox)
    ax.set_title("Implant Site Zoom", fontsize=13)
    ax.axis("off")

    # Row 2: Cross Sections + Tangential
    for i, ci in enumerate(cross_indices):
        draw_cross_section(
            axes[4 + i], image, nerve_mask, ci, by, bz,
            radius_vox, half_len_vox, f"Cross Section x={ci}",
        )

    ax = axes[7]
    ax.imshow(tang_img, cmap="gray", origin="lower", aspect="auto")
    if np.any(tang_mask):
        ax.contour(tang_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.4)
    tang_y, tang_z = bx, bz
    ax.add_patch(Rectangle(
        (tang_z - radius_vox, tang_y - half_len_vox),
        2 * radius_vox, 2 * half_len_vox,
        fill=False, edgecolor="deepskyblue", linewidth=2.4,
    ))
    ax.plot(
        [tang_z, tang_z],
        [tang_y - half_len_vox, tang_y + half_len_vox],
        color="deepskyblue", linewidth=3.0,
    )
    ax.scatter(tang_z, tang_y, c="yellow", s=24, zorder=5)
    ax.set_title("Tangential View", fontsize=13)
    ax.axis("off")

    # Row 3: Panoramic + Summary
    ax_pano.imshow(pano_img, cmap="gray", origin="lower", aspect="auto")
    if np.any(pano_mask):
        ax_pano.contour(pano_mask, levels=[0.5], colors=NERVE_OVERLAY_COLOR, linewidths=1.2)
    ax_pano.add_patch(Rectangle(
        (bz - radius_vox, by - half_len_vox),
        2 * radius_vox, 2 * half_len_vox,
        fill=False, edgecolor="deepskyblue", linewidth=2.2,
    ))
    ax_pano.plot(
        [bz, bz], [by - half_len_vox, by + half_len_vox],
        color="deepskyblue", linewidth=2.8,
    )
    ax_pano.scatter(bz, by, c=IMPLANT_CENTER_COLOR, s=30, zorder=5)
    ax_pano.set_title("Panoramic-like Reconstruction", fontsize=13)
    ax_pano.axis("off")

    # Implant axis arrow on coronal view
    _draw_axis_on_view(axes[1], (bx, bz), (axis_vec[0], axis_vec[2]), half_len_vox)

    # Summary text
    ax_summary.axis("off")

    # Target region label
    target_label = "Whole jaw (automatic)"
    if target_region:
        rtype = target_region.get("type", "")
        if rtype == "side":
            target_label = f"{target_region['value'].capitalize()} side"
        elif rtype == "roi":
            target_label = f"Click ROI {target_region['voxel']}"
        elif rtype == "point":
            target_label = f"Point {target_region['voxel']}"
        elif rtype == "bbox":
            target_label = f"BBox {target_region.get('min')}→{target_region.get('max')}"

    summary_lines = [
        "Planning Summary",
        "==============================",
        f"Target Region: {target_label}",
        f"Implant Center: [{bx}, {by}, {bz}]",
        f"Implant Length:   {length_mm:.2f} mm",
        f"Implant Diameter: {diameter_mm:.2f} mm",
        f"Implant Angle:    {angle_deg:.1f}°",
        f"Nerve Distance:   {distance_to_nerve_mm:.2f} mm",
        "------------------------------",
        "BONE BOUNDARIES",
        f"Wall Distance:  {wall_distance_mm:.2f} mm" if wall_distance_mm else "Wall Distance:  N/A",
        f"Bone Thickness: {bone_thickness_mm:.2f} mm" if bone_thickness_mm else "Bone Thickness: N/A",
        "==============================",
        "BONE DENSITY ALONG PATH",
        "------------------------------",
        f"Mean Density:  {mean_hu:.1f} HU",
        f"Std Deviation: {std_hu:.1f} HU",
        f">>> Misch Class: {bone_class} <<<",
        "",
        "HU Classification (Misch):",
        "  D1: >1250    Dense cortical",
        "  D2: 850-1250 Thick cortical",
        "  D3: 350-850  Thin porous",
        "  D4: <350     Fine trabecular",
        "------------------------------",
        "Diameter refined by density:",
        "  D1 -> 3.5mm  D2 -> 4.0mm",
        "  D3 -> 4.5mm  D4 -> 5.0mm",
        "==============================",
        f"Voxel Spacing: ({sx:.2f}, {sy:.2f}, {sz:.2f})",
        "",
        "Legend:",
        "Orange contour = Predicted nerve",
        "Blue shape  = Implant body",
        "Blue dot    = Implant center",
        "Green arrow = Implant axis",
        "Orange dash = Bone boundary",
        "Cyan shade  = Target region",
        "Green +/box = User click ROI",
        "Yellow dot  = Section center",
        "Overlay     = Bone density map",
    ]
    ax_summary.text(
        0.02, 0.98, "\n".join(summary_lines),
        va="top", ha="left", fontsize=10.8, family="monospace",
    )

    fig.suptitle(
        "Smart Dental Implant Planning Viewer (Bone Density Analysis)",
        fontsize=18, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved viewer layout to: {output_path}")
    plt.show()
