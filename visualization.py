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

def keep_largest_components(mask, num_keep=2, min_size=200):
    """Keep only the N largest connected components above a minimum size."""
    struct = generate_binary_structure(3, 2)
    labeled, num = label(mask, structure=struct)

    if num == 0:
        return mask.astype(np.uint8)

    comps = []
    for i in range(1, num + 1):
        size = int((labeled == i).sum())
        if size >= min_size:
            comps.append((size, i))

    if not comps:
        return mask.astype(np.uint8)

    comps.sort(reverse=True)
    keep_ids = [cid for _, cid in comps[:num_keep]]
    return np.isin(labeled, keep_ids).astype(np.uint8)


def postprocess_nerve_mask(nerve_mask_raw):
    """Clean the predicted nerve mask with morphological operations."""
    struct = generate_binary_structure(3, 1)
    mask = binary_opening(nerve_mask_raw.astype(bool), structure=struct, iterations=1)
    mask = binary_closing(mask, structure=struct, iterations=1)
    mask = binary_fill_holes(mask)
    mask = keep_largest_components(
        mask,
        num_keep=config.NERVE_MASK_KEEP_COMPONENTS,
        min_size=config.NERVE_MASK_MIN_COMPONENT_SIZE,
    )
    return mask.astype(np.uint8)


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
    ax.scatter(cx, cy, c="red", s=40, zorder=5)


def add_implant_coronal(ax, zc, xc, r, x_start, x_end):
    """Draw implant body on a coronal view."""
    body = Rectangle(
        (zc - r, x_start), 2 * r, x_end - x_start,
        fill=False, edgecolor="deepskyblue", linewidth=2.8,
    )
    ax.add_patch(body)
    ax.scatter(zc, xc, c="red", s=40, zorder=5)


def add_implant_sagittal(ax, zc, yc, r):
    """Draw implant cross-section on a sagittal view."""
    ax.add_patch(Circle((zc, yc), r, fill=False, color="deepskyblue", linewidth=2.8))
    ax.scatter(zc, yc, c="red", s=40, zorder=5)


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
        ax.contour(sec_mask, levels=[0.5], colors="red", linewidths=1.4)

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
# Basic 3-view planning figure (pipeline output)
# ═══════════════════════════════════════════════════════════════════════════

def generate_basic_planning_figure(image, nerve_mask, planning_result, spacing,
                                   output_path):
    """Generate the basic 3-panel (axial/coronal/sagittal) planning figure."""
    bx, by, bz = planning_result["implant_center_voxel"]
    length_mm = planning_result["suggested_implant_length_mm"]
    diameter_mm = planning_result["suggested_implant_diameter_mm"]
    angle_deg = planning_result["suggested_implant_angle_deg"]
    distance_to_nerve = planning_result["distance_to_nerve_mm"]

    radius_vox, half_len_vox = compute_implant_voxel_dims(
        diameter_mm, length_mm, spacing
    )
    X = image.shape[0]
    x1 = max(0, bx - half_len_vox)
    x2 = min(X - 1, bx + half_len_vox)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial
    axes[0].imshow(image[:, :, bz], cmap="gray")
    axes[0].imshow(nerve_mask[:, :, bz], alpha=0.25)
    axes[0].scatter(by, bx, c="red", s=70, label="Implant center")
    axes[0].add_patch(
        Circle((by, bx), radius_vox, fill=False, color="yellow", linewidth=2)
    )
    axes[0].set_title(f"Axial z={bz}")
    axes[0].axis("off")
    axes[0].legend(loc="lower right", fontsize=8)

    # Coronal
    axes[1].imshow(image[:, by, :], cmap="gray")
    axes[1].imshow(nerve_mask[:, by, :], alpha=0.25)
    axes[1].scatter(bz, bx, c="red", s=70)
    axes[1].plot([bz, bz], [x1, x2], color="yellow", linewidth=4)
    axes[1].plot(
        [bz - radius_vox, bz + radius_vox], [bx, bx],
        color="yellow", linewidth=3,
    )
    axes[1].set_title(f"Coronal y={by}")
    axes[1].axis("off")

    # Sagittal
    axes[2].imshow(image[bx, :, :], cmap="gray")
    axes[2].imshow(nerve_mask[bx, :, :], alpha=0.25)
    axes[2].scatter(bz, by, c="red", s=70)
    axes[2].add_patch(
        Circle((bz, by), radius_vox, fill=False, color="yellow", linewidth=2)
    )
    axes[2].set_title(f"Sagittal x={bx}")
    axes[2].axis("off")

    summary = (
        f"Center: [{bx}, {by}, {bz}]   "
        f"Length: {length_mm:.2f} mm   "
        f"Diameter: {diameter_mm:.2f} mm   "
        f"Angle: {angle_deg:.1f} deg   "
        f"Distance to nerve: {distance_to_nerve:.2f} mm"
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
    bx, by, bz = planning_meta["implant_center_voxel"]
    length_mm = float(planning_meta["suggested_implant_length_mm"])
    diameter_mm = float(planning_meta["suggested_implant_diameter_mm"])
    angle_deg = float(planning_meta.get("suggested_implant_angle_deg", 90.0))
    distance_to_nerve_mm = float(planning_meta["distance_to_nerve_mm"])
    sx, sy, sz = [float(v) for v in spacing]

    radius_vox, half_len_vox = compute_implant_voxel_dims(
        diameter_mm, length_mm, (sx, sy, sz)
    )
    X = image.shape[0]
    x1 = max(0, bx - half_len_vox)
    x2 = min(X - 1, bx + half_len_vox)

    # Bone density
    mean_hu, std_hu, bone_class = calculate_bone_density(
        image, [bx, by, bz], radius_vox, half_len_vox
    )
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
        ax.contour(axial_mask, levels=[0.5], colors="red", linewidths=1.6)
    add_implant_axial(ax, by, bx, radius_vox)
    ax.set_title(f"Axial View (z={bz})", fontsize=13)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(coronal_img, cmap="gray", origin="lower")
    add_density_overlay(ax, coronal_img, [bz, bx], radius_vox)
    if np.any(coronal_mask):
        ax.contour(coronal_mask, levels=[0.5], colors="red", linewidths=1.6)
    add_implant_coronal(ax, bz, bx, radius_vox, x1, x2)
    ax.set_title(f"Coronal View (y={by})", fontsize=13)
    ax.axis("off")

    ax = axes[2]
    ax.imshow(sagittal_img, cmap="gray", origin="lower")
    add_density_overlay(ax, sagittal_img, [bz, by], radius_vox)
    if np.any(sagittal_mask):
        ax.contour(sagittal_mask, levels=[0.5], colors="red", linewidths=1.6)
    add_implant_sagittal(ax, bz, by, radius_vox)
    ax.set_title(f"Sagittal View (x={bx})", fontsize=13)
    ax.axis("off")

    ax = axes[3]
    local_cx = by - y_min
    local_cy = bx - x_min
    ax.imshow(zoom_axial_img, cmap="gray", origin="lower")
    add_density_overlay(ax, zoom_axial_img, [local_cx, local_cy], radius_vox)
    if np.any(zoom_axial_mask):
        ax.contour(zoom_axial_mask, levels=[0.5], colors="red", linewidths=1.4)
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
        ax.contour(tang_mask, levels=[0.5], colors="red", linewidths=1.4)
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
        ax_pano.contour(pano_mask, levels=[0.5], colors="red", linewidths=1.2)
    ax_pano.add_patch(Rectangle(
        (bz - radius_vox, by - half_len_vox),
        2 * radius_vox, 2 * half_len_vox,
        fill=False, edgecolor="deepskyblue", linewidth=2.2,
    ))
    ax_pano.plot(
        [bz, bz], [by - half_len_vox, by + half_len_vox],
        color="deepskyblue", linewidth=2.8,
    )
    ax_pano.scatter(bz, by, c="red", s=30, zorder=5)
    ax_pano.set_title("Panoramic-like Reconstruction", fontsize=13)
    ax_pano.axis("off")

    # Summary text
    ax_summary.axis("off")
    summary_lines = [
        "Planning Summary",
        "------------------------------",
        f"Implant Center: [{bx}, {by}, {bz}]",
        f"Implant Length: {length_mm:.2f} mm",
        f"Implant Diameter: {diameter_mm:.2f} mm",
        f"Implant Angle: {angle_deg:.1f} deg",
        f"Distance to Nerve: {distance_to_nerve_mm:.2f} mm",
        "------------------------------",
        "BONE DENSITY ANALYSIS",
        "------------------------------",
        f"Mean Density: {mean_hu:.1f} HU",
        f"Std Deviation: {std_hu:.1f} HU",
        f"Classification: {bone_class}",
        "",
        "HU Ranges for Reference:",
        "D1: >1250 HU - Dense cortical",
        "D2: 850-1250 HU - Thick cortical",
        "D3: 350-850 HU - Thin porous",
        "D4: <350 HU - Fine trabecular",
        "------------------------------",
        f"Voxel Spacing: ({sx:.2f}, {sy:.2f}, {sz:.2f})",
        "",
        "Legend:",
        "Red contour = Predicted nerve",
        "Blue shape = Implant body",
        "Red dot = Implant center",
        "Yellow dot = Section center",
        "Overlay = Bone density map",
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
