"""
Implant planning logic — bone-thickness-centre approach.

The implant is placed at the midpoint of bone thickness (equidistant from
both bone walls) using the distance transform, NOT at the geometric centroid
of the bone volume.

Algorithm order:
  1. Load CT volume and predicted nerve mask
  2. Detect jaw bone region using HU thresholds (filter air / noise)
  3. Compute bone thickness map via distance_transform_edt
  4. Select implant centre at the bone-thickness midpoint (~1 mm from walls)
  5. Estimate implant axis / angle from local bone geometry
  6. Determine implant dimensions from bone depth along axis
  7. Evaluate bone density along the planned implant path
  8. Classify density (Misch scale) → refine length/diameter/angle
  9. Validate safety distance from the Inferior Alveolar Nerve
"""

import numpy as np
from scipy.ndimage import (
    distance_transform_edt, sobel, gaussian_filter,
    binary_dilation, binary_closing, binary_fill_holes,
    generate_binary_structure, label,
)
from scipy.spatial import cKDTree

import config


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Nerve distance map
# ═══════════════════════════════════════════════════════════════════════════

def compute_nerve_distance(nerve_mask, spacing):
    """Distance from every voxel to the nearest nerve voxel (mm)."""
    sx, sy, sz = (float(v) for v in spacing)
    return distance_transform_edt(1 - nerve_mask, sampling=(sx, sy, sz))


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Bone detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_bone_region(image):
    """Identify bone voxels using HU thresholds (ignores air / soft tissue
    and metal artefacts)."""
    return (image >= config.BONE_HU_MIN) & (image <= config.BONE_HU_MAX)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Bone thickness map (distance transform)
# ═══════════════════════════════════════════════════════════════════════════

def compute_bone_thickness_map(bone_mask, spacing):
    """Distance transform of the bone mask in mm.

    For every bone voxel the value is its distance to the nearest bone
    boundary (non-bone voxel).  The *maximum* of this map is the point
    deepest inside the bone — the true bone-thickness midpoint.
    """
    sx, sy, sz = (float(v) for v in spacing)
    return distance_transform_edt(bone_mask.astype(bool), sampling=(sx, sy, sz))


def _build_integral_volume(mask):
    """Build a padded integral volume for constant-time box queries."""
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0), (1, 0)))
    return integral.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)


def _query_integral_box(integral, x0, x1, y0, y1, z0, z1):
    """Return the voxel count inside [x0:x1, y0:y1, z0:z1]."""
    return (
        integral[x1, y1, z1]
        - integral[x0, y1, z1]
        - integral[x1, y0, z1]
        - integral[x1, y1, z0]
        + integral[x0, y0, z1]
        + integral[x0, y1, z0]
        + integral[x1, y0, z0]
        - integral[x0, y0, z0]
    )


def _measure_superior_surface_depths(bone_mask, coords, spacing):
    """Measure contiguous bone depth toward the superior side for each voxel."""
    sx = float(spacing[0])
    depths = np.zeros(len(coords), dtype=np.float32)

    for idx, (x, y, z) in enumerate(coords):
        depth_vox = 0
        xi = int(x)
        while xi >= 0 and bone_mask[xi, y, z]:
            depth_vox += 1
            xi -= 1
        depths[idx] = depth_vox * sx

    return depths


def _measure_tooth_above_ratios(image, coords, spacing):
    """Measure dense-tooth occupancy above each candidate voxel."""
    tooth_mask = image >= config.TOOTH_HU_MIN
    integral = _build_integral_volume(tooth_mask)
    shape = image.shape

    sx, sy, sz = (float(v) for v in spacing)
    lookback = max(1, int(np.ceil(config.TOOTH_ABOVE_LOOKBACK_MM / sx)))
    ry = max(1, int(np.ceil(config.TOOTH_NEIGHBORHOOD_RADIUS_MM / sy)))
    rz = max(1, int(np.ceil(config.TOOTH_NEIGHBORHOOD_RADIUS_MM / sz)))

    ratios = np.zeros(len(coords), dtype=np.float32)
    for idx, (x, y, z) in enumerate(coords):
        x0 = max(0, int(x) - lookback)
        x1 = min(shape[0], int(x) + 1)
        y0 = max(0, int(y) - ry)
        y1 = min(shape[1], int(y) + ry + 1)
        z0 = max(0, int(z) - rz)
        z1 = min(shape[2], int(z) + rz + 1)

        dense_voxels = _query_integral_box(integral, x0, x1, y0, y1, z0, z1)
        box_volume = max(1, (x1 - x0) * (y1 - y0) * (z1 - z0))
        ratios[idx] = dense_voxels / float(box_volume)

    return ratios


# ═══════════════════════════════════════════════════════════════════════════
# Step 3b — Edentulous gap detection
# ═══════════════════════════════════════════════════════════════════════════

def _build_tooth_projection(image, bone_mask):
    """Project tooth material (HU >= TOOTH_HU_MIN) onto the Y-Z plane.

    Returns a 2-D float32 array of shape (Y, Z) — count of tooth voxels
    per column along the X (superior-inferior) axis.
    """
    tooth_mask = image >= config.TOOTH_HU_MIN
    return tooth_mask.sum(axis=0).astype(np.float32)


def _detect_edentulous_gaps(tooth_projection, bone_projection, spacing):
    """Detect edentulous gaps in the 2-D tooth projection.

    Strategy:
      1. Binarize columns with significant tooth presence.
      2. Dilate the tooth region to define the *dental arch zone*.
      3. Fill enclosed holes so interior gaps are captured.
      4. Gaps = arch zone with bone but without actual teeth.
      5. Remove tiny noise components.

    Returns a 2-D boolean mask (Y, Z) where True = gap location.
    """
    sy, sz = float(spacing[1]), float(spacing[2])

    has_tooth = tooth_projection >= config.GAP_TOOTH_PROJECTION_MIN_VOXELS
    has_bone = bone_projection >= config.GAP_TOOTH_PROJECTION_MIN_VOXELS

    if not np.any(has_tooth):
        return np.zeros_like(has_tooth, dtype=bool)

    # Dilate tooth footprint to define "dental arch zone"
    dil_y = max(1, int(np.ceil(config.GAP_DILATION_MM / sy)))
    dil_z = max(1, int(np.ceil(config.GAP_DILATION_MM / sz)))
    struct = np.ones((2 * dil_y + 1, 2 * dil_z + 1), dtype=bool)
    arch_zone = binary_dilation(has_tooth, structure=struct)

    # Fill enclosed holes (captures gaps fully surrounded by teeth)
    filled = binary_fill_holes(arch_zone)

    # Gap = inside the arch zone, has bone, but no tooth
    gap_mask = filled & has_bone & ~has_tooth

    # Remove tiny components (noise)
    min_vox = max(1, int((config.GAP_MIN_WIDTH_MM / sy)
                         * (config.GAP_MIN_WIDTH_MM / sz)))
    labeled, n_labels = label(gap_mask)
    for i in range(1, n_labels + 1):
        if np.sum(labeled == i) < min_vox:
            gap_mask[labeled == i] = False

    return gap_mask


# ═══════════════════════════════════════════════════════════════════════════
# Rule 1 — Tooth segmentation & clearance
# ═══════════════════════════════════════════════════════════════════════════

def segment_teeth(image, spacing):
    """Segment teeth as a 3-D binary mask (HU threshold + cleanup).

    Teeth (dentin + enamel) have HU >= TOOTH_HU_MIN.  Small noise
    components below TOOTH_MIN_VOLUME_MM3 are removed.
    """
    raw = image >= config.TOOTH_HU_MIN

    struct = generate_binary_structure(3, 1)
    closed = binary_closing(raw, structure=struct, iterations=2)

    labeled, n = label(closed)
    if n == 0:
        return np.zeros(image.shape, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sx, sy, sz = (float(v) for v in spacing)
    voxel_vol = sx * sy * sz
    min_voxels = max(1, int(config.TOOTH_MIN_VOLUME_MM3 / voxel_vol))

    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if sizes[i] >= min_voxels:
            keep[i] = True
    return keep[labeled]


def build_tooth_exclusion_zone(tooth_mask, spacing, clearance_mm):
    """Dilate tooth mask to create a keep-out zone of *clearance_mm*.

    Uses iterative dilation with a cross structuring element.
    """
    if not np.any(tooth_mask):
        return np.zeros_like(tooth_mask)
    sx, sy, sz = (float(v) for v in spacing)
    iters = max(1, int(np.ceil(clearance_mm / min(sx, sy, sz))))
    struct = generate_binary_structure(3, 1)
    return binary_dilation(tooth_mask, structure=struct, iterations=iters)


def _measure_tooth_clearance(tooth_mask, coords, spacing):
    """Min distance from each candidate to nearest tooth voxel (mm).

    Uses a KD-tree on sub-sampled tooth voxels for efficiency.
    """
    sx, sy, sz = (float(v) for v in spacing)
    tooth_voxels = np.argwhere(tooth_mask)
    if len(tooth_voxels) == 0:
        return np.full(len(coords), 999.0, dtype=np.float32)

    stride = max(1, len(tooth_voxels) // 200000)
    sub = tooth_voxels[::stride].astype(np.float64)
    sub[:, 0] *= sx
    sub[:, 1] *= sy
    sub[:, 2] *= sz

    tree = cKDTree(sub)

    q = coords.astype(np.float64).copy()
    q[:, 0] *= sx
    q[:, 1] *= sy
    q[:, 2] *= sz

    dists, _ = tree.query(q)
    return dists.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Rule 2 — Alveolar ridge (dental arch) constraint
# ═══════════════════════════════════════════════════════════════════════════

def build_alveolar_arch_mask_3d(tooth_mask, bone_mask, spacing):
    """3-D mask restricting candidates to the alveolar ridge zone.

    Projects the tooth footprint onto Y-Z, dilates to define the dental
    arch, fills interior gaps, then broadcasts back into 3-D.
    """
    sy, sz = float(spacing[1]), float(spacing[2])

    tooth_proj = tooth_mask.sum(axis=0).astype(np.float32)
    bone_proj = bone_mask.sum(axis=0).astype(np.float32)

    has_tooth = tooth_proj >= config.GAP_TOOTH_PROJECTION_MIN_VOXELS
    has_bone = bone_proj >= config.GAP_TOOTH_PROJECTION_MIN_VOXELS

    if not np.any(has_tooth):
        return bone_mask.copy()

    dil_y = max(1, int(np.ceil(config.ARCH_DILATION_MM / sy)))
    dil_z = max(1, int(np.ceil(config.ARCH_DILATION_MM / sz)))
    struct_2d = np.ones((2 * dil_y + 1, 2 * dil_z + 1), dtype=bool)

    arch_2d = binary_dilation(has_tooth, structure=struct_2d)
    arch_2d = binary_fill_holes(arch_2d)
    arch_2d &= has_bone

    arch_3d = np.zeros(bone_mask.shape, dtype=bool)
    arch_3d[:] = arch_2d[np.newaxis, :, :]
    arch_3d &= bone_mask
    return arch_3d


# ═══════════════════════════════════════════════════════════════════════════
# Rule 3 — Sinus / cavity proximity
# ═══════════════════════════════════════════════════════════════════════════

def _measure_distance_to_air_above(image, bone_mask, coords, spacing):
    """Walk superiorly from each candidate until hitting air / cavity.

    Returns the distance in mm.  inf means no air was found within
    AIR_ABOVE_SEARCH_MM.
    """
    sx = float(spacing[0])
    max_steps = max(1, int(np.ceil(config.AIR_ABOVE_SEARCH_MM / sx)))
    distances = np.full(len(coords), float('inf'), dtype=np.float32)

    for idx, (x, y, z) in enumerate(coords):
        xi = int(x) - 1
        step = 0
        while xi >= 0 and step < max_steps:
            step += 1
            hu = float(image[xi, y, z])
            if hu <= config.SINUS_AIR_HU_MAX:
                distances[idx] = step * sx
                break
            if not bone_mask[xi, y, z] and hu < config.ALVEOLAR_MIN_HU:
                distances[idx] = step * sx
                break
            xi -= 1

    return distances


# ═══════════════════════════════════════════════════════════════════════════
# Rule 2 supplement — Local neighbourhood density
# ═══════════════════════════════════════════════════════════════════════════

def _measure_local_density(image, coords, spacing, radius_mm):
    """Mean HU in a box neighbourhood around each candidate."""
    sx, sy, sz = (float(v) for v in spacing)
    rx = max(1, int(np.ceil(radius_mm / sx)))
    ry = max(1, int(np.ceil(radius_mm / sy)))
    rz = max(1, int(np.ceil(radius_mm / sz)))
    shape = image.shape

    means = np.empty(len(coords), dtype=np.float32)
    for idx, (x, y, z) in enumerate(coords):
        x0, x1 = max(0, int(x) - rx), min(shape[0], int(x) + rx + 1)
        y0, y1 = max(0, int(y) - ry), min(shape[1], int(y) + ry + 1)
        z0, z1 = max(0, int(z) - rz), min(shape[2], int(z) + rz + 1)
        means[idx] = float(image[x0:x1, y0:y1, z0:z1].mean())
    return means


# ═══════════════════════════════════════════════════════════════════════════
# Rule 5 — Gap-centre preference
# ═══════════════════════════════════════════════════════════════════════════

def _compute_gap_center_distances(gap_2d, coords, spacing):
    """Distance from each candidate to the centroid of its nearest gap (mm).

    Returns inf when no gaps exist.
    """
    sy, sz = float(spacing[1]), float(spacing[2])

    labeled, n_labels = label(gap_2d)
    if n_labels == 0:
        return np.full(len(coords), float('inf'), dtype=np.float32)

    centroids = np.empty((n_labels, 2), dtype=np.float64)
    for i in range(1, n_labels + 1):
        ys, zs = np.where(labeled == i)
        centroids[i - 1, 0] = ys.mean() * sy
        centroids[i - 1, 1] = zs.mean() * sz

    yz_mm = np.empty((len(coords), 2), dtype=np.float64)
    yz_mm[:, 0] = coords[:, 1].astype(np.float64) * sy
    yz_mm[:, 1] = coords[:, 2].astype(np.float64) * sz

    dy = yz_mm[:, 0:1] - centroids[:, 0:1].T
    dz = yz_mm[:, 1:2] - centroids[:, 1:2].T
    dists = np.sqrt(dy ** 2 + dz ** 2)
    return dists.min(axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Occlusal Plane Alignment
# ═══════════════════════════════════════════════════════════════════════════

def compute_occlusal_plane_level(tooth_mask, spacing):
    """Estimate the vertical (X-axis) level of the occlusal plane.

    Uses a low percentile of tooth voxel X-coordinates to approximate
    where tooth crowns sit.  Returns an X-coordinate in voxel units,
    or None when no teeth are present.
    """
    tooth_x = np.where(tooth_mask)[0]
    if len(tooth_x) == 0:
        return None
    return float(np.percentile(tooth_x, config.OCCLUSAL_PERCENTILE))


# ═══════════════════════════════════════════════════════════════════════════
# Buccal-Lingual Centering (coronal-plane 2-D distance transform)
# ═══════════════════════════════════════════════════════════════════════════

def _measure_coronal_centering(bone_mask, coords, spacing):
    """Distance to the nearest bone wall in the coronal (Y-Z) plane.

    For each unique X-slice that contains candidates a 2-D distance
    transform is computed.  Higher value → better centred between the
    buccal and lingual cortical walls.
    """
    sy, sz = float(spacing[1]), float(spacing[2])
    unique_x = np.unique(coords[:, 0].astype(int))

    dt_cache = {}
    for xi in unique_x:
        sl = bone_mask[int(xi), :, :]
        if np.any(sl):
            dt_cache[int(xi)] = distance_transform_edt(
                sl, sampling=(sy, sz),
            )

    scores = np.empty(len(coords), dtype=np.float32)
    for idx in range(len(coords)):
        xi = int(coords[idx, 0])
        yi = int(coords[idx, 1])
        zi = int(coords[idx, 2])
        dt_2d = dt_cache.get(xi)
        scores[idx] = float(dt_2d[yi, zi]) if dt_2d is not None else 0.0
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Implant centre selection (bone-thickness midpoint)
# ═══════════════════════════════════════════════════════════════════════════

def _volume_edge_mask(shape, margin_fraction):
    """Boolean mask that is False within *margin_fraction* of each border."""
    X, Y, Z = shape
    mx = int(margin_fraction * X)
    my = int(margin_fraction * Y)
    mz = int(margin_fraction * Z)
    mask = np.zeros(shape, dtype=bool)
    mask[mx:X - mx, my:Y - my, mz:Z - mz] = True
    return mask


def build_target_mask(shape, target_region, spacing):
    """Create a boolean mask restricting candidate voxels to a target region.

    Supported target_region formats:
      {"type": "roi", "voxel": [x, y, z], "half_size": int}
          Axis-aligned box of ±half_size voxels around the voxel coordinate.
          Default half_size comes from config.CLICK_ROI_HALF_SIZE.
      {"type": "side", "value": "left"|"right"}
          Split the volume in half along the Z axis.
      {"type": "point", "voxel": [x, y, z], "radius_mm": float}
          Sphere of given radius around the voxel coordinate.
      {"type": "bbox", "min": [x0, y0, z0], "max": [x1, y1, z1]}
          Axis-aligned bounding box in voxel coordinates.

    Returns a boolean 3-D mask (True = included in search).
    """
    X, Y, Z = shape
    rtype = target_region.get("type", "").lower()

    if rtype == "roi":
        vx, vy, vz = (int(v) for v in target_region["voxel"])
        hs = int(target_region.get("half_size", config.CLICK_ROI_HALF_SIZE))
        mask = np.zeros(shape, dtype=bool)
        x0, y0, z0 = max(0, vx - hs), max(0, vy - hs), max(0, vz - hs)
        x1 = min(X, vx + hs + 1)
        y1 = min(Y, vy + hs + 1)
        z1 = min(Z, vz + hs + 1)
        mask[x0:x1, y0:y1, z0:z1] = True
        return mask

    if rtype == "side":
        side = target_region.get("value", "").lower()
        mask = np.zeros(shape, dtype=bool)
        mid_z = Z // 2
        if side == "left":
            mask[:, :, :mid_z] = True
        elif side == "right":
            mask[:, :, mid_z:] = True
        else:
            raise ValueError(
                f"Invalid side value '{side}'. Use 'left' or 'right'."
            )
        return mask

    if rtype == "point":
        vx, vy, vz = target_region["voxel"]
        radius_mm = float(target_region.get(
            "radius_mm", config.DEFAULT_TARGET_RADIUS_MM
        ))
        sx, sy, sz = (float(v) for v in spacing)
        # Build distance grid in mm from the target point
        xi = np.arange(X, dtype=np.float64)
        yi = np.arange(Y, dtype=np.float64)
        zi = np.arange(Z, dtype=np.float64)
        dx = (xi - vx)[:, None, None] * sx
        dy = (vy - yi)[None, :, None] * sy
        dz = (zi - vz)[None, None, :] * sz
        dist2 = dx ** 2 + dy ** 2 + dz ** 2
        return dist2 <= radius_mm ** 2

    if rtype == "bbox":
        mn = target_region["min"]
        mx = target_region["max"]
        mask = np.zeros(shape, dtype=bool)
        x0, y0, z0 = max(0, int(mn[0])), max(0, int(mn[1])), max(0, int(mn[2]))
        x1, y1, z1 = min(X, int(mx[0]) + 1), min(Y, int(mx[1]) + 1), min(Z, int(mx[2]) + 1)
        mask[x0:x1, y0:y1, z0:z1] = True
        return mask

    raise ValueError(
        f"Unknown target_region type '{rtype}'. "
        "Supported: 'roi', 'side', 'point', 'bbox'."
    )


def select_implant_center(image, bone_mask, bone_dt, nerve_dist, spacing,
                          tooth_mask, tooth_excl_zone, arch_mask,
                          target_mask=None, occlusal_level=None):
    """Pick the best voxel for implant placement (5-rule evaluation).

    Hard filters:
      Rule 4  – inside bone, ≥ BONE_WALL_MARGIN_MM from walls (centering)
      Rule 3  – ≥ NERVE_SAFETY_MARGIN_MM from the nerve
      Rule 1  – outside tooth exclusion zone (1.5 mm clearance)
      Rule 2  – inside alveolar arch zone
      Edge    – away from scan borders (unless user target is set)
      Target  – user-specified region (if any)

    Per-candidate measurements & scoring:
      Rule 4  – bone_dt (higher = better centred)
      Rule 1  – KD-tree tooth clearance
      Rule 3  – nerve distance + sinus/air proximity above
      Rule 2  – local neighbourhood mean HU ≥ ALVEOLAR_MIN_HU
      Rule 5  – gap detection → lock to gap + prefer gap centre

    Returns (bx, by, bz, wall_distance_mm, diagnostics).
    """
    shape = bone_mask.shape

    # ── Hard filters ──────────────────────────────────────────────────
    valid = bone_mask.copy()

    if target_mask is None:
        valid &= _volume_edge_mask(shape, config.EDGE_MARGIN_FRACTION)

    # Rule 4: bone wall margin (centering)
    valid &= (bone_dt >= config.BONE_WALL_MARGIN_MM)

    # Rule 3: nerve safety
    valid &= (nerve_dist >= config.NERVE_SAFETY_MARGIN_MM)

    # Rule 1: tooth clearance (1.5 mm exclusion zone)
    valid &= ~tooth_excl_zone

    # Rule 2: alveolar arch zone
    valid &= arch_mask

    # Occlusal plane alignment
    if occlusal_level is not None:
        osx = float(spacing[0])
        x_lo = max(0, int(occlusal_level
                          + config.OCCLUSAL_PLANE_MARGIN_MIN_MM / osx))
        x_hi = min(shape[0] - 1, int(occlusal_level
                                      + config.OCCLUSAL_PLANE_MARGIN_MAX_MM / osx))
        occlusal_band = np.zeros(shape, dtype=bool)
        occlusal_band[x_lo:x_hi + 1, :, :] = True
        valid &= occlusal_band

    if target_mask is not None:
        valid &= target_mask

    coords = np.argwhere(valid)
    if len(coords) == 0:
        raise RuntimeError(
            "No valid candidate voxels found.  The bone region may be too "
            "thin, too close to teeth / nerve / sinus / scan border, or "
            "outside the alveolar arch.  Consider relaxing constraints."
        )

    stride = max(1, len(coords) // config.MAX_CANDIDATE_SAMPLES)
    sampled = coords[::stride]

    # ── Bulk reads at candidate positions ─────────────────────────────
    dt_values = bone_dt[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    nerve_values = nerve_dist[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    hu_values = image[sampled[:, 0], sampled[:, 1], sampled[:, 2]]

    # ── Per-candidate measurements ────────────────────────────────────
    print("    Computing crest depths ...")
    crest_depths = _measure_superior_surface_depths(bone_mask, sampled, spacing)

    print("    Computing tooth-above ratios ...")
    tooth_above_ratios = _measure_tooth_above_ratios(image, sampled, spacing)

    print("    Computing local density (Rule 2) ...")
    local_density = _measure_local_density(
        image, sampled, spacing, config.LOCAL_DENSITY_RADIUS_MM,
    )

    print("    Computing sinus/air proximity (Rule 3) ...")
    air_above_dist = _measure_distance_to_air_above(
        image, bone_mask, sampled, spacing,
    )

    print("    Computing tooth clearance via KD-tree (Rule 1) ...")
    tooth_clearance = _measure_tooth_clearance(tooth_mask, sampled, spacing)

    print("    Computing buccal-lingual centering ...")
    bl_centering = _measure_coronal_centering(bone_mask, sampled, spacing)

    # ── Strict / relaxed anatomical filter ────────────────────────────
    density_ok = local_density >= config.DYNAMIC_DENSITY_MIN_HU
    sinus_ok = air_above_dist >= config.SINUS_SAFETY_MARGIN_MM

    strict_mask = (
        density_ok
        & sinus_ok
        & (nerve_values <= config.MAX_SEARCH_NERVE_DISTANCE_MM)
        & (crest_depths >= config.MIN_IMPLANT_CENTER_DEPTH_MM)
        & (crest_depths <= config.MAX_IMPLANT_CENTER_DEPTH_MM)
        & (tooth_above_ratios <= config.MAX_TOOTH_ABOVE_RATIO)
    )
    relaxed_mask = (
        density_ok
        & (crest_depths >= config.MIN_IMPLANT_CENTER_DEPTH_MM)
        & (crest_depths <= config.RELAXED_MAX_IMPLANT_CENTER_DEPTH_MM)
        & (tooth_above_ratios <= config.RELAXED_MAX_TOOTH_ABOVE_RATIO)
    )

    selection_stage = "fallback_all_candidates"
    if np.any(strict_mask):
        selection_mask = strict_mask
        selection_stage = "strict_anatomical_filter"
    elif np.any(relaxed_mask):
        selection_mask = relaxed_mask
        selection_stage = "relaxed_anatomical_filter"
    else:
        selection_mask = np.ones(len(sampled), dtype=bool)

    sampled = sampled[selection_mask]
    dt_values = dt_values[selection_mask]
    nerve_values = nerve_values[selection_mask]
    hu_values = hu_values[selection_mask]
    crest_depths = crest_depths[selection_mask]
    tooth_above_ratios = tooth_above_ratios[selection_mask]
    local_density = local_density[selection_mask]
    air_above_dist = air_above_dist[selection_mask]
    tooth_clearance = tooth_clearance[selection_mask]
    bl_centering = bl_centering[selection_mask]

    # ── Rule 5: Edentulous gap preference ─────────────────────────────
    print("    Detecting edentulous gaps (Rule 5) ...")
    tooth_proj = _build_tooth_projection(image, bone_mask)
    bone_proj = bone_mask.sum(axis=0).astype(np.float32)
    gap_2d = _detect_edentulous_gaps(tooth_proj, bone_proj, spacing)
    gap_regions_found = int(label(gap_2d)[1])

    gap_at_yz = gap_2d[sampled[:, 1], sampled[:, 2]]
    not_tooth_itself = hu_values < config.TOOTH_HU_MIN
    in_gap = gap_at_yz & not_tooth_itself
    n_gap_candidates = int(np.sum(in_gap))

    gap_used = False
    if n_gap_candidates > 0:
        gap_used = True
        sampled = sampled[in_gap]
        dt_values = dt_values[in_gap]
        nerve_values = nerve_values[in_gap]
        hu_values = hu_values[in_gap]
        crest_depths = crest_depths[in_gap]
        tooth_above_ratios = tooth_above_ratios[in_gap]
        local_density = local_density[in_gap]
        air_above_dist = air_above_dist[in_gap]
        tooth_clearance = tooth_clearance[in_gap]
        bl_centering = bl_centering[in_gap]
        selection_stage += "+gap_locked"

    # ── Gap-centre distances (Rule 5) ─────────────────────────────────
    print("    Computing gap-centre distances (Rule 5) ...")
    gap_center_dists = _compute_gap_center_distances(gap_2d, sampled, spacing)

    # ── Multi-factor scoring ──────────────────────────────────────────
    # Rule 4: bone centering (higher bone_dt → more centred)
    thickness_score = np.minimum(dt_values, config.PREFERRED_WALL_DISTANCE_MM)

    # Rule 1: tooth clearance (farther from teeth → better)
    tooth_clr_score = np.minimum(tooth_clearance, 5.0) / 5.0

    # Rule 3: nerve distance
    nerve_score = np.minimum(nerve_values, config.PREFERRED_NERVE_DISTANCE_MM)

    # Rule 5: gap-centre proximity (closer to centre → better)
    gap_center_score = np.clip(
        1.0 - gap_center_dists / config.GAP_CENTER_SCALE_MM, 0.0, 1.0,
    )

    # Buccal-lingual centering (coronal plane)
    bl_score = (
        np.minimum(bl_centering, config.BL_CENTERING_MAX_MM)
        / config.BL_CENTERING_MAX_MM
    )

    # Penalties
    crest_penalty = np.abs(crest_depths - config.IDEAL_IMPLANT_CENTER_DEPTH_MM)
    tooth_penalty = 8.0 * tooth_above_ratios
    dense_core_penalty = np.clip(
        (hu_values - config.TOOTH_CORE_HU_MIN) / 400.0, 0.0, 2.5,
    )
    sinus_penalty = np.clip(
        config.SINUS_SAFETY_MARGIN_MM - air_above_dist, 0.0, 2.0,
    )

    # Dynamic density: penalise candidates outside ideal 300–1200 HU range
    density_range_ok = (
        (local_density >= config.DYNAMIC_DENSITY_IDEAL_MIN_HU)
        & (local_density <= config.DYNAMIC_DENSITY_IDEAL_MAX_HU)
    )
    density_range_penalty = np.where(density_range_ok, 0.0, 1.5)

    scores = (
        2.0 * thickness_score
        + 1.5 * tooth_clr_score
        + 0.25 * nerve_score
        + config.GAP_CENTER_WEIGHT * gap_center_score
        + config.BL_CENTERING_WEIGHT * bl_score
        - 0.9 * crest_penalty
        - tooth_penalty
        - dense_core_penalty
        - 2.0 * sinus_penalty
        - density_range_penalty
    )

    best_idx = int(np.argmax(scores))
    bx, by, bz = sampled[best_idx]
    wall_dist_mm = float(dt_values[best_idx])

    diagnostics = {
        "selection_stage": selection_stage,
        "site_depth_from_crest_mm": round(float(crest_depths[best_idx]), 2),
        "tooth_above_ratio": round(float(tooth_above_ratios[best_idx]), 4),
        "site_hu": round(float(hu_values[best_idx]), 1),
        "site_score": round(float(scores[best_idx]), 3),
        "gap_regions_found": gap_regions_found,
        "gap_candidates": n_gap_candidates,
        "gap_used": gap_used,
        "tooth_clearance_mm": round(float(tooth_clearance[best_idx]), 2),
        "local_density_hu": round(float(local_density[best_idx]), 1),
        "air_above_mm": round(float(air_above_dist[best_idx]), 2),
        "gap_center_dist_mm": round(float(gap_center_dists[best_idx]), 2),
        "bl_centering_mm": round(float(bl_centering[best_idx]), 2),
        "occlusal_level_vox": occlusal_level,
    }
    return int(bx), int(by), int(bz), wall_dist_mm, diagnostics


def measure_bone_boundaries(bone_dt, bx, by, bz, spacing):
    """Return (wall_distance_mm, bone_thickness_mm) at the implant centre.

    wall_distance_mm = bone_dt at the centre (distance to nearest wall).
    bone_thickness_mm ≈ 2 × wall_distance_mm (symmetrical approximation).
    """
    wall_dist = float(bone_dt[bx, by, bz])
    thickness = 2.0 * wall_dist
    return wall_dist, thickness


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Implant axis / angle estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_implant_axis(bone_dt, bx, by, bz, spacing):
    """Estimate insertion axis from the bone-thickness gradient.

    The gradient of the distance-transform points *away* from bone walls.
    At the bone centre its magnitude is small but its direction still
    indicates the principal bone axis.  We use Sobel on a local ROI for
    robustness against noise.

    Returns (angle_deg, axis_unit_vector).
    """
    sx, sy, sz = (float(v) for v in spacing)

    r = config.AXIS_ROI_RADIUS
    x0, x1 = max(0, bx - r), min(bone_dt.shape[0], bx + r + 1)
    y0, y1 = max(0, by - r), min(bone_dt.shape[1], by + r + 1)
    z0, z1 = max(0, bz - r), min(bone_dt.shape[2], bz + r + 1)
    roi = bone_dt[x0:x1, y0:y1, z0:z1].astype(np.float64)

    # Smooth slightly to reduce noise influence
    roi = gaussian_filter(roi, sigma=1.0)

    gx = sobel(roi, axis=0)
    gy = sobel(roi, axis=1)
    gz = sobel(roi, axis=2)

    mean_gx = float(gx.mean())
    mean_gy = float(gy.mean())
    mean_gz = float(gz.mean())
    norm = np.sqrt(mean_gx**2 + mean_gy**2 + mean_gz**2) + 1e-9
    axis_vec = np.array([mean_gx / norm, mean_gy / norm, mean_gz / norm])

    # Angle with the vertical (x-axis in this volume convention).
    # 90° = perfectly vertical (along x); lower = more tilted.
    cos_angle = abs(float(axis_vec[0]))
    deviation_deg = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))
    implant_angle = 90.0 - deviation_deg

    implant_angle = float(np.clip(implant_angle,
                                  config.MIN_ANGLE_DEG,
                                  config.MAX_ANGLE_DEG))
    return implant_angle, axis_vec


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Implant dimensions from available bone depth
# ═══════════════════════════════════════════════════════════════════════════

def _walk_along_axis(mask_3d, bx, by, bz, axis_vec, spacing, direction):
    """Walk from (bx,by,bz) along axis_vec*direction until leaving mask.
    Returns the distance walked in mm."""
    sx, sy, sz = (float(v) for v in spacing)
    step_mm = min(sx, sy, sz) * 0.5
    shape = mask_3d.shape
    dist = 0.0
    for _ in range(500):
        dist += step_mm
        ix = int(round(bx + direction * dist * axis_vec[0] / sx))
        iy = int(round(by + direction * dist * axis_vec[1] / sy))
        iz = int(round(bz + direction * dist * axis_vec[2] / sz))
        if not (0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]):
            break
        if not mask_3d[ix, iy, iz]:
            break
    return dist


def measure_bone_depth(bone_mask, bx, by, bz, axis_vec, spacing):
    """Total bone depth along the implant axis through the centre (mm)."""
    fwd = _walk_along_axis(bone_mask, bx, by, bz, axis_vec, spacing, +1)
    bwd = _walk_along_axis(bone_mask, bx, by, bz, axis_vec, spacing, -1)
    return fwd + bwd


def determine_implant_dimensions(bone_depth_mm, nerve_distance_mm,
                                 wall_distance_mm):
    """Choose implant length and diameter from available space.

    Length  : usable bone depth minus safety margins, clamped 8–14 mm.
    Diameter: constrained so the implant fits within the bone thickness
              with ≥ BONE_WALL_MARGIN_MM clearance on each side.
    """
    usable_depth = bone_depth_mm - config.NERVE_SAFETY_MARGIN_MM
    usable_depth = min(usable_depth,
                       nerve_distance_mm - config.NERVE_SAFETY_MARGIN_MM)
    length = float(np.clip(usable_depth,
                           config.MIN_IMPLANT_LENGTH_MM,
                           config.MAX_IMPLANT_LENGTH_MM))

    # Diameter must fit inside 2 × (wall_distance − margin)
    max_diam_from_bone = 2.0 * (wall_distance_mm - config.BONE_WALL_MARGIN_MM)
    diameter = float(np.clip(max_diam_from_bone,
                             config.MIN_DIAMETER_MM,
                             config.MAX_DIAMETER_MM))
    # Round to nearest 0.5 mm (standard implant increments)
    diameter = round(diameter * 2) / 2.0
    diameter = float(np.clip(diameter, config.MIN_DIAMETER_MM,
                             config.MAX_DIAMETER_MM))

    return round(length, 2), round(diameter, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 + 8 — Density along implant path → Misch classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_bone_density(mean_hu):
    """Misch classification from mean HU."""
    if mean_hu > 1250:
        return "D1"
    elif mean_hu > 850:
        return "D2"
    elif mean_hu > 350:
        return "D3"
    else:
        return "D4"


def evaluate_density_along_path(image, bx, by, bz, axis_vec,
                                length_mm, diameter_mm, spacing):
    """Sample HU values inside the planned implant cylinder.

    Returns (mean_hu, std_hu, misch_class).
    """
    sx, sy, sz = (float(v) for v in spacing)
    half_len = length_mm / 2.0
    radius_mm = diameter_mm / 2.0
    shape = image.shape

    av = np.array(axis_vec, dtype=np.float64)

    # Two perpendicular vectors for radial sampling
    arb = np.array([1, 0, 0]) if abs(av[0]) < 0.9 else np.array([0, 1, 0])
    perp1 = np.cross(av, arb)
    perp1 /= (np.linalg.norm(perp1) + 1e-12)
    perp2 = np.cross(av, perp1)
    perp2 /= (np.linalg.norm(perp2) + 1e-12)

    n_axial = max(10, int(length_mm / 0.5))
    t_vals = np.linspace(-half_len, half_len, n_axial)
    n_radial = 5
    n_angular = 8

    samples = []
    for t in t_vals:
        for frac in np.linspace(0, 1, n_radial):
            r = frac * radius_mm
            angles = np.linspace(0, 2 * np.pi, n_angular, endpoint=False) if frac > 0 else [0.0]
            for ang in angles:
                off = r * (np.cos(ang) * perp1 + np.sin(ang) * perp2)
                px = bx + (t * av[0] + off[0]) / sx
                py = by + (t * av[1] + off[1]) / sy
                pz = bz + (t * av[2] + off[2]) / sz
                ix, iy, iz = int(round(px)), int(round(py)), int(round(pz))
                if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                    samples.append(float(image[ix, iy, iz]))

    if not samples:
        return 0.0, 0.0, "D4"

    mean_hu = float(np.mean(samples))
    std_hu = float(np.std(samples))
    return mean_hu, std_hu, classify_bone_density(mean_hu)


def adjust_params_by_density(bone_class, length, diameter):
    """Refine implant parameters based on Misch bone-density class.

    Softer bone → wider diameter for primary stability.
    Softer bone → prefer shorter implant to avoid fracture risk.
    """
    diam_adj = {"D1": -0.5, "D2": 0.0, "D3": 0.5, "D4": 1.0}
    len_adj  = {"D1": 0.0,  "D2": 0.0, "D3": -1.0, "D4": -2.0}

    diameter = diameter + diam_adj.get(bone_class, 0.0)
    diameter = float(np.clip(diameter, config.MIN_DIAMETER_MM,
                             config.MAX_DIAMETER_MM))

    length = length + len_adj.get(bone_class, 0.0)
    length = float(np.clip(length, config.MIN_IMPLANT_LENGTH_MM,
                           config.MAX_IMPLANT_LENGTH_MM))

    return round(length, 2), round(diameter, 2)


def adjust_angle_by_density(bone_class, base_angle):
    """Softer bone needs more vertical alignment to distribute load."""
    adjustments = {"D1": 0.0, "D2": 0.0, "D3": 2.0, "D4": 4.0}
    angle = base_angle + adjustments.get(bone_class, 0.0)
    return float(np.clip(angle, config.MIN_ANGLE_DEG, config.MAX_ANGLE_DEG))


# ═══════════════════════════════════════════════════════════════════════════
# Step 9 — Nerve safety validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_nerve_safety(nerve_dist, bx, by, bz, length_mm, axis_vec,
                          spacing):
    """Minimum distance from the implant body to the nerve (mm)."""
    sx, sy, sz = (float(v) for v in spacing)
    half_len = length_mm / 2.0
    av = np.array(axis_vec, dtype=np.float64)
    shape = nerve_dist.shape

    n_pts = max(10, int(length_mm / 0.5))
    min_d = float("inf")
    for t in np.linspace(-half_len, half_len, n_pts):
        ix = int(round(bx + t * av[0] / sx))
        iy = int(round(by + t * av[1] / sy))
        iz = int(round(bz + t * av[2] / sz))
        if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
            d = nerve_dist[ix, iy, iz]
            if d < min_d:
                min_d = d
    return float(min_d) if min_d < float("inf") else 0.0


def _auto_adjust_nerve_safety(bone_mask, nerve_dist, bx, by, bz,
                               length_mm, axis_vec, spacing):
    """Shift the implant superiorly or shorten it to satisfy nerve safety.

    Strategy 1: move the centre toward superior (smaller X) in 1-voxel
                steps until the minimum nerve distance along the implant
                body >= NERVE_SAFETY_MARGIN_MM.
    Strategy 2: reduce implant length in 0.5 mm increments.

    Returns (new_bx, new_by, new_bz, new_length, adjusted, adjustments).
    """
    sx = float(spacing[0])
    min_nerve = validate_nerve_safety(
        nerve_dist, bx, by, bz, length_mm, axis_vec, spacing,
    )
    if min_nerve >= config.NERVE_SAFETY_MARGIN_MM:
        return bx, by, bz, length_mm, False, {}

    adjustments = {}
    new_bx = bx

    # Strategy 1 — shift superiorly (decrease X)
    max_shift_vox = max(1, int(np.ceil(
        config.NERVE_ADJUST_MAX_SHIFT_MM / sx,
    )))
    for shift in range(1, max_shift_vox + 1):
        cx = bx - shift
        if cx < 0 or not bone_mask[cx, by, bz]:
            break
        d = validate_nerve_safety(
            nerve_dist, cx, by, bz, length_mm, axis_vec, spacing,
        )
        if d >= config.NERVE_SAFETY_MARGIN_MM:
            adjustments["superior_shift_mm"] = round(shift * sx, 2)
            return cx, by, bz, length_mm, True, adjustments
        new_bx = cx

    if new_bx != bx:
        adjustments["superior_shift_mm"] = round((bx - new_bx) * sx, 2)

    # Strategy 2 — reduce length
    for red_10 in range(5, int(config.NERVE_ADJUST_LENGTH_REDUCTION_MM * 10) + 1, 5):
        red = red_10 / 10.0
        test_len = length_mm - red
        if test_len < config.MIN_IMPLANT_LENGTH_MM:
            break
        d = validate_nerve_safety(
            nerve_dist, new_bx, by, bz, test_len, axis_vec, spacing,
        )
        if d >= config.NERVE_SAFETY_MARGIN_MM:
            adjustments["length_reduction_mm"] = round(red, 2)
            return new_bx, by, bz, round(test_len, 2), True, adjustments

    # Best-effort fallback
    best_len = max(
        config.MIN_IMPLANT_LENGTH_MM,
        length_mm - config.NERVE_ADJUST_LENGTH_REDUCTION_MM,
    )
    adjustments["length_reduction_mm"] = round(length_mm - best_len, 2)
    adjustments["warning"] = "Could not fully satisfy nerve safety"
    return new_bx, by, bz, round(best_len, 2), True, adjustments


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def plan_implant(image, nerve_mask, spacing, target_region=None):
    """Full planning pipeline — bone-thickness-centre approach.

    Parameters:
        image:         3-D CT volume (HU values).
        nerve_mask:    3-D binary mask of the Inferior Alveolar Nerve.
        spacing:       Voxel spacing (sx, sy, sz) in mm.
        target_region: Optional dict restricting the search area.
                       See build_target_mask() for supported formats.

    Returns a dict with all planning results.
    """
    sx, sy, sz = (float(v) for v in spacing)

    # Build target mask (if the user specified a region)
    target_mask = None
    if target_region is not None:
        print("Building target region mask ...")
        target_mask = build_target_mask(image.shape, target_region,
                                        (sx, sy, sz))
        print(f"  Target voxels: {int(target_mask.sum())}")

    # Step 1: nerve distance map
    print("Step 1: Computing nerve distance map ...")
    nerve_dist = compute_nerve_distance(nerve_mask, (sx, sy, sz))

    # Step 2: detect bone
    print("Step 2: Detecting bone region (HU thresholds) ...")
    bone_mask = detect_bone_region(image)
    bone_voxels = int(bone_mask.sum())
    print(f"  Bone voxels: {bone_voxels}")

    # Step 2b: Tooth segmentation (Rule 1)
    print("Step 2b: Segmenting teeth (Rule 1) ...")
    tooth_mask = segment_teeth(image, (sx, sy, sz))
    tooth_voxels = int(tooth_mask.sum())
    print(f"  Tooth voxels: {tooth_voxels}")

    # Step 2b-2: Occlusal plane estimation
    print("Step 2b-2: Estimating occlusal plane level ...")
    occlusal_level = compute_occlusal_plane_level(tooth_mask, (sx, sy, sz))
    if occlusal_level is not None:
        occ_mm = occlusal_level * sx
        print(f"  Occlusal plane at X = {occlusal_level:.1f} vox ({occ_mm:.1f} mm)")
    else:
        print("  No teeth found — occlusal plane filter disabled")

    # Step 2c: Tooth exclusion zone (Rule 1: 2 mm clearance)
    print(f"Step 2c: Building tooth exclusion zone ({config.TOOTH_CLEARANCE_MM} mm) ...")
    tooth_excl = build_tooth_exclusion_zone(
        tooth_mask, (sx, sy, sz), config.TOOTH_CLEARANCE_MM,
    )
    print(f"  Exclusion voxels: {int(tooth_excl.sum())}")

    # Step 2d: Alveolar arch mask (Rule 2)
    print("Step 2d: Building alveolar arch mask (Rule 2) ...")
    arch_mask = build_alveolar_arch_mask_3d(
        tooth_mask, bone_mask, (sx, sy, sz),
    )
    print(f"  Arch voxels: {int(arch_mask.sum())}")

    # Step 3: bone thickness map (distance transform of bone)
    print("Step 3: Computing bone thickness map (distance transform) ...")
    bone_dt = compute_bone_thickness_map(bone_mask, (sx, sy, sz))
    max_half_thickness = float(bone_dt.max())
    print(f"  Max half-thickness: {max_half_thickness:.2f} mm")

    # Step 4: select implant centre at bone-thickness midpoint
    print("Step 4: Selecting implant at bone-thickness midpoint ...")
    bx, by, bz, wall_dist, site_diagnostics = select_implant_center(
        image, bone_mask, bone_dt, nerve_dist, (sx, sy, sz),
        tooth_mask=tooth_mask,
        tooth_excl_zone=tooth_excl,
        arch_mask=arch_mask,
        target_mask=target_mask,
        occlusal_level=occlusal_level,
    )
    wall_dist_at_center, bone_thickness = measure_bone_boundaries(
        bone_dt, bx, by, bz, (sx, sy, sz)
    )
    print(f"  Implant centre:   [{bx}, {by}, {bz}]")
    print(f"  Wall distance:    {wall_dist_at_center:.2f} mm (each side)")
    print(f"  Bone thickness:   {bone_thickness:.2f} mm")
    print(f"  Selection stage:  {site_diagnostics['selection_stage']}")
    print(f"  Crest depth:      {site_diagnostics['site_depth_from_crest_mm']:.2f} mm")
    print(f"  Tooth ratio:      {site_diagnostics['tooth_above_ratio']:.4f}")
    print(f"  Gap regions:      {site_diagnostics.get('gap_regions_found', 0)}")
    print(f"  Gap used:         {site_diagnostics.get('gap_used', False)}")
    print(f"  Gap candidates:   {site_diagnostics.get('gap_candidates', 0)}")
    print(f"  Tooth clearance:  {site_diagnostics.get('tooth_clearance_mm', '?')} mm")
    print(f"  Local density:    {site_diagnostics.get('local_density_hu', '?')} HU")
    print(f"  Air above:        {site_diagnostics.get('air_above_mm', 'inf')} mm")
    print(f"  BL centering:     {site_diagnostics.get('bl_centering_mm', '?')} mm")
    if occlusal_level is not None:
        depth_below = (bx - occlusal_level) * sx
        print(f"  Occlusal depth:   {depth_below:.1f} mm below occlusal plane")
    print(f"  Gap centre dist:  {site_diagnostics.get('gap_center_dist_mm', 'inf')} mm")

    # Step 5: estimate axis / angle
    print("Step 5: Estimating implant axis ...")
    implant_angle, axis_vec = estimate_implant_axis(
        bone_dt, bx, by, bz, (sx, sy, sz)
    )
    print(f"  Axis vector: [{axis_vec[0]:.3f}, {axis_vec[1]:.3f}, {axis_vec[2]:.3f}]")
    print(f"  Base angle:  {implant_angle:.1f}°")

    # Step 6: dimensions from bone depth along axis
    print("Step 6: Determining implant dimensions ...")
    bone_depth = measure_bone_depth(bone_mask, bx, by, bz, axis_vec,
                                    (sx, sy, sz))
    nerve_dist_at_center = float(nerve_dist[bx, by, bz])
    length_mm, diameter_mm = determine_implant_dimensions(
        bone_depth, nerve_dist_at_center, wall_dist_at_center
    )
    print(f"  Bone depth:  {bone_depth:.2f} mm")
    print(f"  Length:      {length_mm} mm")
    print(f"  Diameter:    {diameter_mm} mm")

    # Step 7 + 8: density along path → Misch class → refine params
    print("Step 7-8: Evaluating density along implant path ...")
    mean_hu, std_hu, bone_class = evaluate_density_along_path(
        image, bx, by, bz, axis_vec, length_mm, diameter_mm, (sx, sy, sz)
    )
    length_mm, diameter_mm = adjust_params_by_density(
        bone_class, length_mm, diameter_mm
    )
    implant_angle = adjust_angle_by_density(bone_class, implant_angle)
    print(f"  Mean HU:        {mean_hu:.1f} ± {std_hu:.1f}")
    print(f"  Misch class:    {bone_class}")
    print(f"  Final length:   {length_mm} mm")
    print(f"  Final diameter: {diameter_mm} mm")
    print(f"  Final angle:    {implant_angle:.1f}°")

    # Step 9: nerve safety validation
    print("Step 9: Validating nerve safety ...")
    nerve_distance = validate_nerve_safety(
        nerve_dist, bx, by, bz, length_mm, axis_vec, (sx, sy, sz)
    )
    safe = nerve_distance >= config.NERVE_SAFETY_MARGIN_MM
    print(f"  Min nerve distance: {nerve_distance:.2f} mm")
    print(f"  Safety OK: {safe}")

    # Step 9b: Auto-adjust for nerve safety
    nerve_adjusted = False
    nerve_adjustments = {}
    if config.NERVE_AUTO_ADJUST and not safe:
        print("Step 9b: Auto-adjusting for nerve safety ...")
        bx, by, bz, length_mm, nerve_adjusted, nerve_adjustments = (
            _auto_adjust_nerve_safety(
                bone_mask, nerve_dist, bx, by, bz,
                length_mm, axis_vec, (sx, sy, sz),
            )
        )
        if nerve_adjusted:
            nerve_distance = validate_nerve_safety(
                nerve_dist, bx, by, bz, length_mm, axis_vec, (sx, sy, sz),
            )
            safe = nerve_distance >= config.NERVE_SAFETY_MARGIN_MM
            print(f"  Adjustments: {nerve_adjustments}")
            print(f"  New centre:  [{bx}, {by}, {bz}]")
            print(f"  New length:  {length_mm} mm")
            print(f"  New nerve distance: {nerve_distance:.2f} mm")
            print(f"  Safety OK: {safe}")

    return {
        "implant_center": [bx, by, bz],
        "implant_length": length_mm,
        "implant_diameter": diameter_mm,
        "implant_angle": round(implant_angle, 1),
        "implant_axis_vector": [round(float(v), 4) for v in axis_vec],
        "bone_density_hu": round(mean_hu, 1),
        "bone_density_std_hu": round(std_hu, 1),
        "bone_density_class": bone_class,
        "nerve_distance": round(nerve_distance, 2),
        "bone_depth_mm": round(bone_depth, 2),
        "bone_thickness_mm": round(bone_thickness, 2),
        "wall_distance_mm": round(wall_dist_at_center, 2),
        "bone_voxels": bone_voxels,
        "nerve_safety_ok": safe,
        "nerve_adjusted": nerve_adjusted,
        "nerve_adjustments": nerve_adjustments,
        "voxel_spacing_mm": [sx, sy, sz],
        "target_region": target_region,
        **site_diagnostics,
    }
