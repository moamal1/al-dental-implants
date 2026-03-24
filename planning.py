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
from scipy.ndimage import distance_transform_edt, sobel, gaussian_filter

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


def select_implant_center(bone_mask, bone_dt, nerve_dist, spacing,
                           target_mask=None):
    """Pick the voxel that sits at the midpoint of bone thickness.

    Selection criteria (applied as hard filters then soft scoring):
      • Must be inside the bone (bone_mask == True)
      • Must be away from volume edges (EDGE_MARGIN_FRACTION)
      • Must have ≥ BONE_WALL_MARGIN_MM from both bone walls
      • Must have ≥ NERVE_SAFETY_MARGIN_MM from the nerve
      • If target_mask is provided, must lie within the target region

    Among valid candidates, the voxel with the **highest bone_dt** is
    chosen — that is the point most centred within the bone thickness.

    Returns (bx, by, bz, wall_distance_mm).
    """
    shape = bone_mask.shape

    # Hard filter 1: inside bone
    valid = bone_mask.copy()

    # Hard filter 2: away from scan borders
    # Skip when a user-specified target region is provided — the user
    # explicitly chose this location even if it is near the border.
    if target_mask is None:
        valid &= _volume_edge_mask(shape, config.EDGE_MARGIN_FRACTION)

    # Hard filter 3: ≥ 1 mm from each bone wall
    valid &= (bone_dt >= config.BONE_WALL_MARGIN_MM)

    # Hard filter 4: safe distance from nerve
    valid &= (nerve_dist >= config.NERVE_SAFETY_MARGIN_MM)

    # Hard filter 5: restrict to user-specified target region
    if target_mask is not None:
        valid &= target_mask

    # Also filter air-like voxels: bone_dt must be > 0 (guaranteed by mask)
    coords = np.argwhere(valid)
    if len(coords) == 0:
        raise RuntimeError(
            "No valid candidate voxels found.  The bone region may be too "
            "thin or too close to the nerve / scan border.  Consider "
            "relaxing BONE_WALL_MARGIN_MM or EDGE_MARGIN_FRACTION."
        )

    # Sub-sample for performance
    stride = max(1, len(coords) // config.MAX_CANDIDATE_SAMPLES)
    sampled = coords[::stride]

    # Score: pick the voxel deepest inside the bone (max bone_dt)
    dt_values = bone_dt[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    best_idx = int(np.argmax(dt_values))
    bx, by, bz = sampled[best_idx]
    wall_dist_mm = float(dt_values[best_idx])

    return int(bx), int(by), int(bz), wall_dist_mm


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

    # Step 3: bone thickness map (distance transform of bone)
    print("Step 3: Computing bone thickness map (distance transform) ...")
    bone_dt = compute_bone_thickness_map(bone_mask, (sx, sy, sz))
    max_half_thickness = float(bone_dt.max())
    print(f"  Max half-thickness: {max_half_thickness:.2f} mm")

    # Step 4: select implant centre at bone-thickness midpoint
    print("Step 4: Selecting implant at bone-thickness midpoint ...")
    bx, by, bz, wall_dist = select_implant_center(
        bone_mask, bone_dt, nerve_dist, (sx, sy, sz),
        target_mask=target_mask,
    )
    wall_dist_at_center, bone_thickness = measure_bone_boundaries(
        bone_dt, bx, by, bz, (sx, sy, sz)
    )
    print(f"  Implant centre:   [{bx}, {by}, {bz}]")
    print(f"  Wall distance:    {wall_dist_at_center:.2f} mm (each side)")
    print(f"  Bone thickness:   {bone_thickness:.2f} mm")

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
        "voxel_spacing_mm": [sx, sy, sz],
        "target_region": target_region,
    }
