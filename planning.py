"""
Implant planning logic: candidate region identification, scoring, and suggestion.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

import config


def compute_distance_map(nerve_mask, spacing):
    """Compute distance-from-nerve map in millimeters."""
    distance_vox = distance_transform_edt(1 - nerve_mask)
    mean_spacing = float(np.mean(spacing))
    return distance_vox * mean_spacing


def find_candidate_region(image, nerve_mask, distance_mm, shape):
    """Identify candidate voxels for implant placement.

    Returns:
        (candidate_mask, smooth_image)
    """
    smooth_image = gaussian_filter(image, sigma=1)

    X, Y, Z = shape
    xx, yy, zz = np.indices(shape)

    mf = config.MARGIN_FRACTION
    margin_x, margin_y, margin_z = int(mf * X), int(mf * Y), int(mf * Z)

    inside_margin = (
        (xx > margin_x) & (xx < X - margin_x) &
        (yy > margin_y) & (yy < Y - margin_y) &
        (zz > margin_z) & (zz < Z - margin_z)
    )

    lower_jaw = xx > int(config.LOWER_JAW_FRACTION * X)
    cy_lo, cy_hi = config.CENTRAL_Y_RANGE
    central_y = (yy > int(cy_lo * Y)) & (yy < int(cy_hi * Y))
    mz_lo, mz_hi = config.MID_Z_RANGE
    mid_z = (zz > int(mz_lo * Z)) & (zz < int(mz_hi * Z))

    density_low = np.percentile(smooth_image, config.DENSITY_PERCENTILE_LOW)
    density_high = np.percentile(smooth_image, config.DENSITY_PERCENTILE_HIGH)
    density_region = (smooth_image >= density_low) & (smooth_image <= density_high)

    d_lo, d_hi = config.SAFE_DISTANCE_RANGE_MM
    safe_distance = (distance_mm >= d_lo) & (distance_mm <= d_hi)

    candidate = (
        inside_margin & lower_jaw & central_y & mid_z &
        density_region & safe_distance
    )

    return candidate, smooth_image


def score_candidates(candidate_region, distance_mm, smooth_image, shape):
    """Score candidate voxels and return the best implant center (bx, by, bz)."""
    coords = np.argwhere(candidate_region)
    if len(coords) == 0:
        raise RuntimeError("No candidate region found for implant placement.")

    _, Y, Z = shape
    center_y, center_z = Y / 2.0, Z / 2.0

    stride = max(1, len(coords) // config.MAX_CANDIDATE_SAMPLES)
    sampled = coords[::stride]

    dist_scores = distance_mm[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    dens_scores = smooth_image[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    centrality = (
        config.CENTRALITY_PENALTY_Y * np.abs(sampled[:, 1] - center_y) +
        config.CENTRALITY_PENALTY_Z * np.abs(sampled[:, 2] - center_z)
    )

    scores = (
        config.SCORE_WEIGHT_DISTANCE * dist_scores +
        config.SCORE_WEIGHT_DENSITY * dens_scores -
        centrality
    )

    best_idx = np.argmax(scores)
    bx, by, bz = sampled[best_idx]
    return int(bx), int(by), int(bz), float(scores[best_idx])


def suggest_implant_params(distance_to_nerve_mm):
    """Calculate suggested implant dimensions based on nerve distance."""
    length = min(
        config.MAX_IMPLANT_LENGTH_MM,
        max(config.MIN_IMPLANT_LENGTH_MM,
            distance_to_nerve_mm - config.NERVE_SAFETY_MARGIN_MM)
    )
    return {
        "length_mm": round(length, 2),
        "diameter_mm": config.DEFAULT_DIAMETER_MM,
        "angle_deg": config.DEFAULT_ANGLE_DEG,
    }


def plan_implant(image, nerve_mask, spacing):
    """Full planning pipeline: find optimal implant position and parameters.

    Returns:
        dict with all planning results
    """
    sx, sy, sz = [float(v) for v in spacing]

    distance_mm = compute_distance_map(nerve_mask, (sx, sy, sz))
    candidate_region, smooth_image = find_candidate_region(
        image, nerve_mask, distance_mm, image.shape
    )

    candidate_count = int(np.sum(candidate_region))
    print(f"Candidate voxels: {candidate_count}")

    bx, by, bz, best_score = score_candidates(
        candidate_region, distance_mm, smooth_image, image.shape
    )

    distance_to_nerve = float(distance_mm[bx, by, bz])
    density_value = float(smooth_image[bx, by, bz])
    params = suggest_implant_params(distance_to_nerve)

    print(f"Implant center (voxel): [{bx}, {by}, {bz}]")
    print(f"Distance to nerve: {distance_to_nerve:.2f} mm")
    print(f"Suggested length: {params['length_mm']} mm")

    return {
        "implant_center_voxel": [bx, by, bz],
        "distance_to_nerve_mm": distance_to_nerve,
        "relative_density_value": density_value,
        "suggested_implant_length_mm": params["length_mm"],
        "suggested_implant_diameter_mm": params["diameter_mm"],
        "suggested_implant_angle_deg": params["angle_deg"],
        "candidate_voxels": candidate_count,
        "voxel_spacing_mm": [sx, sy, sz],
    }
