"""
Advanced multi-view dental implant planning viewer.

Loads the pipeline output (NIfTI image, predicted nerve mask, planning JSON)
and generates a detailed 10-panel visualization with bone density analysis.

Usage:
    # Using planning JSON (auto-detects image and mask paths):
    python viewer.py --planning-json outputs/planning_result.json

    # Explicit paths:
    python viewer.py \
        --image path/to/input_case.nii.gz \
        --mask path/to/predicted_ian_same_shape.nii.gz \
        --planning-json path/to/planning_result.json

    # Custom output:
    python viewer.py --planning-json results.json --output viewer_output.png
"""

import os
import sys
import json
import argparse

import numpy as np
import nibabel as nib

import config
from visualization import postprocess_nerve_mask, generate_detailed_viewer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dental Implant Planning Viewer"
    )
    parser.add_argument(
        "--planning-json", type=str, required=True,
        help="Path to the planning_result.json from the pipeline",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to the input NIfTI image (overrides JSON path)",
    )
    parser.add_argument(
        "--mask", type=str, default=None,
        help="Path to the predicted nerve mask NIfTI (overrides JSON path)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (default: outputs/viewer_result.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load planning metadata ────────────────────────────────────────
    with open(args.planning_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # ── Resolve file paths ────────────────────────────────────────────
    image_path = args.image or meta.get("temporary_nifti_path")
    mask_path = args.mask or meta.get("predicted_mask_path")

    if not image_path or not os.path.isfile(image_path):
        print(f"Error: Image file not found: {image_path}")
        print("Use --image to specify the NIfTI image path.")
        sys.exit(1)

    if not mask_path or not os.path.isfile(mask_path):
        print(f"Error: Mask file not found: {mask_path}")
        print("Use --mask to specify the predicted nerve mask path.")
        sys.exit(1)

    output_path = args.output or os.path.join(config.OUTPUT_DIR, "viewer_result.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Load image data ───────────────────────────────────────────────
    print(f"Loading image: {image_path}")
    img_nii = nib.load(image_path)
    image = img_nii.get_fdata().astype(np.float32)
    spacing = img_nii.header.get_zooms()[:3]
    sx, sy, sz = [float(v) for v in spacing]

    print(f"Loading nerve mask: {mask_path}")
    mask_nii = nib.load(mask_path)
    nerve_mask_raw = (mask_nii.get_fdata() > 0).astype(np.uint8)

    print(f"Image shape: {image.shape}")
    print(f"Spacing: ({sx:.3f}, {sy:.3f}, {sz:.3f}) mm")
    print(f"Implant center: {meta['implant_center_voxel']}")

    # ── Post-process nerve mask ───────────────────────────────────────
    print("Post-processing nerve mask...")
    nerve_mask = postprocess_nerve_mask(nerve_mask_raw)
    print(f"Post-processed nerve voxels: {int(nerve_mask.sum())}")

    # ── Generate viewer ───────────────────────────────────────────────
    print("Generating detailed viewer...")
    generate_detailed_viewer(
        image, nerve_mask, meta, (sx, sy, sz), output_path
    )
    print(f"\nViewer saved to: {output_path}")


if __name__ == "__main__":
    main()
