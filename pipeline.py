"""
Main pipeline: DICOM input -> IAN nerve segmentation -> implant planning.

Usage:
    # Interactive mode (file dialog + click on CT image to select target):
    python pipeline.py

    # CLI mode with a single DICOM file:
    python pipeline.py --dicom-file path/to/scan.dcm

    # CLI mode with a DICOM folder:
    python pipeline.py --dicom-folder path/to/series/

    # Select implant target point (±5 voxel ROI by default):
    python pipeline.py --dicom-file scan.dcm --target-point 120,80,64

    # Select target point with custom mm-radius:
    python pipeline.py --dicom-file scan.dcm --target-point 120,80,64 --target-radius 3.0

    # Restrict planning to the left side of the jaw:
    python pipeline.py --dicom-file scan.dcm --target-side left

    # Restrict planning to a bounding box:
    python pipeline.py --dicom-file scan.dcm --target-bbox 50,40,20,180,120,80

    # Specify custom model or output directory:
    python pipeline.py --dicom-file scan.dcm --model model.pth --output-dir ./results
"""

import os
import sys
import json
import argparse
import tempfile

import numpy as np
import nibabel as nib

import config
from dicom_utils import convert_single_dicom_to_nifti, convert_dicom_folder_to_nifti
from inference import load_model, preprocess_image, run_inference, postprocess_prediction
from planning import plan_implant
from visualization import generate_basic_planning_figure, interactive_point_selector


def select_input_interactive():
    """Show a tkinter dialog for input selection (GUI mode)."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askyesno(
        "Input Type Selection",
        "Press Yes to select a single DICOM file (.dcm)\n"
        "Press No to select a DICOM folder.",
    )

    if choice:
        path = filedialog.askopenfilename(
            title="Select DICOM file",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")],
        )
        if not path:
            print("No file selected.")
            sys.exit(1)
        return "single_dcm", path
    else:
        path = filedialog.askdirectory(title="Select DICOM folder")
        if not path:
            print("No folder selected.")
            sys.exit(1)
        return "dicom_folder", path


def select_target_interactive():
    """Show a tkinter dialog asking the user how to select the target region.

    Returns 'click' (to use image-based selector later), a target_region
    dict, or None (automatic / whole jaw).
    """
    import tkinter as tk
    from tkinter import simpledialog, messagebox

    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askyesnocancel(
        "Target Region Selection",
        "Would you like to select the implant location on the image?\n\n"
        "Yes    →  Click on the CT image to choose the exact spot\n"
        "No     →  Plan on the entire jaw (automatic)\n"
        "Cancel →  Abort",
    )

    if choice is None:
        print("Planning cancelled by user.")
        sys.exit(0)

    if choice:
        return "click"  # will show interactive_point_selector later

    return None  # whole jaw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dental Implant Planning Pipeline"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dicom-file", type=str, help="Path to a single DICOM file (.dcm)")
    group.add_argument("--dicom-folder", type=str, help="Path to a DICOM series folder")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    # ── Target region selection ──────────────────────────────────────
    target_grp = parser.add_mutually_exclusive_group()
    target_grp.add_argument(
        "--target-side", type=str, choices=["left", "right"],
        help="Restrict planning to left or right side of the jaw",
    )
    target_grp.add_argument(
        "--target-point", type=str, metavar="X,Y,Z",
        help="Approximate target voxel coordinate, e.g. '120,80,64'",
    )
    target_grp.add_argument(
        "--target-bbox", type=str, metavar="X0,Y0,Z0,X1,Y1,Z1",
        help="Target bounding box in voxel coordinates",
    )
    parser.add_argument(
        "--target-radius", type=float, default=None,
        help="Radius in mm around --target-point (default: %(default)s)",
    )

    return parser.parse_args()


def build_target_region_from_args(args):
    """Convert CLI target arguments into a target_region dict (or None)."""
    if args.target_side:
        return {"type": "side", "value": args.target_side}

    if args.target_point:
        coords = [int(v) for v in args.target_point.split(",")]
        if len(coords) != 3:
            raise ValueError("--target-point must have exactly 3 values: X,Y,Z")
        if args.target_radius is not None:
            # Explicit radius → mm-based sphere
            return {"type": "point", "voxel": coords,
                    "radius_mm": args.target_radius}
        # No radius → tight voxel-based ROI (±5 voxels)
        return {"type": "roi", "voxel": coords}

    if args.target_bbox:
        vals = [int(v) for v in args.target_bbox.split(",")]
        if len(vals) != 6:
            raise ValueError("--target-bbox must have exactly 6 values: X0,Y0,Z0,X1,Y1,Z1")
        return {"type": "bbox", "min": vals[:3], "max": vals[3:]}

    return None


def main():
    args = parse_args()

    # ── Determine input ──────────────────────────────────────────────
    if args.dicom_file:
        input_type, input_path = "single_dcm", args.dicom_file
    elif args.dicom_folder:
        input_type, input_path = "dicom_folder", args.dicom_folder
    else:
        input_type, input_path = select_input_interactive()

    model_path = args.model or config.MODEL_PATH
    output_dir = args.output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ── Output paths ─────────────────────────────────────────────────
    temp_dir = tempfile.mkdtemp(prefix="dicom_pipeline_")
    temp_nifti_path = os.path.join(temp_dir, "input_case.nii.gz")
    pred_mask_path = os.path.join(output_dir, "predicted_ian_same_shape.nii.gz")
    output_png = os.path.join(output_dir, "planning_result.png")
    output_json = os.path.join(output_dir, "planning_result.json")

    # ── Step 1: Convert DICOM to NIfTI ───────────────────────────────
    print("Converting DICOM to NIfTI...")
    print(f"  Input ({input_type}): {input_path}")
    if input_type == "single_dcm":
        convert_single_dicom_to_nifti(input_path, temp_nifti_path)
    else:
        convert_dicom_folder_to_nifti(input_path, temp_nifti_path)
    print(f"  Temporary NIfTI: {temp_nifti_path}")

    # ── Step 2: Load and preprocess image ─────────────────────────────
    orig_nii = nib.load(temp_nifti_path)
    orig_img = orig_nii.get_fdata().astype(np.float32)
    orig_img = np.nan_to_num(orig_img, nan=0.0, posinf=0.0, neginf=0.0)
    orig_shape = orig_img.shape
    spacing = orig_nii.header.get_zooms()[:3]
    sx, sy, sz = [float(v) for v in spacing]

    print(f"  Original shape: {orig_shape}")
    print(f"  Voxel spacing: ({sx:.3f}, {sy:.3f}, {sz:.3f}) mm")

    inputs, downsampled_shape = preprocess_image(orig_img)

    # ── Step 3: Run model inference ───────────────────────────────────
    print("Loading model and running inference...")
    model, device = load_model(model_path)
    pred_small = run_inference(model, inputs, device)
    print(f"  Predicted mask (small): {pred_small.shape}, voxels={int(np.sum(pred_small))}")

    # ── Step 4: Post-process prediction ───────────────────────────────
    pred_mask = postprocess_prediction(pred_small, orig_shape)
    print(f"  Predicted mask (restored): {pred_mask.shape}, voxels={int(np.sum(pred_mask))}")

    pred_nii = nib.Nifti1Image(
        pred_mask.astype(np.uint8), affine=orig_nii.affine, header=orig_nii.header
    )
    nib.save(pred_nii, pred_mask_path)
    print(f"  Saved predicted mask: {pred_mask_path}")

    # ── Step 5: Plan implant ──────────────────────────────────────────
    print("\nPlanning implant placement...")
    target_region = build_target_region_from_args(args)

    # In GUI mode with no CLI target, offer interactive selection
    if target_region is None and not args.dicom_file and not args.dicom_folder:
        user_choice = select_target_interactive()
        if user_choice == "click":
            print("Opening interactive point selector ...")
            clicked = interactive_point_selector(orig_img, pred_mask)
            if clicked is not None:
                bx, by, bz = clicked
                target_region = {"type": "roi", "voxel": [bx, by, bz]}
                print(f"  User selected: [{bx}, {by}, {bz}]")
            else:
                print("  No point selected — using whole jaw.")
        elif isinstance(user_choice, dict):
            target_region = user_choice

    if target_region is not None:
        print(f"  Target region: {target_region}")

    planning_result = plan_implant(orig_img, pred_mask, (sx, sy, sz),
                                   target_region=target_region)

    # ── Step 6: Generate visualization ────────────────────────────────
    print("\nGenerating planning figure...")
    generate_basic_planning_figure(
        orig_img, pred_mask, planning_result, (sx, sy, sz), output_png
    )

    # ── Step 7: Save results ──────────────────────────────────────────
    result = {
        "input_type": input_type,
        "input_path": input_path,
        "temporary_nifti_path": temp_nifti_path,
        "predicted_mask_path": pred_mask_path,
        "original_shape": list(orig_shape),
        "downsampled_shape": list(downsampled_shape),
        "mode": "automatic_pipeline",
        **planning_result,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Saved planning metadata: {output_json}")

    # ── Done ──────────────────────────────────────────────────────────
    print("\nPipeline complete!")
    print(f"  Predicted mask:   {pred_mask_path}")
    print(f"  Planning image:   {output_png}")
    print(f"  Planning results: {output_json}")

    # Show completion dialog in GUI mode
    if not args.dicom_file and not args.dicom_folder:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(
                "Processing Complete",
                f"Pipeline completed successfully.\n\n"
                f"Predicted mask:\n{pred_mask_path}\n\n"
                f"Planning image:\n{output_png}\n\n"
                f"Planning results:\n{output_json}",
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
