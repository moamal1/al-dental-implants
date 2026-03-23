# Smart Dental Implant Planning System

AI-powered dental implant planning tool that uses a UNet neural network to segment the Inferior Alveolar Nerve (IAN) from CBCT/CT scans and automatically suggests optimal implant placement.

## Project Structure

```
pyProject/
├── config.py              # Central configuration (paths, model params, thresholds)
├── dicom_utils.py         # DICOM → NIfTI conversion (single file & folder)
├── inference.py            # UNet model loading and sliding-window inference
├── planning.py            # Implant candidate scoring and placement planning
├── visualization.py       # All visualization functions and nerve mask processing
├── pipeline.py            # Main entry point: DICOM → segmentation → planning
├── viewer.py              # Advanced multi-view viewer with bone density analysis
├── requirements.txt       # Python dependencies
├── ian_unet_model_v2.pth # Pre-trained UNet model weights
└── outputs/               # Default output directory (auto-created)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

**Interactive mode** (opens file dialog):
```bash
python pipeline.py
```

**CLI mode** with a single DICOM file:
```bash
python pipeline.py --dicom-file path/to/scan.dcm
```

**CLI mode** with a DICOM folder:
```bash
python pipeline.py --dicom-folder path/to/series/
```

**Custom model and output directory:**
```bash
python pipeline.py --dicom-file scan.dcm --model custom_model.pth --output-dir ./results
```

### 3. View Results

```bash
python viewer.py --planning-json outputs/planning_result.json
```

**With explicit paths** (if files were moved):
```bash
python viewer.py \
    --planning-json outputs/planning_result.json \
    --image path/to/input_case.nii.gz \
    --mask path/to/predicted_ian_same_shape.nii.gz \
    --output viewer_output.png
```

## Pipeline Workflow

1. **DICOM Conversion**: Converts input DICOM file/folder to NIfTI format
2. **Preprocessing**: Downsamples to max 256 voxels per dimension, normalizes intensity
3. **AI Inference**: Runs a 3D UNet (MONAI) with sliding-window inference to segment the IAN
4. **Post-processing**: Upsamples prediction back to original resolution
5. **Planning**: Computes distance-from-nerve map, identifies safe candidate regions, scores voxels, and selects optimal implant position
6. **Visualization**: Generates a 3-view planning figure (axial/coronal/sagittal)
7. **Output**: Saves planning results as JSON and the visualization as PNG

## Viewer Panels

The advanced viewer (`viewer.py`) generates a 10-panel figure:

| Panel | Description |
|-------|-------------|
| Axial View | CT slice at implant z-coordinate |
| Coronal View | CT slice at implant y-coordinate |
| Sagittal View | CT slice at implant x-coordinate |
| Implant Site Zoom | Zoomed view around implant center |
| Cross Sections (x3) | Slices at offsets from implant center |
| Tangential View | Averaged projection along y-axis |
| Panoramic Reconstruction | Averaged projection along x-axis |
| Planning Summary | Text summary with bone density analysis |

## Key Modules

### config.py
All tunable parameters: model architecture, planning thresholds, scoring weights, viewer settings. Edit this file to adjust behavior without modifying logic.

### planning.py
Implements the implant placement algorithm:
- Computes Euclidean distance transform from the nerve mask
- Filters candidates by: jaw region, density range, safe distance (2–12 mm from nerve)
- Scores candidates using weighted combination of nerve distance, bone density, and centrality
- Suggests implant dimensions based on available bone depth

### visualization.py
Contains bone density classification (Misch scale: D1–D4) and a vectorized density overlay for performance.

## Output Files

| File | Description |
|------|-------------|
| `predicted_ian_same_shape.nii.gz` | Predicted nerve segmentation mask (NIfTI) |
| `planning_result.json` | All planning metadata (implant position, dimensions, density) |
| `planning_result.png` | Basic 3-view planning visualization |
| `viewer_result.png` | Advanced 10-panel viewer (from viewer.py) |

## Notes for Developers

- The model expects 3D volumes normalized to [0, 1] with shape up to 256^3
- Inference uses `sliding_window_inference` with ROI size 64^3
- Bone density is classified per the Misch scale: D1 (>1250 HU) through D4 (<350 HU)
- The `add_density_overlay` function uses vectorized NumPy (not pixel-by-pixel loops) for performance
- All hardcoded paths have been replaced with configurable defaults in `config.py`
- Compressed DICOM formats (JPEG Lossless, JPEG 2000) are supported via `pylibjpeg` + `pylibjpeg-libjpeg` + `pylibjpeg-openjpeg`, with a SimpleITK/GDCM fallback
- DICOM folder slices are sorted by `ImagePositionPatient[2]` (Z-axis) before volume assembly
# -------
# al-dental-implants
