"""
Central configuration for the Dental Implant Planning System.

All configurable paths and parameters are defined here.
Override defaults via command-line arguments in pipeline.py and viewer.py.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "ian_unet_model_v2.pth")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# ── Model / Inference ────────────────────────────────────────────────────────
TARGET_MAX_DIM = 256
ROI_SIZE = (64, 64, 64)
SW_BATCH_SIZE = 1
UNET_CHANNELS = (16, 32, 64, 128)
UNET_STRIDES = (2, 2, 2)
UNET_NUM_RES_UNITS = 2

# ── Bone Detection (HU thresholds) ──────────────────────────────────────────
BONE_HU_MIN = 300       # Minimum HU for cortical/cancellous bone
BONE_HU_MAX = 3000      # Maximum HU (above this is likely metal artifact)

# ── Edge Protection ─────────────────────────────────────────────────────────
EDGE_MARGIN_FRACTION = 0.15   # Hard exclusion zone: 15% of each dimension

# ── Nerve Safety ─────────────────────────────────────────────────────────────
NERVE_SAFETY_MARGIN_MM = 2.0

# ── Bone Wall Margin ─────────────────────────────────────────────────────────
BONE_WALL_MARGIN_MM = 1.0    # Minimum distance from implant surface to bone wall

# ── Implant Dimensions ──────────────────────────────────────────────────────
MIN_IMPLANT_LENGTH_MM = 8.0
MAX_IMPLANT_LENGTH_MM = 14.0
DEFAULT_DIAMETER_MM = 4.0
MIN_DIAMETER_MM = 3.0
MAX_DIAMETER_MM = 5.0

# ── Angle Estimation ────────────────────────────────────────────────────────
AXIS_ROI_RADIUS = 15     # Voxel radius for local gradient estimation
MIN_ANGLE_DEG = 75.0     # Minimum allowed implant angle
MAX_ANGLE_DEG = 95.0     # Maximum allowed implant angle

# ── Candidate Selection ──────────────────────────────────────────────────────
MAX_CANDIDATE_SAMPLES = 15000
# ── Target Region Selection ──────────────────────────────────────────────
CLICK_ROI_HALF_SIZE = 5           # ±N voxels around user-clicked point
DEFAULT_TARGET_RADIUS_MM = 5.0    # Radius in mm around a CLI-specified point
# ── Viewer Parameters ───────────────────────────────────────────────────────
NERVE_MASK_MIN_COMPONENT_SIZE = 150
NERVE_MASK_KEEP_COMPONENTS = 2
ZOOM_RADIUS = 100
CROSS_SECTION_OFFSETS = [-15, 0, 15]
PANORAMIC_WIDTH = 50
TANGENTIAL_WIDTH = 50

# ── Bone Density Classification (Misch) ─────────────────────────────────────
BONE_DENSITY_THRESHOLDS = {
    "D1": (1250, float("inf"), "Dense cortical"),
    "D2": (850, 1250, "Thick cortical/porous"),
    "D3": (350, 850, "Thin porous cortical"),
    "D4": (float("-inf"), 350, "Fine trabecular"),
}
