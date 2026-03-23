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

# ── Planning Parameters ─────────────────────────────────────────────────────
MARGIN_FRACTION = 0.08
LOWER_JAW_FRACTION = 0.40
CENTRAL_Y_RANGE = (0.15, 0.85)
MID_Z_RANGE = (0.25, 0.80)
DENSITY_PERCENTILE_LOW = 55
DENSITY_PERCENTILE_HIGH = 97
SAFE_DISTANCE_RANGE_MM = (2.0, 12.0)
DEFAULT_DIAMETER_MM = 4.0
DEFAULT_ANGLE_DEG = 90.0
MIN_IMPLANT_LENGTH_MM = 8.0
MAX_IMPLANT_LENGTH_MM = 12.0
NERVE_SAFETY_MARGIN_MM = 2.0
MAX_CANDIDATE_SAMPLES = 12000

# ── Scoring Weights ─────────────────────────────────────────────────────────
SCORE_WEIGHT_DISTANCE = 0.55
SCORE_WEIGHT_DENSITY = 0.45
CENTRALITY_PENALTY_Y = 0.03
CENTRALITY_PENALTY_Z = 0.02

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
