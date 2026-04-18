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

# ── Anatomical Site Selection ────────────────────────────────────────────────
MAX_SEARCH_NERVE_DISTANCE_MM = 25.0
MIN_IMPLANT_CENTER_DEPTH_MM = 2.0
IDEAL_IMPLANT_CENTER_DEPTH_MM = 4.5
MAX_IMPLANT_CENTER_DEPTH_MM = 8.0
RELAXED_MAX_IMPLANT_CENTER_DEPTH_MM = 12.0
PREFERRED_WALL_DISTANCE_MM = 5.0
PREFERRED_NERVE_DISTANCE_MM = 8.0
TOOTH_HU_MIN = 1400
TOOTH_CORE_HU_MIN = 1500
TOOTH_ABOVE_LOOKBACK_MM = 8.0
TOOTH_NEIGHBORHOOD_RADIUS_MM = 0.8
MAX_TOOTH_ABOVE_RATIO = 0.08
RELAXED_MAX_TOOTH_ABOVE_RATIO = 0.18

# ── Rule 1: Tooth Segmentation & Clearance ───────────────────────────────────
TOOTH_CLEARANCE_MM = 2.0          # Min distance from implant to any tooth surface (Safety Buffer)
TOOTH_MIN_VOLUME_MM3 = 50.0       # Min volume for a valid tooth component

# ── Rule 2: Alveolar Ridge Constraint ────────────────────────────────────────
ALVEOLAR_MIN_HU = 150             # Min local mean HU (below → cavity / sinus)
LOCAL_DENSITY_RADIUS_MM = 3.0     # Neighbourhood radius for local mean HU check
ARCH_DILATION_MM = 8.0            # Dilation radius defining alveolar arch zone

# ── Rule 3: Sinus / Cavity Safety ────────────────────────────────────────────
SINUS_AIR_HU_MAX = 0              # HU threshold for air / cavity detection
SINUS_SAFETY_MARGIN_MM = 2.0      # Min distance from sinus floor
AIR_ABOVE_SEARCH_MM = 30.0        # How far above candidate to search for air

# ── Gap (Edentulous Space) Detection ─────────────────────────────────────────
GAP_TOOTH_PROJECTION_MIN_VOXELS = 5   # Min tooth voxels per column → "has tooth"
GAP_DILATION_MM = 4.0                  # Dilation radius defining arch zone
GAP_MIN_WIDTH_MM = 2.5                 # Min gap component size to keep (mm/side)

# ── Rule 5: Gap Centre Preference ────────────────────────────────────────────
GAP_CENTER_WEIGHT = 3.0           # Scoring weight for gap-centre proximity
GAP_CENTER_SCALE_MM = 10.0        # Distance normalisation scale (mm)

# ── Occlusal Plane Alignment ────────────────────────────────────────────────
OCCLUSAL_PLANE_MARGIN_MIN_MM = 3.0    # Min depth of implant centre below occlusal plane
OCCLUSAL_PLANE_MARGIN_MAX_MM = 20.0   # Max depth of implant centre below occlusal plane
OCCLUSAL_PERCENTILE = 15              # Percentile of tooth X-coords for crown level

# ── Dynamic Density Filtering ───────────────────────────────────────────────
DYNAMIC_DENSITY_MIN_HU = 250          # Hard exclusion: below → cavity / sinus
DYNAMIC_DENSITY_IDEAL_MIN_HU = 300    # Ideal minimum HU for implant site
DYNAMIC_DENSITY_IDEAL_MAX_HU = 1200   # Ideal maximum HU (above → dense / metal)

# ── Buccal-Lingual Centering ────────────────────────────────────────────────
BL_CENTERING_WEIGHT = 1.5            # Scoring weight for coronal-plane centering
BL_CENTERING_MAX_MM = 5.0            # Normalisation cap (mm)

# ── Nerve Auto-Adjustment ───────────────────────────────────────────────────
NERVE_AUTO_ADJUST = True              # Auto-shift / shorten for nerve safety
NERVE_ADJUST_MAX_SHIFT_MM = 4.0      # Max superior shift (mm)
NERVE_ADJUST_LENGTH_REDUCTION_MM = 3.0  # Max length reduction (mm)

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
NERVE_MASK_CROP_MARGIN = 12
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
