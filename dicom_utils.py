"""
DICOM to NIfTI conversion utilities.

Supports compressed DICOM formats (JPEG Lossless, JPEG 2000, etc.)
via pylibjpeg + pylibjpeg-libjpeg.  If those packages are not installed,
falls back to SimpleITK/GDCM for decompression.

DICOM folder slices are sorted by ImagePositionPatient[2] (Z-axis)
to ensure the 3D volume is correctly ordered.
"""

import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk

# ── Register JPEG transfer-syntax handlers (compressed DICOM) ───────────
# pylibjpeg auto-discovers libjpeg and openjpeg plugins via entry points.
# Simply importing it registers the handlers with pydicom.
try:
    import pylibjpeg  # noqa: F401 – registers JPEG/JPEG2000 handlers
except ImportError:
    pass


def _apply_modality_lut(ds, arr):
    """Apply RescaleSlope/RescaleIntercept if present to get Hounsfield Units."""
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    if slope != 1 or intercept != 0:
        arr = arr * slope + intercept
    return arr


def convert_single_dicom_to_nifti(dcm_path, out_nifti):
    """Convert a single multi-frame 3D DICOM file to NIfTI format.

    Handles compressed transfer syntaxes (JPEG Lossless, JPEG 2000, etc.)
    automatically via pylibjpeg.  Falls back to SimpleITK if pydicom
    cannot decompress the pixel data.
    """
    try:
        ds = pydicom.dcmread(dcm_path)
        tsuid = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        if getattr(tsuid, "is_compressed", False):
            ds.decompress()
        arr = ds.pixel_array.astype(np.float32)
    except Exception as pydicom_err:
        # Fallback: let SimpleITK (which bundles GDCM) read the file
        print(f"  pydicom could not decode pixels ({pydicom_err}); "
              "falling back to SimpleITK...")
        sitk_img = sitk.ReadImage(dcm_path)
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        # SimpleITK returns (Z, Y, X); transpose to (X, Y, Z) for nibabel
        arr = np.transpose(arr, (2, 1, 0))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        spacing = sitk_img.GetSpacing()          # (x, y, z)
        affine = np.eye(4, dtype=np.float32)
        affine[0, 0], affine[1, 1], affine[2, 2] = spacing
        nii = nib.Nifti1Image(arr, affine)
        nib.save(nii, out_nifti)
        return

    if arr.ndim != 3:
        raise RuntimeError(
            f"DICOM file is not a 3D volume. shape = {arr.shape}"
        )

    arr = np.transpose(arr, (2, 1, 0))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = _apply_modality_lut(ds, arr)

    try:
        if hasattr(ds, "PixelSpacing"):
            px_spacing = [float(x) for x in ds.PixelSpacing]
        else:
            px_spacing = [1.0, 1.0]
        slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
        spacing = (px_spacing[0], px_spacing[1], slice_thickness)
    except Exception:
        spacing = (1.0, 1.0, 1.0)

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    nii = nib.Nifti1Image(arr, affine)
    nib.save(nii, out_nifti)


def _sort_dicom_files_by_z(file_list):
    """Sort DICOM file paths by ImagePositionPatient[2] (Z-axis).

    Falls back to InstanceNumber, then filename, if position is missing.
    """
    entries = []
    for fpath in file_list:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "SliceLocation"):
                z = float(ds.SliceLocation)
            elif hasattr(ds, "InstanceNumber"):
                z = float(ds.InstanceNumber)
            else:
                z = 0.0
            entries.append((z, fpath))
        except Exception:
            entries.append((0.0, fpath))

    entries.sort(key=lambda e: e[0])
    return [fpath for _, fpath in entries]


def convert_dicom_folder_to_nifti(folder_path, out_nifti):
    """Convert a DICOM series folder to NIfTI format.

    Slices are sorted by ImagePositionPatient Z-axis so the resulting
    volume has correct anatomical ordering.

    Uses SimpleITK (backed by GDCM) which natively handles compressed
    transfer syntaxes.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)

    if not series_ids:
        raise RuntimeError("No DICOM series found in folder.")

    # Get files for the first (usually only) series
    series_files = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])

    # Sort slices by Z-position for correct 3D ordering
    sorted_files = _sort_dicom_files_by_z(series_files)

    reader.SetFileNames(sorted_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    sitk.WriteImage(image, out_nifti)
