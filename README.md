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
