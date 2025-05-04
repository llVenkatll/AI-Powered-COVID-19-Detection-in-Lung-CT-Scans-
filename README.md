# AI-Powered-COVID-19-Detection-in-Lung-CT-Scans-
Created an advanced image processing and segmentation system to detect COVID-affected regions in lung CT scans using adaptive  thresholding and morphological operations.  Skills: MATLAB, Image Processing, Adaptive Histogram Equalization, Thresholding, Morphological Operations, Region Segmentation, and  Edge Detection Algorithms. 

# COVID-19 Detection in Lung CT Scans

## Overview
This MATLAB-based system provides advanced image processing and analysis for the detection of COVID-19 patterns in lung CT scans. The software implements a sophisticated multi-scale analysis framework with adaptive parameters to identify potential COVID-19 infected regions in lung CT images.

## Features
- **Multi-scale analysis framework** for capturing features at different resolutions
- **Adaptive block processing** with automatic parameter adjustment
- **Advanced texture analysis** using Gabor filters, LBP, and GLCM
- **Lung segmentation** for focusing analysis on relevant regions
- **Anomaly detection and scoring** to highlight potential COVID-19 patterns
- **Comprehensive visualization** of results with heatmaps and 3D visualizations
- **Quantitative analysis** with decision support metrics

## Requirements
- MATLAB R2020a or newer
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

## Installation
1. Clone or download this repository to your local machine
2. Open MATLAB and navigate to the project directory

## Usage
1. Run the main script `covid_ct_analysis.m` in MATLAB
2. When prompted, select your lung CT scan image file
3. The system will process the image and display the results
4. Analysis results will also be printed in the MATLAB Command Window

```matlab
% To run the script
covid_ct_analysis
```

## Input Data
The system accepts the following image formats:
- DICOM (.dcm) - Standard medical imaging format
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)
- BMP (.bmp)

## Output
The system generates multiple visualizations:
1. Original image with segmentation overlay
2. Anomaly score heatmap
3. 3D visualization of anomaly distribution
4. Block-by-block analysis results

Additionally, the Command Window will display metrics including:
- Total potential COVID-19 affected area
- Percentage of lung affected
- Maximum anomaly score
- Decision support suggestion

## Configuration
Parameters can be adjusted in the `config` struct at the beginning of the script:

```matlab
config = struct();
config.denoise = struct('method', 'gaussian', 'sigma', 1.8, 'size', [5 5]);
config.clahe = struct('clip_limit', 0.02, 'tiles', [8 8]);
config.sharpen = struct('radius', 2.5, 'amount', 1.8, 'threshold', 0.4);
config.edge = struct('method', 'canny', 'threshold', [0.04 0.10], 'sigma', 1.5);
config.morph = struct('close_radius', 3, 'open_radius', 2, 'fill_holes', true);
config.segment = struct('num_clusters', 4, 'distance', 'cityblock', 'min_area', 100);
```

## Algorithm Description
The processing pipeline consists of:

1. **Image preprocessing**
   - Grayscale conversion
   - Noise reduction
   - Contrast enhancement with CLAHE

2. **Lung segmentation**
   - Intensity-based thresholding
   - Morphological operations

3. **Block-based processing**
   - Overlapping blocks with 33% overlap
   - Adaptive parameter tuning
   - Feature extraction per block

4. **Anomaly detection**
   - Multi-scale edge detection
   - Texture feature analysis
   - Region properties calculation
   - Anomaly score computation

5. **Visualization and analysis**
   - Heatmap generation
   - 3D visualization
   - Decision support metrics

## Troubleshooting
- **Error reading image**: Ensure the image file is in a supported format
- **Insufficient memory**: Try processing a downsampled version of the image
- **Bilateral filter errors**: The system will automatically fall back to median filtering
- **Poor segmentation results**: Adjust parameters in the `config` struct
