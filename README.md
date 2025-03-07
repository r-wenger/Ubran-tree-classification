# Urban Tree Species Classification using Sentinel-2 and PlanetScope

## Overview

This repository contains the code and datasets used for the research paper **"Urban trees species classification using Sentinel-2 and Planetscope satellite image time series"**, presented at JURSE 2025 in Tunis, Tunisia (May 4-7, 2025). The study focuses on classifying the 20 most representative urban tree species in Strasbourg using deep learning models applied to Satellite Image Time Series (SITS) from Sentinel-2 and PlanetScope.

### Key Features
- Utilizes **InceptionTime, H-InceptionTime, and LITE** models for multivariate Time Series Classification (TSC).
- Implements **sensor fusion** to combine spectral and temporal features from Sentinel-2 and PlanetScope.
- Achieves **69% accuracy** using the best-performing model (H-InceptionTime with sensor fusion).
- Codebase written in **PyTorch**.

## Repository Structure

```
📂 Urban-Tree-Classification
├── main.py          # Main script to run classification
├── models.py        # Definition of deep learning models (InceptionTime, H-InceptionTime, LITE)
├── params.py        # Configuration and hyperparameters
├── test.py          # Evaluation and performance analysis
├── train.py         # Training pipeline
├── utils.py         # Utility functions for data preprocessing and visualization
```

## Dataset
- **Sentinel-2**: 10 spectral bands, 10-20m resolution, 22 cloud-free images (2022).
- **PlanetScope**: 4-band imagery, 3.125m resolution, 53 cloud-free images (2022).
- Data preprocessing includes **zonal statistics**, buffer-based feature extraction, and co-registration.

## Results
| Model        | Sentinel-2 | PlanetScope | Fusion (S2 + PS) |
|-------------|------------|------------|----------------|
| InceptionTime  | 62.9% ± 0.4% | 62.2% ± 1.3% | 67.7% ± 1.4% |
| H-InceptionTime | 63.0% ± 0.3% | 62.4% ± 1.0% | 69.1% ± 0.3% |
| LITE         | 45.7% ± 2.4% | 48.7% ± 2.5% | 56.1% ± 2.5% |

## Citation
If you use this code, please cite:

```
@inproceedings{Latil2025JURSE,
  author    = {Marie Latil, Romain Wenger, David Michéa, Germain Forestier, Anne Puissant},
  title     = {Urban trees species classification using Sentinel-2 and Planetscope satellite image time series},
  booktitle = {JURSE 2025},
  year      = {2025},
  address   = {Tunis, Tunisia}
}
```

## Acknowledgments
- Thanks to Open Data Strasbourg for the urban tree inventory.
- Computational resources provided by Mesocentre Unistra.
- Funded by CNES-TOSCA project AIM-CEE and ANR project M2-BDA (ANR-24-CE23-1130).

## License
This project is licensed under the MIT License.
