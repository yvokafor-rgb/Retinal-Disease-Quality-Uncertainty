# Retinal Disease Detection with Quality Assessment & Uncertainty Estimation

### A research-oriented computer vision pipeline for more reliable retinal fundus image analysis

This repository investigates **multi-label retinal disease classification**, **retinal image quality assessment**, and **prediction uncertainty**. Its purpose is to study how image quality affects model reliability and how uncertainty-aware evaluation can inform referral-oriented decision support.

> **Research prototype only.** This project is not a clinically validated medical device and must not be used to make patient care decisions.

---

## Overview

Retinal fundus photographs can be challenging to interpret automatically when images are blurry, poorly illuminated, compressed, or otherwise degraded. A classifier can produce a confident prediction even when the image is insufficiently clear. This project investigates disease classification alongside quality and uncertainty analysis, with a focus on reliability rather than accuracy alone.

## Research objectives

- Prepare retinal fundus images for consistent model training and evaluation.
- Train and evaluate multi-label retinal disease classification models using RFMiD.
- Investigate image-quality assessment using EyeQ-related experiments.
- Evaluate predictive uncertainty and confidence calibration.
- Measure behavior under simulated image degradations.
- Explore risk–coverage curves and referral-oriented decision rules.

## Conceptual system architecture

```text
Retinal fundus images (RFMiD / EyeQ)
                |
                v
      Preprocessing pipeline
  (crop, resize, normalize)
                |
         +------+------+
         |             |
         v             v
  Disease model    Quality assessment
         |             |
         +------+------+
                |
                v
  Uncertainty & calibration analysis
                |
                v
     Robustness evaluation
     Risk–coverage analysis
                |
                v
  Research-oriented decision support
```

This diagram describes the research workflow; see the implementation and experiment files for the exact components executed.

## Project components

| Component | Purpose | Implementation entry point |
| --- | --- | --- |
| Preprocessing | Prepare retinal images | `src/preprocessing.py` |
| Disease modeling | Train and evaluate classifiers | `src/model_training.py` |
| Image quality | Analyze quality-related outcomes | `src/eyeq_quality.py` |
| Deployment prototype | Application / inference experiments | `src/model_deployment.py` |
| Experiments | Save quantitative results and logs | `results/RFMiD_Results/` |

## Repository structure

```text
Retinal-Disease-Quality-Uncertainty/
├── images/               # Selected visual results for this README
├── src/                  # Core Python scripts
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── eyeq_quality.py
│   └── model_deployment.py
├── results/
│   └── RFMiD_Results/    # Original experiment hierarchy preserved
├── docs/                 # Supplementary research documentation
├── README.md
├── requirements.txt
└── .gitignore
```

Large datasets and model weights are intentionally excluded from Git. The structure above is the target layout; check the current commit for actual available files.

## 🛠️ Technology

- **Language:** Python
- **Domain:** Computer Vision, Deep Learning, Multi-label Classification
- **Data:** RFMiD retinal fundus images; EyeQ image-quality experiments
- **Evaluation focus:** Disease-wise metrics, calibration, robustness, risk–coverage, confidence intervals

For exact library versions, see [`requirements.txt`](requirements.txt).

## Experimental evaluation

The available experiment exports may include:

- Per-disease evaluation metrics and predictions
- Confusion matrices and reliability diagrams
- Bootstrap confidence intervals
- Image-degradation experiments and risk–coverage plots
- Image quality validation metrics

**Insert verified results here after checking the generated CSV/JSON files.** Do not interpret a chart as proving clinical safety or use an unverified result as a headline metric.

### Suggested results table

| Evaluation question | Metric | Verified result |
| --- | --- | --- |
| Disease recognition | AUROC / macro F1 / mAP | To be populated |
| Probability calibration | ECE / Brier score | To be populated |
| Reliability under degradation | Performance change by severity | To be populated |
| Selective prediction | Risk at different coverage levels | To be populated |

## Getting started

Clone the project and install dependencies:

```bash
git clone https://github.com/yvokafor-rgb/Retinal-Disease-Quality-Uncertainty.git
cd Retinal-Disease-Quality-Uncertainty
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Dataset setup:** Acquire RFMiD and EyeQ through their authorized distribution sources and configure dataset paths according to the source scripts. The raw images are not included in the repository.

**Execution:** Review command-line arguments and path assumptions in `src/preprocessing.py`, `src/model_training.py`, `src/eyeq_quality.py`, and `src/model_deployment.py` before running. After moving scripts, imports and relative paths may need updating. Exact run commands should be added only after they have been tested in the reorganized layout.

## Limitations

- Model performance may vary across imaging devices, patient populations, and acquisition settings.
- Image-quality assessment does not replace specialist review.
- Predictive uncertainty is not a guarantee of correctness.
- Clinical validation and external dataset testing are required before any real-world medical use.

## Future directions

- Further validation across image-quality groups and unseen datasets
- Stronger calibration and selective-prediction analysis
- Reproducible model comparisons and automated regression tests
- Clearer documentation of referral thresholds and their tradeoffs

## 👩‍💻 Author

**Yvonne Okafor**
M.S. Computer Science, Georgia Southern University
Interests: Machine Learning, Deep Learning, Computer Vision, Trustworthy AI

---

*This repository documents an ongoing research effort. Reported methods and results should be verified against the code and experiment artifacts.*
