# Brain MRI Tumor Segmentation

This project implements a multi-model ensemble and fusion pipeline for brain tumor segmentation using Attention U-Net and Swin U-Net architectures.

## Project Structure
- `data/`: Contains the raw and processed datasets.
- `models/`: Trained model weights (.pth files).
- `src/`: Core model architectures and utility functions.
- `notebooks/`: Jupyter notebooks for preprocessing, training, and evaluation.
- `scripts/`: Python scripts for inference and parameter tuning.
- `results/`: Visual results and performance metrics.

## Pipeline
1. **Preprocessing**: Intensity normalization, CLAHE enhancement, and data splitting.
2. **Training**:
   - `attention_unet.ipynb`: Training the Attention U-Net.
   - `swin_unet.ipynb`: Training the Swin U-Net.
3. **Ensemble & Fusion**:
   - `ensemble_inference.py`: Weighted ensemble of base models.
   - `meta_learner.py`: Gated Meta-Fusion stacking ensemble.
4. **Analysis**:
   - `test_analysis.ipynb`: Comprehensive final evaluation and visualization.

## Key Features
- **Attention Mechanism**: Visualizing where the model focuses on the MRI.
- **Ensemble Techniques**: Combining global (Swin) and local (Attention) features.
- **Error Mapping**: Insightful visualization of TP, FP, and FN regions.
- **Statistical Rigor**: Comparative study with radar charts and violin plots.

## Setup
Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib opencv-python pillow tqdm scikit-learn scipy
```

## Usage
Refer to the notebooks in the `notebooks/` directory for step-by-step execution of the pipeline. The final evaluation can be found in `notebooks/test_analysis.ipynb`.
