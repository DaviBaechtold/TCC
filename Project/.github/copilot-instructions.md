<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements

- [x] Scaffold the Project

- [x] Customize the Project

- [x] Install Required Extensions

- [x] Compile the Project

- [x] Create and Run Task

- [x] Launch the Project

- [x] Ensure Documentation is Complete

## Project Specific Instructions

This is a TCC thesis project focused on latent space generation with:
- Monocular depth estimation (Depth Anything 2/Depth Pro)
- Human segmentation 
- Multi-view processing
- Video embeddings for temporal analysis
- MediaPipe keypoints integration
- Architecture inspired by MMPose/RTMPose

The project uses PyTorch as the main deep learning framework with modular components for each processing stage.

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run basic test**: `python scripts/test_basic.py`
3. **Debug training**: `python scripts/train.py --debug`
4. **Use VS Code task**: "Train Multimodal Model (Debug)"

## Project Structure

- `src/models/`: Core models (depth, segmentation, pose, fusion, embeddings)
- `src/data/`: Data loaders and preprocessing
- `src/training/`: Training pipeline and utilities
- `src/utils/`: Visualization and helper functions
- `configs/`: YAML configuration files
- `scripts/`: Training and evaluation scripts
- `notebooks/`: Jupyter notebooks for experimentation