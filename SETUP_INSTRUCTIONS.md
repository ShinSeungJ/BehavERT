# BehavERT Repository Setup Instructions

This document provides step-by-step instructions for setting up the BehavERT GitHub repository.

## 1. Initialize Git Repository

```bash
cd /media/sj/linux/behavert/BehavERT
git init
git add .
git commit -m "Initial commit: BehavERT repository structure"
```

## 2. Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click "New repository" or go to https://github.com/new
3. Set repository name: `BehavERT`
4. Add description: "BERT-based Animal Behavior Analysis from Keypoint Data"
5. Set to Public (for IJCV submission sharing)
6. **Do NOT** initialize with README, .gitignore, or license (we already have them)
7. Click "Create repository"

## 3. Connect Local Repository to GitHub

```bash
# Add GitHub remote (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/BehavERT.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 4. Repository Structure

After setup, your repository will have this structure:

```
BehavERT/
├── README.md                 # Comprehensive project documentation
├── LICENSE                   # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── SETUP_INSTRUCTIONS.md    # This file
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation script
│
├── behavert/              # Core BehavERT package
│   ├── __init__.py
│   ├── models/           # Model architectures
│   │   └── __init__.py
│   ├── data/            # Data processing
│   │   └── __init__.py
│   ├── training/        # Training utilities
│   │   └── __init__.py
│   └── utils/           # Utility functions
│       └── __init__.py
│
├── experiments/         # Experiment configurations
│   ├── calms21/
│   ├── mabe22/
│   ├── sbea/
│   ├── deepethogram/
│   └── pair24m/
│
├── scripts/            # Training and evaluation scripts
├── configs/            # Configuration files
│   ├── model_configs/
│   ├── data_configs/
│   └── training_configs/
│
├── notebooks/          # Jupyter notebooks
├── docs/              # Documentation
│   ├── installation.md
│   └── quickstart.md
└── tests/             # Unit tests
```

## 5. Next Steps for Code Migration

### Phase 1: Core Models (Priority 1)
```bash
# Copy core model implementations
cp ../code/sbea/dev/sbea_behavior_prediction.py behavert/models/bert_models.py
cp ../code/calms21/pretraining/calms21_pretrain_mlm.py scripts/pretrain.py
cp ../code/calms21/pretraining/calms21_finetune_from_pretrain.py scripts/finetune.py
```

### Phase 2: Dataset Classes (Priority 2)
```bash
# Copy dataset implementations
cp ../code/sbea/dev/sbea_behavior_prediction.py behavert/data/datasets.py
# Extract and organize dataset classes from other files
```

### Phase 3: Training Utilities (Priority 3)
```bash
# Copy training utilities
cp ../code/sbea/dev/loss_function.py behavert/training/losses.py
# Extract trainer classes from existing code
```

### Phase 4: Experiment Configs (Priority 4)
```bash
# Copy experiment configurations
cp -r ../code/calms21/ experiments/calms21/
cp -r ../code/mabe22/ experiments/mabe22/
cp -r ../code/sbea/ experiments/sbea/
# Clean and organize experiment files
```

## 6. Repository Settings (After GitHub Creation)

### Enable GitHub Pages (Optional)
1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: main, folder: /docs
4. This will create documentation website

### Add Repository Topics
Add these topics to help with discoverability:
- `animal-behavior`
- `deep-learning`
- `transformer`
- `bert`
- `keypoint-analysis`
- `computer-vision`
- `pytorch`
- `behavior-analysis`

### Create Release (After Code Migration)
1. Go to Releases → Create a new release
2. Tag: v1.0.0
3. Title: "BehavERT v1.0.0 - Initial Release"
4. Description: Include key features and datasets supported

## 7. IJCV Submission Preparation

### For Paper Submission
- Repository URL: `https://github.com/yourusername/BehavERT`
- Include this URL in your paper's "Code Availability" section
- Ensure repository is public before submission

### Documentation for Reviewers
- The README.md provides comprehensive overview
- docs/ folder contains detailed documentation
- experiments/ folder shows reproducible experiments
- Clear installation and usage instructions

## 8. Maintenance Commands

```bash
# Check repository status
git status

# Add new files
git add .
git commit -m "Add new feature: [description]"
git push

# Create new branch for features
git checkout -b feature/new-feature
git push -u origin feature/new-feature

# Merge branches
git checkout main
git merge feature/new-feature
git push
```

## 9. Collaboration Setup

### For Team Members
```bash
# Clone repository
git clone https://github.com/yourusername/BehavERT.git
cd BehavERT

# Install in development mode
pip install -e ".[dev]"
pre-commit install

# Create feature branch
git checkout -b feature/my-feature
```

### Branch Protection (Recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Require pull request reviews
4. Require status checks to pass

This setup provides a professional, well-organized repository ready for IJCV submission and future collaboration!
