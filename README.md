MSI_PROJECT - Image Dataset Augmentation for Waste Classification
📁 Project Structure
text
MSI_PROJECT/
│
├── images/                # Original (raw) dataset
│   ├── cardboard/         # Cardboard waste images
│   ├── glass/             # Glass waste images
│   ├── metal/             # Metal waste images
│   ├── paper/             # Paper waste images
│   ├── plastic/           # Plastic waste images
│   └── trash/             # General trash images
│
├── augmented/             # Automatically generated augmented images
│   ├── cardboard/         # Augmented cardboard images
│   ├── glass/             # Augmented glass images
│   ├── metal/             # Augmented metal images
│   ├── paper/             # Augmented paper images
│   ├── plastic/           # Augmented plastic images
│   └── trash/             # Augmented trash images
│
├── *.py                   # Python scripts for the project
│
└── venv/                  # (Optional) Virtual environment for Python
📋 Overview
This project is designed for waste classification and image augmentation. It organizes waste images into 6 categories and provides automated data augmentation to increase dataset size and diversity for machine learning model training.

🗂️ Directory Details
images/ - Raw Dataset
Contains the original, unmodified images organized by waste type:

cardboard/ - Images of cardboard waste (boxes, packaging, etc.)

`glass/`` - Images of glass containers and bottles

metal/ - Images of metal cans, foil, and containers

paper/ - Images of paper waste (newspaper, office paper, etc.)

plastic/ - Images of plastic bottles, containers, and packaging

trash/ - Miscellaneous trash items not fitting other categories

augmented/ - Generated Augmented Images
Contains automatically generated variations of the original images:

Each subfolder corresponds to the same waste categories as images/

Images are created through various augmentation techniques:

Rotation, flipping, scaling

Brightness/contrast adjustments

Color jittering

Gaussian noise addition

Used to expand training datasets for better model generalization

*.py - Python Scripts
Main project scripts (files will vary based on implementation):
