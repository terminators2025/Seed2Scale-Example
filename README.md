# Seed2Scale-Example
Demonstration of the end-to-end trajectory generation results for Seed2Scale, including the robot and scene model assets based on RealMirror Task1_Kitchen_Cleanup, physics replay scripts, and complete trajectories.

# Example Data & Asset
 - [x] Example Data：[Huggingface zte-terminators/seed2scale-example-data](https://huggingface.co/datasets/zte-terminators/seed2scale-example-data)
 - [x] Asset：[Huggingface zte-terminators/seed2scale-example-assets](https://huggingface.co/datasets/zte-terminators/seed2scale-example-assets)

# Data Structure
```
Seed2Scale-Example
├── assets
│   ├── robot
│   │   └── AgiBotA2(USD)
│   └── scenes
│       └── Task1_Kitchen_Cleanup
├── example-data
│   └── zte-terminators/seed2scale-example-data
└── python-script
    └── seed2scale_example_replay.py
```

# Quick Start
```bash
isaacsim/python.sh seed2scale_example_replay.py
```
```bash
==================================================
REPLAY CONTROLS:
S : Play / Pause
R : Reset current rollout
N : Load a random rollout from folder
==================================================
```

# Process Visualization

<p align="center">
  <img src="./example_opt.gif" alt="Demo" width="95%" />
