# BioCAL3D

## Overview

BioCAD3D is a research project investigating the use of intraoral
scanner 3D data and artificial intelligence for objective detection
and quantification of disclosed dental plaque.

The project focuses on colour information captured by intraoral
scanners from laboratory samples with experimentally controlled
plaque accumulation.

## Project objectives

The project investigates:

1. Colour-based detection of disclosed dental plaque.
2. Variation in colour measurements between intraoral scanners.
3. Scanner colour calibration and normalization.
4. Scanner-robust visual feature learning.
5. Spatial segmentation of dental plaque.
6. Quantitative plaque assessment.
7. Multimodal prediction of experimental plaque severity.

## Data

The primary data consist of coloured PLY meshes acquired using
multiple intraoral scanners.

Each PLY file may contain:

- 3D vertex coordinates (XYZ)
- RGB vertex colours
- Triangle connectivity

Raw research data are not stored in this repository.

## Repository structure

- `src/` - reusable project code
- `scripts/` - executable analysis scripts
- `notebooks/` - exploratory analysis
- `tests/` - automated tests
- `configs/` - experiment configuration
- `docs/` - project documentation

## Installation

Create the Conda environment:

```bash
conda create -n biocad3d python=3.11
conda activate biocad3d
pip install -e .
