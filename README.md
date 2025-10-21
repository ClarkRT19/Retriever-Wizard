# 🧙‍♂️ Retriever Wizard
A simple tool to visualize images using premade embeddings and metadata.

## Overview
Retriever Wizard is a Streamlit app for AI-based exploration of large image collections. It compares images by using visual similarity ( with precomputed embeddings) and provides fast nearest-neighbor search, ranked inspection, and simple 2D projections (UMAP or t-SNE). Typical uses include:
- Finding duplicates and near-duplicates across institutions
- Exploring thematic clusters
- Spotting metadata gaps and inconsistencies
- Producing projection plots for research and reporting

It is developed for cross-institutional analysis of educational wall charts, but applicable to any large-scale image dataset with embeddings.  

## Features
- Metadata validation: checks for a `filename` column (auto-derives from alternatives if possible)
- Embeddings integration: loads CLIP/SigLIP/Vision Transformer style CSVs
- File index: builds or loads a CSV map `filename → full_path`
- FAISS search: cosine (normalized IP) or L2
- Stacked view: ranked vertical preview with similarity scores and labels
- Projection: UMAP/t-SNE with coloring by (query/nearest/other) or cosine gradient
- Filters: include/exclude per column and optional `pandas.query()` expression
- Image display: multiple backends (Streamlit path, imageio, OpenCV, PIL, raw bytes)
- Checkpointing: save/load app state


## Installation (Windows)
```bash
git clone https://github.com/ClarkRT19/retriever-wizard.git
cd retriever-wizard

py -3.12 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
# If needed:
pip install faiss-cpu umap-learn scikit-learn plotly imageio pillow opencv-python
