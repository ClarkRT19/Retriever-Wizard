# 🧙‍♂️ Retriever Wizard [WIP]
A small Streamlit app built for my PhD project (IKK, Aarhus University) to make crosscollection-comparisions, via visual similarity using precomputed embeddings and metadata.
The primary use is an adaptive research tool, created for my analytical process. It reflects how I use my data and my research workflow. You’re welcome to study or adapt it, but expect revisions and tweaks.

## Overview
Retriever Wizard loads a metadata CSV and an embeddings CSV (CLIP/SigLIP/Vision Transformer-style). It builds a fast FAISS index for nearest-neighbour search, lets you rank and annotate results, and makes simple 2D projections for communication.

What it helps with:
- find visually similar charts (nearest neighbours)
- inspect clusters and outliers
- annotate candidates for cross-examination
- make simple 2D projections (UMAP/t-SNE) for slides/reports

## Tetset 
The included Examples folder contains a test set (1k images) from a larger collection of educational wall charts.
-Ownership: Danish School of Education (DPU), Aarhus University.
-Source: Digitized by The Royal Danish Library; the selected example images are free of copyright - Hyperlink to source included in the metadata.csv.
-Research context: Processed during an ongoing PhD project. Embeddings were produced with google/siglip2-giant-opt-patch16-384.

## Features
- Metadata validation: checks for a `filename` column (auto-derives from alternatives if possible)
- Embeddings integration: loads premade CLIP/SigLIP/Vision Transformer style CSVs
- File index: builds or loads a CSV map `filename → full_path`
- FAISS search: cosine (normalized IP) or L2
- Stacked view: ranked vertical preview with similarity scores and labels.
- Annotation function with csv output
- Projection: UMAP/t-SNE with coloring by (query/nearest/other) or cosine gradient
- Filters: include/exclude per column and optional `pandas.query()` expression
- Image display: multiple backends (Streamlit path, imageio, OpenCV, PIL, raw bytes)
- Checkpointing: save/load app state

## Missing Features
- Easier navigation between different steps
- Maybe merging step 1-5; removing the redundant checks in the validation of premade data.
  

## Trying it out
The ReWiz.py is naturally set up to run the testset in the Examples folder, try out the testset or change embeddings, metadata and images to your own. 
**Important notes**: The column: *filename* is central to the code and is what FAISS indexes the images after, and is the link between metadata, images and embeddings. All image-names must be different, and you can't use the same named columns. 

## Installation (Windows)
```bash
git clone https://github.com/ClarkRT19/retriever-wizard.git
cd retriever-wizard

py -3.12 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
# If needed:
pip install faiss-cpu umap-learn scikit-learn plotly imageio pillow opencv-python
```

## Running Retriever Wizard:
```bash
streamlit run ReWiz.py
```

### Disclaimer
Parts of this project were drafted or refactored with the assistance of large language models (ChatGPT/GPT-5). 
All prompts/outputs were reviewed and validated by the maintainer.
