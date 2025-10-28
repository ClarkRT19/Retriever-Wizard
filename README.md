# 🧙‍♂️ Retriever Wizard [WIP]
A simple tool to visualize images using premade embeddings and metadata, incl. a small testset of a 1000 images.  

## Overview
Retriever Wizard is a Streamlit tool for AI-based exploration of large image collections. It compares images by using visual similarity (with precomputed embeddings) and provides fast nearest-neighbor search, ranked inspection, annotation marker, and simple 2D projections (UMAP or t-SNE). Typical uses include:
- Exploring thematically similar images
- Spotting gaps and inconsistencies in metadata
- Annotating the images for metadata crossexamination 
- Producing projection plots for research and reporting

It is developed for cross-institutional analysis of educational wall charts and artworks, but applicable to any large-scale image dataset with embeddings.  

## Tetset 
The Images and Metadata used are part of a larger collection of Education Wallcharts, owned by DPU (The Danish School of Education) and Aarhus University. The digitized charts are publicly available from The Danish Royal Library and the images chosen as exampledata are free from copyright. 
The Images and Metadata have been processed as part of a ongoing PhD project, where also the Embeddings have been produced as part of the project, the Embeddings have been made with: google/siglip2-giant-opt-patch16-384.[https://huggingface.co/google/siglip2-giant-opt-patch16-384]


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
The app.py is naturally set up to run the testset in the Examples folder, try out the testset or change embeddings, metadata and images to your own. 
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
streamlit run app.py
```

### Disclaimer
Parts of this project were drafted or refactored with the assistance of large language models (ChatGPT/GPT-5). 
All prompts/outputs were reviewed and validated by the maintainer.
