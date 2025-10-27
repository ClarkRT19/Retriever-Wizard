# app.py — Retriever Wizard (EN) + Annotate Overlay + Export
# Cosine by default (IndexFlatIP + normalization) with UI toggle for L2.
# Robust column handling, contiguous arrays for FAISS, multi-root scanning,
# metadata column picker, cosine metrics, image display, status/progress, checkpointing.
# NEW:
#  - Step 7: Stacked nearest neighbors (vertical list; toggle "Hide metadata" for grid)
#  - Step 8: Annotate overlay (sessions/markers; per-image + batch; persistent)
#  - Step 9: UMAP/t-SNE + Export (merge overlay -> new metadata CSV)

import os, json, csv, traceback, math, random, hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import sys

# ========== FAISS ==========
try:
    import faiss  # pip install faiss-cpu
    _faiss_ok = True
except Exception:
    _faiss_ok = False

# ========== Optional plotting backends ==========
_has_plotly = False
try:
    import plotly.express as px
    try:
        import narwhals  # noqa: F401
    except Exception:
        pass
    _has_plotly = True
except Exception:
    _has_plotly = False

# ========== Optional UMAP ==========
_umap_ok = False
try:
    from umap import UMAP
    _umap_ok = True
except Exception:
    try:
        import umap.umap_ as umap
        UMAP = umap.UMAP
        _umap_ok = True
    except Exception:
        _umap_ok = False

# ========== Optional t-SNE (scikit-learn) ==========
_tsne_ok = False
try:
    from sklearn.manifold import TSNE  # pip install scikit-learn
    _tsne_ok = True
except Exception:
    _tsne_ok = False

st.set_page_config(page_title="Retriever", page_icon="🧙🏻‍♂️", layout="wide")
st.title("🧙🏻‍♂️Retriever Wizard")
st.caption("Flow: metadata → embeddings → images → check → index → pick image → filter → show results → stacked view → annotate → project/export.")

CHECKPOINT = Path(".retriever_wizard_checkpoint.json")
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
CHUNK = 1000  # flush frequency for index writing

# ========== STATE ==========
def _ensure_state():
    ss = st.session_state
    ss.setdefault("step", 1)  # 1..8
    ss.setdefault("meta_path",   str(Path("examples/metadata.csv").resolve()))
    ss.setdefault("embed_path",  str(Path("examples/embeddings.csv").resolve()))
    ss.setdefault("images_root", str(Path("examples/images").resolve()))
    ss.setdefault("output_dir",  str(Path("examples/_index").resolve()))
    ss.setdefault("index_name", "index.csv")
    ss.setdefault("auto_load_index", True)

    ss.setdefault("meta_ok", False)
    ss.setdefault("embed_ok", False)
    ss.setdefault("images_ok", False)
    ss.setdefault("meta_head", pd.DataFrame())
    ss.setdefault("embed_head", pd.DataFrame())
    ss.setdefault("images_count", 0)
    
    # FAISS
    ss.setdefault("faiss_ready", False)
    ss.setdefault("faiss_dim", None)
    ss.setdefault("embed_filenames", [])
    ss.setdefault("k_neighbors", 10)
    ss.setdefault("_faiss_index", None)
    ss.setdefault("index_metric", "Cosine (IP + normalization)")

    # index/filemap
    ss.setdefault("index_csv_path", None)
    ss.setdefault("filemap", {})  # key (basename.lower) -> full_path

    # query image
    ss.setdefault("query_image_path", "")

    # filters
    ss.setdefault("filter_include", {})
    ss.setdefault("filter_exclude", {})
    ss.setdefault("filter_query_text", "")

    # results
    ss.setdefault("result_meta_cols", [])
    ss.setdefault("debug_on", False)
    ss.setdefault("last_results_df", pd.DataFrame())

    # umap cache-ish
    ss.setdefault("umap_df", pd.DataFrame())
    ss.setdefault("umap_params", {})

    # overlay (defaults)
    ss.setdefault("overlay_session", "iconography_2025q4")
    ss.setdefault("overlay_marker", "ikonografi")

_ensure_state()

# --- Namespaced widget keys (avoid collisions across steps) ---
def _k_annot(name: str) -> str:
    return f"annot::{name}"

def _k_export(name: str) -> str:
    return f"export::{name}"

# ========== HELPERS ==========
def save_checkpoint():
    data = {
        "step": st.session_state["step"],
        "meta_path": st.session_state["meta_path"],
        "embed_path": st.session_state["embed_path"],
        "images_root": st.session_state["images_root"],
        "output_dir": st.session_state["output_dir"],
        "index_name": st.session_state["index_name"],
        "auto_load_index": st.session_state["auto_load_index"],
        "meta_ok": st.session_state["meta_ok"],
        "embed_ok": st.session_state["embed_ok"],
        "images_ok": st.session_state["images_ok"],
        "faiss_ready": st.session_state["faiss_ready"],
        "index_csv_path": st.session_state["index_csv_path"],
        "query_image_path": st.session_state["query_image_path"],
        "filter_include": st.session_state["filter_include"],
        "filter_exclude": st.session_state["filter_exclude"],
        "filter_query_text": st.session_state["filter_query_text"],
        "index_metric": st.session_state["index_metric"],
        "result_meta_cols": st.session_state["result_meta_cols"],
        "debug_on": st.session_state["debug_on"],
        "overlay_session": st.session_state["overlay_session"],
        "overlay_marker": st.session_state["overlay_marker"],
    }
    CHECKPOINT.write_text(json.dumps(data), encoding="utf-8")

def load_checkpoint():
    if not CHECKPOINT.exists():
        return False
    try:
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        for k, v in data.items():
            st.session_state[k] = v
        return True
    except Exception:
        return False

def is_hosted_env(output_dir: str) -> bool:
    try:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        t = p / ".write_test"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)
        return False
    except Exception:
        return True

def _clean_pathlike_to_name(x: str) -> str:
    x = str(x).strip().strip('"').strip("'")
    x = x.replace("\\", os.sep).replace("/", os.sep)
    return Path(x).name

def _fname_key(x: str) -> str:
    return _clean_pathlike_to_name(x).lower()

def _normalize_filename_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(_fname_key)

def parse_roots(root_field: str) -> List[str]:
    if not root_field:
        return []
    tmp = root_field
    for sep in [";", "|", ","]:
        tmp = tmp.replace(sep, "\n")
    roots = [Path(p.strip().strip('"').strip("'")).expanduser() for p in tmp.splitlines() if p.strip()]
    return [str(p) for p in roots if p.exists()]

def scan_images_single(root: str) -> List[str]:
    files: List[str] = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        if Path(e.name).suffix.lower() in SUPPORTED_IMG:
                            files.append(e.path)
        except Exception:
            continue
    return files

def scan_images_multi(roots: List[str]) -> List[str]:
    files: List[str] = []
    for r in roots:
        files.extend(scan_images_single(r))
    return files

@st.cache_data(show_spinner=False)
def load_index_df(index_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(index_csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def build_filemap_from_df(index_df: pd.DataFrame) -> Dict[str, str]:
    index_df = _standardize_columns(index_df)
    need = {"filename", "full_path"}
    cols = set(index_df.columns)
    if not need.issubset(cols):
        raise ValueError("Index CSV must contain columns 'filename' and 'full_path'.")
    mapping: Dict[str, str] = {}
    for _, r in index_df.iterrows():
        raw_name = r.get("filename", "") or Path(str(r["full_path"])).name
        key = _fname_key(raw_name)
        mapping[key] = str(r["full_path"])
    return mapping

def try_load_existing_index(output_dir: str, index_name: str) -> Tuple[bool, Dict[str, str], Optional[str]]:
    try:
        if not output_dir or not index_name:
            return False, {}, None
        p = Path(output_dir) / index_name
        if not p.exists():
            return False, {}, None
        df = load_index_df(str(p))
        filemap = build_filemap_from_df(df)
        return True, filemap, str(p)
    except Exception:
        return False, {}, None

# SAFE image display
def display_image(path: str, width: int = 360, caption: Optional[str] = None):
    from pathlib import Path
    p = str(path)
    cap = caption or Path(p).name
    try:
        st.image(p, caption=cap, width=width); return
    except Exception as e0:
        pass
    try:
        import imageio.v2 as iio
        arr = iio.imread(p)
        st.image(arr, caption=cap, width=width); return
    except Exception as e1:
        try:
            import cv2
            arr = cv2.imread(p)
            if arr is None: raise RuntimeError("cv2.imread returned None")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            st.image(arr, caption=cap, width=width); return
        except Exception as e2:
            try:
                from PIL import Image, ImageOps, ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try: Image.MAX_IMAGE_PIXELS = None
                except Exception: pass
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im)
                    st.image(im, caption=cap, width=width)
                return
            except Exception as e3:
                try:
                    with open(p, "rb") as f: data = f.read()
                    st.image(data, caption=cap, width=width); return
                except Exception as e4:
                    st.warning(
                        "Could not display image.\n"
                        f"- st.image(path): {repr(e0)}\n"
                        f"- imageio:        {repr(e1)}\n"
                        f"- OpenCV:         {repr(e2)}\n"
                        f"- PIL:            {repr(e3)}\n"
                        f"- raw bytes:      {repr(e4)}"
                    )

# ========== ROBUST COLUMN HANDLING ==========
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _ensure_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df).copy()
    if "filename" in df.columns:
        df["filename"] = _normalize_filename_series(df["filename"])
        return df
    cols = set(df.columns)
    for c in ["file_name", "original_filename", "image", "img", "name"]:
        if c in cols:
            df["filename"] = _normalize_filename_series(df[c])
            return df
    for c in ["full_path", "path", "filepath", "file_path"]:
        if c in cols:
            df["filename"] = _normalize_filename_series(df[c])
            return df
    return df

def _assert_has_filename(df: pd.DataFrame, origin: str):
    if "filename" not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise ValueError(f"[{origin}] is missing 'filename'. Columns found: {cols}")

def _debug_show_cols(label: str, df: Optional[pd.DataFrame]):
    if not st.session_state.get("debug_on", False):
        return
    try:
        if df is None:
            st.info(f"[DEBUG] {label}: df=None")
        else:
            st.info(f"[DEBUG] {label}: columns = {list(df.columns)} (n={len(df)})")
    except Exception:
        pass

# ========== SIMILARITY / METRICS ==========
@st.cache_data(show_spinner=False)
def load_embeddings_df(embed_csv: str) -> pd.DataFrame:
    df = pd.read_csv(embed_csv, low_memory=False)
    df = _ensure_filename_column(df)
    _assert_has_filename(df, "Embeddings CSV")
    num_cols = [c for c in df.columns if c != "filename"]
    if not num_cols:
        raise ValueError("Embeddings CSV has no vector columns.")
    df[num_cols] = df[num_cols].astype("float32")
    return df

def _safe_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def _cosine_similarity(query_vec: np.ndarray, cand_mat: np.ndarray) -> np.ndarray:
    qn = _safe_norm(query_vec)
    cn = _safe_norm(cand_mat)
    return (qn @ cn.T).ravel()

def _match_label(sim: float) -> str:
    if sim >= 0.90:  return "★★★★★ very close"
    if sim >= 0.80:  return "★★★★ close"
    if sim >= 0.70:  return "★★★ related"
    if sim >= 0.60:  return "★★ loose"
    return "★ weak"

def add_similarity_columns(df_neighbors: pd.DataFrame, embed_df: pd.DataFrame, query_filename: str) -> pd.DataFrame:
    if df_neighbors.empty: return df_neighbors
    embed_df = _ensure_filename_column(embed_df)
    _assert_has_filename(embed_df, "Embeddings DF")
    vec_cols = [c for c in embed_df.columns if c != "filename"]
    qkey = _fname_key(query_filename)
    qrow = embed_df.loc[embed_df["filename"] == qkey]
    if qrow.empty:
        df = df_neighbors.copy()
        df["cosine_similarity"] = np.nan
        df["cosine_distance"]  = np.nan
        df["score_0_100"]      = np.nan
        df["match_quality"]    = "n/a"
        return df
    qvec = np.ascontiguousarray(qrow[vec_cols].to_numpy(dtype="float32").reshape(1, -1), dtype="float32")
    cand = embed_df.set_index("filename").reindex(df_neighbors["filename"].astype(str))
    cmat = np.ascontiguousarray(cand[vec_cols].to_numpy(dtype="float32", copy=False), dtype="float32")
    mask = np.any(np.isnan(cmat), axis=1)
    if mask.any():
        cmat_filled = cmat.copy()
        cmat_filled[np.isnan(cmat_filled)] = 0.0
        sims = _cosine_similarity(qvec, cmat_filled)
        sims[mask] = np.nan
    else:
        sims = _cosine_similarity(qvec, cmat)
    cos_dist = 1.0 - sims
    score = ((sims + 1.0) / 2.0) * 100.0
    df = df_neighbors.copy()
    df["cosine_similarity"] = sims
    df["cosine_distance"]   = cos_dist
    df["score_0_100"]       = np.round(score, 1)
    df["match_quality"]     = [ _match_label(s) if pd.notna(s) else "n/a" for s in sims ]
    return df

# ========== FAISS INDEX BUILD & SEARCH ==========
def _read_embed_matrix(embed_csv: str) -> Tuple[np.ndarray, List[str]]:
    df = load_embeddings_df(embed_csv)
    filenames = df["filename"].astype(str).tolist()
    vec = df.drop(columns=["filename"]).to_numpy(copy=False)
    vec = np.ascontiguousarray(vec, dtype="float32")
    return vec, filenames

def build_faiss_index(embed_csv: str, metric: str, progress_cb=None):
    if not _faiss_ok:
        raise RuntimeError("FAISS not installed. Run: pip install faiss-cpu")
    vec, filenames = _read_embed_matrix(embed_csv)
    vec = np.ascontiguousarray(vec, dtype="float32")
    dim = int(vec.shape[1])
    if "Cosine" in metric:
        faiss.normalize_L2(vec)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    n, B = vec.shape[0], 20000
    for start in range(0, n, B):
        end = min(start + B, n)
        batch = np.ascontiguousarray(vec[start:end], dtype="float32")
        index.add(batch)
        if progress_cb: progress_cb(end, n)
    return index, filenames, dim

def _faiss_search_2tuple(index, qvec: np.ndarray, k: int):
    res = index.search(np.ascontiguousarray(qvec.astype("float32"), dtype="float32"), int(k))
    if isinstance(res, tuple) and len(res) >= 2:
        D, I = res[0], res[1]
    elif hasattr(res, "distances") and hasattr(res, "labels"):
        D, I = res.distances, res.labels
    elif hasattr(res, "D") and hasattr(res, "I"):
        D, I = res.D, res.I
    else:
        raise RuntimeError(f"Unknown FAISS.search return type: {type(res)}")
    D = np.asarray(D); I = np.asarray(I)
    if D.ndim == 1: D = D[None, :]
    if I.ndim == 1: I = I[None, :]
    return D, I

def knn_search_filtered(index,
                        filenames: List[str],
                        query_filename: str,
                        k: int,
                        embed_csv: str,
                        allowed_filenames: Optional[Set[str]] = None,
                        metric: str = "Cosine (IP + normalization)") -> pd.DataFrame:
    emb = load_embeddings_df(embed_csv)
    _debug_show_cols("Embeddings DF (preview)", emb.head(1))
    _assert_has_filename(emb, "Embeddings CSV")

    qkey = _fname_key(query_filename)
    qrow = emb.loc[emb["filename"] == qkey]
    if qrow.empty:
        raise ValueError(f"Did not find '{query_filename}' (key='{qkey}') in embeddings.")
    if len(qrow) > 1:
        qrow = qrow.iloc[[0]]

    vec_cols = [c for c in emb.columns if c != "filename"]
    qvec = np.ascontiguousarray(qrow[vec_cols].to_numpy(dtype="float32").reshape(1, -1), dtype="float32")
    if "Cosine" in metric:
        nrm = np.linalg.norm(qvec, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1e-12
        qvec = np.ascontiguousarray(qvec / nrm, dtype="float32")

    k_search = min(max(int(k) * 400, 100), len(filenames))
    D, I = _faiss_search_2tuple(index, qvec, k_search)
    if st.session_state.get("debug_on", False):
        st.info(f"[DEBUG] FAISS.search shapes: D={D.shape}, I={I.shape}; k_search={k_search}")

    out = []
    D0, I0 = D[0], I[0]
    for pos in range(min(len(D0), len(I0))):
        idx = int(I0[pos])
        if not (0 <= idx < len(filenames)):
            continue
        fn_key = str(filenames[idx])
        if fn_key == qkey:
            continue
        if allowed_filenames is not None and fn_key not in allowed_filenames:
            continue
        dist_raw = float(D0[pos])
        distance = float(1.0 - dist_raw) if "Cosine" in metric else dist_raw
        out.append({"rank": len(out) + 1, "filename": fn_key, "distance": distance})
        if len(out) >= int(k):
            break

    return pd.DataFrame(out)

# ========== OVERLAY STORAGE (ANNOTATIONS) ==========
def _overlay_base_dir(output_dir: str) -> Path:
    base = Path(output_dir or ".") / "_overlay"
    base.mkdir(parents=True, exist_ok=True)
    return base

def overlay_session_dir(output_dir: str, session_name: str) -> Path:
    sdir = _overlay_base_dir(output_dir) / str(session_name)
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir

def overlay_current_path(sess_dir: Path) -> Path:
    return sess_dir / "overlay_current.csv"

def overlay_log_path(sess_dir: Path) -> Path:
    return sess_dir / "overlay_log.csv"

def _atomic_write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)

def overlay_load_current(sess_dir: Path) -> pd.DataFrame:
    p = overlay_current_path(sess_dir)
    if p.exists():
        try:
            df = pd.read_csv(p, low_memory=False)
            return _standardize_columns(df)
        except Exception:
            pass
    return pd.DataFrame(columns=["stable_id","filename","marker","value","timestamp"])

def overlay_log_append(sess_dir: Path, rows: List[Dict[str, object]]):
    lp = overlay_log_path(sess_dir)
    write_header = not lp.exists()
    with open(lp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","action","stable_id","filename","marker","value"])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({
                "timestamp": r.get("timestamp") or datetime.utcnow().isoformat(),
                "action": r.get("action","add"),
                "stable_id": r.get("stable_id",""),
                "filename": r.get("filename",""),
                "marker": r.get("marker",""),
                "value": r.get("value",""),
            })

def overlay_add_value(sess_dir: Path, stable_id: str, filename: str, marker: str, value: str):
    cur = overlay_load_current(sess_dir)
    cur = _standardize_columns(cur)
    row = {
        "stable_id": str(stable_id),
        "filename": _fname_key(filename),
        "marker": str(marker),
        "value": str(value),
        "timestamp": datetime.utcnow().isoformat()
    }
    if not ((cur.get("stable_id","") == row["stable_id"]) &
            (cur.get("marker","") == row["marker"]) &
            (cur.get("value","") == row["value"])).any():
        cur = pd.concat([cur, pd.DataFrame([row])], ignore_index=True)
        _atomic_write_csv(cur, overlay_current_path(sess_dir))
        overlay_log_append(sess_dir, [{"action":"add", **row}])

def overlay_remove_value(sess_dir: Path, stable_id: str, filename: str, marker: str, value: str):
    cur = overlay_load_current(sess_dir)
    cur = _standardize_columns(cur)
    mask = (cur["stable_id"].astype(str) == str(stable_id)) & \
           (cur["marker"].astype(str) == str(marker)) & \
           (cur["value"].astype(str) == str(value))
    if mask.any():
        row = cur.loc[mask].iloc[0].to_dict()
        cur = cur.loc[~mask].copy()
        _atomic_write_csv(cur, overlay_current_path(sess_dir))
        overlay_log_append(sess_dir, [{"action":"remove", **row}])

def overlay_values_map(sess_dir: Path, marker: str) -> Dict[str, Set[str]]:
    cur = overlay_load_current(sess_dir)
    cur = _standardize_columns(cur)
    if cur.empty:
        return {}
    cur = cur[cur.get("marker","") == marker]
    out: Dict[str, Set[str]] = {}
    for fn, vals in cur.groupby("filename")["value"]:
        out[str(fn)] = set(str(v) for v in vals if str(v).strip()!="")
    return out

# ========== STEP FUNCTIONS ==========
def validate_metadata(meta_csv: str) -> Tuple[bool, pd.DataFrame, str]:
    p = Path(meta_csv)
    if not p.exists():
        return False, pd.DataFrame(), "Metadata file does not exist."
    try:
        df = pd.read_csv(p, low_memory=False)
        df = _ensure_filename_column(df)
        msg = "OK"
        if "filename" not in df.columns:
            msg = ("Warning: No column usable as 'filename'. "
                   "Add 'filename' or one of: file_name/original_filename/full_path.")
            return True, df.head(20), msg
        return True, df.head(20), msg
    except Exception as e:
        return False, pd.DataFrame(), f"Could not read metadata: {e}"

def validate_embeddings(embed_csv: str) -> Tuple[bool, pd.DataFrame, str]:
    p = Path(embed_csv)
    if not p.exists():
        return False, pd.DataFrame(), "Embeddings file does not exist."
    try:
        df = pd.read_csv(p, nrows=50, low_memory=False)
        df = _ensure_filename_column(df)
        if "filename" not in df.columns:
            return False, df.head(10), "Embeddings are missing the 'filename' column."
        num_cols = [c for c in df.columns if c != "filename"]
        if not num_cols:
            return False, df.head(10), "No vector columns found."
        df[num_cols] = df[num_cols].astype("float32")
        return True, df.head(20), f"OK (preview shows {len(num_cols)} vector columns)."
    except Exception as e:
        return False, pd.DataFrame(), f"Could not read embeddings: {e}"

def validate_images_root(root_field: str) -> Tuple[bool, int, str]:
    roots = parse_roots(root_field)
    if not roots:
        return False, 0, "No valid image root folders found (check paths)."
    files = scan_images_multi(roots)
    if not files:
        return False, 0, f"No image files found across {len(roots)} root folder(s)."
    return True, len(files), f"Found {len(files):,} images across {len(roots)} root folder(s)."

def run_consistency_checks(meta_csv: str, embed_csv: str, images_root_field: str) -> str:
    lines = []
    try:
        meta = pd.read_csv(meta_csv, low_memory=False)
        meta = _ensure_filename_column(meta)
        meta_set = set(meta["filename"].astype(str)) if "filename" in meta.columns else set()
        n_meta = len(meta_set) if meta_set else -1
    except Exception:
        meta_set, n_meta = set(), -1
    lines.append(f"Metadata: {n_meta:,}" if n_meta >= 0 else "Metadata: unknown (read error)")

    try:
        emb = pd.read_csv(embed_csv, low_memory=False)
        emb = _ensure_filename_column(emb)
        emb_set = set(emb["filename"].astype(str)) if "filename" in emb.columns else set()
        n_emb = len(emb_set) if emb_set else -1
    except Exception:
        emb_set, n_emb = set(), -1
    lines.append(f"Embeddings: {n_emb:,}" if n_emb >= 0 else "Embeddings: unknown (read error)")

    try:
        roots = parse_roots(images_root_field)
        files = scan_images_multi(roots)
        file_set = {Path(p).name.lower() for p in files}
        n_files = len(file_set)
    except Exception:
        file_set, n_files = set(), -1
    lines.append(f"Filesystem images: {n_files:,}" if n_files >= 0 else "Filesystem: unknown (scan error)")

    if meta_set and file_set:
        miss_fs = sorted(list(meta_set - file_set))[:10]
        lines.append("In metadata but missing on disk (top 10): " + str(miss_fs) if miss_fs else "Metadata ≈ files OK.")
    if emb_set and file_set:
        miss_emb = sorted(list(emb_set - file_set))[:10]
        lines.append("In embeddings but missing on disk (top 10): " + str(miss_emb) if miss_emb else "Embeddings ≈ files OK.")
    if meta_set and emb_set:
        miss_me = sorted(list(meta_set - emb_set))[:10]
        miss_em = sorted(list(emb_set - meta_set))[:10]
        if miss_me:
            lines.append("In metadata but not in embeddings (top 10): " + str(miss_me))
        if miss_em:
            lines.append("In embeddings but not in metadata (top 10): " + str(miss_em))
    return "\n".join(lines)

# Add stable_id to index
def _make_stable_id(pp: Path, stt) -> str:
    # fairly stable across renames on same folder; adjust if you prefer hashing file bytes (slower)
    base = f"{pp.name.lower()}|{pp.parent.name.lower()}|{int(stt.st_size)}|{int(stt.st_mtime)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def build_filesystem_index(images_root_field: str, output_dir: Optional[str], index_name: str, progress_cb=None):
    roots = parse_roots(images_root_field)
    if not roots:
        return pd.DataFrame(), 0, None, pd.DataFrame()
    files = scan_images_multi(roots)
    total = len(files)
    hosted = is_hosted_env(output_dir or "")
    out_path: Optional[Path] = None
    csv_writer = None
    csv_file = None
    rows: List[List[str]] = []

    header = ["full_path", "filename", "parent", "size_bytes", "mtime", "stable_id"]

    if not hosted and output_dir:
        out_path = Path(output_dir) / index_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_file = open(out_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

    written = 0
    for i, p in enumerate(files, 1):
        pp = Path(p)
        try:
            stt = pp.stat()
            sid = _make_stable_id(pp, stt)
            row = [str(pp), pp.name, pp.parent.name, stt.st_size, int(stt.st_mtime), sid]
        except Exception:
            row = [str(pp), pp.name, pp.parent.name, -1, 0, hashlib.sha1(str(pp).encode("utf-8")).hexdigest()[:16]]
        if csv_writer:
            csv_writer.writerow(row)
        else:
            rows.append(row)
        written += 1
        if i % CHUNK == 0:
            if csv_file: csv_file.flush()
            if progress_cb: progress_cb(i, total)

    if csv_file:
        csv_file.flush(); csv_file.close()
    if progress_cb: progress_cb(total, total)

    if out_path is not None:
        try:
            head = pd.read_csv(out_path, nrows=20, low_memory=False)
        except Exception:
            head = pd.DataFrame(columns=header)
        return head, written, out_path, None
    else:
        df = pd.DataFrame(rows, columns=header)
        return df.head(20), written, None, df

def build_allowed_set(meta_df: Optional[pd.DataFrame]) -> Optional[Set[str]]:
    if meta_df is None:
        return None
    include_rules = st.session_state.get("filter_include", {}) or {}
    exclude_rules = st.session_state.get("filter_exclude", {}) or {}
    query_text   = (st.session_state.get("filter_query_text", "") or "").strip()
    if not include_rules and not exclude_rules and not query_text:
        return None
    df = _ensure_filename_column(meta_df.copy())
    if "filename" not in df.columns:
        st.info("Metadata has no 'filename' – filters cannot be applied.")
        return None
    for c, vals in include_rules.items():
        if c not in df.columns: continue
        if isinstance(vals, (list, tuple, set)):
            df = df[df[c].astype(str).isin([str(v) for v in vals])]
        else:
            df = df[df[c].astype(str) == str(vals)]
    for c, vals in exclude_rules.items():
        if c not in df.columns: continue
        if isinstance(vals, (list, tuple, set)):
            df = df[~df[c].astype(str).isin([str(v) for v in vals])]
        else:
            df = df[df[c].astype(str) != str(vals)]
    if query_text:
        try:
            df = df.query(query_text, engine="python")
        except Exception as e:
            st.warning(f"Ignoring query due to error: {e}")
    allowed = {
        _fname_key(x)
        for x in df["filename"].astype(str).tolist()
        if str(x).strip() != ""
    }
    return allowed

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("Settings")
    st.text_input("Metadata CSV", key="meta_path")
    st.text_input("Embeddings CSV", key="embed_path")
    st.text_area(
        "Image root folder(s)",
        key="images_root",
        help="Separate multiple roots with ; , | or newlines.",
        height=80
    )
    st.text_input("Output folder (for index)", key="output_dir")
    st.text_input("Index filename", key="index_name")
    st.checkbox(
        "Auto-load existing index if present",
        key="auto_load_index",
        value=st.session_state["auto_load_index"]
    )

    st.markdown("---")
    st.subheader("Index metric")
    st.session_state["index_metric"] = st.radio(
        "Choose metric for FAISS",
        options=["Cosine (IP + normalization)", "L2 (squared)"],
        index=0 if "Cosine" in st.session_state["index_metric"] else 1,
        help="Cosine is recommended for semantics. Cosine metrics are still shown in the results table."
    )

    st.markdown("---")
    st.checkbox("Show debug info (columns)", key="debug_on")

    c1, c2 = st.columns(2)
    if c1.button("💾 Save checkpoint"):
        save_checkpoint(); st.success("Checkpoint saved.")
    if c2.button("📥 Load checkpoint"):
        ok = load_checkpoint()
        st.success("Checkpoint loaded." if ok else "No valid checkpoint file found.")

# ========== AUTO-LOAD EXISTING INDEX ==========
if st.session_state.get("auto_load_index", False):
    if not st.session_state.get("filemap"):
        ok, fm, idx_path = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
        if ok:
            st.session_state["filemap"] = fm
            st.session_state["index_csv_path"] = idx_path
            st.info(f"Loaded existing index: {idx_path} ({len(fm):,} entries).")

# ========== STEPPER ==========
st.markdown("---")
st.subheader(f"Step {st.session_state['step']} of 9")

# --------------------------------
# Step 1 — METADATA
# --------------------------------
if st.session_state["step"] == 1:
    st.markdown("**Step 1: Choose/validate metadata (CSV)**")
    st.info("CSV must contain **filename** (basename). If missing, we try common alternates or derive from full_path.")
    if st.button("🔍 Validate metadata"):
        with st.status("Validating metadata…", expanded=True) as s:
            ok, head, msg = validate_metadata(st.session_state["meta_path"])
            st.session_state["meta_ok"] = ok
            st.session_state["meta_head"] = head
            st.write(msg)
            s.update(label="Metadata OK" if ok else "Metadata NOT OK",
                     state="complete" if ok else "error")
    st.dataframe(st.session_state["meta_head"], width='stretch')
    st.button("➡️ Next", disabled=not st.session_state["meta_ok"],
              on_click=lambda: st.session_state.update(step=2))

# --------------------------------
# Step 2 — EMBEDDINGS
# --------------------------------
elif st.session_state["step"] == 2:
    st.markdown("**Step 2: Choose/validate embeddings (CSV)**")
    if st.button("🔍 Validate embeddings"):
        with st.status("Validating embeddings…", expanded=True) as s:
            ok, head, msg = validate_embeddings(st.session_state["embed_path"])
            st.session_state["embed_ok"] = ok
            st.session_state["embed_head"] = head
            st.write(msg)
            s.update(label="Embeddings OK" if ok else "Embeddings NOT OK",
                     state="complete" if ok else "error")
    st.dataframe(st.session_state["embed_head"], width='stretch')
    cols = st.columns(2)
    cols[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=1))
    cols[1].button("➡️ Next", disabled=not st.session_state["embed_ok"],
                   on_click=lambda: st.session_state.update(step=3))

# --------------------------------
# Step 3 — IMAGES
# --------------------------------
elif st.session_state["step"] == 3:
    st.markdown("**Step 3: Choose/validate image root folder(s)**")
    if st.button("🔍 Check images"):
        with st.status("Scanning image folder(s)…", expanded=True) as s:
            ok, count, msg = validate_images_root(st.session_state["images_root"])
            st.session_state["images_ok"] = ok
            st.session_state["images_count"] = count
            st.write(msg)
            s.update(label="Images OK" if ok else "Images NOT OK",
                     state="complete" if ok else "error")
    st.metric("Images found", f"{st.session_state['images_count']:,}")
    cols = st.columns(2)
    cols[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=2))
    cols[1].button("➡️ Next", disabled=not st.session_state["images_ok"],
                   on_click=lambda: st.session_state.update(step=4))

# --------------------------------
# Step 4 — CONSISTENCY CHECK
# --------------------------------
elif st.session_state["step"] == 4:
    st.markdown("**Step 4: Consistency check (metadata vs. embeddings vs. files)**")
    if st.button("🧪 Run check"):
        with st.status("Running checks…", expanded=True) as s:
            report = run_consistency_checks(
                st.session_state["meta_path"],
                st.session_state["embed_path"],
                st.session_state["images_root"],
            )
            st.session_state["checks_report"] = report
            st.text(report)
            s.update(label="Checks complete", state="complete")
    st.code(st.session_state.get("checks_report", ""), language="text")
    cols = st.columns(2)
    cols[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=3))
    cols[1].button("➡️ Next", on_click=lambda: st.session_state.update(step=5))

# --------------------------------
# Step 5 — INDEX (filename→path, +stable_id)
# --------------------------------
elif st.session_state["step"] == 5:
    st.markdown("**Step 5: Index (key=basename.lower → full_path, + stable_id)**")
    c1, c2 = st.columns(2)
    load_btn = c1.button("📖 Load existing index")
    build_btn = c2.button("🏗️ Build index file")

    if load_btn:
        ok, fm, idx_path = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
        if ok:
            st.session_state["filemap"] = fm
            st.session_state["index_csv_path"] = idx_path
            st.success(f"Loaded index: {idx_path} ({len(fm):,}).")
            try:
                prev_df = load_index_df(idx_path).head(20)
                st.dataframe(prev_df, width='stretch')
            except Exception as e:
                st.warning(f"Could not read preview: {e}")
        else:
            st.warning("No existing index found (or could not be read).")

    if build_btn:
        bar = st.empty()
        def on_prog(done, total):
            pct = int(100 * done / max(1,total))
            bar.progress(pct, text=f"Indexing files: {done}/{total}")
        with st.status("Building index…", expanded=True) as s:
            preview, written, out_path, df_full = build_filesystem_index(
                st.session_state["images_root"],
                st.session_state["output_dir"],
                st.session_state["index_name"],
                progress_cb=on_prog
            )
            st.dataframe(preview, width='stretch')
            if out_path is not None:
                st.success(f"Wrote ~{written:,} rows to: {out_path}")
                st.session_state["index_csv_path"] = str(out_path)
                try:
                    idx_df = load_index_df(str(out_path))
                    st.session_state["filemap"] = build_filemap_from_df(idx_df)
                    st.success(f"Key→full_path is ready ({len(st.session_state['filemap']):,}).")
                except Exception as e:
                    st.warning(f"Could not build filemap from index: {e}")
            else:
                st.success(f"Generated ~{written:,} rows (hosted).")
                st.download_button(
                    "⬇️ Download index.csv",
                    data=df_full.to_csv(index=False).encode("utf-8"),
                    file_name=st.session_state["index_name"],
                    mime="text/csv"
                )
                try:
                    st.session_state["filemap"] = build_filemap_from_df(df_full.rename(columns=str.lower))
                    st.success(f"Key→full_path is ready ({len(st.session_state['filemap']):,}).")
                except Exception as e:
                    st.warning(f"Could not build filemap from in-memory DF: {e}")
            s.update(label="Index ready", state="complete")

    st.markdown("---")
    cols = st.columns(2)
    cols[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=4))
    cols[1].button("➡️ Next", on_click=lambda: st.session_state.update(step=6))

# --------------------------------
# Step 6 — PICK IMAGE + FAISS + FILTERS + SEARCH
# --------------------------------
elif st.session_state["step"] == 6:
    st.markdown("**Step 6: Pick image → set filters → find K nearest**")
    if not _faiss_ok:
        st.error("FAISS not installed. In a terminal, run: pip install faiss-cpu")
    else:
        if st.button("🏗️ Build FAISS index"):
            metric = st.session_state["index_metric"]
            bar = st.empty()
            def on_prog(done, total):
                pct = int(100 * done / max(1,total))
                bar.progress(pct, text=f"Adding vectors ({metric}): {done}/{total}")
            with st.status(f"Building FAISS ({metric})…", expanded=True) as s:
                try:
                    index, filenames, dim = build_faiss_index(
                        st.session_state["embed_path"],
                        metric=metric,
                        progress_cb=on_prog
                    )
                except Exception as e:
                    st.session_state["faiss_ready"] = False
                    s.update(label=f"Error: {e}", state="error")
                else:
                    st.session_state["faiss_ready"] = True
                    st.session_state["faiss_dim"] = dim
                    st.session_state["embed_filenames"] = filenames
                    st.session_state["_faiss_index"] = index
                    s.update(label=f"Index ready ({len(filenames):,} vectors, dim={dim}, metric={metric}).", state="complete")

    st.markdown("**Choose query image**")
    c2, c3 = st.columns(2)

    fm = st.session_state.get("filemap", {})
    uploaded = c2.file_uploader(
        "Upload image",
        type=[e.strip(".") for e in SUPPORTED_IMG],
        help="If filename exists in index, we use the real path from index."
    )
    if uploaded is not None:
        try:
            orig_name = Path(uploaded.name).name
            key = _fname_key(orig_name)
            if fm and key in fm and Path(fm[key]).exists():
                st.session_state["query_image_path"] = fm[key]
                st.success(f"Matched existing file in index: {orig_name}")
            else:
                tmp_dir = Path(st.session_state.get("output_dir") or ".")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / orig_name
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state["query_image_path"] = str(tmp_path)
                st.info(f"Saved temporarily as: {orig_name} (matching by basename)")
        except Exception as e:
            st.error(f"Could not handle upload: {e}")

    if not fm:
        ok, fm2, idx_path2 = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
        if ok:
            st.session_state["filemap"] = fm2
            st.session_state["index_csv_path"] = idx_path2
            fm = fm2
            st.info(f"Loaded index: {idx_path2}")
    if fm:
        all_keys = sorted(list(fm.keys()))
        filt = c3.text_input("Filter filenames (substring)", "")
        matches = [k for k in all_keys if filt.lower() in k.lower()] if filt.strip() else all_keys[:1000]
        sel = c3.selectbox("…or pick from index (filename)", options=["(none)"] + matches, index=0)
        if sel != "(none)" and c3.button("Use selected filename"):
            chosen_path = fm.get(sel)
            if chosen_path and Path(chosen_path).exists():
                st.session_state["query_image_path"] = chosen_path
                st.success(f"Selected (index): {Path(chosen_path).name}")
            else:
                st.error("Could not map key → path from index.")
    else:
        c3.info("No index loaded yet (build/load in Step 5).")

    qpath = st.session_state.get("query_image_path", "")
    st.write("Selected path:")
    st.code(qpath or "(none)", language="text")
    if qpath and Path(qpath).exists():
        display_image(qpath, width=360)

    st.markdown("**Filters (controls which candidates are allowed into KNN)**")
    meta_df: Optional[pd.DataFrame] = None
    if st.session_state.get("meta_ok") and Path(st.session_state["meta_path"]).exists():
        try:
            meta_df = pd.read_csv(st.session_state["meta_path"], low_memory=False)
            meta_df = _ensure_filename_column(meta_df)
            _assert_has_filename(meta_df, "Metadata CSV")
        except Exception as e:
            st.warning(f"Could not robustly read metadata: {e}")
            meta_df = None

    _debug_show_cols("Metadata DF (preview)", meta_df.head(1) if meta_df is not None else None)

    result_meta_cols: List[str] = []
    if meta_df is not None:
        meta_cols_options = [c for c in meta_df.columns if c != "filename"]
        result_meta_cols = st.multiselect(
            "Choose metadata columns to show in the results",
            options=meta_cols_options,
            default=st.session_state.get("result_meta_cols", []) or meta_cols_options[:5],
        )
        result_meta_cols = [c for c in result_meta_cols if c in meta_df.columns]
        st.session_state["result_meta_cols"] = result_meta_cols

    if meta_df is None:
        st.info("No metadata loaded (Step 1). Without filters, all embeddings are considered.")
        include_rules: Dict[str, List[str] | str] = {}
        exclude_rules: Dict[str, List[str] | str] = {}
    else:
        col = st.selectbox("Column to filter on", options=list(meta_df.columns), index=0)
        raw_vals = meta_df[col].dropna().astype(str)
        uniq_vals = sorted(raw_vals.unique().tolist())
        search = st.text_input("Search values (substring, case-insensitive)", "")
        show_vals = [v for v in uniq_vals if search.lower() in v.lower()] if search.strip() else uniq_vals[:400]
        cA, cB = st.columns(2)
        inc_sel = cA.multiselect("➕ Include values", options=show_vals, default=[])
        exc_sel = cB.multiselect("➖ Exclude values", options=show_vals, default=[])

        cC, cD, cE = st.columns(3)
        if cC.button("Save include for column"):
            st.session_state["filter_include"][col] = inc_sel if len(inc_sel) != 1 else inc_sel[0]
        if cD.button("Save exclude for column"):
            st.session_state["filter_exclude"][col] = exc_sel if len(exc_sel) != 1 else exc_sel[0]
        if cE.button("Clear rules for column"):
            st.session_state["filter_include"].pop(col, None)
            st.session_state["filter_exclude"].pop(col, None)

        query_text = st.text_input(
            "Optional pandas.query() expression (e.g., year >= 1900 and year < 1950)",
            value=st.session_state.get("filter_query_text","")
        )
        st.session_state["filter_query_text"] = query_text

        st.write("Active rules (include):", st.session_state["filter_include"] or "—")
        st.write("Active rules (exclude):", st.session_state["filter_exclude"] or "—")

    can_search = bool(st.session_state.get("faiss_ready"))
    if can_search:
        with st.form("search_form"):
            k = st.number_input("K (number of similar images to find)", min_value=1, max_value=200,
                                value=st.session_state["k_neighbors"])
            show_width = st.slider("Display width (px per image)", 120, 1024, 320, step=20)
            submitted = st.form_submit_button("🔎 Find K nearest (with filters)")
    else:
        st.info("Build the FAISS index first (button '🏗️ Build FAISS index').")
        submitted = False

    if submitted:
        st.session_state["k_neighbors"] = int(k)
        if not st.session_state.get("filemap"):
            ok, fm, idx_path = try_load_existing_index(st.session_state["output_dir"], st.session_state["index_name"])
            if ok:
                st.session_state["filemap"] = fm
                st.session_state["index_csv_path"] = idx_path
                st.info(f"Loaded existing index: {idx_path}")

        qpath = st.session_state.get("query_image_path", "")
        if not qpath:
            st.warning("Pick or upload an image first.")
        else:
            qname = Path(qpath).name
            allowed = build_allowed_set(meta_df) if meta_df is not None else None
            try:
                df_res = knn_search_filtered(
                    st.session_state["_faiss_index"],
                    st.session_state["embed_filenames"],
                    qname, int(k),
                    st.session_state["embed_path"],
                    allowed_filenames=allowed,
                    metric=st.session_state["index_metric"]
                )
                try:
                    embed_full = load_embeddings_df(st.session_state["embed_path"])
                    df_res = add_similarity_columns(df_res, embed_full, _fname_key(qname))
                except Exception as e:
                    st.warning(f"Could not compute cosine metrics: {e}")

                if meta_df is not None and st.session_state.get("result_meta_cols"):
                    try:
                        if "filename" in meta_df.columns:
                            meta_cols = ["filename"] + [c for c in st.session_state["result_meta_cols"] if c in meta_df.columns]
                            df_res = df_res.merge(meta_df[meta_cols], on="filename", how="left")
                        else:
                            st.info("Metadata is missing 'filename' – skipping join.")
                    except Exception as e:
                        st.warning(f"Could not join metadata: {e}")

                st.session_state["last_results_df"] = df_res

                if df_res.empty:
                    st.warning("No results match the filters. Try loosening filters or increasing K.")
                else:
                    st.dataframe(df_res, use_container_width=True)
                    st.download_button(
                        "⬇️ Download results (CSV)",
                        data=df_res.to_csv(index=False).encode("utf-8"),
                        file_name="retriever_wizard_results.csv",
                        mime="text/csv",
                    )

                    fm = st.session_state.get("filemap", {}) or {}
                    cols_grid = st.columns(5)
                    for i, r in enumerate(df_res.to_dict("records")):
                        fn = r.get("filename")
                        path = fm.get(fn)
                        with cols_grid[i % 5]:
                            if path and Path(path).exists():
                                cap = f"{i+1}. {fn}"
                                if "cosine_similarity" in df_res.columns and "match_quality" in df_res.columns:
                                    sim_val = r.get("cosine_similarity")
                                    match_label = r.get("match_quality")
                                    try:
                                        cap = f"{i+1}. {fn}\n{match_label}  (cos={float(sim_val):.3f})"
                                    except Exception:
                                        cap = f"{i+1}. {fn}\n{match_label}"
                                display_image(path, width=show_width, caption=cap)
                            else:
                                st.write(f"{i+1}. ❓ {fn}")
            except Exception as e:
                st.error(f"Search failed: {e.__class__.__name__}: {e}")
                st.code(traceback.format_exc())

    st.markdown("---")
    nav = st.columns(2)
    nav[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=5))
    nav[1].button("➡️ Next (Stacked view)", on_click=lambda: st.session_state.update(step=7),
                  disabled=st.session_state.get("last_results_df", pd.DataFrame()).empty)

# --------------------------------
# Step 7 — STACKED NEAREST (toggle: hide/show metadata; grid layout when hidden)
# --------------------------------
elif st.session_state["step"] == 7:
    st.markdown("**Step 7: Stacked view (neighbors + optional metadata)**")
    st.info("Toggle 'Hide metadata' to get a multi-column grid with larger images. When visible, chosen metadata is shown beside each image (incl. query).")

    df_res = st.session_state.get("last_results_df", pd.DataFrame()).copy()
    fm = st.session_state.get("filemap", {}) or {}
    qpath = st.session_state.get("query_image_path", "")

    sel_cols: List[str] = list(st.session_state.get("result_meta_cols", []))
    meta_keep = None
    if Path(st.session_state.get("meta_path", "")).exists():
        try:
            _meta = pd.read_csv(st.session_state["meta_path"], low_memory=False)
            _meta = _ensure_filename_column(_meta)
            if not sel_cols:
                meta_cols_options = [c for c in _meta.columns if c != "filename"]
                sel_cols = meta_cols_options[:8]
                st.session_state["result_meta_cols"] = sel_cols
            keep_cols = ["filename"] + [c for c in sel_cols if c in _meta.columns]
            meta_keep = _meta[keep_cols].copy()
        except Exception as e:
            st.warning(f"Could not load metadata for Step 7: {e}")
            meta_keep = None
    else:
        if not sel_cols: sel_cols = []

    if meta_keep is not None and not df_res.empty:
        missing = [c for c in sel_cols if c not in df_res.columns]
        if missing:
            try:
                df_res = df_res.merge(meta_keep, on="filename", how="left")
            except Exception as e:
                st.warning(f"Could not merge metadata into results: {e}")

    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    hide_meta = c0.checkbox("Hide metadata", value=False)
    show_query = c1.checkbox("Show query image at the top", value=True)
    show_top_n = c2.number_input("How many neighbors", 1, max(1, len(df_res)), min(30, len(df_res)) if len(df_res) else 1)
    show_width = c3.slider("Image width (px)", 160, 1600, 520, step=20)
    grid_cols = 3
    if hide_meta:
        grid_cols = st.slider("Columns (grid when metadata hidden)", 1, 8, 3)

    st.session_state.setdefault("step7_params", {})
    st.session_state["step7_params"].update(
        dict(top_n=int(show_top_n), width=int(show_width), show_query=bool(show_query), hide_meta=bool(hide_meta), grid_cols=int(grid_cols))
    )

    if "rank" in df_res.columns:
        df_sorted = df_res.sort_values("rank", ascending=True).head(int(show_top_n))
    elif "distance" in df_res.columns:
        df_sorted = df_res.sort_values("distance", ascending=True).head(int(show_top_n))
    else:
        df_sorted = df_res.head(int(show_top_n))

    def _render_meta(md: Dict[str, object], cols: List[str]):
        if not cols:
            st.caption("No metadata columns selected in Step 6."); return
        lines = []
        for c in cols:
            v = md.get(c, "—")
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = "—"
            if isinstance(v, (list, dict)):
                try: v = json.dumps(v, ensure_ascii=False)[:1000]
                except Exception: v = str(v)[:1000]
            else:
                v = str(v)[:1000]
            lines.append(f"- **{c}**: {v}")
        st.markdown("\n".join(lines))

    def _chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    total_to_render = len(df_sorted) + (1 if (show_query and qpath and Path(qpath).exists()) else 0)
    prog = st.progress(0, text=f"Rendering 0/{total_to_render}")
    rendered = 0

    if show_query and qpath and Path(qpath).exists():
        st.subheader("Query image")
        if hide_meta:
            display_image(qpath, width=show_width, caption=Path(qpath).name)
        else:
            colL, colR = st.columns([1, 2])
            with colL:
                display_image(qpath, width=show_width, caption=Path(qpath).name)
            with colR:
                if meta_keep is not None:
                    try:
                        qkey = _fname_key(Path(qpath).name)
                        qrow = meta_keep.loc[meta_keep["filename"] == qkey]
                        qmd = qrow.iloc[0].to_dict() if not qrow.empty else {}
                    except Exception:
                        qmd = {}
                    _render_meta(qmd, sel_cols)
                else:
                    st.caption("No metadata available.")
        rendered += 1
        prog.progress(int(rendered * 100 / max(1, total_to_render)), text=f"Rendering {rendered}/{total_to_render}")

    st.subheader("Nearest neighbors")
    recs = df_sorted.to_dict("records")

    if hide_meta:
        def _caption(i, r):
            fn = r.get("filename")
            cap = f"#{i} — {fn}"
            if "cosine_similarity" in df_sorted.columns and "match_quality" in df_sorted.columns:
                sim_val = r.get("cosine_similarity")
                match_label = r.get("match_quality")
                try: cap += f"\n{match_label} (cos={float(sim_val):.3f})"
                except Exception: cap += f"\n{match_label}"
            return cap

        idx = 1
        for row in _chunk(recs, grid_cols):
            cols = st.columns(len(row))
            for c, r in zip(cols, row):
                fn = r.get("filename")
                cap = _caption(idx, r)
                path = fm.get(fn)
                with c:
                    if path and Path(path).exists():
                        display_image(path, width=show_width, caption=cap)
                    else:
                        st.write(f"{cap}\n(path not found)")
                rendered += 1
                prog.progress(int(rendered * 100 / max(1, total_to_render)), text=f"Rendering {rendered}/{total_to_render}")
                idx += 1
    else:
        for i, r in enumerate(recs, start=1):
            fn = r.get("filename"); path = fm.get(fn)
            cap = f"#{i} — {fn}"
            if "cosine_similarity" in df_sorted.columns and "match_quality" in df_sorted.columns:
                sim_val = r.get("cosine_similarity"); match_label = r.get("match_quality")
                try: cap += f"\n{match_label} (cos={float(sim_val):.3f})"
                except Exception: cap += f"\n{match_label}"

            colL, colR = st.columns([1, 2])
            with colL:
                if path and Path(path).exists():
                    display_image(path, width=show_width, caption=cap)
                else:
                    st.write(f"{cap}\n(path not found)")
            with colR:
                md = {c: r.get(c, "—") for c in sel_cols}
                if meta_keep is not None and (not md or any(m == "—" for m in md.values())):
                    try:
                        mrow = meta_keep.loc[meta_keep["filename"] == str(fn)]
                        if not mrow.empty:
                            md_full = mrow.iloc[0].to_dict()
                            for c in sel_cols:
                                if md.get(c, "—") == "—":
                                    md[c] = md_full.get(c, "—")
                    except Exception:
                        pass
                _render_meta(md, sel_cols)

            rendered += 1
            prog.progress(int(rendered * 100 / max(1, total_to_render)), text=f"Rendering {rendered}/{total_to_render}")

    if not df_sorted.empty:
        try:
            export_cols = []
            if "rank" in df_sorted.columns: export_cols += ["rank"]
            export_cols += ["filename"]
            if "cosine_similarity" in df_sorted.columns: export_cols += ["cosine_similarity"]
            if "match_quality" in df_sorted.columns: export_cols += ["match_quality"]
            export_cols += [c for c in sel_cols if c in df_sorted.columns]
            csv_df = df_sorted[export_cols].copy()
            st.download_button(
                "⬇️ Download stacked metadata (CSV)",
                data=csv_df.to_csv(index=False).encode("utf-8"),
                file_name="stacked_nearest_with_metadata.csv",
                mime="text/csv",
            )
        except Exception:
            pass

    st.markdown("---")
    nav = st.columns(2)
    nav[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=6))
    nav[1].button("➡️ Next (Annotate)", on_click=lambda: st.session_state.update(step=8),
                  disabled=st.session_state.get("last_results_df", pd.DataFrame()).empty)

# --------------------------------
# Step 8 — ANNOTATE (overlay)
# --------------------------------
elif st.session_state["step"] == 8:
    st.markdown("**Step 8: Annotate (overlay, persistent)**")
    st.info("Annotate works with new metadata values in a separate overlay that doesn't overwrites the original metadata. Everything is saved in output_dir/_overlay/<session>/.")

    # Controls (namespaced keys)
    c0, c1, c2, c3 = st.columns([1,1,1,1])
    session = c0.text_input("Session name", value=st.session_state.get("overlay_session","iconography_2025q4"), key=_k_annot("session"))
    marker  = c1.text_input("Marker field", value=st.session_state.get("overlay_marker","ikonografi"), key=_k_annot("marker"))
    mode    = c2.selectbox("Mode", ["multi", "single"], index=0, key=_k_annot("mode"))
    new_val = c3.text_input("Value", value="", key=_k_annot("value"))

    # sync to global so Step 9 can read defaults
    st.session_state["overlay_session"] = session
    st.session_state["overlay_marker"]  = marker

    # Data we need
    df_res = st.session_state.get("last_results_df", pd.DataFrame()).copy()
    idx_df = pd.DataFrame()
    if st.session_state.get("index_csv_path"):
        try:
            idx_df = load_index_df(st.session_state["index_csv_path"])
        except Exception:
            idx_df = pd.DataFrame()
    idx_df = _standardize_columns(idx_df)

    # filename -> stable_id map
    if "stable_id" in idx_df.columns:
        sid_map = dict(zip(_normalize_filename_series(idx_df["filename"]), idx_df["stable_id"].astype(str)))
    else:
        sid_map = {}  # fallback to filename as sid

    fm = st.session_state.get("filemap", {}) or {}
    sess_dir = overlay_session_dir(st.session_state["output_dir"], session)
    current_map = overlay_values_map(sess_dir, marker)  # filename -> set(values)

    # UI for page rendering controls
    c4, c5, c6 = st.columns([1,1,1])
    top_n = c4.number_input("How many from results to show", 1, max(1, len(df_res)), min(50, len(df_res)) if len(df_res) else 1, key=_k_annot("topn"))
    img_w = c5.slider("Image width (px)", 160, 1200, 420, step=20, key=_k_annot("imgw"))
    cols  = c6.slider("Columns", 1, 6, 3, key=_k_annot("cols"))

    # Sort like step 7
    if "rank" in df_res.columns:
        df_sorted = df_res.sort_values("rank", ascending=True).head(int(top_n))
    elif "distance" in df_res.columns:
        df_sorted = df_res.sort_values("distance", ascending=True).head(int(top_n))
    else:
        df_sorted = df_res.head(int(top_n))

    # Helpers
    def _sid_for(fn: str) -> str:
        return sid_map.get(fn, fn)  # fallback: filename as sid

    # palette suggestions from current overlay
    palette = sorted({v for vals in current_map.values() for v in vals}) if current_map else []
    with st.expander("Existing values (palette)"):
        st.write(", ".join(palette) if palette else "—")

    # Batch area
    st.markdown("---")
    st.subheader("Batch tools")
    cB1, cB2, cB3 = st.columns([1,1,1])
    select_all = cB1.checkbox("Select all on page", value=False, key=_k_annot("select_all"))
    batch_val = cB2.text_input("Batch value", value=new_val, key=_k_annot("batch_val"))
    _ = cB3.empty()

    # Selection state for this page
    page_keys = [str(x) for x in df_sorted["filename"].astype(str).tolist()]
    sel_default = page_keys if select_all else []
    selected = st.multiselect("Selected filenames (this page)", options=page_keys, default=sel_default, key=_k_annot("selected"))

    cAct1, cAct2 = st.columns([1,1])
    if cAct1.button("➕ Add value to selected", disabled=(len(selected)==0 or not batch_val.strip()), key=_k_annot("batch_add")):
        with st.status("Adding values…", expanded=True) as s:
            prog = st.progress(0, text="0")
            for i, fn in enumerate(selected, 1):
                sid = _sid_for(fn)
                if mode == "single":
                    # remove existing values for this marker first
                    for v in list(current_map.get(fn, set())):
                        overlay_remove_value(sess_dir, sid, fn, marker, v)
                overlay_add_value(sess_dir, sid, fn, marker, batch_val.strip())
                current_map.setdefault(fn, set()).add(batch_val.strip())
                prog.progress(int(i*100/max(1,len(selected))), text=f"{i}/{len(selected)}")
            s.update(label=f"Added to {len(selected)} item(s).", state="complete")

    if cAct2.button("➖ Remove value from selected", disabled=(len(selected)==0 or not batch_val.strip()), key=_k_annot("batch_remove")):
        with st.status("Removing values…", expanded=True) as s:
            prog = st.progress(0, text="0")
            for i, fn in enumerate(selected, 1):
                sid = _sid_for(fn)
                overlay_remove_value(sess_dir, sid, fn, marker, batch_val.strip())
                if fn in current_map and batch_val.strip() in current_map[fn]:
                    current_map[fn].remove(batch_val.strip())
                prog.progress(int(i*100/max(1,len(selected))), text=f"{i}/{len(selected)}")
            s.update(label=f"Removed from {len(selected)} item(s).", state="complete")

    st.markdown("---")
    st.subheader("Per-image annotate")

    # Render grid with per-image controls
    def _chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    recs = df_sorted.to_dict("records")
    total = len(recs)
    progR = st.progress(0, text=f"0/{total}")
    count = 0

    for row in _chunk(recs, int(cols)):
        cols_row = st.columns(len(row))
        for c, r in zip(cols_row, row):
            fn = str(r.get("filename"))
            sid = _sid_for(fn)
            path = st.session_state.get("filemap", {}).get(fn)
            with c:
                if path and Path(path).exists():
                    cap = fn
                    if "cosine_similarity" in df_sorted.columns and "match_quality" in df_sorted.columns:
                        sim_val = r.get("cosine_similarity"); match_label = r.get("match_quality")
                        try: cap += f"\n{match_label} (cos={float(sim_val):.3f})"
                        except Exception: cap += f"\n{match_label}"
                    display_image(path, width=img_w, caption=cap)
                else:
                    st.write(f"{fn}\n(path not found)")

                cur_vals = sorted(list(current_map.get(fn, set())))
                st.caption("Current: " + ("; ".join(cur_vals) if cur_vals else "—"))

                cbtn1, cbtn2 = st.columns(2)
                if cbtn1.button("Mark", key=_k_annot(f"mark_{sid}")):
                    if new_val.strip():
                        if mode == "single":
                            for v in list(current_map.get(fn, set())):
                                overlay_remove_value(sess_dir, sid, fn, marker, v)
                            current_map[fn] = set()
                        overlay_add_value(sess_dir, sid, fn, marker, new_val.strip())
                        current_map.setdefault(fn, set()).add(new_val.strip())
                        st.success("Marked.")
                    else:
                        st.warning("Enter a value first.")

                if cbtn2.button("Unmark", key=_k_annot(f"unmark_{sid}")):
                    if new_val.strip():
                        overlay_remove_value(sess_dir, sid, fn, marker, new_val.strip())
                        if fn in current_map and new_val.strip() in current_map[fn]:
                            current_map[fn].remove(new_val.strip())
                        st.info("Removed.")
                    else:
                        st.warning("Enter a value to remove.")

            count += 1
            progR.progress(int(count*100/max(1,total)), text=f"{count}/{total}")

    st.markdown("---")
    nav = st.columns(3)
    nav[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=7))
    nav[1].button("Save checkpoint", on_click=lambda: (save_checkpoint(), st.success("Saved.")))
    nav[2].button("➡️ Next (Project/Export)", on_click=lambda: st.session_state.update(step=9))

# --------------------------------
# Step 9 — PROJECTION (UMAP/t-SNE) + EXPORT
# --------------------------------
elif st.session_state["step"] == 9:
    st.markdown("**Step 9: Projection (UMAP/t-SNE) and Export overlay → merged CSV**")
    st.info("Generate 2D projection; and export a merged metadata CSV (original + overlay).")

    # ======== PROJECTION ========
    try:
        has_plotly = True
        import plotly.graph_objects as go
    except Exception:
        has_plotly = False

    umap_available = False
    try:
        _ = UMAP
        umap_available = True
    except Exception:
        try:
            from umap import UMAP as _UMAP2
            UMAP = _UMAP2
            umap_available = True
        except Exception:
            umap_available = False

    tsne_available = False
    TSNE = None
    try:
        from sklearn.manifold import TSNE as _TSNE
        TSNE = _TSNE
        tsne_available = True
    except Exception:
        tsne_available = False

    ctop = st.columns(2)
    do_projection = ctop[0].checkbox("Show projection", value=True, key=_k_export("do_projection"))
    do_export = ctop[1].checkbox("Enable export overlay→merged CSV", value=True, key=_k_export("do_export"))

    if do_projection:
        if not (umap_available or tsne_available):
            st.error("Neither UMAP nor t-SNE available. Install with: pip install umap-learn scikit-learn")
        else:
            c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
            proj_options = []
            if umap_available: proj_options.append("UMAP")
            if tsne_available: proj_options.append("t-SNE")
            proj_method = c0.selectbox("Projection method", proj_options, index=0, key=_k_export("proj_method"))
            max_points = c1.number_input("Max points (sampled)", min_value=200, max_value=200000, value=10000, step=500, key=_k_export("max_points"))
            if proj_method == "UMAP":
                nn = c2.number_input("UMAP n_neighbors", min_value=5, max_value=200, value=30, step=5, key=_k_export("umap_nn"))
                md = c3.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key=_k_export("umap_md"))
                metric = st.selectbox("UMAP metric", ["cosine", "euclidean"], index=0, key=_k_export("umap_metric"))
            else:
                perplexity = c2.slider("t-SNE perplexity", 5.0, 100.0, 30.0, 1.0, key=_k_export("tsne_perp"))
                n_iter = c3.slider("t-SNE iterations", 250, 5000, 1000, 250, key=_k_export("tsne_iters"))
                col_tsne = st.columns(3)
                early_exag = col_tsne[0].slider("Early exaggeration", 4.0, 20.0, 12.0, 0.5, key=_k_export("tsne_ee"))
                lr_mode = col_tsne[1].selectbox("Learning rate", ["auto", "custom"], index=0, key=_k_export("tsne_lrmode"))
                if lr_mode == "custom":
                    learning_rate = col_tsne[2].slider("LR value", 10.0, 2000.0, 200.0, 10.0, key=_k_export("tsne_lrval"))
                else:
                    learning_rate = "auto"
                tsne_cap = st.number_input(
                    "t-SNE max points (cap)", min_value=1000, max_value=50000, value=12000, step=1000,
                    help="Safety cap; t-SNE is O(n²). Query/nearest are always kept.", key=_k_export("tsne_cap")
                )

            st.markdown("---")
            color_mode = st.radio(
                "Coloring",
                ["Clear categories (query/nearest/other)", "Gradient by similarity to query (cosine)"],
                index=1, key=_k_export("color_mode")
            )
            if color_mode.startswith("Gradient"):
                cA, cB = st.columns(2)
                pct_lo, pct_hi = cA.slider(
                    "Gradient focus (percentile window on similarity)",
                    0.0, 100.0, (80.0, 100.0), 0.5, key=_k_export("grad_window")
                )
                gamma = cB.slider("Contrast (gamma, >1 = focus high end)", 0.2, 5.0, 2.0, 0.1, key=_k_export("grad_gamma"))
            else:
                pct_lo, pct_hi, gamma = 0.0, 100.0, 1.0

            with st.status("Preparing data for projection…", expanded=False) as s:
                try:
                    emb = load_embeddings_df(st.session_state["embed_path"])
                    vec_cols = [c for c in emb.columns if c != "filename"]
                    meta_df = None
                    if st.session_state.get("meta_ok") and Path(st.session_state["meta_path"]).exists():
                        try:
                            meta_df = pd.read_csv(st.session_state["meta_path"], low_memory=False)
                            meta_df = _ensure_filename_column(meta_df); _assert_has_filename(meta_df, "Metadata CSV")
                        except Exception:
                            meta_df = None
                    allowed = build_allowed_set(meta_df) if meta_df is not None else None

                    if allowed is None:
                        pool = set(emb["filename"].astype(str))
                    else:
                        pool = set([f for f in emb["filename"].astype(str) if f in allowed])

                    qpath = st.session_state.get("query_image_path", "")
                    qkey = _fname_key(Path(qpath).name) if qpath else None
                    if qkey and qkey in set(emb["filename"].astype(str)):
                        pool.add(qkey)
                    df_res = st.session_state.get("last_results_df", pd.DataFrame())
                    nearest_set = set(df_res["filename"].astype(str)) if not df_res.empty else set()
                    pool |= nearest_set

                    if not pool:
                        raise RuntimeError("No filenames available for projection (check filters and embeddings).")

                    pool_list = list(pool)
                    random.shuffle(pool_list)
                    if len(pool_list) > int(max_points):
                        keep = set(pool_list[: int(max_points)])
                        if qkey: keep.add(qkey)
                        keep |= nearest_set
                    else:
                        keep = set(pool_list)

                    sub = emb[emb["filename"].astype(str).isin(keep)].copy()
                    if sub.empty:
                        raise RuntimeError("No rows matched after sampling; try increasing Max points.")

                    def _label_fn(x):
                        x = str(x)
                        if qkey and x == qkey: return "query"
                        if x in nearest_set: return "nearest"
                        return "other"
                    sub["label"] = sub["filename"].astype(str).map(_label_fn)

                    X_full = np.ascontiguousarray(sub[vec_cols].to_numpy(dtype="float32"), dtype="float32")

                    if qkey and (emb["filename"] == qkey).any():
                        qrow = emb.loc[emb["filename"] == qkey].iloc[[0]]
                        qvec = np.ascontiguousarray(qrow[vec_cols].to_numpy(dtype="float32"), dtype="float32")
                        sim_raw = _cosine_similarity(qvec, X_full)
                        sim01_full = ((sim_raw + 1.0) / 2.0).astype("float32")
                    else:
                        sim01_full = np.full(shape=(X_full.shape[0],), fill_value=np.nan, dtype="float32")

                    if proj_method == "t-SNE":
                        n = X_full.shape[0]; cap = int(tsne_cap)
                        if n > cap:
                            idx = np.arange(n)
                            labels_np = sub["label"].to_numpy()
                            keep_mask = (labels_np == "query") | (labels_np == "nearest")
                            idx_keep = idx[keep_mask]; idx_rest = idx[~keep_mask]
                            need = max(0, cap - len(idx_keep))
                            if need < len(idx_rest):
                                sel = np.random.RandomState(42).choice(idx_rest, size=need, replace=False)
                                idx_final = np.concatenate([idx_keep, sel])
                            else:
                                idx_final = idx
                            X = X_full[idx_final]; sub = sub.iloc[idx_final].reset_index(drop=True); sim01 = sim01_full[idx_final]
                        else:
                            X = X_full; sim01 = sim01_full
                    else:
                        X = X_full; sim01 = sim01_full

                    s.update(label=f"Prepared {len(sub):,} items. Computing {proj_method}…", state="running")

                    if proj_method == "UMAP":
                        umap_model = UMAP(n_neighbors=int(nn), min_dist=float(md), metric=metric, random_state=42)
                        coords = umap_model.fit_transform(X)
                    else:
                        from inspect import signature
                        tsne_kwargs = dict(
                            n_components=2,
                            perplexity=float(min(perplexity, max(5.0, (X.shape[0]-1)/3.0))),
                            early_exaggeration=float(early_exag),
                            init="pca",
                            metric="euclidean",
                            random_state=42,
                        )
                        params = signature(TSNE.__init__).parameters
                        if "n_iter" in params: tsne_kwargs["n_iter"] = int(n_iter)
                        elif "max_iter" in params: tsne_kwargs["max_iter"] = int(n_iter)
                        if "learning_rate" in params:
                            tsne_kwargs["learning_rate"] = learning_rate if isinstance(learning_rate, str) else float(learning_rate)
                        try:
                            tsne = TSNE(**tsne_kwargs)
                        except TypeError:
                            tsne_kwargs["learning_rate"] = 200.0
                            tsne_kwargs.pop("n_iter", None); tsne_kwargs.pop("max_iter", None)
                            tsne = TSNE(**tsne_kwargs)
                        coords = tsne.fit_transform(X).astype("float32")

                    plot_df = pd.DataFrame({
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "filename": sub["filename"].astype(str).values,
                        "label": sub["label"].values,
                        "sim_to_query": sim01,
                    })

                    st.session_state["umap_df"] = plot_df
                    st.session_state["umap_params"] = dict(
                        method=proj_method, max_points=int(max_points),
                        n_neighbors=int(nn) if proj_method=="UMAP" else None,
                        min_dist=float(md) if proj_method=="UMAP" else None,
                        metric=metric if proj_method=="UMAP" else None,
                        perplexity=float(tsne_kwargs.get("perplexity", np.nan)) if proj_method=="t-SNE" else None,
                        n_iter=int(tsne_kwargs.get("n_iter", tsne_kwargs.get("max_iter", np.nan))) if proj_method=="t-SNE" else None,
                        learning_rate=tsne_kwargs.get("learning_rate", None) if proj_method=="t-SNE" else None,
                        early_exaggeration=float(tsne_kwargs.get("early_exaggeration", np.nan)) if proj_method=="t-SNE" else None,
                    )
                    s.update(label=f"Computed {proj_method} for {len(plot_df):,} points.", state="complete")
                except Exception as e:
                    s.update(label=f"Error preparing data: {e}", state="error")
                    st.stop()

            plot_df = st.session_state.get("umap_df", pd.DataFrame())
            if plot_df.empty:
                st.warning("Nothing to plot.")
            else:
                st.markdown("**Projection**")
                plotted = False
                if has_plotly:
                    try:
                        fig = go.Figure()
                        if color_mode.startswith("Clear"):
                            ddq = plot_df[plot_df["label"] == "query"]
                            if not ddq.empty:
                                fig.add_trace(go.Scattergl(x=ddq["x"], y=ddq["y"], mode="markers", name="query",
                                                           text=ddq["filename"], hoverinfo="text",
                                                           marker=dict(color="green", size=12, line=dict(width=1, color="black"), symbol="diamond")))
                            ddn = plot_df[plot_df["label"] == "nearest"]
                            if not ddn.empty:
                                fig.add_trace(go.Scattergl(x=ddn["x"], y=ddn["y"], mode="markers", name="nearest",
                                                           text=ddn["filename"], hoverinfo="text", marker=dict(color="yellow", size=8)))
                            ddo = plot_df[plot_df["label"] == "other"]
                            if not ddo.empty:
                                fig.add_trace(go.Scattergl(x=ddo["x"], y=ddo["y"], mode="markers", name="other",
                                                           text=ddo["filename"], hoverinfo="text", marker=dict(color="red", size=6)))
                            fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Group")
                            st.plotly_chart(fig, use_container_width=True)
                            plotted = True
                        else:
                            base = plot_df[plot_df["label"] != "query"]["sim_to_query"].dropna().to_numpy(dtype="float32")
                            if base.size >= 2:
                                lo = float(np.percentile(base, pct_lo)); hi = float(np.percentile(base, pct_hi))
                                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: lo, hi = 0.0, 1.0
                            else:
                                lo, hi = 0.0, 1.0
                            sim_raw = plot_df["sim_to_query"].fillna((lo + hi) / 2.0).to_numpy(dtype="float32")
                            sim_norm = np.clip((sim_raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                            sim_focus = 1.0 - np.power(1.0 - sim_norm, float(gamma))
                            plot_df["sim_color"] = sim_focus

                            ddq = plot_df[plot_df["label"] == "query"]
                            if not ddq.empty:
                                fig.add_trace(go.Scattergl(
                                    x=ddq["x"], y=ddq["y"], mode="markers", name="query",
                                    text=ddq["filename"], hoverinfo="text",
                                    marker=dict(color="green", size=12, line=dict(width=1, color="black"), symbol="diamond"),
                                ))
                            fig.add_trace(go.Scattergl(
                                x=plot_df["x"], y=plot_df["y"], mode="markers", name="similarity",
                                text=plot_df["filename"], hoverinfo="text",
                                marker=dict(
                                    size=7, color=plot_df["sim_color"], cmin=0, cmax=1,
                                    colorscale=[[0.0, "darkred"], [0.5, "orange"], [1.0, "yellow"]],
                                    showscale=True,
                                    colorbar=dict(title=f"cosine norm\n{pct_lo:.0f}–{pct_hi:.0f} pctl\nγ={gamma:.1f}"),
                                ),
                            ))
                            fig.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                            plotted = True
                    except Exception as e:
                        st.warning(f"Plotly failed ({e.__class__.__name__}: {e}). Using Vega-Lite fallback…")

                if not plotted:
                    if color_mode.startswith("Clear"):
                        spec = {
                            "layer": [
                                {"mark": {"type": "point", "filled": True, "tooltip": True},
                                 "transform": [{"filter": "datum.label == 'other'"}],
                                 "encoding": {"x": {"field": "x", "type": "quantitative"},
                                              "y": {"field": "y", "type": "quantitative"},
                                              "color": {"value": "red"}, "size": {"value": 30},
                                              "tooltip": [{"field": "filename", "type": "nominal"}]}},
                                {"mark": {"type": "point", "filled": True, "tooltip": True},
                                 "transform": [{"filter": "datum.label == 'nearest'"}],
                                 "encoding": {"x": {"field": "x", "type": "quantitative"},
                                              "y": {"field": "y", "type": "quantitative"},
                                              "color": {"value": "yellow"}, "size": {"value": 40},
                                              "tooltip": [{"field": "filename", "type": "nominal"}]}},
                                {"mark": {"type": "point", "filled": True, "tooltip": True, "shape": "diamond"},
                                 "transform": [{"filter": "datum.label == 'query'"}],
                                 "encoding": {"x": {"field": "x", "type": "quantitative"},
                                              "y": {"field": "y", "type": "quantitative"},
                                              "color": {"value": "green"}, "size": {"value": 60},
                                              "tooltip": [{"field": "filename", "type": "nominal"}]}},
                            ],
                            "height": 720,
                        }
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)
                    else:
                        base = plot_df[plot_df["label"] != "query"]["sim_to_query"].dropna().to_numpy(dtype="float32")
                        if base.size >= 2:
                            lo = float(np.percentile(base, pct_lo)); hi = float(np.percentile(base, pct_hi))
                            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: lo, hi = 0.0, 1.0
                        else:
                            lo, hi = 0.0, 1.0
                        sim_raw = plot_df["sim_to_query"].fillna((lo + hi) / 2.0).to_numpy(dtype="float32")
                        sim_norm = np.clip((sim_raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                        sim_focus = 1.0 - np.power(1.0 - sim_norm, float(gamma))
                        plot_df["sim_color"] = sim_focus

                        spec = {
                            "layer": [
                                {"mark": {"type": "point", "tooltip": True},
                                 "encoding": {"x": {"field": "x", "type": "quantitative"},
                                              "y": {"field": "y", "type": "quantitative"},
                                              "color": {"field": "sim_color", "type": "quantitative",
                                                        "scale": {"domain": [0, 1], "range": ["darkred", "orange", "yellow"]},
                                                        "legend": {"title": f"cosine norm {pct_lo:.0f}–{pct_hi:.0f} pctl, γ={gamma:.1f}"}} ,
                                              "size": {"value": 36},
                                              "tooltip": [{"field": "filename", "type": "nominal"}]}},
                                {"transform": [{"filter": "datum.label == 'query'"}],
                                 "mark": {"type": "point", "filled": True, "shape": "diamond"},
                                 "encoding": {"x": {"field": "x", "type": "quantitative"},
                                              "y": {"field": "y", "type": "quantitative"},
                                              "color": {"value": "green"}, "size": {"value": 80}}},
                            ],
                            "height": 720,
                        }
                        st.vega_lite_chart(plot_df, spec, use_container_width=True)

                with st.expander("Data (first rows)"):
                    st.dataframe(plot_df.head(100), use_container_width=True)
                st.download_button(
                    "⬇️ Download projection (CSV)",
                    data=plot_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{st.session_state.get('umap_params',{}).get('method','projection').lower()}_projection.csv",
                    mime="text/csv",
                )

    # ======== EXPORT OVERLAY -> MERGED CSV ========
    if do_export:
        st.markdown("---")
        st.subheader("Export overlay → merged metadata.csv")
        session = st.text_input("Session to export", value=st.session_state.get("overlay_session","iconography_2025q4"), key=_k_export("session"))
        marker = st.text_input("Marker field to export", value=st.session_state.get("overlay_marker","ikonografi"), key=_k_export("marker"))
        style = st.radio("Export style", ["Wide: single 'marker_list' column", "Binary columns per value"], index=0, key=_k_export("style"))
        include_extras = st.checkbox("Include rows not in base metadata (append at bottom)", value=False, key=_k_export("extras"))
        btn = st.button("📦 Build merged CSV", key=_k_export("build"))

        if btn:
            with st.status("Building merged CSV…", expanded=True) as s:
                # Load base metadata
                try:
                    meta = pd.read_csv(st.session_state["meta_path"], low_memory=False)
                    meta = _ensure_filename_column(meta)
                except Exception as e:
                    s.update(label=f"Error reading base metadata: {e}", state="error")
                    st.stop()

                # Load index (for sid->filename mapping, optional)
                idx_df = pd.DataFrame()
                if st.session_state.get("index_csv_path"):
                    try:
                        idx_df = load_index_df(st.session_state["index_csv_path"])
                        idx_df = _standardize_columns(idx_df)
                    except Exception:
                        idx_df = pd.DataFrame()

                # Load overlay current
                sess_dir = overlay_session_dir(st.session_state["output_dir"], session)
                cur = overlay_load_current(sess_dir)
                cur = _standardize_columns(cur)
                cur = cur[cur.get("marker","") == marker]
                if cur.empty:
                    s.update(label=f"No overlay entries for session '{session}' and marker '{marker}'.", state="error")
                    st.stop()

                # Map sid->filename if possible
                if "stable_id" in idx_df.columns:
                    idx_small = idx_df[["stable_id","filename"]].copy()
                    idx_small["filename"] = _normalize_filename_series(idx_small["filename"])
                    cur = cur.merge(idx_small, on="stable_id", how="left", suffixes=("",""))
                    # if filename missing, keep cur.filename
                    cur["filename"] = cur["filename"].fillna(cur.get("filename_x", cur["filename"]))
                # else: assume cur.filename is already normalized

                # Build wide or binary
                grp = cur.groupby(_normalize_filename_series(cur["filename"]))["value"].agg(lambda v: sorted(set([str(x) for x in v if str(x).strip()!=""])))
                wide = grp.reset_index().rename(columns={"filename":"filename","value": f"{marker}_list"})
                wide[f"{marker}_list"] = wide[f"{marker}_list"].apply(lambda vs: ";".join(vs))

                if style.startswith("Binary"):
                    all_vals = sorted({v for vs in grp for v in vs})
                    bin_df = pd.DataFrame({"filename": grp.index})
                    for v in all_vals:
                        col = f"{marker}_{v}"
                        bin_df[col] = [int(v in set(vs)) for vs in grp]
                    merged = meta.merge(bin_df, on="filename", how="left")
                else:
                    merged = meta.merge(wide, on="filename", how="left")

                if include_extras:
                    meta_keys = set(meta["filename"].astype(str))
                    extra = merged[~merged["filename"].astype(str).isin(meta_keys)]
                    if not extra.empty:
                        st.info(f"Appending {len(extra)} extra rows not present in base metadata.")
                        merged = pd.concat([merged, extra], ignore_index=True)

                out_name = f"metadata_merged_{session}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button("⬇️ Download merged CSV",
                                   data=merged.to_csv(index=False).encode("utf-8"),
                                   file_name=out_name, mime="text/csv")
                s.update(label=f"Merged CSV ready ({len(merged):,} rows).", state="complete")

    st.markdown("---")
    nav = st.columns(2)
    nav[0].button("⬅️ Back", on_click=lambda: st.session_state.update(step=8))

    nav[1].button("🔁 Recompute", on_click=lambda: st.session_state.update(step=9))
