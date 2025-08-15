import pathlib
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
import pandas as pd
from benchmark.utils.chem import classical_tanimoto
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity

import dreams.utils.spectra as su
from dreams.definitions import SPECTRUM, PRECURSOR_MZ

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any



def _assign_scalar_attr(obj: dict, key: str, value) -> None:
    """
    Assign only simple scalars supported by GraphML. Skip None, NaN/Inf, lists/arrays, and empty/'nan'/'none' strings.
    Convert numpy scalars to Python scalars.
    """
    if value is None:
        return
    # skip containers & arrays
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return
    # convert numpy 0-d scalars
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    # drop non-finite floats
    if isinstance(value, float) and not np.isfinite(value):
        return
    # drop empty-ish strings
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none"):
        return
    # keep: str, bool, int, float
    if isinstance(value, (str, bool, int, float)):
        obj[key] = value

def build_query_knn_graph(embs_q: np.ndarray, k: int, thld: float) -> nx.Graph:
    """k-NN on query embeddings → cosine distance -> similarity (1 - d) → threshold → undirected graph."""
    A = kneighbors_graph(embs_q, k, mode="distance", metric="cosine", include_self=False)
    A = A.toarray().astype(float)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0.0:
                A[i, j] = 1.0 - A[i, j]
            if A[i, j] < thld:
                A[i, j] = 0.0
    G = nx.from_numpy_array(A)
    for i in G.nodes():
        G.nodes[i]["node_type"] = "query"
        G.nodes[i]["id"] = f"Q_{i}"
    return G


def annotate_edges_modcos_qq(G: nx.Graph, msdata_q, mz_tolerance: float = 0.05) -> None:
    """Rename weight->DreaMS_similarity and compute modified cosine for Q-Q edges."""
    cos_sim_pl = su.PeakListModifiedCosine(mz_tolerance=mz_tolerance)
    for u, v, d in G.edges(data=True):
        if "weight" in d:
            d["DreaMS_similarity"] = float(d["weight"])
            del d["weight"]
        d["edge_type"] = "Q-Q"
        try:
            mc = cos_sim_pl(
                spec1=msdata_q[SPECTRUM][u], prec_mz1=msdata_q[PRECURSOR_MZ][u],
                spec2=msdata_q[SPECTRUM][v], prec_mz2=msdata_q[PRECURSOR_MZ][v],
            )
            if isinstance(mc, (int, float)) and np.isfinite(float(mc)):
                d["modified_cosine_similarity"] = float(mc)
        except Exception:
            pass

def find_q2lib_neighbors(
    embs_q: np.ndarray,
    embs_lib: np.ndarray,
    sim_thld: float,
    topk_per_q: Optional[int] = 5,
) -> tuple[Dict[int, List[tuple[int, float]]], set[int]]:
    """For each query qi, return [(lib_idx, sim)] with sim ≥ sim_thld, sorted desc and truncated to topk_per_q."""
    sims = cosine_similarity(embs_q, embs_lib)  # (nQ, nL)
    q2lib: Dict[int, List[tuple[int, float]]] = {}
    lib_keep: set[int] = set()
    for qi in range(sims.shape[0]):
        hits = np.where(sims[qi] >= sim_thld)[0]
        if hits.size:
            hits = hits[np.argsort(sims[qi, hits])[::-1]]
            if topk_per_q is not None:
                hits = hits[:topk_per_q]
            q2lib[qi] = [(int(j), float(sims[qi, j])) for j in hits]
            lib_keep.update(hits.tolist())
    return q2lib, lib_keep

def augment_with_library_nodes(
    G: nx.Graph,
    q2lib: Dict[int, List[tuple[int, float]]],
    lib_keep: set[int],
    msdata_q,
    msdata_lib,
    safe_lib_cols: Optional[Set[str]] = None,  # kept for signature; None = no filtering
    mz_tolerance: float = 0.05,
) -> None:
    """Add only selected library nodes and Q-L edges with similarities."""
    offset = len(msdata_q)
    lib_idx_to_node: Dict[int, int] = {}

    # add library nodes
    for j in sorted(lib_keep):
        nid = offset + j
        G.add_node(nid)
        G.nodes[nid]["node_type"] = "library"
        G.nodes[nid]["id"] = f"L_{j}"
        row = msdata_lib.at(j, plot_spec=False, plot_mol=False)
        for k, v in row.items():
            if safe_lib_cols is not None and k not in safe_lib_cols:
                continue
            _assign_scalar_attr(G.nodes[nid], k, v)
        # label
        label = row.get("IDENTIFIER", None) or row.get("INCHIKEY", None) or f"L_{j}"
        _assign_scalar_attr(G.nodes[nid], "label", label)
        lib_idx_to_node[j] = nid

    # edges Q-L with sims + optional mod cosine
    cos_sim_pl = su.PeakListModifiedCosine(mz_tolerance=mz_tolerance)
    for qi, hits in q2lib.items():
        u = qi
        for (lj, sim) in hits:
            v = lib_idx_to_node[lj]
            if not G.has_edge(u, v):
                G.add_edge(u, v)
            d = G[u][v]
            d["edge_type"] = "Q-L"
            if d.get("DreaMS_similarity", 0.0) < sim:
                d["DreaMS_similarity"] = float(sim)
            try:
                mc = cos_sim_pl(
                    spec1=msdata_q[SPECTRUM][u],   prec_mz1=msdata_q[PRECURSOR_MZ][u],
                    spec2=msdata_lib[SPECTRUM][lj], prec_mz2=msdata_lib[PRECURSOR_MZ][lj],
                )
                if isinstance(mc, (int, float)) and np.isfinite(float(mc)):
                    d["modified_cosine_similarity"] = float(mc)
            except Exception:
                pass

def tag_top1_assignment(
    G,
    embs_q,
    embs_lib,
    msdata_lib,
    sim_thld: float,
    overwrite_smiles: bool = False,
) -> None:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    sims = cosine_similarity(embs_q, embs_lib)
    top1_idx = sims.argmax(axis=1)
    top1_sim = sims[np.arange(sims.shape[0]), top1_idx]

    # columns() works with MSData; if it's a property in your version, change to msdata_lib.columns
    lib_cols = set(msdata_lib.columns())
    lib_has_ident = "IDENTIFIER" in lib_cols
    lib_has_ikey  = "INCHIKEY"   in lib_cols
    lib_has_smi   = "smiles"     in lib_cols

    for qi in range(sims.shape[0]):
        sim = float(top1_sim[qi])
        lj  = int(top1_idx[qi])

        if sim >= sim_thld:
            # identifier
            ident = None
            if lib_has_ident:
                try:
                    ident = msdata_lib.get_values("IDENTIFIER", lj)
                except Exception:
                    ident = None
            if not ident and lib_has_ikey:
                try:
                    ident = msdata_lib.get_values("INCHIKEY", lj)
                except Exception:
                    ident = None
            if not isinstance(ident, str) or ident.strip().lower() in ("", "nan", "none"):
                ident = f"L_{lj}"

            G.nodes[qi]["has_lib_hit"] = "hit"
            G.nodes[qi]["best_lib_similarity"] = sim
            G.nodes[qi]["best_lib_identifier"] = ident

            if lib_has_smi:
                try:
                    lib_smiles = msdata_lib.get_values("smiles", lj)
                except Exception:
                    lib_smiles = None

                if isinstance(lib_smiles, str) and lib_smiles.strip().lower() not in ("", "nan", "none"):
                    existing = G.nodes[qi].get("smiles", None)
                    exists = isinstance(existing, str) and existing.strip().lower() not in ("", "nan", "none")
                    if overwrite_smiles or not exists:
                        G.nodes[qi]["smiles"] = lib_smiles
        else:
            G.nodes[qi]["has_lib_hit"] = "no_hit"

def export_graphs(G: nx.Graph, out_base):
    """Write GraphML directly. Attributes must already be scalar-safe."""
    from pathlib import Path
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    graphml_path = out_base.with_suffix(".graphml")
    nx.write_graphml(G, graphml_path)

    print(f"Saved GraphML → {graphml_path}")
    return graphml_path

def to_clean_str(x: Any) -> str | None:
    """Bytes→str, strip; None for common placeholders."""
    missing_strings = {"none"}
    if x is None:
        return None
    if isinstance(x, bytes):
        try:
            x = x.decode()
        except Exception:
            x = x.decode("latin1", errors="ignore")
    s = str(x).strip()
    if s.lower() in missing_strings:
        return None
    return s

def inchikey_first_block(x: Any) -> str | None:
    """Uppercase INCHIKEY and return the block before the first '-'. Require ≥14 chars."""
    s = to_clean_str(x)
    if s is None:
        return None
    blk = s.upper().split("-")[0]
    return blk if len(blk) >= 14 else None

def build_rank1_annotations(df_matches: pd.DataFrame, msdata_q, msdata_lib) -> pd.DataFrame:
    """
    From df_matches (with columns: query_index, library_index, rank, DreaMS_similarity, Modified_cosine_similarity, library_IDENTIFIER),
    build a dataframe for rank==1 pairs with cleaned SMILES/INCHIKEYs, an annotation match flag, and classical tanimoto.
    """
    df1 = df_matches[df_matches["rank"] == 1].copy()

    # Pull raw annotations from MSData
    df1["q_SMILES_raw"]   = df1["query_index"].apply(lambda i: msdata_q.get_values("smiles", int(i)))
    df1["l_SMILES_raw"]   = df1["library_index"].apply(lambda j: msdata_lib.get_values("smiles", int(j)))
    df1["q_INCHIKEY_raw"] = df1["query_index"].apply(lambda i: msdata_q.get_values("INCHIKEY", int(i)))
    df1["l_INCHIKEY_raw"] = df1["library_index"].apply(lambda j: msdata_lib.get_values("INCHIKEY", int(j)))

    # Clean + normalize
    df1["q_SMILES"] = df1["q_SMILES_raw"].map(to_clean_str)
    df1["l_SMILES"] = df1["l_SMILES_raw"].map(to_clean_str)
    df1["q_IK_block"] = df1["q_INCHIKEY_raw"].map(inchikey_first_block)
    df1["l_IK_block"] = df1["l_INCHIKEY_raw"].map(inchikey_first_block)

    # Keep only pairs with both INCHIKEYs
    annot = df1.dropna(subset=["q_IK_block", "l_IK_block"]).copy()

    # Annotation agreement flag
    annot["annotation_match"] = (annot["q_IK_block"] == annot["l_IK_block"])

    # Classical Tanimoto (Morgan r=2) — name it as requested
    annot["Tanimoto_similarity"] = annot.apply(
        lambda r: classical_tanimoto(r["q_SMILES"], r["l_SMILES"]),
        axis=1
    )

    return annot

def annotate_mgf_with_label(
        input_mgf: pathlib.Path,
        output_mgf: pathlib.Path,
        label_fn
):
    """
    Reads spectra from `input_mgf`, computes label_fn(metadata) for each,
    writes new MGF with a "LABEL=<float>" entry in each block.

    Uses `Spectrum.set()` to properly update metadata without altering the spectra.
    """
    # Load all spectra
    specs = list(load_from_mgf(str(input_mgf)))

    # Annotate each spectrum
    for spec in specs:
        lbl = float(label_fn(spec.metadata))
        spec.set("LABEL", str(lbl))

    # Save back out
    save_as_mgf(specs, str(output_mgf))