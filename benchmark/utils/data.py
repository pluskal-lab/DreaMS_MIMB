import pathlib
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
import pandas as pd
from typing import Any

from benchmark.utils.chem import classical_tanimoto


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