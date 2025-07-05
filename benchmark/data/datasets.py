import abc
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from typing import Callable, Optional

from paths import PROJECT_ROOT
from benchmark.utils.plots import init_plotting, get_palette

from dreams.utils.spectra import plot_spectrum as su_plot
from rdkit.Chem import MolFromSmiles
from IPython.display import display

class BenchmarkDataset(Dataset, abc.ABC):
    """
    Abstract base for any MGF/TSV‐based benchmark.
    Subclasses must implement `load_data()` and `compute_labels()`.
    """
    def __init__(
        self,
        pth: Path = None,
        spec_transform=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.pth = pth or (PROJECT_ROOT / "data" / "massspecgym" / "MassSpecGym.mgf")
        self.spec_transform = spec_transform
        self.dtype = dtype
        self.load_data()
        if "label" not in self.metadata.columns:
            self.compute_labels()

    @abc.abstractmethod
    def load_data(self):
        """Load self.spectra (list of matchms.Spectrum) and self.metadata (pd.DataFrame)."""

    @abc.abstractmethod
    def compute_labels(self):
        """Compute self.metadata['label'] (float) for each spectrum."""

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx: int) -> dict:
        spec = self.spectra[idx]
        x = self.spec_transform(spec) if self.spec_transform else spec
        lbl = float(self.metadata.iloc[idx]["label"])
        item = {
            "spec": torch.as_tensor(x, dtype=self.dtype) if not isinstance(x, dict) else x,
            "label": torch.tensor(lbl, dtype=self.dtype),
            "identifier": str(self.metadata.iloc[idx].get("identifier", idx)),
        }
        return item

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)


class BinaryDetectionDataset(Dataset):
    """
    General binary classification dataset for MGF data.

    - If your MGF already has a 'LABEL' metadata entry, it will be used directly.
    - Otherwise, call `annotate_mgf` to generate a labeled file before instantiation.
    """
    def __init__(self, pth: Path, spec_transform=None, dtype=None):
        self.pth = pth
        self.spec_transform = spec_transform
        self.dtype = dtype
        self.load_data()
        self.compute_labels()

    @staticmethod
    def annotate_mgf(input_pth: Path,
                     output_pth: Path,
                     label_fn) -> None:
        """
        Annotate an unlabeled MGF with 'LABEL' metadata and save to `output_pth`.

        Prints only the overall numeric distribution.
        """
        if output_pth.exists():
            print(f"{output_pth} already exists; skipping annotation.")
            return

        specs = list(load_from_mgf(str(input_pth)))
        if all(('LABEL' in s.metadata or 'label' in s.metadata) for s in specs):
            print(f"{input_pth} already contains LABEL metadata; skipping annotation.")
            return

        meta = [dict(s.metadata) for s in specs]
        df = pd.DataFrame(meta)
        df['LABEL'] = df.apply(lambda row: str(int(label_fn(row))), axis=1)
        for spec, lbl in zip(specs, df['LABEL']):
            spec.set('LABEL', lbl)

        save_as_mgf(specs, str(output_pth))
        counts = df['LABEL'].value_counts().sort_index().astype(int)
        total = counts.sum()
        print("Overall label distribution:")
        for lbl, cnt in counts.items():
            print(f"  {lbl}: {cnt} ({cnt/total*100:.1f}%)")

    @staticmethod
    def plot_fold_distribution(mgf_pth: Path) -> None:
        """
        Reads LABEL and fold columns (any case) from an annotated MGF and
        plots stacked bar charts of counts and percentages by fold.
        """

        # Load metadata
        specs = list(load_from_mgf(str(mgf_pth)))
        df = pd.DataFrame([s.metadata for s in specs])

        # Case‐insensitive lookup of the two required columns
        cols_lower = {c.lower(): c for c in df.columns}
        if 'label' not in cols_lower or 'fold' not in cols_lower:
            raise KeyError("Need both 'label' and 'fold' (any case) in metadata.")
        label_col = cols_lower['label']
        fold_col = cols_lower['fold']

        # Build count table
        fc = df.groupby([fold_col, label_col]).size().unstack(fill_value=0)

        # Prepare plotting style
        init_plotting(figsize=(6, 4), font_scale=1.0, style='whitegrid', cmap='nature')
        palette = get_palette('nature')

        # Plot counts
        fig, ax = plt.subplots()
        fc.plot(kind='bar', stacked=True, ax=ax,
                color=[palette[i] for i in (0, 2)])
        ax.set_title('Label counts by fold')
        ax.set_xlabel(fold_col.capitalize())
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)
        plt.show()

        # Plot percentages
        pct = fc.div(fc.sum(axis=1), axis=0) * 100
        fig2, ax2 = plt.subplots()
        pct.plot(kind='bar', stacked=True, ax=ax2,
                 color=[palette[i] for i in (0, 2)])
        ax2.set_title('Label % by fold')
        ax2.set_xlabel(fold_col.capitalize())
        ax2.set_ylabel('Percent')
        ax2.tick_params(axis='x', rotation=0)
        plt.show()

    @staticmethod
    def visualize_examples(mgf_pth: Path, smiles_key: str = 'library_SMILES') -> None:
        """
        Displays one negative and one positive example:
          - Plots the spectrum (via su_plot) using mz/int arrays.
          - Renders the molecule (RDKit).
          - Prints the full metadata dict.
        Case‐insensitive on the 'label' column.
        """
        specs = list(load_from_mgf(str(mgf_pth)))
        df = pd.DataFrame([s.metadata for s in specs])

        # Map lowercase → actual column names
        cols_lower = {c.lower(): c for c in df.columns}
        if 'label' not in cols_lower:
            raise KeyError("No 'label' (any case) in metadata; run annotate_mgf first.")
        label_col = cols_lower['label']

        for lbl in ['0', '1']:
            subset = df[df[label_col] == lbl]
            if subset.empty:
                continue

            idx = subset.index[0]
            row = subset.iloc[0].to_dict()
            print(f"\n=== Example label {lbl} ===")

            # Build a (2, n_peaks) array for su_plot
            spec_obj = specs[idx]
            mzs = spec_obj.peaks.mz
            ints = spec_obj.peaks.intensities
            spec_array = np.vstack([mzs, ints])

            # Plot just the query spectrum
            prec = row.get('precursor_mz') or row.get('PRECURSOR_MZ')
            su_plot(spec=spec_array, prec_mz=prec)

            # Show molecule
            if smiles_key in row:
                display(MolFromSmiles(row[smiles_key]))

            # Print metadata
            print(row)

    def load_data(self):
        if self.pth.suffix.lower() != ".mgf":
            raise ValueError(f"Unsupported file type: {self.pth.suffix}")
        from matchms.importing import load_from_mgf
        self.spectra = list(load_from_mgf(str(self.pth)))
        self.metadata = pd.DataFrame([s.metadata for s in self.spectra])

    def compute_labels(self):
        if 'LABEL' not in self.metadata.columns:
            raise KeyError("No 'LABEL' metadata; please run `annotate_mgf` first.")
        self.metadata['label'] = self.metadata['LABEL'].astype(float)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spec = self.spectra[idx]
        x = self.spec_transform(spec) if self.spec_transform else spec
        lbl = float(self.metadata.iloc[idx]['label'])
        return {
            'spec': x,
            'label': lbl,
            'identifier': str(self.metadata.iloc[idx].get('identifier', idx))
        }

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)


# import abc
# import pandas as pd
# import torch
# from pathlib import Path
# from matchms.importing import load_from_mgf
# from torch.utils.data import Dataset
# from torch.utils.data.dataloader import default_collate
# from benchmark import utils
# from paths import PROJECT_ROOT
#
#
# class BenchmarkDataset(Dataset, abc.ABC):
#     """
#     Abstract base for any MGF/TSV‐based benchmark.
#     Subclasses must implement `load_data()` and `compute_labels()`.
#     """
#
#     def __init__(
#         self,
#         pth: Path = None,
#         spec_transform=None,
#         dtype: torch.dtype = torch.float32,
#     ):
#         # Default to original MassSpecGym mgf if not provided
#         self.pth = pth or (PROJECT_ROOT / "data" / "massspecgym" / "MassSpecGym.mgf")
#         self.spec_transform = spec_transform
#         self.dtype = dtype
#         self.load_data()
#         if "label" not in self.metadata.columns:
#             self.compute_labels()
#
#     @abc.abstractmethod
#     def load_data(self):
#         """Load self.spectra (list of matchms.Spectrum) and self.metadata (pd.DataFrame)."""
#
#     @abc.abstractmethod
#     def compute_labels(self):
#         """Compute self.metadata['label'] (float) for each spectrum."""
#
#     def __len__(self):
#         return len(self.spectra)
#
#     def __getitem__(self, idx: int) -> dict:
#         spec = self.spectra[idx]
#         x = self.spec_transform(spec) if self.spec_transform else spec
#         lbl = float(self.metadata.iloc[idx]["label"])
#         item = {
#             "spec": torch.as_tensor(x, dtype=self.dtype) if not isinstance(x, dict) else x,
#             "label": torch.tensor(lbl, dtype=self.dtype),
#             "identifier": str(self.metadata.iloc[idx].get("identifier", idx)),
#         }
#         return item
#
#     @staticmethod
#     def collate_fn(batch):
#         return default_collate(batch)
#
#
# class ChlorineDetectionDataset(BenchmarkDataset):
#     """
#     Example: binary classification of chlorine presence.
#     Reads 'LABEL' if in MGF, else infers from 'FORMULA'.
#     """
#
#     def load_data(self):
#         if self.pth.suffix == ".mgf":
#             self.spectra = list(load_from_mgf(str(self.pth)))
#             self.metadata = pd.DataFrame([s.metadata for s in self.spectra])
#         else:
#             raise ValueError(f"Unsupported file type: {self.pth.suffix}")
#
#     def compute_labels(self):
#         # If already labeled in MGF
#         if "LABEL" in self.metadata.columns:
#             self.metadata["label"] = self.metadata["LABEL"].astype(float)
#         # Else infer from FORMULA entry
#         elif "FORMULA" in self.metadata.columns:
#             self.metadata["label"] = (
#                 self.metadata["FORMULA"].astype(str)
#                 .str.contains("Cl")
#                 .astype(float)
#             )
#         else:
#             raise KeyError("No 'LABEL' or 'FORMULA' column to derive chlorine label.")