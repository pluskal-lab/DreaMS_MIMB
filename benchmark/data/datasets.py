import abc
import pandas as pd
import torch
from pathlib import Path
from matchms.importing import load_from_mgf
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from benchmark import utils
from paths import PROJECT_ROOT


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
        # Default to original MassSpecGym mgf if not provided
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


class ChlorineDetectionDataset(BenchmarkDataset):
    """
    Example: binary classification of chlorine presence.
    Reads 'LABEL' if in MGF, else infers from 'FORMULA'.
    """

    def load_data(self):
        if self.pth.suffix == ".mgf":
            self.spectra = list(load_from_mgf(str(self.pth)))
            self.metadata = pd.DataFrame([s.metadata for s in self.spectra])
        else:
            raise ValueError(f"Unsupported file type: {self.pth.suffix}")

    def compute_labels(self):
        # If already labeled in MGF
        if "LABEL" in self.metadata.columns:
            self.metadata["label"] = self.metadata["LABEL"].astype(float)
        # Else infer from FORMULA entry
        elif "FORMULA" in self.metadata.columns:
            self.metadata["label"] = (
                self.metadata["FORMULA"].astype(str)
                .str.contains("Cl")
                .astype(float)
            )
        else:
            raise KeyError("No 'LABEL' or 'FORMULA' column to derive chlorine label.")