import pathlib
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf

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