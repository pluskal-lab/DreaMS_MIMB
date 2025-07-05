import pathlib
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf

# Directories
BENCHMARK_ROOT_DIR = pathlib.Path(__file__).parent.absolute()
BENCHMARK_REPO_DIR = BENCHMARK_ROOT_DIR.parent
BENCHMARK_DATA_DIR = BENCHMARK_REPO_DIR / 'data'
BENCHMARK_ASSETS_DIR = BENCHMARK_REPO_DIR / 'assets'

# Chemistry: elements 1–118
CHEM_ELEMS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac",
    "Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh",
    "Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]


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