"""
Plot4PV: Solar Data Visualization Tool
--------------------------------------

This Streamlit dashboard allows users to visualise photovoltaic
performance data—including extracted parameters, IPCE curves and JV
characteristics—using interactive plots.  It combines clean UI design
with powerful customisation options such as mismatch correction,
variation colouring, smoothing, curve normalisation and exportable
graphics.

Select a plot type in the sidebar (Box Plot, IPCE Curve or JV Curve) to
reveal the relevant inputs.  The box plot mode supports grouped and
individual variation views of Jsc, Voc, FF and PCE distributions with
custom colours and markers.  Mismatch correction is applied only to
Jsc, and corrected PCE is recalculated as

    PCE = (Voc × Jsc_corr × FF) / 100

with optional toggles to enable or disable this correction.  IPCE
curves are plotted on dual axes (IPCE vs wavelength and integrated
current density on a secondary axis) with the ability to overlay
multiple devices or view them separately.  JV curves display forward
and reverse scans in different colours with optional smoothing and
normalisation.

The interface has been carefully designed to minimise errors and
clipping: slider values are clamped to valid ranges, invalid Plotly
properties have been removed, and default themes are configurable.  All
plots are exportable in high resolution via Kaleido.
"""

import io
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import importlib
import subprocess
import sys

def _ensure_module(mod_name: str, pip_name: Optional[str] = None):
    """Import a module, installing it via pip on ImportError.

    Returns the imported module object.
    """
    pip_name = pip_name or mod_name
    try:
        return importlib.import_module(mod_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(mod_name)

# Ensure commonly missing interactive/plotting packages are available at runtime.
px = _ensure_module("plotly.express", "plotly")
go = _ensure_module("plotly.graph_objects", "plotly")
make_subplots = _ensure_module("plotly.subplots", "plotly").make_subplots
st = _ensure_module("streamlit")

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects  # For glow effects on markers
import seaborn as sns
# Import matplotlib.colors for RGBA conversion when setting box face colours with opacity
import matplotlib.colors as mcolors
import zipfile
import os  # Added for directory scanning
import itertools  # For cycling marker templates

# ---------------------------------------------------------------------------
# Helper functions for automatic data discovery
#
# The tool can now automatically detect summary parameter files, JV curves and
# IPCE curves located in the same directory (or any subdirectory) as this
# script.  When the dashboard is launched (for example by double‑clicking
# this file), the directory is scanned once and the discovered files are
# cached in ``st.session_state['folder_data_cache']``.  These cached files
# are then used as defaults in the Box Plot, IPCE Curve and JV Curve modes
# if the user does not manually upload any files.  This behaviour preserves
# all existing plot settings and UI elements while enabling a “drop‑in”
# workflow: place the script alongside your data files and open it to see
# immediate visualisations.

def _classify_single_txt(path: str) -> Tuple[str, Optional[str]]:
    """Classify a single text file into summary, JV or IPCE.

    Parameters
    ----------
    path : str
        Absolute path to the file.

    Returns
    -------
    tuple
        A tuple of (kind, subtype) where ``kind`` is one of ``'summary'``,
        ``'jv'``, ``'ipce'`` or ``None`` if the file could not be parsed.
        For summary files, the ``subtype`` indicates the scan direction
        (``'rv'`` for reverse, ``'fw'`` for forward or ``None`` if
        indeterminate).  For JV and IPCE files, ``subtype`` is ``None``.

    Notes
    -----
    The classification relies on the file name to detect summary parameter
    files (those containing ``summary`` in the name).  Reverse and forward
    summaries are distinguished by searching for common abbreviations such as
    ``rv``/``reverse`` and ``fw``/``forward``.  Files that are not summary
    parameters are tentatively parsed using :func:`parse_ipce` and
    :func:`parse_jv` in that order.  If parsing succeeds without raising an
    exception and returns a non‑empty DataFrame, the file is classified
    accordingly.  Files that cannot be parsed into any known type are
    ignored.
    """
    name_lower = os.path.basename(path).lower()
    # Detect summary files by presence of 'summary' in the file name
    if name_lower.endswith('.txt') and 'summary' in name_lower:
        # Determine scan direction from common abbreviations
        subtype: Optional[str] = None
        if any(s in name_lower for s in ['rv', 'reverse']):
            subtype = 'rv'
        elif any(s in name_lower for s in ['fw', 'for', 'forward']):
            subtype = 'fw'
        return 'summary', subtype
    # Try IPCE parsing
    try:
        with open(path, 'rb') as fh:
            df_ipce = parse_ipce(fh)
        # IPCE parser returns an empty DataFrame on failure
        if not df_ipce.empty and 'Wavelength' in df_ipce.columns:
            return 'ipce', None
    except Exception:
        pass
    # Try JV parsing
    try:
        with open(path, 'rb') as fh:
            df_jv, _ = parse_jv(fh)
        if not df_jv.empty and (('V_FW' in df_jv.columns and 'J_FW' in df_jv.columns) or
                                ('V_RV' in df_jv.columns and 'J_RV' in df_jv.columns)):
            return 'jv', None
    except Exception:
        pass
    return None, None


def scan_directory_for_data(base_dir: str) -> Dict[str, List[str]]:
    """Recursively scan ``base_dir`` for PV data files.

    Parameters
    ----------
    base_dir : str
        The directory to scan.  All subdirectories are explored.

    Returns
    -------
    dict
        A dictionary with keys ``'summary_rv'``, ``'summary_fw'``, ``'jv'``
        and ``'ipce'``, each containing a list of absolute file paths for
        that category.  Files that cannot be classified are omitted.

    Notes
    -----
    The scanning function skips this script itself to avoid attempting
    to parse code as data.  Duplicate file names across nested folders
    are included separately.
    """
    result = {'summary_rv': [], 'summary_fw': [], 'jv': [], 'ipce': []}
    script_path = os.path.abspath(__file__)
    for root, _, files in os.walk(base_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            # Skip this script file
            if os.path.abspath(full_path) == script_path:
                continue
            # Only examine text files (and optionally csv files if they mimic txt)
            if not fname.lower().endswith(('.txt', '.csv')):
                continue
            kind, subtype = _classify_single_txt(full_path)
            if kind == 'summary':
                # If scan direction cannot be determined, include in both lists
                if subtype == 'rv':
                    result['summary_rv'].append(full_path)
                elif subtype == 'fw':
                    result['summary_fw'].append(full_path)
                else:
                    result['summary_rv'].append(full_path)
                    result['summary_fw'].append(full_path)
            elif kind == 'jv':
                result['jv'].append(full_path)
            elif kind == 'ipce':
                result['ipce'].append(full_path)
    return result

def extract_variation(file_name: str) -> str:
    """Extract a variation identifier from a file name.

    Many device summary files include a variation code embedded in
    the final underscore‑delimited segment of the file name.  For
    example, in a file name such as::

        0001_2025-06-28_16.12.41_Stability (JV)_ITO-CU20-23-1A.txt

    the variation should be ``ITO-CU20`` rather than the entire
    ``ITO-CU20-23-1A`` suffix.  This helper splits on underscores to
    isolate the trailing segment, then drops the last two hyphen
    components (which typically encode device number and replicate
    identifiers).  If there are fewer than three hyphen‑separated
    tokens, the first token is returned.

    Parameters
    ----------
    file_name : str
        The full path or name of the file.

    Returns
    -------
    str
        The inferred variation identifier, or an empty string if the
        input is blank.
    """
    if not file_name:
        return ""
    # Strip any directory components and extension
    base = file_name.split("/")[-1].strip()
    stem = base.rsplit(".", 1)[0] if "." in base else base
    # Extract the last underscore‑delimited segment (e.g. 'ITO-CU20-23-1A')
    parts_underscore = stem.split("_")
    tail = parts_underscore[-1] if parts_underscore else stem
    # Split the tail on hyphens to remove device and replicate identifiers
    hyphen_parts = tail.split("-")
    if len(hyphen_parts) > 2:
        # Remove last two tokens (e.g. '23', '1A')
        variation = "-".join(hyphen_parts[:-2])
    elif hyphen_parts:
        # Fallback to the first token if there are too few parts
        variation = hyphen_parts[0]
    else:
        variation = tail
    return variation

# -----------------------------------------------------------------------------
# Additional helper functions for JV processing
#
def parse_variation_device_pixel(file_name: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract variation, device number, and pixel identifier from a JV file name.

    JV file names typically embed these identifiers in the final underscore‑delimited
    segment.  For example::

        0001_2025-09-16_16.27.35_Stability (JV)_ITO-SAM2-ADD-MGF-ST-28-1D.txt

    yields variation ``ITO-SAM2-ADD-MGF-ST``, device ``28`` and pixel ``1D``.
    If the tail does not contain at least three hyphen‑separated tokens,
    the variation is returned and device/pixel are set to ``None``.

    Parameters
    ----------
    file_name : str
        Name or path of the JV file.

    Returns
    -------
    Tuple[str, Optional[str], Optional[str]]
        A tuple of (variation, device, pixel).  Device and pixel may be ``None``
        if the file name does not encode these fields.
    """
    if not file_name:
        return "", None, None
    base = file_name.split("/")[-1].strip()
    stem = base.rsplit(".", 1)[0] if "." in base else base
    parts_underscore = stem.split("_")
    tail = parts_underscore[-1] if parts_underscore else stem
    hyphen_parts = tail.split("-")
    if len(hyphen_parts) >= 3:
        variation = "-".join(hyphen_parts[:-2])
        device = hyphen_parts[-2]
        pixel = hyphen_parts[-1]
        return variation, device, pixel
    # Fallback: return the tail as the variation and no device/pixel when
    # fewer than three hyphen‑separated tokens are present.
    return tail, None, None


def parse_scan_number(file_name: str) -> Optional[str]:
    """
    Extract the leading scan number from a data file name.

    Many summary and JV file names begin with a numerical scan identifier
    followed by an underscore and the date/time stamp, for example::

        0001_2025-09-16_16.24.56_Stability (JV)_ITO-SAM2-ADD-MGF-OP-30-1A.txt

    Here ``0001`` is the scan number.  This helper uses a regular
    expression to capture any leading digits before the first non‑digit
    character in the base file name.  If no such digits exist, ``None``
    is returned.

    Parameters
    ----------
    file_name : str
        Name or path of the file.

    Returns
    -------
    Optional[str]
        The scan number as a string, or ``None`` if it cannot be
        determined.
    """
    if not file_name:
        return None
    base = os.path.basename(file_name).strip()
    # Extract leading digits before any underscore or non‑digit
    import re  # Import locally to avoid polluting the global namespace
    m = re.match(r"^(\d+)", base)
    if m:
        return m.group(1)
    return None

def parse_txt(file_buffer) -> pd.DataFrame:
    """Parses a tab‑delimited summary file into a tidy DataFrame.

    Attempts multiple encodings and standardises column names.  A
    ``Variation`` column is derived from the file name.
    """
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1"]
    df: Optional[pd.DataFrame] = None
    for enc in encodings:
        try:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, sep="\t", encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        file_buffer.seek(0)
        raw = file_buffer.read()
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        from io import StringIO
        df = pd.read_csv(StringIO(text), sep="\t")
    rename_map = {
        "File": "File Name",
        "Voc (V)": "Voc",
        "Jsc (mA/cm²)": "Jsc",
        "V_MPP (V)": "V_MPP",
        "J_MPP (mA/cm²)": "J_MPP",
        "P_MPP (mW/cm²)": "P_MPP",
        "Rs (Ohm)": "Rs",
        "R// (Ohm)": "R//",
        "FF (%)": "FF",
        "Eff (%)": "Eff",
    }
    df = df.rename(columns=rename_map)
    for col in ["Voc", "Jsc", "V_MPP", "J_MPP", "P_MPP", "Rs", "R//", "FF", "Eff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Variation"] = df["File Name"].apply(extract_variation)
    return df

def parse_excel_raw(file_buffer) -> Dict[str, Tuple[float, pd.DataFrame]]:
    """Parses sun‑simulator Excel workbooks for reverse and forward data."""
    result: Dict[str, Tuple[float, pd.DataFrame]] = {}
    xls = pd.ExcelFile(file_buffer)
    mapping = {"RV Data": "RV", "For Data": "FW"}
    for sheet_name, key in mapping.items():
        if sheet_name not in xls.sheet_names:
            continue
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        factor = 1.0
        try:
            cell = raw.iloc[1, 11]
            if isinstance(cell, (int, float)):
                factor = float(cell)
        except Exception:
            pass
        df = pd.read_excel(xls, sheet_name=sheet_name, header=2, skiprows=[3])
        df = df.dropna(axis=1, how="all")
        rename_map = {
            "Voc (V)": "Voc",
            "Jsc (mA/cm²)": "Jsc",
            "V_MPP (V)": "V_MPP",
            "J_MPP (mA/cm²)": "J_MPP",
            "P_MPP (mW/cm²)": "P_MPP",
            "Rs (Ohm)": "Rs",
            "R// (Ohm)": "R//",
            "FF (%)": "FF",
            "Eff (%)": "Eff",
            "Eff.1": "Eff (Corr)",
        }
        df = df.rename(columns=rename_map)
        df = df[df["File Name"].astype(str).str.lower() != "file"]
        for col in ["Voc", "Jsc", "V_MPP", "J_MPP", "P_MPP", "Rs", "R//", "FF", "Eff", "Jsc Corrected", "Eff (Corr)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Variation"] = df["File Name"].apply(extract_variation)
        result[key] = (factor, df)
    return result

def compute_corrected_values(df: pd.DataFrame, mismatch: float) -> pd.DataFrame:
    """Applies mismatch correction to Jsc and recalculates PCE.

    Correction is applied only to the magnitude of Jsc.  PCE is
    recomputed as `(Voc × Jsc_corr × FF) / 100` using FF in percent.
    """
    df = df.copy()
    if mismatch != 0:
        df["Jsc_corrected"] = df["Jsc"].abs() / mismatch
    else:
        df["Jsc_corrected"] = df["Jsc"].abs()
    df["PCE_corrected"] = (df["Voc"] * df["Jsc_corrected"] * df["FF"]) / 100.0
    return df

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics by variation and scan."""
    metrics = {
        "Jsc_corrected": "Jsc_corrected",
        "Voc": "Voc",
        "FF": "FF",
        "PCE_corrected": "PCE_corrected",
    }
    agg = {m: ["mean", "median", "std", "min", "max"] for m in metrics.values()}
    summary = df.groupby(["Variation", "Scan"]).agg(agg)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()

@st.cache_data(show_spinner=False)
def parse_ipce(file_buffer) -> pd.DataFrame:
    """Parse IPCE .txt files to extract wavelength, IPCE and integrated current."""
    file_buffer.seek(0)
    raw = file_buffer.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    lines = text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Wavelength"):
            header_idx = i
            break
    data_lines = [l for l in lines[header_idx:] if l.strip()]
    if not data_lines:
        return pd.DataFrame()
    csv_str = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_str), sep="\t")
    rename_map = {
        "Wavelength (nm)": "Wavelength",
        "IPCE (%)": "IPCE",
        "J integrated (mA/cm2)": "J_integrated",
        "J integrated (mA/cm²)": "J_integrated",
    }
    df = df.rename(columns=rename_map)
    for col in ["Wavelength", "IPCE", "J_integrated"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Wavelength", "IPCE"])
    return df[[c for c in ["Wavelength", "IPCE", "J_integrated"] if c in df.columns]]

@st.cache_data(show_spinner=False)
def parse_jv(file_buffer) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Parse JV summary logs to extract forward and reverse curves."""
    file_buffer.seek(0)
    raw = file_buffer.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("V_FW") or line.startswith("V_FW (V)"):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(), {}
    data_lines = [l for l in lines[header_idx:] if l.strip()]
    csv_str = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_str), sep="\t")
    rename_map = {
        "V_FW (V)": "V_FW",
        "J_FW (mA/cm)": "J_FW",
        "J_FW (mA/cm²)": "J_FW",
        "V_RV (V)": "V_RV",
        "J_RV (mA/cm)": "J_RV",
        "J_RV (mA/cm²)": "J_RV",
    }
    df = df.rename(columns=rename_map)
    for col in ["V_FW", "J_FW", "V_RV", "J_RV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[[c for c in ["V_FW", "J_FW", "V_RV", "J_RV"] if c in df.columns]], {}

def main() -> None:
    """Main entry for Streamlit app."""
    st.set_page_config(page_title="Plot4PV: Solar Data Visualization Tool", layout="wide")
    st.title("Plot4PV: Solar Data Visualization Tool")
    st.markdown(
        """
        Visualize photovoltaic performance with Jsc, Voc, FF, PCE, IPCE, and JV curves.  Upload Excel or
        TXT files to generate clean, export‑ready plots.  Use the sidebar to select a plot type and
        customise your analysis.
        """
    )
    # On first run, scan the directory containing this script for data files
    # and cache the results.  This enables automatic plotting when the
    # application is opened in a folder containing PV data (summaries,
    # JV and IPCE).  The cache is stored in ``st.session_state`` so that
    # toggling between plot types does not require reloading the same files.
    if 'folder_data_cache' not in st.session_state:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            st.session_state['folder_data_cache'] = scan_directory_for_data(base_dir)
        except Exception:
            st.session_state['folder_data_cache'] = {'summary_rv': [], 'summary_fw': [], 'jv': [], 'ipce': []}

    # Global theme selection
    theme_choice = st.sidebar.selectbox("Theme", options=["Light", "Dark"], index=0)
    template_name = "plotly_white" if theme_choice == "Light" else "plotly_dark"

    # Initialise global font settings with sensible defaults.  These will be
    # overridden when the user adjusts font settings in the Box Plot mode.
    font_size_global: int = 12
    bold_title_global: bool = False
    italic_axes_global: bool = False
    font_family_global: str = "Arial"

    # Select plot mode
    mode = st.sidebar.radio("What would you like to plot?", options=["Box Plot", "IPCE Curve", "JV Curve", "JVcorr"], index=0)
    # Colour palette for variations
    from plotly.colors import qualitative as qc
    base_palette = qc.Dark24

    if mode == "Box Plot":
        # Upload data
        st.sidebar.header("Upload Summary Files")
        excel_files = st.sidebar.file_uploader("Excel (.xlsx) file(s)", type=["xlsx"], accept_multiple_files=True)
        rv_files = st.sidebar.file_uploader("Reverse scan summary (.txt) file(s)", type=["txt"], accept_multiple_files=True)
        fw_files = st.sidebar.file_uploader("Forward scan summary (.txt) file(s)", type=["txt"], accept_multiple_files=True)
        # Optional folder upload (.zip) containing summary parameter files and JV data.  If provided,
        # the app will automatically extract any summary files for reverse and forward scans.
        folder_zip = st.sidebar.file_uploader(
            "Upload data folder (.zip) for summaries and JV", type=["zip"], accept_multiple_files=False
        )
        # Prepare combined lists for reverse and forward summary files.  Start with any individually
        # uploaded files, then extend with files discovered in the zip archive (if present).
        rv_files_combined: List = []
        fw_files_combined: List = []
        if rv_files:
            rv_files_combined.extend(list(rv_files))
        if fw_files:
            fw_files_combined.extend(list(fw_files))
        if folder_zip is not None:
            try:
                folder_zip.seek(0)
                with zipfile.ZipFile(folder_zip) as zf:
                    for nm in zf.namelist():
                        nm_lower = nm.lower()
                        # Match summary parameter files for reverse (RV) and forward (FW) scans.  The
                        # naming convention uses "JV Summary_Parameters RV.txt" and
                        # "JV Summary_Parameters FW.txt".  We check only the file suffix to be
                        # case‑insensitive and avoid matching unintended files.
                        if nm_lower.endswith("jv summary_parameters rv.txt"):
                            data = zf.read(nm)
                            buf = io.BytesIO(data)
                            # Assign a name attribute for display and variation extraction.  Use only
                            # the base filename to avoid exposing directory paths.
                            buf.name = nm.split("/")[-1]
                            rv_files_combined.append(buf)
                        elif nm_lower.endswith("jv summary_parameters fw.txt"):
                            data = zf.read(nm)
                            buf = io.BytesIO(data)
                            buf.name = nm.split("/")[-1]
                            fw_files_combined.append(buf)
            except Exception as e:
                # Display an error in the sidebar if the zip cannot be processed
                st.sidebar.error(f"Could not process the uploaded zip folder: {e}")

        # Extend the lists with any automatically detected summary files from the
        # folder the script resides in.  These files are discovered once at
        # startup via ``scan_directory_for_data`` and stored in
        # ``st.session_state['folder_data_cache']``.  They are only added
        # when the user has not uploaded explicit files to avoid duplicates.
        folder_cache = st.session_state.get('folder_data_cache', None)
        if folder_cache:
            # Only extend from the cache when the user has not uploaded any files
            # to prevent duplicates.  If the corresponding combined list is empty,
            # populate it with the automatically discovered summaries.
            if not rv_files_combined:
                for p in folder_cache.get('summary_rv', []):
                    try:
                        f = open(p, 'rb')
                        rv_files_combined.append(f)
                    except Exception:
                        continue
            if not fw_files_combined:
                for p in folder_cache.get('summary_fw', []):
                    try:
                        f = open(p, 'rb')
                        fw_files_combined.append(f)
                    except Exception:
                        continue
        # Correction toggle and factors
        st.sidebar.header("Mismatch Correction")
        apply_corr = st.sidebar.checkbox("Apply mismatch correction", value=True)
        # Default factors from first Excel file if present
        default_rv = 1.0
        default_fw = 1.0
        excel_data_list: List[Dict[str, Tuple[float, pd.DataFrame]]] = []
        if excel_files:
            for ef in excel_files:
                try:
                    ed = parse_excel_raw(ef)
                    excel_data_list.append(ed)
                    if default_rv == 1.0 and "RV" in ed:
                        default_rv = ed["RV"][0]
                    if default_fw == 1.0 and "FW" in ed:
                        default_fw = ed["FW"][0]
                except Exception as e:
                    st.sidebar.warning(f"Could not parse Excel file '{ef.name}': {e}")
        # Initialise mismatch factors using defaults.  When correction is enabled,
        # present a single input that applies the same factor to both reverse
        # and forward data.  This avoids having separate inputs and ensures
        # consistent correction across scans.
        rv_factor = default_rv
        fw_factor = default_fw
        if apply_corr:
            mismatch_factor = st.sidebar.number_input(
                "Mismatch factor",
                value=float(default_rv),
                min_value=0.0,
                step=0.01,
                help="Single correction factor applied to both reverse and forward scans"
            )
            rv_factor = mismatch_factor
            fw_factor = mismatch_factor
        # Load and correct data
        frames: List[pd.DataFrame] = []
        raw_previews: List[Tuple[str, pd.DataFrame]] = []
        for ed in excel_data_list:
            for key, (fac, df) in ed.items():
                scan = "Reverse" if key == "RV" else "Forward"
                corr = rv_factor if scan == "Reverse" else fw_factor
                raw_previews.append((f"{scan} (Excel)", df.copy()))
                frames.append(compute_corrected_values(df, corr if apply_corr else 1.0).assign(Scan=scan))
        # Iterate over combined reverse summary files (uploaded individually and extracted from zip)
        if rv_files_combined:
            for f in rv_files_combined:
                try:
                    dfr = parse_txt(f)
                    raw_previews.append((f"Reverse (TXT) - {f.name}", dfr.copy()))
                    frames.append(compute_corrected_values(dfr, rv_factor if apply_corr else 1.0).assign(Scan="Reverse"))
                except Exception as e:
                    st.sidebar.error(f"Failed to parse {f.name}: {e}")
        # Iterate over combined forward summary files (uploaded individually and extracted from zip)
        if fw_files_combined:
            for f in fw_files_combined:
                try:
                    dff = parse_txt(f)
                    raw_previews.append((f"Forward (TXT) - {f.name}", dff.copy()))
                    frames.append(compute_corrected_values(dff, fw_factor if apply_corr else 1.0).assign(Scan="Forward"))
                except Exception as e:
                    st.sidebar.error(f"Failed to parse {f.name}: {e}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame()
        # Show raw preview in a collapsible section.  The outer expander is
        # collapsed by default to reduce initial clutter.  Each raw dataset
        # remains individually expandable within the section.
        if not combined.empty:
            with st.expander("Raw Data Preview Before Correction", expanded=False):
                for name, df in raw_previews:
                    with st.expander(name, expanded=False):
                        st.dataframe(df)
        else:
            st.info("Upload at least one summary file to begin.")
        if combined.empty:
            return
        # Variation definition
        with st.sidebar.expander("Define Variations and Visuals", expanded=False):
            st.write(
                "Enter the number of variations and assign labels, colours and markers.  Rows are matched by substring search on the file name."
            )
            nvars = st.number_input("Number of variations", min_value=1, max_value=100, value=5, step=1)
            marker_opts = [
                "circle", "square", "diamond", "triangle-up", "triangle-down", 
                "triangle-left", "triangle-right", "pentagon", "hexagon", "hexagon2", 
                "star", "star-square", "star-diamond", "cross", "x", "asterisk", 
                "bowtie", "hourglass"
            ]
            user_vars: List[Tuple[str, str, str]] = []
            for i in range(1, int(nvars) + 1):
                default_col = base_palette[(i - 1) % len(base_palette)]
                lbl = st.text_input(f"Variation {i} label", key=f"var_lbl_{i}").strip()
                col = st.color_picker(f"Colour for variation {i}", default_col, key=f"var_col_{i}")
                mrk = st.selectbox(f"Marker for variation {i}", marker_opts, index=(i - 1) % len(marker_opts), key=f"var_mrk_{i}")
                if lbl:
                    user_vars.append((lbl, col, mrk))
        var_color_map: Dict[str, str] = {}
        var_marker_map: Dict[str, str] = {}
        if user_vars:
            def assign(fname: str) -> str:
                fname_lower = str(fname).lower() if pd.notna(fname) else ""
                for lbl, _, _ in user_vars:
                    if lbl.lower() in fname_lower:
                        return lbl
                return "Unassigned"
            combined["Variation"] = combined["File Name"].apply(assign)
            for lbl, col, mrk in user_vars:
                var_color_map[lbl] = col
                var_marker_map[lbl] = mrk
            combined = combined[combined["Variation"] != "Unassigned"].copy()
        # Plot settings
        # Collapse the plot settings section by default; users can expand when needed.
        with st.sidebar.expander("Plot Settings", expanded=False):
            scans = sorted(combined["Scan"].unique())
            selected_scans = st.multiselect("Select scan directions", options=scans, default=scans)
            variations = sorted(combined["Variation"].unique())
            # Provide variation checkboxes instead of a multiselect.  Each variation
            # can be toggled individually; by default all are selected.  States
            # are stored in ``st.session_state`` using a unique key.
            selected_variations: List[str] = []
            for v in variations:
                key_var = f"box_var_include_{v.replace(' ', '_')}"
                include_v = st.checkbox(
                    f"Include {v}",
                    value=True,
                    key=key_var
                )
                if include_v:
                    selected_variations.append(v)
            metric_map = {"Jsc": "Jsc_corrected", "Voc": "Voc", "FF": "FF", "PCE": "PCE_corrected"}
            selected_metrics = st.multiselect("Select metrics", list(metric_map.keys()), default=list(metric_map.keys()))
            mode_choice = st.radio("Combine scans or separate", ["Combine", "Separate"], index=0)
            show_boxes = st.checkbox("Show box outlines", value=True)
            # Thickness of the box border
            outline_px = st.slider("Box border thickness (px)", min_value=1, max_value=6, value=2)
            # Whisker width clamp to [0.05,1.0].  Currently unused but retained for UI completeness
            whisker_frac = st.slider("Whisker width (0–1)", min_value=0.05, max_value=1.0, value=0.4)
            # Overlay marker size and opacity
            # Overlay marker size: default increased to 6 for better visibility
            marker_sz = st.slider("Overlay marker size", min_value=2, max_value=12, value=6)
            marker_opacity = st.slider("Overlay marker opacity", min_value=0.2, max_value=1.0, value=0.7, step=0.05)
            # Overlay style: scatter points, marker shapes using variation markers, or none
            overlay_style = st.selectbox(
                "Overlay style for data points",
                options=["Scatter", "Markers", "None"],
                index=1,
                help="Choose how to display individual data points on the box plot"
            )
            # Jitter spread to avoid overlapping points when overlaying data.  0 = no jitter, 1 = maximal jitter
            jitter_spread = st.slider(
                "Jitter spread (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="Controls horizontal spreading of data points when overlaying"
            )
            # Opacity for the filled portion of each box (0=transparent, 1=opaque)
            box_opacity = st.slider(
                "Box opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                help="Controls the transparency of the coloured boxes"
            )

            # Allow users to choose among different box plot templates.  Each option corresponds
            # to a distinct visual style inspired by the provided examples.  The
            # default style preserves the existing behaviour.  Template names are kept short
            # and descriptive to appear succinctly in the UI.
            template_options = [
                "Classic",  # existing coloured boxes with overlay points
                "Raincloud",  # violin + box combination with jittered points
                "Outline",   # white interior boxes with coloured outlines
                "Mono",      # grey boxes with coloured outlines
                "Notched",   # notched boxes emphasising median
                "Density"    # density curves alongside boxes with offset scatter
            ]
            template_choice = st.selectbox(
                "Box plot template", options=template_options, index=0,
                help="Select the visual style for the box plots"
            )
            # Whether to display a legend for the variations and where to place it
            show_legend = st.checkbox(
                "Show legend for variations", value=True,
                help="Toggle display of the variation legend with coloured markers"
            )
            legend_position = st.selectbox(
                "Legend position",
                options=["bottom", "top", "left", "right"],
                index=1,
                help="Select where to place the legend on the figure"
            )
            # Control visibility of axis labels.  X‑axis labels are hidden by default to avoid
            # overlap with legends; Y‑axis labels are shown by default.  Users can toggle
            # these settings to suit their preferences.
            show_x_label_global = st.checkbox("Show x-axis labels", value=False)
            show_y_label_global = st.checkbox("Show y-axis labels", value=True)
            # Spacing between box groups
            spacing_val = st.slider("Adjust spacing between box groups (1–100)", 1, 100, 10, help="Smaller values produce tighter plots")
            box_gap = spacing_val / 100.0
            plot_width = st.slider("Plot width (inches)", 4, 20, 10)
            plot_height = st.slider("Plot height (inches)", 3, 15, 6)
            download_fmt = st.selectbox("Download format", ["PNG", "SVG"], index=0).lower()
            export_base = st.text_input("Export file base name", value="plot4pv")
            show_combined = st.checkbox("Show Combined Box Plot (All Selected Variations)", value=True, help="Display all selected variations together for each metric")
            # Legend spacing control: adjust the distance between the legend
            # and the plot.  Smaller values bring the legend closer to the
            # axes while larger values push it further away.  Default has
            # been reduced for a more compact layout.
            legend_distance = st.slider(
                "Legend spacing", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                help="Adjust the distance between the legend and the plot. Smaller values bring the legend closer."
            )

            # Font customization
            st.markdown("**Font Settings**")
            font_size = st.slider("Font size", min_value=10, max_value=24, value=12)
            bold_title = st.checkbox("Bold plot titles", value=False)
            italic_axes = st.checkbox("Italic axis labels", value=False)
            font_family = st.selectbox("Font family", options=["Arial", "Times New Roman", "Roboto", "Courier New"], index=0)
            x_axis_rotation = st.slider("X-axis label rotation (degrees)", 0, 90, 30)
            # Propagate font settings to global variables for use in other modes
            font_size_global = font_size
            bold_title_global = bold_title
            italic_axes_global = italic_axes
            font_family_global = font_family
        # Filters and axes
        with st.sidebar.expander("Filters and Axis Limits", expanded=False):
            filter_ranges: Dict[str, Tuple[float, float]] = {}
            axis_limits: Dict[str, Optional[Tuple[float, float]]] = {}
            # Default filter ranges for each metric.  These defaults will be
            # clipped to the actual data min/max values so that the initial
            # filter does not exceed the data range.  Users can adjust
            # these values as desired.
            default_filter_values = {
                "Jsc": (4.0, 25.0),
                "Voc": (0.2, 1.8),
                "FF": (20.0, 90.0),
                "PCE": (4.0, 25.0),
            }
            for m_key, col in metric_map.items():
                if col not in combined.columns:
                    continue
                vals = combined[col].dropna()
                if vals.empty:
                    continue
                vmin, vmax = float(vals.min()), float(vals.max())
                # Determine the default filter bounds.  Use specified defaults
                # where available, clipped to the data range.
                dmin, dmax = default_filter_values.get(m_key, (vmin, vmax))
                def_min = max(vmin, dmin)
                def_max = min(vmax, dmax)
                # Handle edge case where default bounds are outside the data range
                if def_min > def_max:
                    def_min, def_max = vmin, vmax
                #
                # Replace the slider used to select a filter range with two number
                # input boxes for the minimum and maximum values.  Users can
                # specify the lower and upper bounds directly.  The default
                # values (def_min and def_max) are clipped to the data range.  If
                # the user enters a minimum greater than the maximum, the
                # values are swapped internally to ensure a valid range.
                #
                col_min = st.number_input(
                    f"{m_key} filter min", value=def_min, min_value=vmin, max_value=vmax, key=f"fmin_{m_key}"
                )
                col_max = st.number_input(
                    f"{m_key} filter max", value=def_max, min_value=vmin, max_value=vmax, key=f"fmax_{m_key}"
                )
                # Ensure that the minimum does not exceed the maximum; if it
                # does, swap the values.  This prevents an empty filter range.
                if col_min > col_max:
                    col_min, col_max = col_max, col_min
                filter_ranges[col] = (col_min, col_max)
                manual_lim = st.checkbox(f"Set {m_key} axis limits", value=False, key=f"lim_{m_key}")
                if manual_lim:
                    a_min = st.number_input(f"{m_key} axis min", value=vmin, key=f"amin_{m_key}")
                    a_max = st.number_input(f"{m_key} axis max", value=vmax, key=f"amax_{m_key}")
                    if a_min > a_max:
                        a_min, a_max = a_max, a_min
                    axis_limits[col] = (a_min, a_max)
                else:
                    axis_limits[col] = None
        # Custom labels
        with st.sidebar.expander("Custom Labels", expanded=False):
            custom_labels: Dict[str, str] = {}
            for m in ["Jsc", "Voc", "FF", "PCE"]:
                custom_labels[m] = st.text_input(f"Label for {m} axis", value=m, key=f"lab_{m}")
            x_label = st.text_input("Label for x‑axis", value="Variation & Scan")
        # Apply filters
        data = combined[combined["Scan"].isin(selected_scans) & combined["Variation"].isin(selected_variations)].copy()
        for col, (mi, ma) in filter_ranges.items():
            data = data[(data[col] >= mi) & (data[col] <= ma)]
        # Data previews and summary
        # Wrap corrected data preview in a collapsible expander to keep the
        # interface tidy by default.  Users can expand to inspect the
        # corrected dataset.
        with st.expander("Corrected Data Preview", expanded=False):
            st.dataframe(data)
        # Per‑Variation data preview.  The entire section is collapsed by
        # default; within it, each variation has its own sub‑expander.
        with st.expander("Per‑Variation Preview", expanded=False):
            for v in selected_variations:
                with st.expander(f"{v} data", expanded=False):
                    st.dataframe(data[data["Variation"] == v])
        # Display summary statistics in a collapsible section.  Only maximum values
        # for Jsc, Voc, FF and PCE per variation (along with the device and pixel
        # where these maxima occur) are shown initially.  The full summary
        # remains accessible via a nested expander.
        with st.expander("Summary Statistics", expanded=False):
            summary_table_df = build_summary_table(data)
            best_rows: List[Dict[str, Any]] = []
            # Determine the best (maximum) values for each metric within each selected variation
            for v in selected_variations:
                dfv = data[data["Variation"] == v]
                if dfv.empty:
                    continue
                row: Dict[str, Any] = {"Variation": v}
                metric_info = [
                    ("Jsc_max", "Jsc_corrected"),
                    ("Voc_max", "Voc"),
                    ("FF_max", "FF"),
                    ("PCE_max", "PCE_corrected")
                ]
                for metric_key, col_name in metric_info:
                    if col_name in dfv.columns:
                        try:
                            idx_max = dfv[col_name].idxmax()
                        except ValueError:
                            continue
                        max_val = dfv.loc[idx_max, col_name]
                        fname = str(dfv.loc[idx_max, "File Name"]) if "File Name" in dfv.columns else ""
                        # Extract variation, device and pixel identifiers
                        _, dev, pix = parse_variation_device_pixel(fname)
                        # Also derive the scan number from the file name so that
                        # users can easily map a pixel back to its acquisition
                        # sequence.  The scan number corresponds to the
                        # leading digits in the file name, for example the
                        # ``0001`` in ``0001_2025-09-16_16.24.56_…``.  When
                        # available, append it to the pixel identifier in
                        # parentheses so that the best table reads like
                        # ``1A (0001)``.  If either component is missing,
                        # fall back to whichever piece is present.
                        scan_num = parse_scan_number(fname)
                        row[metric_key] = max_val
                        row[f"{metric_key}_device"] = dev or ""
                        # Keep a raw pixel value (without scan number) for internal
                        # matching (e.g. for preselecting JV curves).  The raw
                        # pixel may be None when not encoded in the file name.
                        raw_pixel = pix or ""
                        # Construct the display pixel: include scan number in
                        # parentheses when both pixel and scan are known.  This
                        # field will be shown in the summary table.
                        if raw_pixel and scan_num:
                            display_pixel = f"{raw_pixel} ({scan_num})"
                        elif raw_pixel:
                            display_pixel = raw_pixel
                        elif scan_num:
                            display_pixel = f"({scan_num})"
                        else:
                            display_pixel = ""
                        row[f"{metric_key}_pixel"] = display_pixel
                        # Store the raw pixel separately for computing defaults later
                        row[f"{metric_key}_pixel_raw"] = raw_pixel
                        # Store the scan number for the metric so we can enforce
                        # same‑scan defaults in the JV overlay.  May be None when
                        # not encoded in the file name.
                        row[f"{metric_key}_scan"] = scan_num
                best_rows.append(row)
            # Convert best rows into a DataFrame.  Remove any internal
            # columns used for bookkeeping (those ending in '_pixel_raw') so
            # they are not shown to the user.
            best_df = pd.DataFrame(best_rows)
            cols_to_drop = [c for c in best_df.columns if c.endswith('_pixel_raw') or c.endswith('_scan')]
            if cols_to_drop:
                best_df = best_df.drop(columns=cols_to_drop)
            st.dataframe(best_df)
            # Save best device/pixel defaults for JV overlay in session_state.  These
            # defaults will be used in the JV plot to preselect the best curves by
            # variation.  Each variation maps to a dictionary with optional 'jsc'
            # and 'pce' entries storing the (device, pixel) tuple producing the
            # maximum Jsc and PCE, respectively.
            best_defaults: Dict[str, Dict[str, Tuple[Optional[str], Optional[str]]]] = {}
            for r in best_rows:
                v = r.get("Variation")
                if not v:
                    continue
                bd: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
                # Use the raw pixel values (without scan number) for defaults so
                # that JV overlay preselection works correctly.  The display
                # pixel (with scan number) is used only for showing in the
                # summary table.
                if "Jsc_max_device" in r:
                    dev_raw = r.get("Jsc_max_device")
                    pix_raw = r.get("Jsc_max_pixel_raw") if "Jsc_max_pixel_raw" in r else r.get("Jsc_max_pixel")
                    scan_raw = r.get("Jsc_max_scan")
                    # Store device, pixel and scan number (may be None) for Jsc
                    bd['jsc'] = (dev_raw, pix_raw, scan_raw)
                if "PCE_max_device" in r:
                    dev_raw = r.get("PCE_max_device")
                    pix_raw = r.get("PCE_max_pixel_raw") if "PCE_max_pixel_raw" in r else r.get("PCE_max_pixel")
                    scan_raw = r.get("PCE_max_scan")
                    # Store device, pixel and scan number (may be None) for PCE
                    bd['pce'] = (dev_raw, pix_raw, scan_raw)
                if bd:
                    best_defaults[v] = bd
            if best_defaults:
                st.session_state['best_jv_defaults'] = best_defaults
            # Nested expander for the full summary statistics (mean, median, std, etc.)
            with st.expander("Show full summary statistics", expanded=False):
                st.dataframe(summary_table_df)
        # Colour and marker fallback
        c_map: Dict[str, str] = {}
        m_map: Dict[str, str] = {}
        for i, v in enumerate(selected_variations):
            c_map[v] = var_color_map.get(v, base_palette[i % len(base_palette)])
            m_map[v] = var_marker_map.get(v, marker_opts[i % len(marker_opts)])
        # Box plots
        st.subheader("Box Plots")
        # Set Matplotlib style based on theme
        if theme_choice == "Dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        # Determine layout spacing (for grouping in plots)
        gap = box_gap
        group_gap = box_gap / 2
        # Define marker shape mapping from Plotly names to Matplotlib
        marker_map_plotly_to_mpl = {
            "circle": "o", "square": "s", "diamond": "D", "triangle-up": "^", 
            "triangle-down": "v", "triangle-left": "<", "triangle-right": ">", 
            "pentagon": "p", "hexagon": "h", "hexagon2": "H", "star": "*", 
            "star-square": "*", "star-diamond": "*", "cross": "+", "x": "x", 
            "asterisk": "*", "bowtie": "X", "hourglass": "X"
        }
        # Configure outlier (flier) and mean point style for box plots
        # Configure outlier (flier) size: hide flier markers when overlaying data points to avoid clutter
        flierprops = {
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            # If overlay_style is not 'None', suppress flier markers by setting their size to zero
            "markersize": (0 if overlay_style != "None" else marker_sz)
        }
        meanprops = {"marker": "D", "markerfacecolor": "white", "markeredgecolor": ("white" if theme_choice == "Dark" else "black"), "markersize": marker_sz + 2}
        # Compute box_width from spacing for Seaborn (width of boxes)
        box_width = 1.0 - box_gap
        if box_width < 0.05:
            box_width = 0.05
        # Combined metrics figure
        # When requested, display a 2×2 grid of box plots for the four key metrics (Voc, Jsc, FF, PCE).
        # Each subplot shows the distribution of the selected metric for all variations and scans, with
        # scan labels (RV/FW) on the x‑axis and variation colours represented in the legend.  Variation
        # markers and colours are used for overlay points when chosen.  The legend location and
        # visibility are configurable.
        if show_combined:
            # Determine the ordered list of metrics to display in the grid.  Even if only a subset
            # of metrics is selected, the grid will allocate up to four panels and hide unused ones.
            all_metrics_order = ["Voc", "Jsc", "FF", "PCE"]
            metrics_to_plot = [m for m in all_metrics_order if m in selected_metrics]
            # Create a 2×2 subplot grid.  The size of the figure is scaled up relative to a single
            # plot to maintain readability when multiple subplots are present.
            fig_grid, axes_grid = plt.subplots(2, 2, figsize=(plot_width * 2, plot_height * 2))
            axes_flat = axes_grid.flatten()
            # Build a mapping from combined category (Variation + Scan) to colours based on variation.
            # This mapping is reused for all metrics.
            var_scan_order: List[str] = []
            for v in selected_variations:
                for s in selected_scans:
                    # Only include categories present in the data
                    if not data[(data["Variation"] == v) & (data["Scan"] == s)].empty:
                        var_scan_order.append(f"{v} {s}")
            colour_mapping = {cat: c_map[cat.split()[0]] for cat in var_scan_order}
            # Loop through each of the four possible metric positions
            for idx, metric_name in enumerate(all_metrics_order):
                ax = axes_flat[idx]
                if metric_name not in metrics_to_plot:
                    # Hide unused subplot
                    ax.axis('off')
                    continue
                col_name = metric_map[metric_name]
                # Prepare DataFrame for plotting: combine variation and scan into a single category
                df_plot = data[["Variation", "Scan", col_name]].copy()
                df_plot["VarScan"] = df_plot["Variation"] + " " + df_plot["Scan"]
                # Drop rows without data for this metric
                df_plot = df_plot.dropna(subset=[col_name])
                # Draw according to the selected template
                if template_choice == "Classic":
                    # Standard coloured boxes with jittered points
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=colour_mapping,
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    # Colour each box and set opacity
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    # Overlay points
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Raincloud":
                    # Raincloud: symmetric violin (full density), narrow box overlay, jittered points around centre
                    for pos, cat in enumerate(var_scan_order):
                        var_name = cat.split()[0]
                        sub_vals = df_plot[df_plot["VarScan"] == cat][col_name].dropna()
                        if sub_vals.empty:
                            continue
                        try:
                            from scipy.stats import gaussian_kde
                        except Exception:
                            gaussian_kde = None
                        if gaussian_kde and len(sub_vals) > 1:
                            kde = gaussian_kde(sub_vals)
                            y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                            dens = kde(y_vals_dens)
                            if dens.max() > 0:
                                # Normalise density to half the total width on each side
                                dens_scaled = dens / dens.max() * (box_width / 2.0)
                                x_left = pos - dens_scaled
                                x_right = pos + dens_scaled
                                ax.fill_betweenx(
                                    y_vals_dens,
                                    x_left,
                                    x_right,
                                    color=c_map.get(var_name, base_palette[0]),
                                    alpha=0.4,
                                    linewidth=0
                                )
                    # Narrow central boxplot
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=colour_mapping,
                        width=box_width * 0.25,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    # Set box opacity
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    # Overlay points jittered around the centre
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            # Jitter range spans the violin width
                            jitter_range = jitter_spread * (box_width / 2.0)
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Outline":
                    # Outline: white interior with coloured borders and diamond points; global dashed line
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=["white"] * len(var_scan_order),
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='white', markersize=marker_sz)
                    )
                    # Set border colour for each box
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        # interior white with alpha applied to box_opacity (for subtle shading)
                        rgba = mcolors.to_rgba('white', alpha=box_opacity)
                        patch.set_facecolor(rgba)
                        patch.set_edgecolor(colour)
                        patch.set_linewidth(outline_px if show_boxes else 0)
                    # Draw overall mean dashed line
                    overall_mean = df_plot[col_name].mean()
                    ax.axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
                    # Overlay points with diamond markers
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            # diamond marker shape
                            if overlay_style == "Scatter":
                                mshape = 'D'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'diamond'), 'D')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Mono":
                    # Mono: grey boxes with coloured outlines
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=["white"] * len(var_scan_order),
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    # Set grey interior and coloured edges
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        # Compute a light grey by mixing colour with white
                        r, g, b = mcolors.to_rgb(colour)
                        grey = (0.8, 0.8, 0.8)
                        # Mix grey with colour based on opacity for subtle tint
                        mix_colour = tuple(0.6 * c + 0.4 * g_ for c, g_ in zip((r, g, b), grey))
                        rgba = (*mix_colour, box_opacity)
                        patch.set_facecolor(rgba)
                        patch.set_edgecolor(colour)
                        patch.set_linewidth(outline_px if show_boxes else 0)
                    # Overlay points
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Notched":
                    # Notched: emphasise the median with notches.  Draw the
                    # notched boxes and lighten the interior while keeping
                    # coloured edges for clarity.
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=colour_mapping,
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops,
                        notch=True
                    )
                    # Lighten the box interior: blend the variation colour with white.
                    lighten_factor = 0.5
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        r, g, b = mcolors.to_rgb(colour)
                        mix_colour = tuple((1 - lighten_factor) * c + lighten_factor * 1.0 for c in (r, g, b))
                        rgba = (*mix_colour, box_opacity)
                        patch.set_facecolor(rgba)
                        patch.set_edgecolor(colour)
                        patch.set_linewidth(outline_px if show_boxes else 0)
                    # Overlay points
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Density":
                    # Density: show side density curves, box and scatter with mean marker
                    # Draw density curves on the left of the box
                    for pos, cat in enumerate(var_scan_order):
                        var_name = cat.split()[0]
                        sub_vals = df_plot[df_plot["VarScan"] == cat][col_name].dropna()
                        if sub_vals.empty:
                            continue
                        try:
                            from scipy.stats import gaussian_kde
                        except Exception:
                            gaussian_kde = None
                        if gaussian_kde and len(sub_vals) > 1:
                            kde = gaussian_kde(sub_vals)
                            y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                            dens = kde(y_vals_dens)
                            if dens.max() > 0:
                                dens_scaled = dens / dens.max() * (box_width / 2.0)
                                x_curve = pos - dens_scaled
                                ax.plot(x_curve, y_vals_dens, color=c_map.get(var_name, base_palette[0]), linewidth=1.5)
                                # Optionally fill under the curve lightly
                                ax.fill_betweenx(
                                    y_vals_dens,
                                    pos,
                                    x_curve,
                                    color=c_map.get(var_name, base_palette[0]),
                                    alpha=0.2
                                )
                    # Draw boxplot with slightly narrower width
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=colour_mapping,
                        width=box_width * 0.6,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=False,
                        meanprops=meanprops
                    )
                    # Colour each box and set opacity
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    # Draw mean as a small square inside the box
                    for pos, cat in enumerate(var_scan_order):
                        sub_vals = df_plot[df_plot["VarScan"] == cat][col_name].dropna()
                        if sub_vals.empty:
                            continue
                        mean_val = sub_vals.mean()
                        var_name = cat.split()[0]
                        ax.scatter(
                            [pos], [mean_val],
                            s=marker_sz ** 2,
                            marker='s',
                            color='white',
                            edgecolor=c_map.get(var_name, base_palette[0]),
                            linewidth=1.0
                        )
                    # Overlay scatter points shifted to the right
                    if overlay_style != "None":
                        scatter_shift = box_width / 2.0
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + scatter_shift + j for p, j in zip(positions, jitter_offsets)]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                else:
                    # Fallback: if unknown template, use classic
                    sns.boxplot(
                        x="VarScan",
                        y=col_name,
                        data=df_plot,
                        order=var_scan_order,
                        ax=ax,
                        palette=colour_mapping,
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    for patch, cat in zip(ax.patches, var_scan_order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [var_scan_order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col_name].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            if overlay_style == "Scatter":
                                mshape = 'o'
                            else:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )

                # Common axis formatting applied regardless of template
                # X-axis labels: hide on the top row to prevent overlap with the legend.
                # On the bottom row, show 'Scan' if the global toggle is enabled; otherwise hide.
                if idx < 2:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(
                        "Scan" if show_x_label_global else "",
                        fontsize=font_size,
                        fontname=font_family,
                        fontstyle=('italic' if italic_axes else 'normal')
                    )
                # Y-axis labels: controlled by global toggle.  Use custom label if provided.
                ax.set_ylabel(
                    (custom_labels.get(metric_name, metric_name) if show_y_label_global else ""),
                    fontsize=font_size,
                    fontname=font_family,
                    fontstyle=('italic' if italic_axes else 'normal')
                )
                # Set custom x‑tick labels to only the scan portion (e.g. RV/FW)
                ax.set_xticks(range(len(var_scan_order)))
                ax.set_xticklabels([cat.split()[1] for cat in var_scan_order], rotation=x_axis_rotation)
                # Adjust tick label alignment if rotated
                if x_axis_rotation:
                    for tick in ax.get_xticklabels():
                        tick.set_ha('right')
                # Apply custom font to all tick labels
                ax.tick_params(axis='both', labelsize=font_size)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontname(font_family)
                # Apply axis limits if set
                if axis_limits.get(col_name):
                    ax.set_ylim(axis_limits[col_name])
            # Remove any individual axes legends to avoid duplication
            for ax in axes_flat:
                if hasattr(ax, 'get_legend') and ax.get_legend():
                    ax.get_legend().remove()
            # Create a global legend for variations if requested.  The legend is
            # drawn on the figure rather than individual axes and is
            # positioned outside the plotting area.  For top/bottom
            # placement, the legend is arranged horizontally with up to
            # four entries per row; for left/right placement, entries
            # are stacked vertically.
            if show_legend:
                # Determine which variations actually appear in the combined box plot.  A variation
                # appears only if it has at least one data point for the selected scans.  Build
                # this list in the order of ``selected_variations`` to maintain colour mapping.
                variations_present: List[str] = []
                for v in selected_variations:
                    # Check if this variation is represented in any of the plotted categories
                    if any(cat.split()[0] == v for cat in var_scan_order):
                        variations_present.append(v)
                legend_handles: List[plt.Line2D] = []
                legend_labels: List[str] = []
                for v in variations_present:
                    mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                    handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                        markerfacecolor=c_map[v], markeredgecolor=c_map[v], markersize=marker_sz,
                                        label=v)
                    legend_handles.append(handle)
                    legend_labels.append(v)
                # Limit the number of columns to avoid cramped legends.  If there
                # are more entries than max_cols, they will wrap to the next row.
                max_cols = 4
                if legend_position in ['top', 'bottom']:
                    ncol_legend = min(len(legend_labels), max_cols)
                else:
                    ncol_legend = 1
                # Compute dynamic offsets based on the legend_distance slider.  This
                # allows the user to control the space between the legend and
                # the plotting area.
                if legend_position == 'top':
                    loc = 'lower center'
                    xoff = 0.5
                    yoff = 1 + legend_distance
                elif legend_position == 'bottom':
                    loc = 'upper center'
                    xoff = 0.5
                    yoff = -legend_distance
                elif legend_position == 'left':
                    loc = 'center right'
                    xoff = -legend_distance
                    yoff = 0.5
                elif legend_position == 'right':
                    loc = 'center left'
                    xoff = 1 + legend_distance
                    yoff = 0.5
                else:
                    # Default to bottom if unknown
                    loc = 'upper center'
                    xoff = 0.5
                    yoff = -legend_distance
                fig_grid.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc=loc,
                    bbox_to_anchor=(xoff, yoff),
                    ncol=ncol_legend,
                    frameon=False,
                    fontsize=font_size
                )
            fig_grid.tight_layout()
            st.pyplot(fig_grid)
            # Offer download of the combined figure
            buf = io.BytesIO()
            fig_grid.savefig(buf, format=download_fmt, dpi=300)
            st.download_button(
                label="Download combined box plot",
                data=buf.getvalue(),
                file_name=f"{export_base}_combined_box_plot.{download_fmt}",
                mime=f"image/{download_fmt}"
            )
            plt.close(fig_grid)
        # Individual metric plots
        for metric in selected_metrics:
            col = metric_map[metric]
            if mode_choice == "Combine":
                # Combined scans in one plot per metric.  Each variation may have multiple scan directions (e.g. RV/FW).
                df_plot = data.copy()
                df_plot["VarScan"] = df_plot["Variation"] + " " + df_plot["Scan"]
                # Establish the order of categories for plotting (variation and scan)
                order = []
                for v in selected_variations:
                    for s in selected_scans:
                        category = f"{v} {s}"
                        if not df_plot[df_plot["VarScan"] == category].empty:
                            order.append(category)
                # If no categories, skip this metric
                if not order:
                    continue
                fig_metric, ax_metric = plt.subplots(figsize=(plot_width, plot_height))
                # Create palette mapping from category to variation colour
                palette_mapping = {cat: c_map[cat.split()[0]] for cat in order}
                # Plot based on template
                if template_choice == "Classic":
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=palette_mapping,
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    # Colour boxes
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        r, g, b = mcolors.to_rgb(colour)
                        mix_colour = tuple(0.6 * c + 0.4 * 1.0 for c in (r, g, b))
                        rgba = (*mix_colour, box_opacity)
                        patch.set_facecolor(rgba)
                    # Overlay points
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Raincloud":
                    # Draw symmetric density for each category
                    for pos, cat in enumerate(order):
                        var_name = cat.split()[0]
                        sub_vals = df_plot[df_plot["VarScan"] == cat][col].dropna()
                        if sub_vals.empty:
                            continue
                        try:
                            from scipy.stats import gaussian_kde
                        except Exception:
                            gaussian_kde = None
                        if gaussian_kde and len(sub_vals) > 1:
                            kde = gaussian_kde(sub_vals)
                            y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                            dens = kde(y_vals_dens)
                            if dens.max() > 0:
                                dens_scaled = dens / dens.max() * (box_width / 2.0)
                                x_left = pos - dens_scaled
                                x_right = pos + dens_scaled
                                ax_metric.fill_betweenx(
                                    y_vals_dens,
                                    x_left,
                                    x_right,
                                    color=c_map.get(var_name, base_palette[0]),
                                    alpha=0.4,
                                    linewidth=0
                                )
                    # Narrow central boxplot
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=palette_mapping,
                        width=box_width * 0.25,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * (box_width / 2.0)
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Outline":
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=["white"] * len(order),
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='white', markersize=marker_sz)
                    )
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba('white', alpha=box_opacity)
                        patch.set_facecolor(rgba)
                        patch.set_edgecolor(colour)
                        patch.set_linewidth(outline_px if show_boxes else 0)
                    overall_mean = df_plot[col].mean()
                    ax_metric.axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'D' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'diamond'), 'D')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Mono":
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=["white"] * len(order),
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        r, g, b = mcolors.to_rgb(colour)
                        grey = (0.8, 0.8, 0.8)
                        mix_colour = tuple(0.6 * c + 0.4 * g_ for c, g_ in zip((r, g, b), grey))
                        rgba = (*mix_colour, box_opacity)
                        patch.set_facecolor(rgba)
                        patch.set_edgecolor(colour)
                        patch.set_linewidth(outline_px if show_boxes else 0)
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Notched":
                    # Draw notched boxes and lighten the interior
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=palette_mapping,
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops,
                        notch=True
                    )
                    lighten_factor = 0.5
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        r, g, b = mcolors.to_rgb(colour)
                        mix_colour = tuple((1 - lighten_factor) * c + lighten_factor * 1.0 for c in (r, g, b))
                        rgba = (*mix_colour, box_opacity)
                        patch.set_facecolor(rgba)
                        patch.set_edgecolor(colour)
                        patch.set_linewidth(outline_px if show_boxes else 0)
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                elif template_choice == "Density":
                    # Density: density curves on the left, boxes and scatter with mean markers
                    for pos, cat in enumerate(order):
                        var_name = cat.split()[0]
                        sub_vals = df_plot[df_plot["VarScan"] == cat][col].dropna()
                        if sub_vals.empty:
                            continue
                        try:
                            from scipy.stats import gaussian_kde
                        except Exception:
                            gaussian_kde = None
                        if gaussian_kde and len(sub_vals) > 1:
                            kde = gaussian_kde(sub_vals)
                            y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                            dens = kde(y_vals_dens)
                            if dens.max() > 0:
                                dens_scaled = dens / dens.max() * (box_width / 2.0)
                                x_curve = pos - dens_scaled
                                ax_metric.plot(x_curve, y_vals_dens, color=c_map.get(var_name, base_palette[0]), linewidth=1.5)
                                ax_metric.fill_betweenx(
                                    y_vals_dens,
                                    pos,
                                    x_curve,
                                    color=c_map.get(var_name, base_palette[0]),
                                    alpha=0.2
                                )
                    # Boxplot narrower
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=palette_mapping,
                        width=box_width * 0.6,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=False
                    )
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    # Mean square marker
                    for pos, cat in enumerate(order):
                        sub_vals = df_plot[df_plot["VarScan"] == cat][col].dropna()
                        if sub_vals.empty:
                            continue
                        mean_val = sub_vals.mean()
                        var_name = cat.split()[0]
                        ax_metric.scatter(
                            [pos], [mean_val],
                            s=marker_sz ** 2,
                            marker='s',
                            color='white',
                            edgecolor=c_map.get(var_name, base_palette[0]),
                            linewidth=1.0
                        )
                    if overlay_style != "None":
                        scatter_shift = box_width / 2.0
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + scatter_shift + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                else:
                    # Unknown template fallback to Classic
                    sns.boxplot(
                        x="VarScan",
                        y=col,
                        data=df_plot,
                        order=order,
                        ax=ax_metric,
                        palette=palette_mapping,
                        width=box_width,
                        linewidth=(outline_px if show_boxes else 0),
                        flierprops=flierprops,
                        showmeans=True,
                        meanprops=meanprops
                    )
                    for patch, cat in zip(ax_metric.patches, order):
                        var_name = cat.split()[0]
                        colour = c_map.get(var_name, base_palette[0])
                        rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                        patch.set_facecolor(rgba)
                    if overlay_style != "None":
                        for v in selected_variations:
                            sub_v = df_plot[df_plot["Variation"] == v]
                            if sub_v.empty:
                                continue
                            positions = [order.index(f"{v} {s}") for s in sub_v["Scan"]]
                            y_vals = sub_v[col].tolist()
                            jitter_range = jitter_spread * box_width / 2.0
                            jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(positions))
                            x_jittered = [p + j for p, j in zip(positions, jitter_offsets)]
                            mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            ax_metric.scatter(
                                x_jittered,
                                y_vals,
                                s=marker_sz ** 2,
                                marker=mshape,
                                color=c_map[v],
                                alpha=marker_opacity,
                                edgecolors='none'
                            )
                # Common axis formatting for metric plots
                ax_metric.set_xlabel(
                    "Scan" if show_x_label_global else "",
                    fontsize=font_size,
                    fontname=font_family,
                    fontstyle=('italic' if italic_axes else 'normal')
                )
                ax_metric.set_ylabel(
                    (custom_labels.get(metric, metric) if show_y_label_global else ""),
                    fontsize=font_size,
                    fontname=font_family,
                    fontstyle=('italic' if italic_axes else 'normal')
                )
                if axis_limits.get(col):
                    ax_metric.set_ylim(axis_limits[col])
                ax_metric.set_xticks(range(len(order)))
                ax_metric.set_xticklabels([cat.split()[1] for cat in order], rotation=x_axis_rotation)
                if x_axis_rotation:
                    for tick in ax_metric.get_xticklabels():
                        tick.set_ha('right')
                ax_metric.tick_params(axis='both', labelsize=font_size)
                for tick in ax_metric.get_xticklabels() + ax_metric.get_yticklabels():
                    tick.set_fontname(font_family)
                # Remove default legend created by seaborn
                if ax_metric.get_legend():
                    ax_metric.get_legend().remove()
                # Create a legend for variations if requested.  For top/bottom
                # positions the legend is arranged horizontally with up to four
                # items per row.  For left/right positions the legend is
                # vertical.  The legend is drawn on the axes rather than the
                # figure, so we push it outside the plotting area via
                # bbox_to_anchor offsets.
                if show_legend:
                    legend_handles: List[plt.Line2D] = []
                    legend_labels: List[str] = []
                    for v in selected_variations:
                        if df_plot[df_plot["Variation"] == v].empty:
                            continue
                        mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                        handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                            markerfacecolor=c_map[v], markeredgecolor=c_map[v], markersize=marker_sz,
                                            label=v)
                        legend_handles.append(handle)
                        legend_labels.append(v)
                    # Limit columns for horizontal legends.  If more items exist
                    # than max_cols they will wrap to additional rows.
                    max_cols = 4
                    if legend_position in ['top', 'bottom']:
                        ncol_legend = min(len(legend_labels), max_cols)
                    else:
                        ncol_legend = 1
                    # Compute dynamic offsets based on legend_distance.  This
                    # slider controls the spacing between the legend and the plot.
                    if legend_position == 'top':
                        loc = 'lower center'
                        xoff = 0.5
                        yoff = 1 + legend_distance
                    elif legend_position == 'bottom':
                        loc = 'upper center'
                        xoff = 0.5
                        yoff = -legend_distance
                    elif legend_position == 'left':
                        loc = 'center right'
                        xoff = -legend_distance
                        yoff = 0.5
                    elif legend_position == 'right':
                        loc = 'center left'
                        xoff = 1 + legend_distance
                        yoff = 0.5
                    else:
                        loc = 'upper center'
                        xoff = 0.5
                        yoff = -legend_distance
                    ax_metric.legend(
                        handles=legend_handles,
                        labels=legend_labels,
                        loc=loc,
                        bbox_to_anchor=(xoff, yoff),
                        ncol=ncol_legend,
                        frameon=False,
                        fontsize=font_size
                    )
                fig_metric.tight_layout()
                st.pyplot(fig_metric)
                buf = io.BytesIO()
                fig_metric.savefig(buf, format=download_fmt, dpi=300)
                st.download_button(label=f"Download {metric} box plot", data=buf.getvalue(),
                                   file_name=f"{export_base}_{metric}_box_plot.{download_fmt}", mime=f"image/{download_fmt}")
                plt.close(fig_metric)
            else:
                # Separate plots per scan
                for s in selected_scans:
                    sub_df = data[data["Scan"] == s]
                    if sub_df.empty:
                        continue
                    fig_metric, ax_metric = plt.subplots(figsize=(plot_width, plot_height))
                    # Define the order of variations for this scan based on data availability
                    var_order = [v for v in selected_variations if not sub_df[sub_df["Variation"] == v].empty]
                    # Plot based on selected template
                    if template_choice == "Classic":
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette={v: c_map[v] for v in var_order},
                            width=box_width,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=True,
                            meanprops=meanprops
                        )
                        for patch, v in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v, base_palette[0])
                            r, g, b = mcolors.to_rgb(colour)
                            mix_colour = tuple(0.6 * c + 0.4 * 1.0 for c in (r, g, b))
                            rgba = (*mix_colour, box_opacity)
                            patch.set_facecolor(rgba)
                        if overlay_style != "None":
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * box_width / 2.0
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + j for j in jitter_offsets]
                                mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    elif template_choice == "Raincloud":
                        # Symmetric density + narrow box + scatter for each variation
                        for pos, v in enumerate(var_order):
                            sub_vals = sub_df[sub_df["Variation"] == v][col].dropna()
                            if sub_vals.empty:
                                continue
                            try:
                                from scipy.stats import gaussian_kde
                            except Exception:
                                gaussian_kde = None
                            if gaussian_kde and len(sub_vals) > 1:
                                kde = gaussian_kde(sub_vals)
                                y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                                dens = kde(y_vals_dens)
                                if dens.max() > 0:
                                    dens_scaled = dens / dens.max() * (box_width / 2.0)
                                    x_left = pos - dens_scaled
                                    x_right = pos + dens_scaled
                                    ax_metric.fill_betweenx(
                                        y_vals_dens,
                                        x_left,
                                        x_right,
                                        color=c_map.get(v, base_palette[0]),
                                        alpha=0.4,
                                        linewidth=0
                                    )
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette={v: c_map[v] for v in var_order},
                            width=box_width * 0.25,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=True,
                            meanprops=meanprops
                        )
                        for patch, v in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v, base_palette[0])
                            rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                            patch.set_facecolor(rgba)
                        if overlay_style != "None":
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * (box_width / 2.0)
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + j for j in jitter_offsets]
                                mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    elif template_choice == "Outline":
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette=["white"] * len(var_order),
                            width=box_width,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='white', markersize=marker_sz)
                        )
                        for patch, v in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v, base_palette[0])
                            rgba = mcolors.to_rgba('white', alpha=box_opacity)
                            patch.set_facecolor(rgba)
                            patch.set_edgecolor(colour)
                            patch.set_linewidth(outline_px if show_boxes else 0)
                        overall_mean = sub_df[col].mean()
                        ax_metric.axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
                        if overlay_style != "None":
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * box_width / 2.0
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + j for j in jitter_offsets]
                                mshape = 'D' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'diamond'), 'D')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    elif template_choice == "Mono":
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette=["white"] * len(var_order),
                            width=box_width,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=True,
                            meanprops=meanprops
                        )
                        for patch, v in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v, base_palette[0])
                            r, g, b = mcolors.to_rgb(colour)
                            grey = (0.8, 0.8, 0.8)
                            mix_colour = tuple(0.6 * c + 0.4 * g_ for c, g_ in zip((r, g, b), grey))
                            rgba = (*mix_colour, box_opacity)
                            patch.set_facecolor(rgba)
                            patch.set_edgecolor(colour)
                            patch.set_linewidth(outline_px if show_boxes else 0)
                        if overlay_style != "None":
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * box_width / 2.0
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + j for j in jitter_offsets]
                                mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    elif template_choice == "Notched":
                        # Notched boxes per variation: lighten interior and retain coloured edges
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette={v: c_map[v] for v in var_order},
                            width=box_width,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=True,
                            meanprops=meanprops,
                            notch=True
                        )
                        lighten_factor = 0.5
                        for patch, v_name in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v_name, base_palette[0])
                            r, g, b = mcolors.to_rgb(colour)
                            mix_colour = tuple((1 - lighten_factor) * c + lighten_factor * 1.0 for c in (r, g, b))
                            rgba = (*mix_colour, box_opacity)
                            patch.set_facecolor(rgba)
                            patch.set_edgecolor(colour)
                            patch.set_linewidth(outline_px if show_boxes else 0)
                        if overlay_style != "None":
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * box_width / 2.0
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + j for j in jitter_offsets]
                                mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    elif template_choice == "Density":
                        # Density: density curve to the left, narrow box, scatter right, mean square
                        for pos, v in enumerate(var_order):
                            sub_vals = sub_df[sub_df["Variation"] == v][col].dropna()
                            if sub_vals.empty:
                                continue
                            try:
                                from scipy.stats import gaussian_kde
                            except Exception:
                                gaussian_kde = None
                            if gaussian_kde and len(sub_vals) > 1:
                                kde = gaussian_kde(sub_vals)
                                y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                                dens = kde(y_vals_dens)
                                if dens.max() > 0:
                                    dens_scaled = dens / dens.max() * (box_width / 2.0)
                                    x_curve = pos - dens_scaled
                                    ax_metric.plot(x_curve, y_vals_dens, color=c_map.get(v, base_palette[0]), linewidth=1.5)
                                    ax_metric.fill_betweenx(
                                        y_vals_dens,
                                        pos,
                                        x_curve,
                                        color=c_map.get(v, base_palette[0]),
                                        alpha=0.2
                                    )
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette={v: c_map[v] for v in var_order},
                            width=box_width * 0.6,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=False
                        )
                        for patch, v in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v, base_palette[0])
                            rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                            patch.set_facecolor(rgba)
                        # Mean square markers
                        for pos, v in enumerate(var_order):
                            sub_vals = sub_df[sub_df["Variation"] == v][col].dropna()
                            if sub_vals.empty:
                                continue
                            mean_val = sub_vals.mean()
                            ax_metric.scatter(
                                [pos], [mean_val],
                                s=marker_sz ** 2,
                                marker='s',
                                color='white',
                                edgecolor=c_map.get(v, base_palette[0]),
                                linewidth=1.0
                            )
                        if overlay_style != "None":
                            scatter_shift = box_width / 2.0
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * box_width / 2.0
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + scatter_shift + j for j in jitter_offsets]
                                mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    else:
                        sns.boxplot(
                            x="Variation",
                            y=col,
                            data=sub_df,
                            ax=ax_metric,
                            order=var_order,
                            palette={v: c_map[v] for v in var_order},
                            width=box_width,
                            linewidth=(outline_px if show_boxes else 0),
                            flierprops=flierprops,
                            showmeans=True,
                            meanprops=meanprops
                        )
                        for patch, v in zip(ax_metric.patches, var_order):
                            colour = c_map.get(v, base_palette[0])
                            rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                            patch.set_facecolor(rgba)
                        if overlay_style != "None":
                            for v in var_order:
                                sub_v = sub_df[sub_df["Variation"] == v]
                                if sub_v.empty:
                                    continue
                                x_pos = var_order.index(v)
                                y_vals = sub_v[col].tolist()
                                jitter_range = jitter_spread * box_width / 2.0
                                jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                x_jittered = [x_pos + j for j in jitter_offsets]
                                mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                                ax_metric.scatter(
                                    x_jittered,
                                    y_vals,
                                    s=marker_sz ** 2,
                                    marker=mshape,
                                    color=c_map[v],
                                    alpha=marker_opacity,
                                    edgecolors='none'
                                )
                    # Common axis formatting for separate scans
                    ax_metric.set_xlabel(
                        "Variation" if show_x_label_global else "",
                        fontsize=font_size,
                        fontname=font_family,
                        fontstyle=('italic' if italic_axes else 'normal')
                    )
                    ax_metric.set_ylabel(
                        (custom_labels.get(metric, metric) if show_y_label_global else ""),
                        fontsize=font_size,
                        fontname=font_family,
                        fontstyle=('italic' if italic_axes else 'normal')
                    )
                    if axis_limits.get(col):
                        ax_metric.set_ylim(axis_limits[col])
                    ax_metric.set_xticks(range(len(var_order)))
                    ax_metric.set_xticklabels(var_order, rotation=x_axis_rotation)
                    if x_axis_rotation:
                        for tick in ax_metric.get_xticklabels():
                            tick.set_ha('right')
                    ax_metric.tick_params(axis='both', labelsize=font_size)
                    for tick in ax_metric.get_xticklabels() + ax_metric.get_yticklabels():
                        tick.set_fontname(font_family)
                    # Remove default legend
                    if ax_metric.get_legend():
                        ax_metric.get_legend().remove()
                    # Legend for variations if requested.
                    if show_legend:
                        legend_handles: List[plt.Line2D] = []
                        legend_labels: List[str] = []
                        for v in var_order:
                            mshape = marker_map_plotly_to_mpl.get(m_map.get(v, 'circle'), 'o')
                            handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                                markerfacecolor=c_map[v], markeredgecolor=c_map[v], markersize=marker_sz,
                                                label=v)
                            legend_handles.append(handle)
                            legend_labels.append(v)
                        max_cols = 4
                        if legend_position in ['top', 'bottom']:
                            ncol_legend = min(len(legend_labels), max_cols)
                        else:
                            ncol_legend = 1
                        if legend_position == 'top':
                            loc = 'lower center'
                            xoff = 0.5
                            yoff = 1 + legend_distance
                        elif legend_position == 'bottom':
                            loc = 'upper center'
                            xoff = 0.5
                            yoff = -legend_distance
                        elif legend_position == 'left':
                            loc = 'center right'
                            xoff = -legend_distance
                            yoff = 0.5
                        elif legend_position == 'right':
                            loc = 'center left'
                            xoff = 1 + legend_distance
                            yoff = 0.5
                        else:
                            loc = 'upper center'
                            xoff = 0.5
                            yoff = -legend_distance
                        ax_metric.legend(
                            handles=legend_handles,
                            labels=legend_labels,
                            loc=loc,
                            bbox_to_anchor=(xoff, yoff),
                            ncol=ncol_legend,
                            frameon=False,
                            fontsize=font_size
                        )
                    fig_metric.tight_layout()
                    st.pyplot(fig_metric)
                    buf = io.BytesIO()
                    fig_metric.savefig(buf, format=download_fmt, dpi=300)
                    st.download_button(label=f"Download {metric} ({s}) box plot", data=buf.getvalue(),
                                       file_name=f"{export_base}_{metric}_{s}_box_plot.{download_fmt}", mime=f"image/{download_fmt}")
                    plt.close(fig_metric)
        # Individual Variation Analysis
        if selected_variations:
            st.subheader("Individual Variation Analysis")
            indiv_var = st.selectbox("Select Individual Variation for Analysis", options=selected_variations, index=0)
            if indiv_var:
                var_df = data[data["Variation"] == indiv_var]
                if var_df.empty:
                    st.info("No data available for the selected variation.")
                else:
                    metric_tabs = st.tabs([m for m in selected_metrics])
                    for (tab, metric_name) in zip(metric_tabs, selected_metrics):
                        with tab:
                            col_name = metric_map[metric_name]
                            if col_name not in var_df.columns or var_df[col_name].dropna().empty:
                                st.write("No data for this metric.")
                                continue
                            fig_var, ax_var = plt.subplots(figsize=(plot_width, plot_height))
                            # Determine scan list order for this variation
                            scan_list = [s for s in selected_scans if not var_df[var_df["Scan"] == s].empty]
                            # Plot based on template
                            if template_choice == "Classic":
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    color=c_map.get(indiv_var, base_palette[0]),
                                    width=box_width,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=True,
                                    meanprops=meanprops
                                )
                                # Apply face colour opacity
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    r, g, b = mcolors.to_rgb(colour)
                                    mix_colour = tuple(0.6 * c + 0.4 * 1.0 for c in (r, g, b))
                                    rgba = (*mix_colour, box_opacity)
                                    patch.set_facecolor(rgba)
                                # Overlay points
                                if overlay_style != "None":
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * box_width / 2.0
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + j for j in jitter_offsets]
                                        mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            elif template_choice == "Raincloud":
                                # Symmetric density for each scan
                                for pos, s in enumerate(scan_list):
                                    sub_vals = var_df[var_df["Scan"] == s][col_name].dropna()
                                    if sub_vals.empty:
                                        continue
                                    try:
                                        from scipy.stats import gaussian_kde
                                    except Exception:
                                        gaussian_kde = None
                                    if gaussian_kde and len(sub_vals) > 1:
                                        kde = gaussian_kde(sub_vals)
                                        y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                                        dens = kde(y_vals_dens)
                                        if dens.max() > 0:
                                            dens_scaled = dens / dens.max() * (box_width / 2.0)
                                            x_left = pos - dens_scaled
                                            x_right = pos + dens_scaled
                                            ax_var.fill_betweenx(
                                                y_vals_dens,
                                                x_left,
                                                x_right,
                                                color=c_map.get(indiv_var, base_palette[0]),
                                                alpha=0.4,
                                                linewidth=0
                                            )
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    color=c_map.get(indiv_var, base_palette[0]),
                                    width=box_width * 0.25,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=True,
                                    meanprops=meanprops
                                )
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                                    patch.set_facecolor(rgba)
                                if overlay_style != "None":
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * (box_width / 2.0)
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + j for j in jitter_offsets]
                                        mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            elif template_choice == "Outline":
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    palette=["white"] * len(scan_list),
                                    width=box_width,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=True,
                                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='white', markersize=marker_sz)
                                )
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    rgba = mcolors.to_rgba('white', alpha=box_opacity)
                                    patch.set_facecolor(rgba)
                                    patch.set_edgecolor(colour)
                                    patch.set_linewidth(outline_px if show_boxes else 0)
                                overall_mean = var_df[col_name].mean()
                                ax_var.axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
                                if overlay_style != "None":
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * box_width / 2.0
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + j for j in jitter_offsets]
                                        mshape = 'D' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'diamond'), 'D')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            elif template_choice == "Mono":
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    palette=["white"] * len(scan_list),
                                    width=box_width,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=True,
                                    meanprops=meanprops
                                )
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    r, g, b = mcolors.to_rgb(colour)
                                    grey = (0.8, 0.8, 0.8)
                                    mix_colour = tuple(0.6 * c + 0.4 * g_ for c, g_ in zip((r, g, b), grey))
                                    rgba = (*mix_colour, box_opacity)
                                    patch.set_facecolor(rgba)
                                    patch.set_edgecolor(colour)
                                    patch.set_linewidth(outline_px if show_boxes else 0)
                                if overlay_style != "None":
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * box_width / 2.0
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + j for j in jitter_offsets]
                                        mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            elif template_choice == "Notched":
                                # Notched boxplot for each scan: lighten interior and keep coloured edges
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    color=c_map.get(indiv_var, base_palette[0]),
                                    width=box_width,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=True,
                                    meanprops=meanprops,
                                    notch=True
                                )
                                lighten_factor = 0.5
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    r, g, b = mcolors.to_rgb(colour)
                                    mix_colour = tuple((1 - lighten_factor) * c + lighten_factor * 1.0 for c in (r, g, b))
                                    rgba = (*mix_colour, box_opacity)
                                    patch.set_facecolor(rgba)
                                    patch.set_edgecolor(colour)
                                    patch.set_linewidth(outline_px if show_boxes else 0)
                                if overlay_style != "None":
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * box_width / 2.0
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + j for j in jitter_offsets]
                                        mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            elif template_choice == "Density":
                                # Density curves for each scan on the left side
                                for pos, s in enumerate(scan_list):
                                    sub_vals = var_df[var_df["Scan"] == s][col_name].dropna()
                                    if sub_vals.empty:
                                        continue
                                    try:
                                        from scipy.stats import gaussian_kde
                                    except Exception:
                                        gaussian_kde = None
                                    if gaussian_kde and len(sub_vals) > 1:
                                        kde = gaussian_kde(sub_vals)
                                        y_vals_dens = np.linspace(sub_vals.min(), sub_vals.max(), 200)
                                        dens = kde(y_vals_dens)
                                        if dens.max() > 0:
                                            dens_scaled = dens / dens.max() * (box_width / 2.0)
                                            x_curve = pos - dens_scaled
                                            ax_var.plot(x_curve, y_vals_dens, color=c_map.get(indiv_var, base_palette[0]), linewidth=1.5)
                                            ax_var.fill_betweenx(
                                                y_vals_dens,
                                                pos,
                                                x_curve,
                                                color=c_map.get(indiv_var, base_palette[0]),
                                                alpha=0.2
                                            )
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    color=c_map.get(indiv_var, base_palette[0]),
                                    width=box_width * 0.6,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=False
                                )
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                                    patch.set_facecolor(rgba)
                                # Mean square markers
                                for pos, s in enumerate(scan_list):
                                    sub_vals = var_df[var_df["Scan"] == s][col_name].dropna()
                                    if sub_vals.empty:
                                        continue
                                    mean_val = sub_vals.mean()
                                    ax_var.scatter(
                                        [pos], [mean_val],
                                        s=marker_sz ** 2,
                                        marker='s',
                                        color='white',
                                        edgecolor=c_map.get(indiv_var, base_palette[0]),
                                        linewidth=1.0
                                    )
                                if overlay_style != "None":
                                    scatter_shift = box_width / 2.0
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * box_width / 2.0
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + scatter_shift + j for j in jitter_offsets]
                                        mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            else:
                                # Fallback to Classic
                                sns.boxplot(
                                    x="Scan",
                                    y=col_name,
                                    data=var_df,
                                    ax=ax_var,
                                    order=scan_list,
                                    color=c_map.get(indiv_var, base_palette[0]),
                                    width=box_width,
                                    linewidth=(outline_px if show_boxes else 0),
                                    flierprops=flierprops,
                                    showmeans=True,
                                    meanprops=meanprops
                                )
                                for patch in ax_var.patches:
                                    colour = c_map.get(indiv_var, base_palette[0])
                                    rgba = mcolors.to_rgba(colour, alpha=box_opacity)
                                    patch.set_facecolor(rgba)
                                if overlay_style != "None":
                                    for s in scan_list:
                                        sub_s = var_df[var_df["Scan"] == s]
                                        if sub_s.empty:
                                            continue
                                        x_pos = scan_list.index(s)
                                        y_vals = sub_s[col_name].tolist()
                                        jitter_range = jitter_spread * box_width / 2.0
                                        jitter_offsets = np.random.uniform(-jitter_range, jitter_range, size=len(y_vals))
                                        x_jittered = [x_pos + j for j in jitter_offsets]
                                        mshape = 'o' if overlay_style == "Scatter" else marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                        ax_var.scatter(
                                            x_jittered,
                                            y_vals,
                                            s=marker_sz ** 2,
                                            marker=mshape,
                                            color=c_map.get(indiv_var, base_palette[0]),
                                            alpha=marker_opacity,
                                            edgecolors='none'
                                        )
                            # Common formatting for individual variation
                            ax_var.set_xlabel(
                                "Scan" if show_x_label_global else "",
                                fontsize=font_size,
                                fontname=font_family,
                                fontstyle=('italic' if italic_axes else 'normal')
                            )
                            ax_var.set_ylabel(
                                (custom_labels.get(metric_name, metric_name) if show_y_label_global else ""),
                                fontsize=font_size,
                                fontname=font_family,
                                fontstyle=('italic' if italic_axes else 'normal')
                            )
                            if axis_limits.get(col_name):
                                ax_var.set_ylim(axis_limits[col_name])
                            ax_var.set_xticks(range(len(scan_list)))
                            ax_var.set_xticklabels(scan_list, rotation=x_axis_rotation)
                            if x_axis_rotation:
                                for tick in ax_var.get_xticklabels():
                                    tick.set_ha('right')
                            ax_var.tick_params(axis='both', labelsize=font_size)
                            for tick in ax_var.get_xticklabels() + ax_var.get_yticklabels():
                                tick.set_fontname(font_family)
                            # Remove default legend
                            if ax_var.get_legend():
                                ax_var.get_legend().remove()
                            # Show legend if requested (single variation)
                            if show_legend:
                                mshape = marker_map_plotly_to_mpl.get(m_map.get(indiv_var, 'circle'), 'o')
                                handle = plt.Line2D([], [], linestyle='None', marker=mshape,
                                                    markerfacecolor=c_map.get(indiv_var, base_palette[0]),
                                                    markeredgecolor=c_map.get(indiv_var, base_palette[0]),
                                                    markersize=marker_sz,
                                                    label=indiv_var)
                                if legend_position == 'top':
                                    loc = 'lower center'
                                    xoff = 0.5
                                    yoff = 1 + legend_distance
                                elif legend_position == 'bottom':
                                    loc = 'upper center'
                                    xoff = 0.5
                                    yoff = -legend_distance
                                elif legend_position == 'left':
                                    loc = 'center right'
                                    xoff = -legend_distance
                                    yoff = 0.5
                                elif legend_position == 'right':
                                    loc = 'center left'
                                    xoff = 1 + legend_distance
                                    yoff = 0.5
                                else:
                                    loc = 'upper center'
                                    xoff = 0.5
                                    yoff = -legend_distance
                                ax_var.legend(
                                    handles=[handle],
                                    labels=[indiv_var],
                                    loc=loc,
                                    bbox_to_anchor=(xoff, yoff),
                                    ncol=1,
                                    frameon=False,
                                    fontsize=font_size
                                )
                            fig_var.tight_layout()
                            st.pyplot(fig_var)
                            buf = io.BytesIO()
                            fig_var.savefig(buf, format=download_fmt, dpi=300)
                            st.download_button(
                                label=f"Download {metric_name} plot for {indiv_var}",
                                data=buf.getvalue(),
                                file_name=f"{export_base}_{metric_name}_{indiv_var}_box_plot.{download_fmt}",
                                mime=f"image/{download_fmt}"
                            )
                            plt.close(fig_var)
    elif mode == "IPCE Curve":
        # (IPCE Curve code remains unchanged)
        st.sidebar.header("Upload IPCE Files")
        ipce_files = st.sidebar.file_uploader("IPCE .txt file(s)", type=["txt"], accept_multiple_files=True)
        # Convert the upload object to a list for extension
        # Build a list of IPCE files from user uploads.  Persist this list
        # across reruns using ``st.session_state`` so that selections are
        # preserved when switching between different plot modes.  If new
        # uploads occur, the cache is overwritten.  If no uploads occur, the
        # previously cached list is reused.  Only if both are empty do we
        # attempt to load files from the automatically detected folder cache.
        ipce_files_list: List = []
        if ipce_files:
            ipce_files_list.extend(list(ipce_files))
        # Initialise the cache on first run
        if 'ipce_files_cache' not in st.session_state:
            st.session_state['ipce_files_cache'] = []
        # Update the cache with newly uploaded files
        if ipce_files_list:
            st.session_state['ipce_files_cache'] = ipce_files_list
        # Retrieve the cached list for use
        ipce_files_combined = st.session_state['ipce_files_cache']
        # If nothing in cache and no uploads, try to populate from folder cache
        if not ipce_files_list and not ipce_files_combined:
            folder_cache = st.session_state.get('folder_data_cache', None)
            if folder_cache:
                for p in folder_cache.get('ipce', []):
                    try:
                        f = open(p, 'rb')
                        ipce_files_combined.append(f)
                    except Exception:
                        continue
                # Update the cached list after loading from folder cache
                st.session_state['ipce_files_cache'] = ipce_files_combined
        # Use the combined list for downstream logic
        ipce_files = st.session_state['ipce_files_cache']
        if not ipce_files:
            st.info("Upload IPCE data files to generate curves or place them alongside the script.")
            return
        ipce_mode = st.sidebar.radio(
            "Plot mode",
            ["Overlay all variations", "Separate plots"],
            index=0,
            key="ipce_plot_mode"
        )
        labels: List[str] = []
        colours: List[str] = []
        markers: List[str] = []
        # Wrap label and colour inputs in an expander so the sidebar remains clean by default.
        with st.sidebar.expander("Labels, Colours & Shapes", expanded=False):
            # Provide shape options for markers when using Matplotlib.  These shapes
            # influence the appearance of the IPCE curves in publication‑style
            # figures.  Users can select from common marker styles.
            # Marker options for per‑dataset customisation.  The first option
            # represents "No marker" (simple line) by using None.  Users can
            # choose from a variety of marker shapes for each dataset.  Adding
            # additional shapes helps create visually distinct curves in
            # publication‑style plots.
            marker_opts = [
                ("No marker", None),
                ("Circle", "o"),
                ("Square", "s"),
                ("Diamond", "D"),
                ("Triangle Up", "^"),
                ("Triangle Down", "v"),
                ("Triangle Left", "<"),
                ("Triangle Right", ">"),
                ("Pentagon", "p"),
                ("Hexagon", "h"),
                ("Star", "*"),
                ("X", "x"),
                ("Plus", "+")
            ]
            for i, f in enumerate(ipce_files, start=1):
                default_lbl = extract_variation(f.name)
                lbl = st.text_input(
                    f"Label for file {i} ({f.name})",
                    value=default_lbl,
                    key=f"ipce_lbl_{i}"
                )
                col = st.color_picker(
                    f"Colour for {lbl}",
                    base_palette[(i - 1) % len(base_palette)],
                    key=f"ipce_col_{i}"
                )
                # Select marker shape for the Matplotlib plot.  Use the displayed
                # name as the option but store the marker code internally.
                shape_names = [name for name, code in marker_opts]
                shape_selection = st.selectbox(
                    f"Marker for {lbl}",
                    options=shape_names,
                    index=0,
                    key=f"ipce_mark_{i}"
                )
                # Map the selected shape name back to its marker code
                marker_code = dict(marker_opts).get(shape_selection, "o")
                labels.append(lbl)
                colours.append(col)
                markers.append(marker_code)
        normalize_ipce = st.sidebar.checkbox(
            "Normalize IPCE curves to 100%",
            value=False,
            key="ipce_normalize_ipce"
        )
        normalize_jint = st.sidebar.checkbox(
            "Normalize integrated current",
            value=False,
            key="ipce_normalize_jint"
        )
        scale_factor = st.sidebar.number_input(
            "Apply scaling factor to integrated current (default = 1)",
            value=1.0,
            step=0.1,
            key="ipce_scale_factor"
        )
        # Capture the selected export format and preserve it across reruns
        ipce_export_fmt_sel = st.sidebar.selectbox(
            "Download format",
            ["PNG", "SVG"],
            index=0,
            key="ipce_export_fmt"
        )
        export_fmt = ipce_export_fmt_sel.lower()

        # The dashboard will render both Plotly and Matplotlib versions of each
        # IPCE curve.  No need for the user to choose a single library; both
        # visualisations are displayed concurrently.
        # Configuration for Matplotlib plots.  These settings are displayed
        # within an expander to reduce clutter.  They are only applied when
        # the Matplotlib plot library is selected.  Users can adjust figure
        # size, line width, font parameters, axis labels, axis limits, legend
        # location and choose from several style templates.
        with st.sidebar.expander("Matplotlib Plot Settings", expanded=False):
            # Available style templates
            style_templates = [
                "default",
                "seaborn-v0_8-darkgrid",
                "seaborn-v0_8-whitegrid",
                "ggplot",
                "Solarize_Light2"
            ]
            mpl_style = st.selectbox(
                "Style template",
                options=style_templates,
                index=0,
                key="ipce_mpl_style"
            )
            mpl_fig_width = st.slider(
                "Figure width (inches)",
                min_value=4,
                max_value=16,
                value=8,
                key="ipce_mpl_fig_width"
            )
            mpl_fig_height = st.slider(
                "Figure height (inches)",
                min_value=3,
                max_value=10,
                value=5,
                key="ipce_mpl_fig_height"
            )
            mpl_line_width = st.slider(
                "Line width (px)",
                min_value=1,
                max_value=5,
                value=2,
                key="ipce_mpl_line_width"
            )
            mpl_font_size = st.slider(
                "Font size", min_value=8, max_value=24, value=12, key="ipce_mpl_font_size"
            )
            mpl_font_family = st.selectbox(
                "Font family",
                options=["Arial", "Times New Roman", "Roboto", "Courier New"],
                index=0,
                key="ipce_mpl_font_family"
            )

            # ------------------------------------------------------------------
            # Additional Matplotlib settings: marker scheme, axis labels and limits,
            # legend configuration and X-axis limits.  These were previously
            # defined outside this expander; moving them here keeps all
            # Matplotlib configuration controls in one place.
            # Marker scheme: global templates override per‑dataset markers
            marker_templates = [
                "Custom (per dataset)",
                "Triangles",
                "Squares",
                "Diamonds",
                "Stars (glowing)"
            ]
            mpl_marker_template = st.selectbox(
                "Marker template",
                options=marker_templates,
                index=0,
                key="ipce_mpl_marker_template"
            )
            # Axis labels (X and both Y axes)
            default_y1_label = f"IPCE (%){' (norm.)' if normalize_ipce else ''}"
            default_y2_label = "J_int (mA/cm²)" if not normalize_jint else "J_int (norm.)"
            mpl_x_label = st.text_input(
                "X-axis label",
                value="Wavelength (nm)",
                key="ipce_mpl_x_label"
            )
            mpl_y1_label = st.text_input(
                "Left Y-axis label",
                value=default_y1_label,
                key="ipce_mpl_y1_label"
            )
            mpl_y2_label = st.text_input(
                "Right Y-axis label",
                value=default_y2_label,
                key="ipce_mpl_y2_label"
            )
            # Axis limits for the left Y axis
            y1_lim_set = st.checkbox(
                "Set left Y-axis limits",
                value=False,
                key="ipce_mpl_y1_lim_set"
            )
            if y1_lim_set:
                y1_min = st.number_input(
                    "Y1 minimum",
                    value=0.0,
                    key="ipce_mpl_y1_min"
                )
                y1_max = st.number_input(
                    "Y1 maximum",
                    value=100.0,
                    key="ipce_mpl_y1_max"
                )
            else:
                y1_min = None
                y1_max = None
            # Axis limits for the right Y axis
            y2_lim_set = st.checkbox(
                "Set right Y-axis limits",
                value=False,
                key="ipce_mpl_y2_lim_set"
            )
            if y2_lim_set:
                y2_min = st.number_input(
                    "Y2 minimum",
                    value=0.0,
                    key="ipce_mpl_y2_min"
                )
                y2_max = st.number_input(
                    "Y2 maximum",
                    value=100.0,
                    key="ipce_mpl_y2_max"
                )
            else:
                y2_min = None
                y2_max = None
            # Legend positioning and configuration.  Additional controls for
            # horizontal legends (top/bottom) allow the number of columns and
            # spacing between columns to be specified.
            # Legend positioning and configuration.  The default selection is now
            # "Outside top" so that the legend sits above the plot without
            # encroaching on the axes area.  Users can still choose other
            # positions as needed.  Setting the ``index`` argument here
            # determines the initial state prior to any session_state override.
            legend_mode = st.selectbox(
                "Legend position",
                [
                    "Inside plot",
                    "Outside left",
                    "Outside right",
                    "Outside top",
                    "Outside bottom"
                ],
                index=3,  # Default to "Outside top"
                key="ipce_mpl_legend_mode"
            )
            # Initialise legend parameters
            legend_x = None
            legend_y = None
            legend_spacing = None
            legend_cols = 1
            legend_col_spacing = 0.2
            if legend_mode == "Inside plot":
                legend_x = st.slider(
                    "Legend X position (0–1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    key="ipce_mpl_legend_x"
                )
                legend_y = st.slider(
                    "Legend Y position (0–1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    key="ipce_mpl_legend_y"
                )
            elif legend_mode in ["Outside top", "Outside bottom"]:
                legend_cols = st.slider(
                    "Legend columns",
                    min_value=1,
                    max_value=6,
                    value=2,
                    key="ipce_mpl_legend_cols"
                )
                legend_col_spacing = st.slider(
                    "Legend column spacing",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key="ipce_mpl_legend_col_spacing"
                )
                legend_spacing = st.slider(
                    "Legend spacing from plot",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    key="ipce_mpl_legend_spacing"
                )
            else:
                legend_spacing = st.slider(
                    "Legend spacing from plot",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    key="ipce_mpl_legend_spacing"
                )
            # X-axis limits
            x_lim_set = st.checkbox(
                "Set X-axis limits",
                value=False,
                key="ipce_mpl_x_lim_set"
            )
            if x_lim_set:
                x_min = st.number_input(
                    "X minimum",
                    value=0.0,
                    key="ipce_mpl_x_min"
                )
                x_max = st.number_input(
                    "X maximum",
                    value=0.0,
                    key="ipce_mpl_x_max"
                )
            else:
                x_min = None
                x_max = None
        # Allow the user to choose a global marker template.  When set to
        # "Custom (per dataset)", the individual marker selections defined in
        # the "Labels, Colours & Shapes" section are used.  Otherwise, all
        # curves will use a predefined marker scheme suitable for publication
        # (e.g. triangles, squares, diamonds or stars with glow).  These
        # templates override per‑dataset marker selections to ensure
        # consistency across all curves.
        # Retrieve previously selected Matplotlib settings from session_state.
        # These assignments avoid rendering duplicate controls outside the expander.
        mpl_marker_template = st.session_state.get("ipce_mpl_marker_template", "Custom (per dataset)")
        default_y1_label = f"IPCE (%){' (norm.)' if normalize_ipce else ''}"
        default_y2_label = "J_int (mA/cm²)" if not normalize_jint else "J_int (norm.)"
        mpl_x_label = st.session_state.get("ipce_mpl_x_label", "Wavelength (nm)")
        mpl_y1_label = st.session_state.get("ipce_mpl_y1_label", default_y1_label)
        mpl_y2_label = st.session_state.get("ipce_mpl_y2_label", default_y2_label)
        y1_lim_set = st.session_state.get("ipce_mpl_y1_lim_set", False)
        if y1_lim_set:
            y1_min = st.session_state.get("ipce_mpl_y1_min")
            y1_max = st.session_state.get("ipce_mpl_y1_max")
        else:
            y1_min = None
            y1_max = None
        y2_lim_set = st.session_state.get("ipce_mpl_y2_lim_set", False)
        if y2_lim_set:
            y2_min = st.session_state.get("ipce_mpl_y2_min")
            y2_max = st.session_state.get("ipce_mpl_y2_max")
        else:
            y2_min = None
            y2_max = None
        # Retrieve the legend mode from session state; default to "Outside top" so
        # that, on first load, the legend does not overlap the plotting area.
        legend_mode = st.session_state.get("ipce_mpl_legend_mode", "Outside top")
        legend_x = st.session_state.get("ipce_mpl_legend_x")
        legend_y = st.session_state.get("ipce_mpl_legend_y")
        legend_spacing = st.session_state.get("ipce_mpl_legend_spacing")
        # Retrieve additional legend parameters if defined
        legend_cols = st.session_state.get("ipce_mpl_legend_cols", 1)
        legend_col_spacing = st.session_state.get("ipce_mpl_legend_col_spacing", 0.2)
        x_lim_set = st.session_state.get("ipce_mpl_x_lim_set", False)
        if x_lim_set:
            x_min = st.session_state.get("ipce_mpl_x_min")
            x_max = st.session_state.get("ipce_mpl_x_max")
        else:
            x_min = None
            x_max = None
        mpl_settings = {
            "style": mpl_style,
            "fig_width": mpl_fig_width,
            "fig_height": mpl_fig_height,
            "line_width": mpl_line_width,
            "font_size": mpl_font_size,
            "font_family": mpl_font_family,
            "x_label": mpl_x_label,
            "y1_label": mpl_y1_label,
            "y2_label": mpl_y2_label,
            "y1_min": y1_min,
            "y1_max": y1_max,
            "y2_min": y2_min,
            "y2_max": y2_max,
            "y1_lim_set": y1_lim_set,
            "y2_lim_set": y2_lim_set,
            "x_min": x_min,
            "x_max": x_max,
            "x_lim_set": x_lim_set,
            "legend_mode": legend_mode,
            "legend_x": legend_x,
            "legend_y": legend_y,
            "legend_spacing": legend_spacing,
            "legend_cols": legend_cols,
            "legend_col_spacing": legend_col_spacing,
            "marker_template": mpl_marker_template,
        }
        export_name = st.sidebar.text_input(
            "Export file name base",
            value="ipce_plot",
            key="ipce_export_name"
        )

        # ------------------------------------------------------------------
        # Plotly customisation settings for IPCE curves
        #
        # These controls allow the user to tweak the appearance of Plotly
        # IPCE curves, including legend placement, font family/size, axis
        # limits and figure margins.  When expanded, the settings persist
        # across reruns via ``st.session_state``.
        with st.sidebar.expander("Plotly Plot Settings (IPCE)", expanded=False):
            # Legend position options.  The default is outside the top of
            # the figure to keep the legend clear of the data area.  The
            # "Inside" option uses Plotly's automatic positioning.
            plotly_ipce_legend_loc = st.selectbox(
                "Legend location",
                ["Outside top", "Outside bottom", "Outside left", "Outside right", "Inside"],
                index=0,
                key="ipce_plotly_legend_loc"
            )
            # Legend orientation: horizontal or vertical
            plotly_ipce_legend_orient = st.selectbox(
                "Legend orientation",
                ["Horizontal", "Vertical"],
                index=0,
                key="ipce_plotly_legend_orient"
            )
            # Font customisation
            plotly_ipce_font_size = st.number_input(
                "Font size",
                value=font_size_global,
                min_value=6,
                max_value=32,
                step=1,
                key="ipce_plotly_font_size"
            )
            plotly_ipce_font_family = st.text_input(
                "Font family",
                value=font_family_global,
                key="ipce_plotly_font_family"
            )
            # Axis limits for wavelength (X axis)
            plotly_ipce_x_lim_set = st.checkbox(
                "Set X-axis limits (Wavelength)",
                value=False,
                key="ipce_plotly_x_lim_set"
            )
            if plotly_ipce_x_lim_set:
                plotly_ipce_x_min = st.number_input(
                    "X minimum (nm)",
                    value=0.0,
                    key="ipce_plotly_x_min"
                )
                plotly_ipce_x_max = st.number_input(
                    "X maximum (nm)",
                    value=0.0,
                    key="ipce_plotly_x_max"
                )
            else:
                plotly_ipce_x_min = None
                plotly_ipce_x_max = None
            # Y1 axis limits (IPCE)
            plotly_ipce_y1_lim_set = st.checkbox(
                "Set IPCE Y-axis limits",
                value=False,
                key="ipce_plotly_y1_lim_set"
            )
            if plotly_ipce_y1_lim_set:
                plotly_ipce_y1_min = st.number_input(
                    "IPCE Y minimum",
                    value=0.0,
                    key="ipce_plotly_y1_min"
                )
                plotly_ipce_y1_max = st.number_input(
                    "IPCE Y maximum",
                    value=100.0,
                    key="ipce_plotly_y1_max"
                )
            else:
                plotly_ipce_y1_min = None
                plotly_ipce_y1_max = None
            # Y2 axis limits (J_int)
            plotly_ipce_y2_lim_set = st.checkbox(
                "Set J_int Y-axis limits",
                value=False,
                key="ipce_plotly_y2_lim_set"
            )
            if plotly_ipce_y2_lim_set:
                plotly_ipce_y2_min = st.number_input(
                    "J_int Y minimum",
                    value=0.0,
                    key="ipce_plotly_y2_min"
                )
                plotly_ipce_y2_max = st.number_input(
                    "J_int Y maximum",
                    value=100.0,
                    key="ipce_plotly_y2_max"
                )
            else:
                plotly_ipce_y2_min = None
                plotly_ipce_y2_max = None
            # Figure margins
            plotly_ipce_margin_left = st.number_input(
                "Left margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="ipce_plotly_margin_left"
            )
            plotly_ipce_margin_right = st.number_input(
                "Right margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="ipce_plotly_margin_right"
            )
            plotly_ipce_margin_bottom = st.number_input(
                "Bottom margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="ipce_plotly_margin_bottom"
            )
            plotly_ipce_margin_top = st.number_input(
                "Top margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="ipce_plotly_margin_top"
            )
        # End of Plotly customisation expander
        # Build a list of parsed IPCE dataframes.  Persist parsed results by file
        # name in ``st.session_state['ipce_data_cache']`` to avoid re‑parsing files
        # when switching between plot modes.  If a file has already been parsed,
        # reuse the cached dataframe.  Apply the user’s scaling factor on a
        # temporary copy so the cached original remains unchanged.
        curves = []
        # Initialise the IPCE data cache if it does not exist
        if 'ipce_data_cache' not in st.session_state:
            st.session_state['ipce_data_cache'] = {}
        for f in ipce_files:
            fname = getattr(f, 'name', None)
            # Use cached dataframe if available
            if fname in st.session_state['ipce_data_cache']:
                df_cached = st.session_state['ipce_data_cache'][fname]
                df = df_cached.copy()
            else:
                df = parse_ipce(f)
                # Store the original dataframe in the cache for subsequent runs
                st.session_state['ipce_data_cache'][fname] = df.copy()
                df = df.copy()
            # Apply scaling factor to integrated current on a per‑run copy
            if not df.empty and "J_integrated" in df.columns and scale_factor != 1.0:
                df["J_integrated"] = df["J_integrated"] * scale_factor
            curves.append(df)
        # Display parsed IPCE files within a collapsible section.  Each uploaded
        # file is shown under its label for inspection along with a checkbox to
        # include or exclude the curve from the overlay plot.  Collapsing
        # the outer expander provides a cleaner UI by default.
        with st.expander("Parsed IPCE File (Before Plotting)", expanded=False):
            for idx, (lbl, df) in enumerate(zip(labels, curves)):
                # Checkbox to select whether this curve should be included in the overlay
                chk_key = f"ipce_overlay_include_{idx}"
                include_default = True
                st.checkbox(
                    f"Include {lbl}",
                    value=include_default,
                    key=chk_key
                )
                with st.expander(lbl, expanded=False):
                    st.dataframe(df)
        if ipce_mode == "Overlay all variations":
            # Determine which curves to include in the overlay using the
            # dataset‑level checkboxes defined in the parsed data section.  If
            # nothing is selected, default to including all curves.  The
            # `st.session_state` dictionary holds the checkbox values keyed by
            # ``ipce_overlay_include_{idx}``.
            selected_idx: List[int] = [
                i for i in range(len(curves))
                if st.session_state.get(f"ipce_overlay_include_{i}", True)
            ]
            if not selected_idx:
                selected_idx = list(range(len(curves)))
            # --- Plotly overlay ---
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for i, df in enumerate(curves):
                if i not in selected_idx or df.empty:
                    continue
                ipce_y = df["IPCE"]
                jint_y = df["J_integrated"] if "J_integrated" in df.columns else None
                if normalize_ipce and not ipce_y.empty:
                    ipce_y = ipce_y / ipce_y.max() * 100.0
                if jint_y is not None and normalize_jint and not jint_y.empty:
                    jint_y = jint_y / jint_y.max() * 100.0
                fig.add_trace(
                    go.Scatter(
                        x=df["Wavelength"],
                        y=ipce_y,
                        name=f"{labels[i]} IPCE",
                        line=dict(color=colours[i], width=3)
                    ),
                    secondary_y=False
                )
                if jint_y is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df["Wavelength"],
                            y=jint_y,
                            name=f"{labels[i]} J_int",
                            line=dict(color=colours[i], width=3, dash="dash")
                        ),
                        secondary_y=True
                    )
            fig.update_xaxes(title_text="Wavelength (nm)")
            fig.update_yaxes(title_text="IPCE (%)" + (" (norm.)" if normalize_ipce else ""), secondary_y=False)
            fig.update_yaxes(title_text=("J_int (mA/cm²)" if not normalize_jint else "J_int (norm.)"), secondary_y=True)
            # Determine legend positioning based on the user's Plotly settings.
            # The dictionary below maps each location option to appropriate
            # anchor settings for Plotly legends.  Orientation is chosen
            # separately.
            legend_orientation = "h" if plotly_ipce_legend_orient == "Horizontal" else "v"
            if plotly_ipce_legend_loc == "Outside top":
                legend_dict = dict(
                    orientation=legend_orientation,
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                )
            elif plotly_ipce_legend_loc == "Outside bottom":
                legend_dict = dict(
                    orientation=legend_orientation,
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                )
            elif plotly_ipce_legend_loc == "Outside left":
                legend_dict = dict(
                    orientation=legend_orientation,
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=-0.02,
                )
            elif plotly_ipce_legend_loc == "Outside right":
                legend_dict = dict(
                    orientation=legend_orientation,
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                )
            else:  # Inside (default automatic)
                legend_dict = dict(orientation=legend_orientation)
            fig.update_layout(
                title="IPCE and Integrated Current vs Wavelength",
                template=template_name,
                legend=legend_dict,
                margin=dict(
                    l=plotly_ipce_margin_left,
                    r=plotly_ipce_margin_right,
                    b=plotly_ipce_margin_bottom,
                    t=plotly_ipce_margin_top,
                ),
                font=dict(size=plotly_ipce_font_size, family=plotly_ipce_font_family),
                title_font=dict(size=plotly_ipce_font_size + 2, family=plotly_ipce_font_family)
            )
            # Apply axis limits if the user has specified them
            if plotly_ipce_x_min is not None and plotly_ipce_x_max is not None:
                fig.update_xaxes(range=[plotly_ipce_x_min, plotly_ipce_x_max])
            if plotly_ipce_y1_min is not None and plotly_ipce_y1_max is not None:
                fig.update_yaxes(range=[plotly_ipce_y1_min, plotly_ipce_y1_max], secondary_y=False)
            if plotly_ipce_y2_min is not None and plotly_ipce_y2_max is not None:
                fig.update_yaxes(range=[plotly_ipce_y2_min, plotly_ipce_y2_max], secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
            st.download_button(
                label="Download IPCE plot",
                data=img,
                file_name=f"{export_name}.{export_fmt}",
                mime=f"image/{export_fmt}",
                key="download_ipce_overlay"
            )
            # --- Matplotlib overlay ---
            plt.style.use(mpl_settings["style"])
            # Prepare Matplotlib figure and axes with a dedicated legend axes when placing
            # the legend outside of the plot.  This prevents the legend from
            # compressing the plotting area by reserving a separate subplot for it.
            legend_mode = mpl_settings.get("legend_mode", "Outside right")
            # Use the user provided legend_spacing ratio when available; default to 0.2
            spacing = mpl_settings.get("legend_spacing", 0.2) if mpl_settings.get("legend_spacing") is not None else 0.2
            if legend_mode == "Inside plot":
                # Single axes when legend is inside the plot area
                fig2, ax1 = plt.subplots(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                ax2 = ax1.twinx()
                legend_ax = None
            else:
                # Create a GridSpec layout to allocate space for the legend outside the plot.
                if legend_mode in ["Outside right", "Outside left"]:
                    fig2 = plt.figure(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                    # Define column widths based on the spacing ratio.  The legend occupies ``spacing``
                    # fraction of the width, leaving the remainder for the plot.  If the legend is on
                    # the right, the order of subplots is [plot, legend]; if on the left, [legend, plot].
                    if legend_mode == "Outside right":
                        gs = fig2.add_gridspec(1, 2, width_ratios=[1 - spacing, spacing])
                        ax_main = fig2.add_subplot(gs[0])
                        legend_ax = fig2.add_subplot(gs[1])
                    else:
                        gs = fig2.add_gridspec(1, 2, width_ratios=[spacing, 1 - spacing])
                        legend_ax = fig2.add_subplot(gs[0])
                        ax_main = fig2.add_subplot(gs[1])
                    ax1 = ax_main
                    ax2 = ax1.twinx()
                    legend_ax.axis('off')
                elif legend_mode in ["Outside top", "Outside bottom"]:
                    fig2 = plt.figure(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                    # Define row heights for top or bottom legend.  The legend occupies ``spacing``
                    # fraction of the height.  For a top legend, the order is [legend, plot]; for
                    # a bottom legend, [plot, legend].
                    if legend_mode == "Outside top":
                        gs = fig2.add_gridspec(2, 1, height_ratios=[spacing, 1 - spacing])
                        legend_ax = fig2.add_subplot(gs[0])
                        ax_main = fig2.add_subplot(gs[1])
                    else:
                        gs = fig2.add_gridspec(2, 1, height_ratios=[1 - spacing, spacing])
                        ax_main = fig2.add_subplot(gs[0])
                        legend_ax = fig2.add_subplot(gs[1])
                    ax1 = ax_main
                    ax2 = ax1.twinx()
                    legend_ax.axis('off')
                else:
                    # Fallback to a single axes if an unknown legend mode is provided
                    fig2, ax1 = plt.subplots(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                    ax2 = ax1.twinx()
                    legend_ax = None
            # Determine marker assignments based on the selected template.  If
            # "Custom (per dataset)" is chosen, use the per‑file marker
            # selections stored in the ``markers`` list.  Otherwise, cycle
            # through a predefined set of markers appropriate to the chosen
            # template.  A glow effect is applied when the template is
            # "Stars (glowing)".
            marker_template = mpl_settings.get("marker_template", "Custom (per dataset)")
            if marker_template != "Custom (per dataset)":
                if marker_template == "Triangles":
                    templ_markers = ["^", "v", "<", ">"]
                elif marker_template == "Squares":
                    templ_markers = ["s"]
                elif marker_template == "Diamonds":
                    templ_markers = ["D"]
                elif marker_template == "Stars (glowing)":
                    templ_markers = ["*", "X"]
                else:
                    templ_markers = ["o"]
                marker_cycle = itertools.cycle(templ_markers)
                markers_overlay = [next(marker_cycle) for _ in selected_idx]
            else:
                markers_overlay = [markers[i] for i in selected_idx]
            # Loop over selected curves and plot IPCE and J_int on twin axes
            for count, i in enumerate(selected_idx):
                df = curves[i]
                if df.empty:
                    continue
                ipce_y = df["IPCE"]
                jint_y = df["J_integrated"] if "J_integrated" in df.columns else None
                if normalize_ipce and not ipce_y.empty:
                    ipce_y = ipce_y / ipce_y.max() * 100.0
                if jint_y is not None and normalize_jint and not jint_y.empty:
                    jint_y = jint_y / jint_y.max() * 100.0
                # Determine marker for this curve
                this_marker = markers_overlay[count]
                # Plot IPCE on left axis
                line1, = ax1.plot(
                    df["Wavelength"],
                    ipce_y,
                    label=f"{labels[i]} IPCE",
                    color=colours[i],
                    linewidth=mpl_settings["line_width"],
                    marker=this_marker
                )
                # Plot J_int on right axis if available
                line2 = None
                if jint_y is not None:
                    line2, = ax2.plot(
                        df["Wavelength"],
                        jint_y,
                        label=f"{labels[i]} J_int",
                        color=colours[i],
                        linewidth=mpl_settings["line_width"],
                        linestyle="--",
                        marker=this_marker
                    )
                # Apply glow effect to both lines for the stars template
                if marker_template == "Stars (glowing)":
                    glow = [patheffects.Stroke(linewidth=4, foreground="white"), patheffects.Normal()]
                    line1.set_path_effects(glow)
                    if line2 is not None:
                        line2.set_path_effects(glow)
            # Axis labels and limits
            ax1.set_xlabel(mpl_settings["x_label"], fontsize=mpl_settings["font_size"], fontname=mpl_settings["font_family"])
            ax1.set_ylabel(mpl_settings["y1_label"], fontsize=mpl_settings["font_size"], fontname=mpl_settings["font_family"])
            ax2.set_ylabel(mpl_settings["y2_label"], fontsize=mpl_settings["font_size"], fontname=mpl_settings["font_family"])
            if mpl_settings["y1_lim_set"] and mpl_settings["y1_min"] is not None and mpl_settings["y1_max"] is not None:
                ax1.set_ylim(mpl_settings["y1_min"], mpl_settings["y1_max"])
            if mpl_settings["y2_lim_set"] and mpl_settings["y2_min"] is not None and mpl_settings["y2_max"] is not None:
                ax2.set_ylim(mpl_settings["y2_min"], mpl_settings["y2_max"])
            # Apply X-axis limits if requested
            if mpl_settings.get("x_lim_set") and mpl_settings.get("x_min") is not None and mpl_settings.get("x_max") is not None:
                ax1.set_xlim(mpl_settings["x_min"], mpl_settings["x_max"])
                ax2.set_xlim(mpl_settings["x_min"], mpl_settings["x_max"])
            # Legend combining both axes with configurable placement
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # Place the legend either inside the plot or on a dedicated legend axes
            if legend_mode == "Inside plot" or legend_ax is None:
                lx = mpl_settings.get("legend_x", 0.8)
                ly = mpl_settings.get("legend_y", 0.9)
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc="center",
                    bbox_to_anchor=(lx, ly),
                    prop={"size": mpl_settings["font_size"], "family": mpl_settings["font_family"]}
                )
                fig2.tight_layout()
            else:
                # Outside legends use the reserved legend axes.  For top/bottom legends,
                # use multiple columns if specified; left/right legends remain vertical.
                if legend_mode in ["Outside top", "Outside bottom"]:
                    ncols = mpl_settings.get("legend_cols", 1)
                    col_spacing = mpl_settings.get("legend_col_spacing", 0.2)
                    legend_ax.legend(
                        lines1 + lines2,
                        labels1 + labels2,
                        loc="center",
                        ncol=ncols,
                        columnspacing=col_spacing,
                        prop={"size": mpl_settings["font_size"], "family": mpl_settings["font_family"]}
                    )
                else:
                    legend_ax.legend(
                        lines1 + lines2,
                        labels1 + labels2,
                        loc="center",
                        prop={"size": mpl_settings["font_size"], "family": mpl_settings["font_family"]}
                    )
                # Apply tight_layout to prevent overlap between the plot and legend axes
                fig2.tight_layout()
            st.pyplot(fig2)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format=export_fmt, dpi=300)
            st.download_button(
                label="Download IPCE plot (Matplotlib)",
                data=buf2.getvalue(),
                file_name=f"{export_name}_mat.{export_fmt}",
                mime=f"image/{export_fmt}",
                key="download_ipce_overlay_mat"
            )
            plt.close(fig2)
        else:
            tabs = st.tabs([lbl for lbl in labels])
            for i, (tab, df) in enumerate(zip(tabs, curves)):
                with tab:
                    if df.empty:
                        st.info("No data parsed.")
                        continue
                    # Plot each IPCE curve separately and display both Plotly and Matplotlib versions.
                    # --- Plotly separate plot ---
                    fig_sep = make_subplots(specs=[[{"secondary_y": True}]])
                    ipce_y = df["IPCE"].copy()
                    jint_y = df["J_integrated"].copy() if "J_integrated" in df.columns else None
                    if normalize_ipce and not ipce_y.empty:
                        ipce_y = ipce_y / ipce_y.max() * 100.0
                    if jint_y is not None and normalize_jint and not jint_y.empty:
                        jint_y = jint_y / jint_y.max() * 100.0
                    fig_sep.add_trace(
                        go.Scatter(
                            x=df["Wavelength"],
                            y=ipce_y,
                            name="IPCE",
                            line=dict(color=colours[i], width=3)
                        ),
                        secondary_y=False
                    )
                    if jint_y is not None:
                        fig_sep.add_trace(
                            go.Scatter(
                                x=df["Wavelength"],
                                y=jint_y,
                                name="J_int",
                                line=dict(color=colours[i], width=3, dash="dash")
                            ),
                            secondary_y=True
                        )
                    fig_sep.update_xaxes(title_text="Wavelength (nm)")
                    fig_sep.update_yaxes(title_text="IPCE (%)" + (" (norm.)" if normalize_ipce else ""), secondary_y=False)
                    fig_sep.update_yaxes(title_text=("J_int (mA/cm²)" if not normalize_jint else "J_int (norm.)"), secondary_y=True)
                    # Apply user customisations to the Plotly IPCE separate plot.
                    # Determine legend placement similar to the overlay case.
                    legend_orientation_sep = "h" if plotly_ipce_legend_orient == "Horizontal" else "v"
                    if plotly_ipce_legend_loc == "Outside top":
                        legend_sep = dict(
                            orientation=legend_orientation_sep,
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        )
                    elif plotly_ipce_legend_loc == "Outside bottom":
                        legend_sep = dict(
                            orientation=legend_orientation_sep,
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                        )
                    elif plotly_ipce_legend_loc == "Outside left":
                        legend_sep = dict(
                            orientation=legend_orientation_sep,
                            yanchor="middle",
                            y=0.5,
                            xanchor="right",
                            x=-0.02,
                        )
                    elif plotly_ipce_legend_loc == "Outside right":
                        legend_sep = dict(
                            orientation=legend_orientation_sep,
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.02,
                        )
                    else:
                        legend_sep = dict(orientation=legend_orientation_sep)
                    fig_sep.update_layout(
                        title=f"{labels[i]}: IPCE and Integrated Current",
                        template=template_name,
                        legend=legend_sep,
                        margin=dict(
                            l=plotly_ipce_margin_left,
                            r=plotly_ipce_margin_right,
                            b=plotly_ipce_margin_bottom,
                            t=plotly_ipce_margin_top,
                        ),
                        font=dict(size=plotly_ipce_font_size, family=plotly_ipce_font_family),
                        title_font=dict(size=plotly_ipce_font_size + 2, family=plotly_ipce_font_family)
                    )
                    # Apply axis limits when specified
                    if plotly_ipce_x_min is not None and plotly_ipce_x_max is not None:
                        fig_sep.update_xaxes(range=[plotly_ipce_x_min, plotly_ipce_x_max])
                    if plotly_ipce_y1_min is not None and plotly_ipce_y1_max is not None:
                        fig_sep.update_yaxes(range=[plotly_ipce_y1_min, plotly_ipce_y1_max], secondary_y=False)
                    if plotly_ipce_y2_min is not None and plotly_ipce_y2_max is not None:
                        fig_sep.update_yaxes(range=[plotly_ipce_y2_min, plotly_ipce_y2_max], secondary_y=True)
                    st.plotly_chart(fig_sep, use_container_width=True)
                    img_sep = fig_sep.to_image(format=export_fmt, width=1600, height=900, scale=2)
                    st.download_button(
                        label="Download plot (Plotly)",
                        data=img_sep,
                        file_name=f"{export_name}_{labels[i]}.{export_fmt}",
                        mime=f"image/{export_fmt}",
                        key=f"download_ipce_plot_{i}"
                    )
                    # --- Matplotlib separate plot ---
                    plt.style.use(mpl_settings["style"])
                    # Prepare a Matplotlib figure and axes with a dedicated legend axes when the legend is outside.
                    legend_mode = mpl_settings.get("legend_mode", "Outside right")
                    spacing = mpl_settings.get("legend_spacing", 0.2) if mpl_settings.get("legend_spacing") is not None else 0.2
                    if legend_mode == "Inside plot":
                        fig_m, ax1_m = plt.subplots(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                        ax2_m = ax1_m.twinx()
                        legend_ax_m = None
                    else:
                        if legend_mode in ["Outside right", "Outside left"]:
                            fig_m = plt.figure(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                            if legend_mode == "Outside right":
                                gs_m = fig_m.add_gridspec(1, 2, width_ratios=[1 - spacing, spacing])
                                ax_main_m = fig_m.add_subplot(gs_m[0])
                                legend_ax_m = fig_m.add_subplot(gs_m[1])
                            else:
                                gs_m = fig_m.add_gridspec(1, 2, width_ratios=[spacing, 1 - spacing])
                                legend_ax_m = fig_m.add_subplot(gs_m[0])
                                ax_main_m = fig_m.add_subplot(gs_m[1])
                            ax1_m = ax_main_m
                            ax2_m = ax1_m.twinx()
                            legend_ax_m.axis('off')
                        elif legend_mode in ["Outside top", "Outside bottom"]:
                            fig_m = plt.figure(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                            if legend_mode == "Outside top":
                                gs_m = fig_m.add_gridspec(2, 1, height_ratios=[spacing, 1 - spacing])
                                legend_ax_m = fig_m.add_subplot(gs_m[0])
                                ax_main_m = fig_m.add_subplot(gs_m[1])
                            else:
                                gs_m = fig_m.add_gridspec(2, 1, height_ratios=[1 - spacing, spacing])
                                ax_main_m = fig_m.add_subplot(gs_m[0])
                                legend_ax_m = fig_m.add_subplot(gs_m[1])
                            ax1_m = ax_main_m
                            ax2_m = ax1_m.twinx()
                            legend_ax_m.axis('off')
                        else:
                            fig_m, ax1_m = plt.subplots(figsize=(mpl_settings["fig_width"], mpl_settings["fig_height"]))
                            ax2_m = ax1_m.twinx()
                            legend_ax_m = None
                    ipce_y_m = df["IPCE"].copy()
                    jint_y_m = df["J_integrated"].copy() if "J_integrated" in df.columns else None
                    if normalize_ipce and not ipce_y_m.empty:
                        ipce_y_m = ipce_y_m / ipce_y_m.max() * 100.0
                    if jint_y_m is not None and normalize_jint and not jint_y_m.empty:
                        jint_y_m = jint_y_m / jint_y_m.max() * 100.0
                    # Determine marker for this curve based on the template
                    marker_template = mpl_settings.get("marker_template", "Custom (per dataset)")
                    if marker_template != "Custom (per dataset)":
                        if marker_template == "Triangles":
                            templ_markers = ["^", "v", "<", ">"]
                        elif marker_template == "Squares":
                            templ_markers = ["s"]
                        elif marker_template == "Diamonds":
                            templ_markers = ["D"]
                        elif marker_template == "Stars (glowing)":
                            templ_markers = ["*", "X"]
                        else:
                            templ_markers = ["o"]
                        marker_cycle = itertools.cycle(templ_markers)
                        this_marker = next(marker_cycle)
                    else:
                        this_marker = markers[i]
                    # Plot IPCE on left axis
                    line1_m, = ax1_m.plot(
                        df["Wavelength"],
                        ipce_y_m,
                        label="IPCE",
                        color=colours[i],
                        linewidth=mpl_settings["line_width"],
                        marker=this_marker
                    )
                    # Plot J_int on right axis if available
                    line2_m = None
                    if jint_y_m is not None:
                        line2_m, = ax2_m.plot(
                            df["Wavelength"],
                            jint_y_m,
                            label="J_int",
                            color=colours[i],
                            linewidth=mpl_settings["line_width"],
                            linestyle="--",
                            marker=this_marker
                        )
                    # Apply glow effect for stars template
                    if marker_template == "Stars (glowing)":
                        glow_m = [patheffects.Stroke(linewidth=4, foreground="white"), patheffects.Normal()]
                        line1_m.set_path_effects(glow_m)
                        if line2_m is not None:
                            line2_m.set_path_effects(glow_m)
                    # Labels and limits
                    ax1_m.set_xlabel(mpl_settings["x_label"], fontsize=mpl_settings["font_size"], fontname=mpl_settings["font_family"])
                    ax1_m.set_ylabel(mpl_settings["y1_label"], fontsize=mpl_settings["font_size"], fontname=mpl_settings["font_family"])
                    ax2_m.set_ylabel(mpl_settings["y2_label"], fontsize=mpl_settings["font_size"], fontname=mpl_settings["font_family"])
                    if mpl_settings["y1_lim_set"] and mpl_settings["y1_min"] is not None and mpl_settings["y1_max"] is not None:
                        ax1_m.set_ylim(mpl_settings["y1_min"], mpl_settings["y1_max"])
                    if mpl_settings["y2_lim_set"] and mpl_settings["y2_min"] is not None and mpl_settings["y2_max"] is not None:
                        ax2_m.set_ylim(mpl_settings["y2_min"], mpl_settings["y2_max"])
                    # X limits
                    if mpl_settings.get("x_lim_set") and mpl_settings.get("x_min") is not None and mpl_settings.get("x_max") is not None:
                        ax1_m.set_xlim(mpl_settings["x_min"], mpl_settings["x_max"])
                        ax2_m.set_xlim(mpl_settings["x_min"], mpl_settings["x_max"])
                    # Legend configuration for Matplotlib separate plot.  Draw the legend either
                    # inside the main axes or on the reserved legend axes depending on the
                    # selected legend mode.
                    lines1_m, labels1_m = ax1_m.get_legend_handles_labels()
                    lines2_m, labels2_m = ax2_m.get_legend_handles_labels()
                    # Access the legend_mode and legend_ax_m variables defined during figure creation
                    legend_mode_m = legend_mode
                    if legend_mode_m == "Inside plot" or legend_ax_m is None:
                        lx = mpl_settings.get("legend_x", 0.8)
                        ly = mpl_settings.get("legend_y", 0.9)
                        ax1_m.legend(
                            lines1_m + lines2_m,
                            labels1_m + labels2_m,
                            loc="center",
                            bbox_to_anchor=(lx, ly),
                            prop={"size": mpl_settings["font_size"], "family": mpl_settings["font_family"]}
                        )
                        fig_m.tight_layout()
                    else:
                        # Outside legends: use horizontal layout for top/bottom and vertical for left/right
                        if legend_mode_m in ["Outside top", "Outside bottom"]:
                            ncols_m = mpl_settings.get("legend_cols", 1)
                            col_sp_m = mpl_settings.get("legend_col_spacing", 0.2)
                            legend_ax_m.legend(
                                lines1_m + lines2_m,
                                labels1_m + labels2_m,
                                loc="center",
                                ncol=ncols_m,
                                columnspacing=col_sp_m,
                                prop={"size": mpl_settings["font_size"], "family": mpl_settings["font_family"]}
                            )
                        else:
                            legend_ax_m.legend(
                                lines1_m + lines2_m,
                                labels1_m + labels2_m,
                                loc="center",
                                prop={"size": mpl_settings["font_size"], "family": mpl_settings["font_family"]}
                            )
                        fig_m.tight_layout()
                    st.pyplot(fig_m)
                    buf_m = io.BytesIO()
                    fig_m.savefig(buf_m, format=export_fmt, dpi=300)
                    st.download_button(
                        label="Download plot (Matplotlib)",
                        data=buf_m.getvalue(),
                        file_name=f"{export_name}_{labels[i]}_mat.{export_fmt}",
                        mime=f"image/{export_fmt}",
                        key=f"download_ipce_plot_{i}_mat"
                    )
                    plt.close(fig_m)
    elif mode == "JV Curve":
        # (JV Curve code unchanged)
        st.sidebar.header("Upload JV Files")
        jv_files = st.sidebar.file_uploader("JV .txt file(s)", type=["txt"], accept_multiple_files=True)
        # Optional zip upload for JV files.  When provided, all .txt files within the archive
        # (excluding summary parameter files) will be added to the list of JV files.  This allows
        # bulk upload of an entire folder structure containing device subfolders and their JV curves.
        jv_folder_zip = st.sidebar.file_uploader("Upload JV folder (.zip)", type=["zip"], accept_multiple_files=False)
        # Build a list of JV files from user uploads and optionally from the uploaded zip.  Persist
        # this list in ``st.session_state`` so that files and subsequent label/colour selections
        # are retained when switching between plot types.  New uploads overwrite the cached list.
        jv_files_combined: List = []
        # Add individually uploaded files (if any)
        if jv_files:
            jv_files_combined.extend(list(jv_files))
        # Extract any JV files from the uploaded zip archive
        if jv_folder_zip is not None:
            try:
                jv_folder_zip.seek(0)
                with zipfile.ZipFile(jv_folder_zip) as zf:
                    for nm in zf.namelist():
                        nm_lower = nm.lower()
                        # Only include .txt files that are not summary parameter files.  Summary files
                        # typically contain "summary" in their name and should be excluded to avoid
                        # plotting aggregated parameters as JV curves.
                        if nm_lower.endswith(".txt") and "summary" not in nm_lower:
                            data = zf.read(nm)
                            buf = io.BytesIO(data)
                            # Assign a name attribute for display and variation extraction
                            buf.name = nm.split("/")[-1]
                            jv_files_combined.append(buf)
            except Exception as e:
                st.sidebar.error(f"Could not process the JV zip folder: {e}")
        # Initialise the JV cache on first run
        if 'jv_files_cache' not in st.session_state:
            st.session_state['jv_files_cache'] = []
        # If any files were uploaded (via uploader or zip), update the cache
        if jv_files_combined:
            st.session_state['jv_files_cache'] = jv_files_combined
        # Retrieve the cached list for processing
        jv_files_combined = st.session_state['jv_files_cache']
        # If the cache is empty (no uploads), attempt to source files from the automatically
        # discovered folder cache.  This allows the dashboard to show JV curves immediately
        # when the script resides alongside device folders.  Update the cache accordingly.
        if not jv_files_combined:
            folder_cache = st.session_state.get('folder_data_cache', None)
            if folder_cache:
                for p in folder_cache.get('jv', []):
                    try:
                        f = open(p, 'rb')
                        jv_files_combined.append(f)
                    except Exception:
                        continue
                st.session_state['jv_files_cache'] = jv_files_combined
        # Only keep JV files that correspond to stability (JV) tests.  Files for tracking or
        # parameter measurements are excluded by checking for 'Stability (JV)' in the file name
        # (case‑insensitive).  If no such files are found, the resulting list will be empty and
        # the user will be prompted to upload valid JV files.
        jv_files = [buf for buf in jv_files_combined if "stability (jv)" in getattr(buf, "name", "").lower()]
        if not jv_files:
            st.info("Upload JV files or a zip folder to generate curves or place them alongside the script.")
            return
        plot_mode = st.sidebar.radio(
            "Plot mode",
            ["Overlay all variations", "Separate plots"],
            index=0,
            key="jv_plot_mode"
        )
        st.sidebar.header("Curve Options")
        smooth_toggle = st.sidebar.checkbox(
            "Apply smoothing (moving average)",
            value=False,
            key="jv_smooth_toggle"
        )
        smooth_win = st.sidebar.slider(
            "Smoothing window",
            3,
            21,
            5,
            step=2,
            key="jv_smooth_win"
        )
        normalize_j = st.sidebar.checkbox(
            "Normalize current to 100%",
            value=False,
            key="jv_normalize_j"
        )
        line_px = st.sidebar.slider(
            "Line thickness (px)",
            1,
            6,
            3,
            key="jv_line_px"
        )
        # Capture the selected export format and persist it across reruns
        jv_export_fmt_sel = st.sidebar.selectbox(
            "Download format",
            ["PNG", "SVG"],
            index=0,
            key="jv_export_fmt"
        )
        export_fmt = jv_export_fmt_sel.lower()
        export_name = st.sidebar.text_input(
            "Export file name base",
            value="jv_plot",
            key="jv_export_name"
        )

        # ------------------------------------------------------------------
        # Plotly customisation settings for JV curves
        #
        # Users can adjust legend location and orientation, font properties,
        # axis limits and margins for the Plotly JV plots.  These settings
        # persist via ``st.session_state`` and are applied to both overlay
        # and per-device plots.
        with st.sidebar.expander("Plotly Plot Settings (JV)", expanded=False):
            plotly_jv_legend_loc = st.selectbox(
                "Legend location",
                ["Outside right", "Outside left", "Outside top", "Outside bottom", "Inside"],
                index=0,
                key="jv_plotly_legend_loc"
            )
            plotly_jv_legend_orient = st.selectbox(
                "Legend orientation",
                ["Horizontal", "Vertical"],
                index=0,
                key="jv_plotly_legend_orient"
            )
            plotly_jv_font_size = st.number_input(
                "Font size",
                value=font_size_global,
                min_value=6,
                max_value=32,
                step=1,
                key="jv_plotly_font_size"
            )
            plotly_jv_font_family = st.text_input(
                "Font family",
                value=font_family_global,
                key="jv_plotly_font_family"
            )
            # Axis limits for voltage (X) and current (Y)
            plotly_jv_x_lim_set = st.checkbox(
                "Set X-axis limits (Voltage)",
                value=False,
                key="jv_plotly_x_lim_set"
            )
            if plotly_jv_x_lim_set:
                plotly_jv_x_min = st.number_input(
                    "X minimum (V)",
                    value=0.0,
                    key="jv_plotly_x_min"
                )
                plotly_jv_x_max = st.number_input(
                    "X maximum (V)",
                    value=0.0,
                    key="jv_plotly_x_max"
                )
            else:
                plotly_jv_x_min = None
                plotly_jv_x_max = None
            plotly_jv_y_lim_set = st.checkbox(
                "Set Y-axis limits (Current Density)",
                value=False,
                key="jv_plotly_y_lim_set"
            )
            if plotly_jv_y_lim_set:
                plotly_jv_y_min = st.number_input(
                    "Y minimum (mA/cm²)",
                    value=0.0,
                    key="jv_plotly_y_min"
                )
                plotly_jv_y_max = st.number_input(
                    "Y maximum (mA/cm²)",
                    value=0.0,
                    key="jv_plotly_y_max"
                )
            else:
                plotly_jv_y_min = None
                plotly_jv_y_max = None
            # Margins for the figure
            plotly_jv_margin_left = st.number_input(
                "Left margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jv_plotly_margin_left"
            )
            plotly_jv_margin_right = st.number_input(
                "Right margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jv_plotly_margin_right"
            )
            plotly_jv_margin_bottom = st.number_input(
                "Bottom margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jv_plotly_margin_bottom"
            )
            plotly_jv_margin_top = st.number_input(
                "Top margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jv_plotly_margin_top"
            )

        # Provide Matplotlib plot settings for JV curves.  These controls mirror
        # the IPCE plot settings and allow the user to configure the appearance
        # of Matplotlib‑based JV curves.  All values are stored in
        # ``st.session_state`` using the keys prefixed with ``jv_mpl_``.
        with st.sidebar.expander("Matplotlib Plot Settings", expanded=False):
            # Choose a Matplotlib style template.  Only valid styles are offered
            # to avoid runtime errors.
            jv_style_options = [
                "default",
                "seaborn-v0_8-darkgrid",
                "seaborn-v0_8-whitegrid",
                "ggplot",
                "Solarize_Light2",
            ]
            st.selectbox(
                "Style template",
                jv_style_options,
                index=0,
                key="jv_mpl_style",
            )
            # Figure dimensions
            st.slider(
                "Figure width (inches)",
                4.0,
                12.0,
                8.0,
                step=0.5,
                key="jv_mpl_fig_width",
            )
            st.slider(
                "Figure height (inches)",
                3.0,
                8.0,
                5.0,
                step=0.5,
                key="jv_mpl_fig_height",
            )
            # Line width and marker size
            st.slider(
                "Line width (px)",
                1,
                6,
                3,
                key="jv_mpl_line_width",
            )
            st.slider(
                "Marker size",
                2,
                20,
                6,
                key="jv_mpl_marker_size",
            )
            # Global font settings
            st.slider(
                "Font size",
                6,
                20,
                12,
                key="jv_mpl_font_size",
            )
            st.selectbox(
                "Font family",
                ["sans-serif", "serif", "monospace"],
                index=0,
                key="jv_mpl_font_family",
            )
            # Marker template for automatic marker assignment across datasets
            st.selectbox(
                "Marker template",
                [
                    "Custom (per dataset)",
                    "Triangles",
                    "Squares",
                    "Diamonds",
                    "Stars (glowing)",
                    "Circles",
                ],
                index=0,
                key="jv_mpl_marker_template",
            )
            # Axis labels
            st.text_input(
                "X-axis label",
                "Voltage (V)",
                key="jv_mpl_x_label",
            )
            st.text_input(
                "Y-axis label",
                "Current Density (mA/cm²)",
                key="jv_mpl_y_label",
            )
            # Axis limits toggles
            st.checkbox(
                "Set Y-axis limits",
                False,
                key="jv_mpl_y_lim_set",
            )
            st.number_input(
                "Y-axis minimum",
                value=0.0,
                key="jv_mpl_y_min",
            )
            st.number_input(
                "Y-axis maximum",
                value=100.0,
                key="jv_mpl_y_max",
            )
            st.checkbox(
                "Set X-axis limits",
                False,
                key="jv_mpl_x_lim_set",
            )
            st.number_input(
                "X-axis minimum",
                value=0.0,
                key="jv_mpl_x_min",
            )
            st.number_input(
                "X-axis maximum",
                value=1.0,
                key="jv_mpl_x_max",
            )
            # Legend placement and layout
            st.selectbox(
                "Legend location",
                [
                    "Inside plot",
                    "Outside left",
                    "Outside right",
                    "Outside top",
                    "Outside bottom",
                ],
                # Default to "Outside right" for JV curves to keep the legend outside
                index=2,
                key="jv_mpl_legend_mode",
            )
            st.slider(
                "Legend x-position",
                0.0,
                1.0,
                0.8,
                step=0.05,
                key="jv_mpl_legend_x",
            )
            st.slider(
                "Legend y-position",
                0.0,
                1.0,
                0.9,
                step=0.05,
                key="jv_mpl_legend_y",
            )
            st.slider(
                "Legend spacing",
                0.05,
                1.0,
                0.1,
                step=0.05,
                key="jv_mpl_legend_spacing",
            )
            st.slider(
                "Legend columns",
                1,
                4,
                1,
                step=1,
                key="jv_mpl_legend_cols",
            )
            st.slider(
                "Legend column spacing",
                0.1,
                2.0,
                0.2,
                step=0.1,
                key="jv_mpl_legend_col_spacing",
            )
        # Labels, colours & shapes
        labels: List[str] = []
        colours: List[str] = []
        markers: List[str] = []
        st.sidebar.header("Labels, Colours & Shapes")
        # Define marker options for per‑dataset selection.  The keys are
        # displayed to the user and map to Matplotlib marker codes (or an
        # empty string for no marker).
        jv_marker_options: Dict[str, str] = {
            "No marker": "",
            "Circle": "o",
            "Triangle Up": "^",
            "Triangle Down": "v",
            "Square": "s",
            "Diamond": "D",
            "Star": "*",
            "X": "X",
            "Plus": "+",
            "Cross": "x",
        }
        marker_display_keys = list(jv_marker_options.keys())
        for i, f in enumerate(jv_files, start=1):
            default_lbl = extract_variation(f.name)
            lbl = st.sidebar.text_input(
                f"Label for file {i} ({f.name})",
                value=default_lbl,
                key=f"jv_lbl_{i}"
            )
            col = st.sidebar.color_picker(
                f"Colour for {lbl}",
                base_palette[(i - 1) % len(base_palette)],
                key=f"jv_col_{i}"
            )
            # Allow the user to choose a marker for this dataset.  A blank value
            # (``"No marker"``) will result in lines with no markers in the
            # Matplotlib plot.
            marker_choice = st.sidebar.selectbox(
                f"Marker for {lbl}",
                marker_display_keys,
                index=0,
                key=f"jv_marker_{i}"
            )
            labels.append(lbl)
            colours.append(col)
            markers.append(jv_marker_options[marker_choice])
        # Parse all JV curves and cache the results by file name so that repeated
        # parsing is avoided when switching between plot types.  Each dataframe is
        # retrieved from ``st.session_state['jv_data_cache']`` if it exists; otherwise
        # it is parsed once and stored.  A copy is used here to avoid modifying
        # the cached original during smoothing or normalisation operations later on.
        curves: List[pd.DataFrame] = []
        # Initialise the JV data cache if necessary
        if 'jv_data_cache' not in st.session_state:
            st.session_state['jv_data_cache'] = {}
        for f, lbl in zip(jv_files, labels):
            fname = getattr(f, 'name', None)
            # Fetch from cache if available
            if fname in st.session_state['jv_data_cache']:
                df_cached = st.session_state['jv_data_cache'][fname]
                df = df_cached.copy()
            else:
                df, _ = parse_jv(f)
                st.session_state['jv_data_cache'][fname] = df.copy()
                df = df.copy()
            curves.append(df)
        # Retrieve Matplotlib settings for JV curves from session state.  These values
        # are defined in the "Matplotlib Plot Settings" sidebar expander and are
        # used to configure the appearance of Matplotlib‑based JV plots.
        jv_mpl_style = st.session_state.get("jv_mpl_style", "default")
        jv_mpl_fig_width = st.session_state.get("jv_mpl_fig_width", 8.0)
        jv_mpl_fig_height = st.session_state.get("jv_mpl_fig_height", 5.0)
        jv_mpl_line_width = st.session_state.get("jv_mpl_line_width", 3)
        jv_mpl_marker_size = st.session_state.get("jv_mpl_marker_size", 6)
        jv_mpl_font_size = st.session_state.get("jv_mpl_font_size", 12)
        jv_mpl_font_family = st.session_state.get("jv_mpl_font_family", "sans-serif")
        jv_mpl_marker_template = st.session_state.get("jv_mpl_marker_template", "Custom (per dataset)")
        jv_mpl_x_label = st.session_state.get("jv_mpl_x_label", "Voltage (V)")
        jv_mpl_y_label = st.session_state.get("jv_mpl_y_label", "Current Density (mA/cm²)")
        jv_mpl_y_lim_set = st.session_state.get("jv_mpl_y_lim_set", False)
        jv_mpl_y_min = st.session_state.get("jv_mpl_y_min")
        jv_mpl_y_max = st.session_state.get("jv_mpl_y_max")
        jv_mpl_x_lim_set = st.session_state.get("jv_mpl_x_lim_set", False)
        jv_mpl_x_min = st.session_state.get("jv_mpl_x_min")
        jv_mpl_x_max = st.session_state.get("jv_mpl_x_max")
        # Default legend placement for JV curves is outside to the right.  This
        # default is applied if no previous selection is stored in session_state.
        jv_mpl_legend_mode = st.session_state.get("jv_mpl_legend_mode", "Outside right")
        jv_mpl_legend_x = st.session_state.get("jv_mpl_legend_x")
        jv_mpl_legend_y = st.session_state.get("jv_mpl_legend_y")
        jv_mpl_legend_spacing = st.session_state.get("jv_mpl_legend_spacing")
        jv_mpl_legend_cols = st.session_state.get("jv_mpl_legend_cols", 1)
        jv_mpl_legend_col_spacing = st.session_state.get("jv_mpl_legend_col_spacing", 0.2)
        jv_mpl_settings = {
            "style": jv_mpl_style,
            "fig_width": jv_mpl_fig_width,
            "fig_height": jv_mpl_fig_height,
            "line_width": jv_mpl_line_width,
            "marker_size": jv_mpl_marker_size,
            "font_size": jv_mpl_font_size,
            "font_family": jv_mpl_font_family,
            "marker_template": jv_mpl_marker_template,
            "x_label": jv_mpl_x_label,
            "y_label": jv_mpl_y_label,
            "y_lim_set": jv_mpl_y_lim_set,
            "y_min": jv_mpl_y_min,
            "y_max": jv_mpl_y_max,
            "x_lim_set": jv_mpl_x_lim_set,
            "x_min": jv_mpl_x_min,
            "x_max": jv_mpl_x_max,
            "legend_mode": jv_mpl_legend_mode,
            "legend_x": jv_mpl_legend_x,
            "legend_y": jv_mpl_legend_y,
            "legend_spacing": jv_mpl_legend_spacing,
            "legend_cols": jv_mpl_legend_cols,
            "legend_col_spacing": jv_mpl_legend_col_spacing,
        }
        # raw_jv_previews is no longer used since previews are generated below
        # Show parsed JV files grouped by variation, device and pixel.  The entire
        # section is initially collapsed; the user can expand it to view the raw
        # data along with checkboxes for selecting curves to include in the overlay.
        st.subheader("Parsed JV Data (Before Plotting)")
        with st.expander("JV Data", expanded=False):
            # Build a mapping from variation -> device -> list of entries for preview.
            # Each entry contains the pixel, dataframe and its overall index (for checkbox keys).
            variation_map_preview: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for idx, (fbuf, df) in enumerate(zip(jv_files, curves)):
                file_name = getattr(fbuf, "name", "")
                var, dev, pix = parse_variation_device_pixel(file_name)
                # Ensure non‑None strings for device and pixel
                dev = dev or ""
                pix = pix or f"File{idx + 1}"
                # Determine scan number from the file name.  Use empty string
                # when not encoded.  This will be used in display names and
                # to enforce same‑scan defaults.
                scan_num = parse_scan_number(file_name) or ""
                if var not in variation_map_preview:
                    variation_map_preview[var] = {}
                if dev not in variation_map_preview[var]:
                    variation_map_preview[var][dev] = []
                variation_map_preview[var][dev].append({"pixel": pix, "df": df, "index": idx, "scan": scan_num})
            # Display the grouped data.  Provide a variation‑level checkbox that
            # allows toggling all curves for a given variation.  Individual
            # pixel‑level checkboxes remain for fine‑grained control.  The
            # variation‑level checkbox state is stored in ``st.session_state``
            # under a key derived from the variation name.
            # Build a mapping from dataset index to its variation for later use
            index_to_var: Dict[int, str] = {}
            # Retrieve best device/pixel defaults computed from Box Plot (if available)
            best_defaults = st.session_state.get('best_jv_defaults', {})
            for vname, dev_dict in variation_map_preview.items():
                # Derive a key for the variation checkbox (replace spaces with underscores)
                var_key = f"jv_overlay_var_include_{vname.replace(' ', '_')}"
                # Determine the default for the variation checkbox.  If the key is not yet in
                # session_state, default to True only when best defaults exist for this
                # variation; otherwise default to False.  When the key exists,
                # session_state will override the value passed here.
                var_default = True if (vname in best_defaults and best_defaults[vname]) else False
                st.checkbox(
                    f"Include variation {vname}",
                    value=var_default,
                    key=var_key
                )
                # Heading for the variation
                st.markdown(f"### Variation: {vname}")
                for dname, entries in dev_dict.items():
                    st.markdown(f"**Device {dname}**")
                    for entry in entries:
                        idx = entry["index"]
                        pixel = entry["pixel"]
                        scan_entry = entry.get("scan", "")
                        index_to_var[idx] = vname
                        # Compose a display name combining scan number, variation, device and pixel.
                        # Format: "0001_ITO-CU20-23-1A".  Only include the scan prefix
                        # when available; otherwise omit it.
                        if dname:
                            if scan_entry:
                                display_name = f"{scan_entry}_{vname}-{dname}-{pixel}"
                            else:
                                display_name = f"{vname}-{dname}-{pixel}"
                        else:
                            if scan_entry:
                                display_name = f"{scan_entry}_{vname}-{pixel}"
                            else:
                                display_name = f"{vname}-{pixel}"
                        # Determine default for pixel checkbox.  If the key is not yet in
                        # session_state, set to True only if this device/pixel matches the
                        # best PCE for this variation.  Jsc maxima are no longer used
                        # to select JV curves by default.
                        pix_key = f"jv_overlay_include_{idx}"
                        if pix_key in st.session_state:
                            pix_default = st.session_state[pix_key]
                        else:
                            # Default based on best defaults (only PCE)
                            pix_default = False
                            if vname in best_defaults:
                                bd = best_defaults[vname]
                                if 'pce' in bd:
                                    pce_tuple = bd['pce']
                                    # Unpack device, pixel and scan number (if present)
                                    dev_best_pce = pix_best_pce = scan_best_pce = None
                                    if len(pce_tuple) >= 3:
                                        dev_best_pce, pix_best_pce, scan_best_pce = pce_tuple
                                    elif len(pce_tuple) == 2:
                                        dev_best_pce, pix_best_pce = pce_tuple
                                    # Select only the best PCE curve that matches device, pixel and scan (when available)
                                    if dev_best_pce is not None and pix_best_pce is not None:
                                        if (dname == dev_best_pce) and (pixel == pix_best_pce):
                                            # If scan is specified for the best PCE, ensure it matches the entry's scan
                                            if scan_best_pce is None or scan_best_pce == scan_entry:
                                                pix_default = True
                        st.checkbox(
                            display_name,
                            value=pix_default,
                            key=pix_key
                        )
                        # Show the dataframe for this curve in its own sub‑expander
                        with st.expander(f"{display_name} data", expanded=False):
                            st.dataframe(entry["df"])
            # Store the index_to_var mapping in session_state for access when computing includes
            st.session_state['jv_index_to_var'] = index_to_var

        # Add checkboxes to allow the user to include or exclude forward and reverse scans
        # in the JV overlay.  These toggles affect only the plotted curves and do not
        # influence which datasets are displayed in the parsed data preview.  Defaults
        # are set to include both scan directions.
        include_fw = st.checkbox(
            "Include Forward Scans (FW)", value=True, key="jv_include_fw"
        )
        include_rv = st.checkbox(
            "Include Reverse Scans (RV)", value=True, key="jv_include_rv"
        )
        # Determine which curves to include in the overlay by reading the
        # per‑pixel checkbox values and the variation‑level checkboxes.  A curve
        # is included only if both its pixel checkbox and its variation
        # checkbox are True.  The mapping from curve index to variation is
        # stored in ``st.session_state['jv_index_to_var']``.
        include_flags: List[bool] = []
        index_to_var_map = st.session_state.get('jv_index_to_var', {})
        for i in range(len(curves)):
            pix_flag = st.session_state.get(f"jv_overlay_include_{i}", True)
            var_name = index_to_var_map.get(i, None)
            if var_name is not None:
                var_key = f"jv_overlay_var_include_{var_name.replace(' ', '_')}"
                var_flag = st.session_state.get(var_key, True)
            else:
                var_flag = True
            include_flags.append(pix_flag and var_flag)
        if plot_mode == "Overlay all variations":
            fig = go.Figure()
            for i, df in enumerate(curves):
                # Skip curves that the user has chosen to exclude or that contain no data
                if not include_flags[i] or df.empty:
                    continue
                # Forward scan: plot only when forward scans are enabled
                if include_fw and "V_FW" in df.columns and "J_FW" in df.columns:
                    j_fw = df["J_FW"].to_numpy()
                    v_fw = df["V_FW"].to_numpy()
                    if smooth_toggle:
                        j_fw = pd.Series(j_fw).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                    if normalize_j and np.max(np.abs(j_fw)) != 0:
                        j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                    fig.add_trace(
                        go.Scatter(
                            x=v_fw,
                            y=j_fw,
                            name=f"{labels[i]} FW",
                            line=dict(color=colours[i], width=line_px)
                        )
                    )
                # Reverse scan: plot only when reverse scans are enabled
                if include_rv and "V_RV" in df.columns and "J_RV" in df.columns:
                    j_rv = df["J_RV"].to_numpy()
                    v_rv = df["V_RV"].to_numpy()
                    if smooth_toggle:
                        j_rv = pd.Series(j_rv).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                    if normalize_j and np.max(np.abs(j_rv)) != 0:
                        j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                    fig.add_trace(
                        go.Scatter(
                            x=v_rv,
                            y=j_rv,
                            name=f"{labels[i]} RV",
                            line=dict(color=colours[i], width=line_px, dash="dash")
                        )
                    )
            # Update figure layout and display
            # Apply Plotly JV customisation settings for legend, margins, fonts and axis ranges
            legend_orientation_jv = "h" if plotly_jv_legend_orient == "Horizontal" else "v"
            if plotly_jv_legend_loc == "Outside right":
                legend_jv = dict(orientation=legend_orientation_jv, yanchor="middle", y=0.5, xanchor="left", x=1.02)
            elif plotly_jv_legend_loc == "Outside left":
                legend_jv = dict(orientation=legend_orientation_jv, yanchor="middle", y=0.5, xanchor="right", x=-0.05)
            elif plotly_jv_legend_loc == "Outside top":
                legend_jv = dict(orientation=legend_orientation_jv, yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            elif plotly_jv_legend_loc == "Outside bottom":
                legend_jv = dict(orientation=legend_orientation_jv, yanchor="top", y=-0.2, xanchor="center", x=0.5)
            else:
                legend_jv = dict(orientation=legend_orientation_jv)
            fig.update_layout(
                title="JV Curves",
                xaxis_title="Voltage (V)",
                yaxis_title="Current Density (mA/cm²" + (" (norm.)" if normalize_j else ")"),
                template=template_name,
                legend=legend_jv,
                margin=dict(
                    l=plotly_jv_margin_left,
                    r=plotly_jv_margin_right,
                    b=plotly_jv_margin_bottom,
                    t=plotly_jv_margin_top,
                ),
                font=dict(size=plotly_jv_font_size, family=plotly_jv_font_family),
                title_font=dict(size=plotly_jv_font_size + 2, family=plotly_jv_font_family, color="black"),
            )
            # Apply axis ranges if set
            if plotly_jv_x_min is not None and plotly_jv_x_max is not None:
                fig.update_xaxes(range=[plotly_jv_x_min, plotly_jv_x_max])
            if plotly_jv_y_min is not None and plotly_jv_y_max is not None:
                fig.update_yaxes(range=[plotly_jv_y_min, plotly_jv_y_max])
            st.plotly_chart(fig, use_container_width=True)
            img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
            st.download_button(
                label="Download JV plot",
                data=img,
                file_name=f"{export_name}.{export_fmt}",
                mime=f"image/{export_fmt}"
            )
            # --- Matplotlib overlay ---
            # Build a list of indices corresponding to curves selected for the overlay.
            selected_indices_jv: List[int] = [
                idx for idx, flag in enumerate(include_flags)
                if flag and not curves[idx].empty
            ]
            if selected_indices_jv:
                # Determine markers for each selected curve.  If a marker template
                # other than "Custom (per dataset)" is chosen, assign markers
                # from a predefined list cycling through the required number of
                # curves; otherwise use the per‑file markers.
                if jv_mpl_settings.get("marker_template") != "Custom (per dataset)":
                    templ_name = jv_mpl_settings.get("marker_template")
                    if templ_name == "Triangles":
                        templ_markers = ["^", "v", "<", ">"]
                    elif templ_name == "Squares":
                        templ_markers = ["s"]
                    elif templ_name == "Diamonds":
                        templ_markers = ["D"]
                    elif templ_name == "Stars (glowing)":
                        templ_markers = ["*", "X"]
                    elif templ_name == "Circles":
                        templ_markers = ["o"]
                    else:
                        templ_markers = ["o"]
                    marker_cycle = itertools.cycle(templ_markers)
                    markers_overlay = [next(marker_cycle) for _ in selected_indices_jv]
                else:
                    markers_overlay = [markers[i] for i in selected_indices_jv]
                # Create Matplotlib figure and axes.  When placing the legend outside
                # of the plot, allocate a dedicated subplot for it so that the
                # plotting area is not compressed.
                plt.style.use(jv_mpl_settings["style"])
                jv_legend_mode = jv_mpl_settings.get("legend_mode", "Outside right")
                jv_spacing = jv_mpl_settings.get("legend_spacing", 0.2) if jv_mpl_settings.get("legend_spacing") is not None else 0.2
                if jv_legend_mode == "Inside plot":
                    fig_m, ax_m = plt.subplots(
                        figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"])
                    )
                    legend_ax_m = None
                else:
                    if jv_legend_mode in ["Outside right", "Outside left"]:
                        fig_m = plt.figure(figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"]))
                        if jv_legend_mode == "Outside right":
                            gs_jv = fig_m.add_gridspec(1, 2, width_ratios=[1 - jv_spacing, jv_spacing])
                            ax_main_m = fig_m.add_subplot(gs_jv[0])
                            legend_ax_m = fig_m.add_subplot(gs_jv[1])
                        else:
                            gs_jv = fig_m.add_gridspec(1, 2, width_ratios=[jv_spacing, 1 - jv_spacing])
                            legend_ax_m = fig_m.add_subplot(gs_jv[0])
                            ax_main_m = fig_m.add_subplot(gs_jv[1])
                        ax_m = ax_main_m
                        legend_ax_m.axis('off')
                    elif jv_legend_mode in ["Outside top", "Outside bottom"]:
                        fig_m = plt.figure(figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"]))
                        if jv_legend_mode == "Outside top":
                            gs_jv = fig_m.add_gridspec(2, 1, height_ratios=[jv_spacing, 1 - jv_spacing])
                            legend_ax_m = fig_m.add_subplot(gs_jv[0])
                            ax_main_m = fig_m.add_subplot(gs_jv[1])
                        else:
                            gs_jv = fig_m.add_gridspec(2, 1, height_ratios=[1 - jv_spacing, jv_spacing])
                            ax_main_m = fig_m.add_subplot(gs_jv[0])
                            legend_ax_m = fig_m.add_subplot(gs_jv[1])
                        ax_m = ax_main_m
                        legend_ax_m.axis('off')
                    else:
                        fig_m, ax_m = plt.subplots(
                            figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"])
                        )
                        legend_ax_m = None
                # Plot each selected JV curve (forward and reverse)
                for cidx, idx in enumerate(selected_indices_jv):
                    df = curves[idx]
                    # Forward scan: plot only when forward scans are enabled
                    if include_fw and "V_FW" in df.columns and "J_FW" in df.columns:
                        j_fw = df["J_FW"].to_numpy()
                        v_fw = df["V_FW"].to_numpy()
                        if smooth_toggle:
                            j_fw = (
                                pd.Series(j_fw)
                                .rolling(window=smooth_win, min_periods=1, center=True)
                                .mean()
                                .to_numpy()
                            )
                        if normalize_j and np.max(np.abs(j_fw)) != 0:
                            j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                        ax_m.plot(
                            v_fw,
                            j_fw,
                            label=f"{labels[idx]} FW",
                            color=colours[idx],
                            linewidth=jv_mpl_settings["line_width"],
                            marker=markers_overlay[cidx] if markers_overlay[cidx] != "" else None,
                            markersize=jv_mpl_settings["marker_size"],
                        )
                    # Reverse scan: plot only when reverse scans are enabled
                    if include_rv and "V_RV" in df.columns and "J_RV" in df.columns:
                        j_rv = df["J_RV"].to_numpy()
                        v_rv = df["V_RV"].to_numpy()
                        if smooth_toggle:
                            j_rv = (
                                pd.Series(j_rv)
                                .rolling(window=smooth_win, min_periods=1, center=True)
                                .mean()
                                .to_numpy()
                            )
                        if normalize_j and np.max(np.abs(j_rv)) != 0:
                            j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                        ax_m.plot(
                            v_rv,
                            j_rv,
                            label=f"{labels[idx]} RV",
                            color=colours[idx],
                            linewidth=jv_mpl_settings["line_width"],
                            linestyle="--",
                            marker=markers_overlay[cidx] if markers_overlay[cidx] != "" else None,
                            markersize=jv_mpl_settings["marker_size"],
                        )
                    # Apply glow effect when the stars template is selected
                    if jv_mpl_settings.get("marker_template") == "Stars (glowing)":
                        # The last up to two lines correspond to the current dataset
                        glow_lines = ax_m.get_lines()[-2:]
                        for gline in glow_lines:
                            gline.set_path_effects(
                                [
                                    patheffects.Stroke(linewidth=4, foreground="white"),
                                    patheffects.Normal(),
                                ]
                            )
                # Axis labels
                ax_m.set_xlabel(
                    jv_mpl_settings["x_label"],
                    fontsize=jv_mpl_settings["font_size"],
                    fontname=jv_mpl_settings["font_family"],
                )
                ax_m.set_ylabel(
                    jv_mpl_settings["y_label"],
                    fontsize=jv_mpl_settings["font_size"],
                    fontname=jv_mpl_settings["font_family"],
                )
                # Axis limits
                if jv_mpl_settings.get("y_lim_set") and jv_mpl_settings.get("y_min") is not None and jv_mpl_settings.get("y_max") is not None:
                    ax_m.set_ylim(jv_mpl_settings["y_min"], jv_mpl_settings["y_max"])
                if jv_mpl_settings.get("x_lim_set") and jv_mpl_settings.get("x_min") is not None and jv_mpl_settings.get("x_max") is not None:
                    ax_m.set_xlim(jv_mpl_settings["x_min"], jv_mpl_settings["x_max"])
                # Title
                ax_m.set_title(
                    "JV Curves",
                    fontsize=jv_mpl_settings["font_size"] + 2,
                    fontname=jv_mpl_settings["font_family"],
                    color="black",
                )
                # Legend configuration
                legend_handles, legend_labels = ax_m.get_legend_handles_labels()
                if jv_legend_mode == "Inside plot" or legend_ax_m is None:
                    lx = jv_mpl_settings.get("legend_x", 0.8)
                    ly = jv_mpl_settings.get("legend_y", 0.9)
                    ax_m.legend(
                        legend_handles,
                        legend_labels,
                        loc="center",
                        bbox_to_anchor=(lx, ly),
                        prop={
                            "size": jv_mpl_settings["font_size"],
                            "family": jv_mpl_settings["font_family"],
                        },
                    )
                    # Use tight_layout when legend is inside
                    fig_m.tight_layout()
                else:
                    # Place legend on the reserved legend axes.  Top/bottom legends use
                    # multiple columns when specified; left/right remain vertical.
                    if jv_legend_mode in ["Outside top", "Outside bottom"]:
                        ncols_jv = jv_mpl_settings.get("legend_cols", 1)
                        col_sp_jv = jv_mpl_settings.get("legend_col_spacing", 0.2)
                        legend_ax_m.legend(
                            legend_handles,
                            legend_labels,
                            loc="center",
                            ncol=ncols_jv,
                            columnspacing=col_sp_jv,
                            prop={
                                "size": jv_mpl_settings["font_size"],
                                "family": jv_mpl_settings["font_family"],
                            },
                        )
                    else:
                        legend_ax_m.legend(
                            legend_handles,
                            legend_labels,
                            loc="center",
                            prop={
                                "size": jv_mpl_settings["font_size"],
                                "family": jv_mpl_settings["font_family"],
                            },
                        )
                    fig_m.tight_layout()
                st.pyplot(fig_m)
                # Save the Matplotlib figure to a bytes buffer and provide a download button
                buf_m = io.BytesIO()
                fig_m.savefig(buf_m, format=export_fmt, dpi=300)
                st.download_button(
                    label="Download JV plot (Matplotlib)",
                    data=buf_m.getvalue(),
                    file_name=f"{export_name}_mat.{export_fmt}",
                    mime=f"image/{export_fmt}",
                    key="download_jv_overlay_mat",
                )
                plt.close(fig_m)
        else:
            # Group separate JV curves by variation and device.  Within each device,
            # multiple pixel curves are plotted on the same figure with legend entries
            # allowing the user to toggle visibility.  Variation and device names are
            # derived from the file name (see ``parse_variation_device_pixel``).
            # Build a mapping: variation -> device -> list of (pixel, df, colour)
            variation_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for idx, (fbuf, df) in enumerate(zip(jv_files, curves)):
                # Skip empty dataframes
                if df.empty:
                    continue
                file_name = getattr(fbuf, "name", "")
                var, dev, pix = parse_variation_device_pixel(file_name)
                # Fallback values when device or pixel are missing
                dev = dev or ""
                pix = pix or f"File{idx + 1}"
                # Derive the scan number from the file name (if encoded) so that
                # it can be appended to the pixel display name.  When no scan
                # number is present, leave it blank.
                scan_num = parse_scan_number(file_name) or ""
                # Construct the display name combining pixel and scan number
                pixel_display = f"{pix} ({scan_num})" if scan_num else pix
                if var not in variation_map:
                    variation_map[var] = {}
                if dev not in variation_map[var]:
                    variation_map[var][dev] = []
                variation_map[var][dev].append({
                    "pixel": pix,
                    "pixel_display": pixel_display,
                    "scan": scan_num,
                    "df": df,
                    "colour": colours[idx],
                    "index": idx,
                })
            # Inform the user if no valid data was parsed
            if not variation_map:
                st.info("No data parsed.")
            else:
                # Top level tabs for each variation
                var_names = list(variation_map.keys())
                var_tabs = st.tabs(var_names)
                for vname, vtab in zip(var_names, var_tabs):
                    with vtab:
                        device_dict = variation_map[vname]
                        dev_names = list(device_dict.keys())
                        # Tabs for each device within the current variation
                        dev_tabs = st.tabs(dev_names)
                        for dname, dtab in zip(dev_names, dev_tabs):
                            with dtab:
                                pixel_entries = device_dict[dname]
                                # Show the pixel along with its scan number in the selection list.  The
                                # ``pixel_display`` field is constructed when building ``variation_map``.
                                pixel_names = [entry["pixel_display"] for entry in pixel_entries]
                                # Multi-select to choose which pixel curves to display.  Default selects
                                # all available pixel names.
                                selected_pixels = st.multiselect(
                                    f"Select pixel(s) for device {dname}",
                                    pixel_names,
                                    default=pixel_names,
                                    key=f"jv_pixel_select_{vname}_{dname}"
                                )
                                # Forward/reverse scan checkboxes for each device.  These allow users to
                                # independently toggle the inclusion of forward and reverse scans for
                                # separate plots.  Use unique keys per device and variation to persist state.
                                include_fw_sep = st.checkbox(
                                    "Include Forward Scans (FW)",
                                    value=True,
                                    key=f"jv_sep_include_fw_{vname}_{dname}"
                                )
                                include_rv_sep = st.checkbox(
                                    "Include Reverse Scans (RV)",
                                    value=True,
                                    key=f"jv_sep_include_rv_{vname}_{dname}"
                                )
                                fig = go.Figure()
                                # Add traces for each selected pixel
                                for entry in pixel_entries:
                                    # Only include curves whose display name is selected
                                    if entry["pixel_display"] not in selected_pixels:
                                        continue
                                    dfp = entry["df"]
                                    colour = entry["colour"]
                                    # Forward scan: plot only when forward scans are enabled
                                    if include_fw_sep and "V_FW" in dfp.columns and "J_FW" in dfp.columns:
                                        j_fw = dfp["J_FW"].to_numpy()
                                        v_fw = dfp["V_FW"].to_numpy()
                                        if smooth_toggle:
                                            j_fw = pd.Series(j_fw).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                                        if normalize_j and np.max(np.abs(j_fw)) != 0:
                                            j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                                        fig.add_trace(
                                            go.Scatter(
                                                x=v_fw,
                                                y=j_fw,
                                                name=f"{entry['pixel']} FW",
                                                line=dict(color=colour, width=line_px)
                                            )
                                        )
                                    # Reverse scan: plot only when reverse scans are enabled
                                    if include_rv_sep and "V_RV" in dfp.columns and "J_RV" in dfp.columns:
                                        j_rv = dfp["J_RV"].to_numpy()
                                        v_rv = dfp["V_RV"].to_numpy()
                                        if smooth_toggle:
                                            j_rv = pd.Series(j_rv).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                                        if normalize_j and np.max(np.abs(j_rv)) != 0:
                                            j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                                        fig.add_trace(
                                            go.Scatter(
                                                x=v_rv,
                                                y=j_rv,
                                                name=f"{entry['pixel']} RV",
                                                line=dict(color=colour, width=line_px, dash="dash")
                                            )
                                        )
                                # Display info if no traces were added
                                if not fig.data:
                                    st.info("No curves found for the selected pixel(s).")
                                else:
                                    # Apply Plotly JV customisation settings to per-device plots
                                    legend_orientation_dev = "h" if plotly_jv_legend_orient == "Horizontal" else "v"
                                    if plotly_jv_legend_loc == "Outside right":
                                        legend_dev = dict(orientation=legend_orientation_dev, yanchor="middle", y=0.5, xanchor="left", x=1.02)
                                    elif plotly_jv_legend_loc == "Outside left":
                                        legend_dev = dict(orientation=legend_orientation_dev, yanchor="middle", y=0.5, xanchor="right", x=-0.05)
                                    elif plotly_jv_legend_loc == "Outside top":
                                        legend_dev = dict(orientation=legend_orientation_dev, yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                                    elif plotly_jv_legend_loc == "Outside bottom":
                                        legend_dev = dict(orientation=legend_orientation_dev, yanchor="top", y=-0.2, xanchor="center", x=0.5)
                                    else:
                                        legend_dev = dict(orientation=legend_orientation_dev)
                                    fig.update_layout(
                                        title=f"{vname} Device {dname} JV Curves",
                                        xaxis_title="Voltage (V)",
                                        yaxis_title="Current Density (mA/cm²" + (" (norm.)" if normalize_j else ")"),
                                        template=template_name,
                                        legend=legend_dev,
                                        margin=dict(
                                            l=plotly_jv_margin_left,
                                            r=plotly_jv_margin_right,
                                            b=plotly_jv_margin_bottom,
                                            t=plotly_jv_margin_top,
                                        ),
                                        font=dict(size=plotly_jv_font_size, family=plotly_jv_font_family),
                                        title_font=dict(size=plotly_jv_font_size + 2, family=plotly_jv_font_family, color="black"),
                                    )
                                    # Apply axis ranges if set
                                    if plotly_jv_x_min is not None and plotly_jv_x_max is not None:
                                        fig.update_xaxes(range=[plotly_jv_x_min, plotly_jv_x_max])
                                    if plotly_jv_y_min is not None and plotly_jv_y_max is not None:
                                        fig.update_yaxes(range=[plotly_jv_y_min, plotly_jv_y_max])
                                    st.plotly_chart(fig, use_container_width=True)
                                    # Export the figure
                                    img = fig.to_image(format=export_fmt, width=1600, height=900, scale=2)
                                    # Ensure a unique key for each download button by including variation and device
                                    st.download_button(
                                        label="Download JV plot",
                                        data=img,
                                        file_name=f"{export_name}_{vname}_{dname}.{export_fmt}",
                                        mime=f"image/{export_fmt}",
                                        key=f"download_jv_plot_{vname}_{dname}"
                                    )

                                    # --- Matplotlib plot for this device ---
                                    # Build a list of entries corresponding to the selected pixels
                                    # Build a list of entries corresponding to the selected pixels using
                                    # the display name (which includes the scan number).  This allows
                                    # users to distinguish between pixels with the same letter but
                                    # different scan numbers.
                                    selected_entries = [
                                        entry for entry in pixel_entries
                                        if entry["pixel_display"] in selected_pixels
                                    ]
                                    if selected_entries:
                                        # Determine markers for each entry based on the global marker template
                                        if jv_mpl_settings.get("marker_template") != "Custom (per dataset)":
                                            templ_name = jv_mpl_settings.get("marker_template")
                                            if templ_name == "Triangles":
                                                templ_markers = ["^", "v", "<", ">"]
                                            elif templ_name == "Squares":
                                                templ_markers = ["s"]
                                            elif templ_name == "Diamonds":
                                                templ_markers = ["D"]
                                            elif templ_name == "Stars (glowing)":
                                                templ_markers = ["*", "X"]
                                            elif templ_name == "Circles":
                                                templ_markers = ["o"]
                                            else:
                                                templ_markers = ["o"]
                                            marker_cycle = itertools.cycle(templ_markers)
                                            markers_dev = [next(marker_cycle) for _ in selected_entries]
                                        else:
                                            # Use per-dataset markers from the top-level selection
                                            markers_dev = [markers[entry["index"]] for entry in selected_entries]
                                        # Create Matplotlib figure for this device.  When the legend is placed
                                        # outside of the plot, allocate a dedicated subplot for it to prevent
                                        # compression of the main plot area.  Otherwise, use a single axes.
                                        plt.style.use(jv_mpl_settings["style"])
                                        jv_legend_mode = jv_mpl_settings.get("legend_mode", "Outside right")
                                        jv_spacing = jv_mpl_settings.get("legend_spacing", 0.2) if jv_mpl_settings.get("legend_spacing") is not None else 0.2
                                        if jv_legend_mode == "Inside plot":
                                            fig_md, ax_md = plt.subplots(
                                                figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"])
                                            )
                                            legend_ax_md = None
                                        else:
                                            if jv_legend_mode in ["Outside right", "Outside left"]:
                                                fig_md = plt.figure(figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"]))
                                                if jv_legend_mode == "Outside right":
                                                    gs_md = fig_md.add_gridspec(1, 2, width_ratios=[1 - jv_spacing, jv_spacing])
                                                    ax_main_md = fig_md.add_subplot(gs_md[0])
                                                    legend_ax_md = fig_md.add_subplot(gs_md[1])
                                                else:
                                                    gs_md = fig_md.add_gridspec(1, 2, width_ratios=[jv_spacing, 1 - jv_spacing])
                                                    legend_ax_md = fig_md.add_subplot(gs_md[0])
                                                    ax_main_md = fig_md.add_subplot(gs_md[1])
                                                ax_md = ax_main_md
                                                legend_ax_md.axis('off')
                                            elif jv_legend_mode in ["Outside top", "Outside bottom"]:
                                                fig_md = plt.figure(figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"]))
                                                if jv_legend_mode == "Outside top":
                                                    gs_md = fig_md.add_gridspec(2, 1, height_ratios=[jv_spacing, 1 - jv_spacing])
                                                    legend_ax_md = fig_md.add_subplot(gs_md[0])
                                                    ax_main_md = fig_md.add_subplot(gs_md[1])
                                                else:
                                                    gs_md = fig_md.add_gridspec(2, 1, height_ratios=[1 - jv_spacing, jv_spacing])
                                                    ax_main_md = fig_md.add_subplot(gs_md[0])
                                                    legend_ax_md = fig_md.add_subplot(gs_md[1])
                                                ax_md = ax_main_md
                                                legend_ax_md.axis('off')
                                            else:
                                                fig_md, ax_md = plt.subplots(
                                                    figsize=(jv_mpl_settings["fig_width"], jv_mpl_settings["fig_height"])
                                                )
                                                legend_ax_md = None
                                        for count, entry in enumerate(selected_entries):
                                            dfp = entry["df"]
                                            colour = entry["colour"]
                                            marker_code = markers_dev[count]
                                            # Forward scan: only plot when forward scans are enabled
                                            if include_fw_sep and "V_FW" in dfp.columns and "J_FW" in dfp.columns:
                                                j_fw = dfp["J_FW"].to_numpy()
                                                v_fw = dfp["V_FW"].to_numpy()
                                                if smooth_toggle:
                                                    j_fw = (
                                                        pd.Series(j_fw)
                                                        .rolling(window=smooth_win, min_periods=1, center=True)
                                                        .mean()
                                                        .to_numpy()
                                                    )
                                                if normalize_j and np.max(np.abs(j_fw)) != 0:
                                                    j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                                                ax_md.plot(
                                                    v_fw,
                                                    j_fw,
                                                    label=f"{entry['pixel']} FW",
                                                    color=colour,
                                                    linewidth=jv_mpl_settings["line_width"],
                                                    marker=marker_code if marker_code != "" else None,
                                                    markersize=jv_mpl_settings["marker_size"],
                                                )
                                            # Reverse scan: only plot when reverse scans are enabled
                                            if include_rv_sep and "V_RV" in dfp.columns and "J_RV" in dfp.columns:
                                                j_rv = dfp["J_RV"].to_numpy()
                                                v_rv = dfp["V_RV"].to_numpy()
                                                if smooth_toggle:
                                                    j_rv = (
                                                        pd.Series(j_rv)
                                                        .rolling(window=smooth_win, min_periods=1, center=True)
                                                        .mean()
                                                        .to_numpy()
                                                    )
                                                if normalize_j and np.max(np.abs(j_rv)) != 0:
                                                    j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                                                ax_md.plot(
                                                    v_rv,
                                                    j_rv,
                                                    label=f"{entry['pixel']} RV",
                                                    color=colour,
                                                    linewidth=jv_mpl_settings["line_width"],
                                                    linestyle="--",
                                                    marker=marker_code if marker_code != "" else None,
                                                    markersize=jv_mpl_settings["marker_size"],
                                                )
                                            # Apply glow effect for stars template
                                            if jv_mpl_settings.get("marker_template") == "Stars (glowing)":
                                                # The last up to two lines correspond to the current dataset
                                                glow_lines_dev = ax_md.get_lines()[-2:]
                                                for gline in glow_lines_dev:
                                                    gline.set_path_effects(
                                                        [
                                                            patheffects.Stroke(linewidth=4, foreground="white"),
                                                            patheffects.Normal(),
                                                        ]
                                                    )
                                        # Axis labels
                                        ax_md.set_xlabel(
                                            jv_mpl_settings["x_label"],
                                            fontsize=jv_mpl_settings["font_size"],
                                            fontname=jv_mpl_settings["font_family"],
                                        )
                                        ax_md.set_ylabel(
                                            jv_mpl_settings["y_label"],
                                            fontsize=jv_mpl_settings["font_size"],
                                            fontname=jv_mpl_settings["font_family"],
                                        )
                                        # Axis limits
                                        if jv_mpl_settings.get("y_lim_set") and jv_mpl_settings.get("y_min") is not None and jv_mpl_settings.get("y_max") is not None:
                                            ax_md.set_ylim(jv_mpl_settings["y_min"], jv_mpl_settings["y_max"])
                                        if jv_mpl_settings.get("x_lim_set") and jv_mpl_settings.get("x_min") is not None and jv_mpl_settings.get("x_max") is not None:
                                            ax_md.set_xlim(jv_mpl_settings["x_min"], jv_mpl_settings["x_max"])
                                        # Title
                                        ax_md.set_title(
                                            f"{vname} Device {dname} JV Curves",
                                            fontsize=jv_mpl_settings["font_size"] + 2,
                                            fontname=jv_mpl_settings["font_family"],
                                            color="black",
                                        )
                                        # Legend handling
                                        legend_handles_dev, legend_labels_dev = ax_md.get_legend_handles_labels()
                                        if jv_legend_mode == "Inside plot" or legend_ax_md is None:
                                            lx = jv_mpl_settings.get("legend_x", 0.8)
                                            ly = jv_mpl_settings.get("legend_y", 0.9)
                                            ax_md.legend(
                                                legend_handles_dev,
                                                legend_labels_dev,
                                                loc="center",
                                                bbox_to_anchor=(lx, ly),
                                                prop={
                                                    "size": jv_mpl_settings["font_size"],
                                                    "family": jv_mpl_settings["font_family"],
                                                },
                                            )
                                            fig_md.tight_layout()
                                        else:
                                            if jv_legend_mode in ["Outside top", "Outside bottom"]:
                                                ncols_md = jv_mpl_settings.get("legend_cols", 1)
                                                col_sp_md = jv_mpl_settings.get("legend_col_spacing", 0.2)
                                                legend_ax_md.legend(
                                                    legend_handles_dev,
                                                    legend_labels_dev,
                                                    loc="center",
                                                    ncol=ncols_md,
                                                    columnspacing=col_sp_md,
                                                    prop={
                                                        "size": jv_mpl_settings["font_size"],
                                                        "family": jv_mpl_settings["font_family"],
                                                    },
                                                )
                                            else:
                                                legend_ax_md.legend(
                                                    legend_handles_dev,
                                                    legend_labels_dev,
                                                    loc="center",
                                                    prop={
                                                        "size": jv_mpl_settings["font_size"],
                                                        "family": jv_mpl_settings["font_family"],
                                                    },
                                                )
                                            fig_md.tight_layout()
                                        st.pyplot(fig_md)
                                        # Save Matplotlib figure and provide download button
                                        buf_md = io.BytesIO()
                                        fig_md.savefig(buf_md, format=export_fmt, dpi=300)
                                        st.download_button(
                                            label="Download JV plot (Matplotlib)",
                                            data=buf_md.getvalue(),
                                            file_name=f"{export_name}_{vname}_{dname}_mat.{export_fmt}",
                                            mime=f"image/{export_fmt}",
                                            key=f"download_jv_plot_{vname}_{dname}_mat",
                                        )
                                        plt.close(fig_md)

    elif mode == "JVcorr":
        """
        Manual JV curve plotting with mismatch correction and custom axis tick steps.

        This section replicates the core functionality of the default JV Curve mode but allows the user
        to manually upload JV files (TXT or ZIP containing TXT files) without any automatic filtering
        on the file names.  An optional mismatch correction factor can be applied to the current density
        data, and users can specify custom tick steps for both axes.  All session keys are prefixed
        with ``jvcorr_`` to avoid clashing with the default JV curve settings.
        """
        # Upload JV files manually
        st.sidebar.header("Upload JV Files (Manual)")
        jvcorr_files = st.sidebar.file_uploader(
            "JV .txt file(s)",
            type=["txt"],
            accept_multiple_files=True,
            key="jvcorr_jv_files",
        )
        # Optional zip upload to support bulk upload of JV files.  Only .txt files are extracted
        # and summary parameter files containing "summary" in their names are ignored.
        jvcorr_folder_zip = st.sidebar.file_uploader(
            "Upload JV folder (.zip)",
            type=["zip"],
            accept_multiple_files=False,
            key="jvcorr_jv_folder_zip",
        )
        # Build a list of JV buffers from uploaded files and the zip archive
        jvcorr_files_combined: List = []
        if jvcorr_files:
            jvcorr_files_combined.extend(list(jvcorr_files))
        if jvcorr_folder_zip is not None:
            try:
                jvcorr_folder_zip.seek(0)
                with zipfile.ZipFile(jvcorr_folder_zip) as zf:
                    for nm in zf.namelist():
                        # only include .txt files, skip summary parameter files
                        nm_lower = nm.lower()
                        if nm_lower.endswith(".txt") and "summary" not in nm_lower:
                            data = zf.read(nm)
                            buf = io.BytesIO(data)
                            buf.name = nm.split("/")[-1]
                            jvcorr_files_combined.append(buf)
            except Exception as e:
                st.sidebar.error(f"Could not process the JV zip folder: {e}")
        # Initialise the JVcorr cache if necessary
        if 'jvcorr_files_cache' not in st.session_state:
            st.session_state['jvcorr_files_cache'] = []
        # When new files are uploaded, overwrite the cache
        if jvcorr_files_combined:
            st.session_state['jvcorr_files_cache'] = jvcorr_files_combined
        # Retrieve cached files for processing
        jvcorr_files_combined = st.session_state['jvcorr_files_cache']
        # Use all uploaded JV files; do not filter by stability keyword.  If none are present,
        # inform the user and return early.
        jv_files = jvcorr_files_combined
        if not jv_files:
            st.info("Upload one or more JV .txt files or a zip folder to plot JV curves.")
            return
        # Select plot mode: overlay all variations or separate per variation/device
        plot_mode = st.sidebar.radio(
            "Plot mode",
            ["Overlay all variations", "Separate plots"],
            index=0,
            key="jvcorr_plot_mode",
        )
        st.sidebar.header("Curve Options")
        # Smoothing and normalisation controls
        smooth_toggle = st.sidebar.checkbox(
            "Apply smoothing (moving average)",
            value=False,
            key="jvcorr_smooth_toggle",
        )
        smooth_win = st.sidebar.slider(
            "Smoothing window",
            3,
            21,
            5,
            step=2,
            key="jvcorr_smooth_win",
        )
        normalize_j = st.sidebar.checkbox(
            "Normalize current to 100%",
            value=False,
            key="jvcorr_normalize_j",
        )
        line_px = st.sidebar.slider(
            "Line thickness (px)",
            1,
            6,
            3,
            key="jvcorr_line_px",
        )
        # Mismatch correction controls
        st.sidebar.header("Mismatch Correction")
        # Toggle to enable or disable mismatch correction.  When enabled,
        # mismatch factors can be specified individually for each uploaded
        # JV file (or variation) below.  A global default value may still be
        # provided here for convenience; it is used as the initial value
        # for per-file inputs but does not apply universally to all curves.
        apply_corr = st.sidebar.checkbox(
            "Apply mismatch correction",
            value=False,
            key="jvcorr_apply_corr",
        )
        global_mismatch_factor = st.sidebar.number_input(
            "Default mismatch factor",
            value=1.0,
            min_value=0.0,
            step=0.01,
            key="jvcorr_mismatch_factor",
        )
        # Export format and base name
        jvcorr_export_fmt_sel = st.sidebar.selectbox(
            "Download format",
            ["PNG", "SVG"],
            index=0,
            key="jvcorr_export_fmt",
        )
        export_fmt = jvcorr_export_fmt_sel.lower()
        export_name = st.sidebar.text_input(
            "Export file name base",
            value="jvcorr_plot",
            key="jvcorr_export_name",
        )
        # Plotly customisation settings for JVcorr curves
        with st.sidebar.expander("Plotly Plot Settings (JVcorr)", expanded=False):
            # Legend configuration
            jvcorr_plotly_legend_loc = st.selectbox(
                "Legend location",
                ["Outside right", "Outside left", "Outside top", "Outside bottom", "Inside"],
                index=0,
                key="jvcorr_plotly_legend_loc",
            )
            jvcorr_plotly_legend_orient = st.selectbox(
                "Legend orientation",
                ["Horizontal", "Vertical"],
                index=0,
                key="jvcorr_plotly_legend_orient",
            )
            jvcorr_plotly_font_size = st.number_input(
                "Font size",
                value=font_size_global,
                min_value=6,
                max_value=32,
                step=1,
                key="jvcorr_plotly_font_size",
            )
            jvcorr_plotly_font_family = st.text_input(
                "Font family",
                value=font_family_global,
                key="jvcorr_plotly_font_family",
            )
            # Axis limits toggles and inputs
            jvcorr_plotly_x_lim_set = st.checkbox(
                "Set X-axis limits (Voltage)",
                value=False,
                key="jvcorr_plotly_x_lim_set",
            )
            if jvcorr_plotly_x_lim_set:
                jvcorr_plotly_x_min = st.number_input(
                    "X minimum (V)",
                    value=0.0,
                    key="jvcorr_plotly_x_min",
                )
                jvcorr_plotly_x_max = st.number_input(
                    "X maximum (V)",
                    value=0.0,
                    key="jvcorr_plotly_x_max",
                )
            else:
                jvcorr_plotly_x_min = None
                jvcorr_plotly_x_max = None
            jvcorr_plotly_y_lim_set = st.checkbox(
                "Set Y-axis limits (Current Density)",
                value=False,
                key="jvcorr_plotly_y_lim_set",
            )
            if jvcorr_plotly_y_lim_set:
                jvcorr_plotly_y_min = st.number_input(
                    "Y minimum (mA/cm²)",
                    value=0.0,
                    key="jvcorr_plotly_y_min",
                )
                jvcorr_plotly_y_max = st.number_input(
                    "Y maximum (mA/cm²)",
                    value=0.0,
                    key="jvcorr_plotly_y_max",
                )
            else:
                jvcorr_plotly_y_min = None
                jvcorr_plotly_y_max = None
            # Axis tick step controls
            jvcorr_plotly_x_step_set = st.checkbox(
                "Set X-axis tick step",
                value=False,
                key="jvcorr_plotly_x_step_set",
            )
            if jvcorr_plotly_x_step_set:
                jvcorr_plotly_x_step = st.number_input(
                    "X-axis step (V)",
                    value=0.2,
                    step=0.05,
                    key="jvcorr_plotly_x_step",
                )
            else:
                jvcorr_plotly_x_step = None
            jvcorr_plotly_y_step_set = st.checkbox(
                "Set Y-axis tick step",
                value=False,
                key="jvcorr_plotly_y_step_set",
            )
            if jvcorr_plotly_y_step_set:
                jvcorr_plotly_y_step = st.number_input(
                    "Y-axis step (mA/cm²)",
                    value=10.0,
                    step=1.0,
                    key="jvcorr_plotly_y_step",
                )
            else:
                jvcorr_plotly_y_step = None
            # Margins
            jvcorr_plotly_margin_left = st.number_input(
                "Left margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jvcorr_plotly_margin_left",
            )
            jvcorr_plotly_margin_right = st.number_input(
                "Right margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jvcorr_plotly_margin_right",
            )
            jvcorr_plotly_margin_bottom = st.number_input(
                "Bottom margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jvcorr_plotly_margin_bottom",
            )
            jvcorr_plotly_margin_top = st.number_input(
                "Top margin (px)",
                min_value=0,
                max_value=300,
                value=80,
                step=5,
                key="jvcorr_plotly_margin_top",
            )
        # Matplotlib customisation for JVcorr curves
        with st.sidebar.expander("Matplotlib Plot Settings (JVcorr)", expanded=False):
            # Style selection
            jvcorr_style_options = [
                "default",
                "seaborn-v0_8-darkgrid",
                "seaborn-v0_8-whitegrid",
                "ggplot",
                "Solarize_Light2",
            ]
            st.selectbox(
                "Style template",
                jvcorr_style_options,
                index=0,
                key="jvcorr_mpl_style",
            )
            # Figure dimensions
            st.slider(
                "Figure width (inches)",
                4.0,
                12.0,
                8.0,
                step=0.5,
                key="jvcorr_mpl_fig_width",
            )
            st.slider(
                "Figure height (inches)",
                3.0,
                8.0,
                5.0,
                step=0.5,
                key="jvcorr_mpl_fig_height",
            )
            # Line width and marker size
            st.slider(
                "Line width (px)",
                1,
                6,
                3,
                key="jvcorr_mpl_line_width",
            )
            st.slider(
                "Marker size",
                2,
                20,
                6,
                key="jvcorr_mpl_marker_size",
            )
            # Font settings
            st.slider(
                "Font size",
                6,
                20,
                12,
                key="jvcorr_mpl_font_size",
            )
            st.selectbox(
                "Font family",
                ["sans-serif", "serif", "monospace"],
                index=0,
                key="jvcorr_mpl_font_family",
            )
            # Marker template for automatic assignment
            st.selectbox(
                "Marker template",
                [
                    "Custom (per dataset)",
                    "Triangles",
                    "Squares",
                    "Diamonds",
                    "Stars (glowing)",
                    "Circles",
                ],
                index=0,
                key="jvcorr_mpl_marker_template",
            )
            # Axis labels
            st.text_input(
                "X-axis label",
                "Voltage (V)",
                key="jvcorr_mpl_x_label",
            )
            st.text_input(
                "Y-axis label",
                "Current Density (mA/cm²)",
                key="jvcorr_mpl_y_label",
            )
            # Axis limits toggles and values
            st.checkbox(
                "Set Y-axis limits",
                False,
                key="jvcorr_mpl_y_lim_set",
            )
            st.number_input(
                "Y-axis minimum",
                value=0.0,
                key="jvcorr_mpl_y_min",
            )
            st.number_input(
                "Y-axis maximum",
                value=100.0,
                key="jvcorr_mpl_y_max",
            )
            st.checkbox(
                "Set X-axis limits",
                False,
                key="jvcorr_mpl_x_lim_set",
            )
            st.number_input(
                "X-axis minimum",
                value=0.0,
                key="jvcorr_mpl_x_min",
            )
            st.number_input(
                "X-axis maximum",
                value=1.0,
                key="jvcorr_mpl_x_max",
            )
            # Tick step controls for Matplotlib
            st.checkbox(
                "Set Y-axis tick step",
                False,
                key="jvcorr_mpl_y_step_set",
            )
            st.number_input(
                "Y-axis step (mA/cm²)",
                value=10.0,
                key="jvcorr_mpl_y_step",
            )
            st.checkbox(
                "Set X-axis tick step",
                False,
                key="jvcorr_mpl_x_step_set",
            )
            st.number_input(
                "X-axis step (V)",
                value=0.2,
                key="jvcorr_mpl_x_step",
            )
            # Legend placement and layout
            st.selectbox(
                "Legend location",
                [
                    "Inside plot",
                    "Outside left",
                    "Outside right",
                    "Outside top",
                    "Outside bottom",
                ],
                index=2,
                key="jvcorr_mpl_legend_mode",
            )
            st.slider(
                "Legend x-position",
                0.0,
                1.0,
                0.8,
                step=0.05,
                key="jvcorr_mpl_legend_x",
            )
            st.slider(
                "Legend y-position",
                0.0,
                1.0,
                0.9,
                step=0.05,
                key="jvcorr_mpl_legend_y",
            )
            st.slider(
                "Legend spacing",
                0.05,
                1.0,
                0.1,
                step=0.05,
                key="jvcorr_mpl_legend_spacing",
            )
            st.slider(
                "Legend columns",
                1,
                4,
                1,
                step=1,
                key="jvcorr_mpl_legend_cols",
            )
            st.slider(
                "Legend column spacing",
                0.1,
                2.0,
                0.2,
                step=0.1,
                key="jvcorr_mpl_legend_col_spacing",
            )
        # Labels, colours & markers section
        labels: List[str] = []
        colours: List[str] = []
        markers: List[str] = []
        # Per-file mismatch factors.  This list will be populated in the
        # loop below.  Each entry corresponds to a JV file and determines
        # the mismatch correction applied to that file.  Defaults to the
        # global mismatch factor but can be customised individually.
        mismatch_factors: List[float] = []
        st.sidebar.header("Labels, Colours & Shapes (JVcorr)")
        # Define marker options
        jvcorr_marker_options: Dict[str, str] = {
            "No marker": "",
            "Circle": "o",
            "Triangle Up": "^",
            "Triangle Down": "v",
            "Square": "s",
            "Diamond": "D",
            "Star": "*",
            "X": "X",
            "Plus": "+",
            "Cross": "x",
        }
        marker_keys = list(jvcorr_marker_options.keys())
        for i, f in enumerate(jv_files, start=1):
            default_lbl = extract_variation(getattr(f, "name", ""))
            # Label for the current JV file
            lbl = st.sidebar.text_input(
                f"Label for file {i} ({getattr(f, 'name', '')})",
                value=default_lbl,
                key=f"jvcorr_lbl_{i}",
            )
            # Colour selection for the current label
            col = st.sidebar.color_picker(
                f"Colour for {lbl}",
                base_palette[(i - 1) % len(base_palette)],
                key=f"jvcorr_col_{i}",
            )
            # Marker selection for the current label
            marker_choice = st.sidebar.selectbox(
                f"Marker for {lbl}",
                marker_keys,
                index=0,
                key=f"jvcorr_marker_{i}",
            )
            # Mismatch factor for the current file.  Use the global default as
            # the starting value.  If mismatch correction is disabled,
            # assign a factor of 1.0 so that no correction is applied.
            if apply_corr:
                mf = st.sidebar.number_input(
                    f"Mismatch factor for {lbl}",
                    value=float(global_mismatch_factor),
                    min_value=0.0,
                    step=0.01,
                    key=f"jvcorr_mismatch_factor_{i}",
                )
            else:
                mf = 1.0
            mismatch_factors.append(mf)
            labels.append(lbl)
            colours.append(col)
            markers.append(jvcorr_marker_options[marker_choice])
        # Parse all JV curves and cache results.  Use a separate cache for JVcorr.
        curves: List[pd.DataFrame] = []
        if 'jvcorr_data_cache' not in st.session_state:
            st.session_state['jvcorr_data_cache'] = {}
        # Iterate over files, labels and their corresponding mismatch factors
        for f, lbl, mf in zip(jv_files, labels, mismatch_factors):
            fname = getattr(f, 'name', None)
            if fname in st.session_state['jvcorr_data_cache']:
                df_cached = st.session_state['jvcorr_data_cache'][fname]
                df = df_cached.copy()
            else:
                df, _ = parse_jv(f)
                st.session_state['jvcorr_data_cache'][fname] = df.copy()
                df = df.copy()
            # Apply mismatch correction if enabled for this file
            if apply_corr and mf and mf != 0:
                if "J_FW" in df.columns:
                    df["J_FW"] = df["J_FW"].astype(float) / mf
                if "J_RV" in df.columns:
                    df["J_RV"] = df["J_RV"].astype(float) / mf
            curves.append(df)
        # Retrieve Matplotlib settings into a dictionary
        jvcorr_mpl_settings: Dict[str, Any] = {}
        jvcorr_mpl_settings["style"] = st.session_state.get("jvcorr_mpl_style", "default")
        jvcorr_mpl_settings["fig_width"] = st.session_state.get("jvcorr_mpl_fig_width", 8.0)
        jvcorr_mpl_settings["fig_height"] = st.session_state.get("jvcorr_mpl_fig_height", 5.0)
        jvcorr_mpl_settings["line_width"] = st.session_state.get("jvcorr_mpl_line_width", 3)
        jvcorr_mpl_settings["marker_size"] = st.session_state.get("jvcorr_mpl_marker_size", 6)
        jvcorr_mpl_settings["font_size"] = st.session_state.get("jvcorr_mpl_font_size", 12)
        jvcorr_mpl_settings["font_family"] = st.session_state.get("jvcorr_mpl_font_family", "sans-serif")
        jvcorr_mpl_settings["marker_template"] = st.session_state.get("jvcorr_mpl_marker_template", "Custom (per dataset)")
        jvcorr_mpl_settings["x_label"] = st.session_state.get("jvcorr_mpl_x_label", "Voltage (V)")
        jvcorr_mpl_settings["y_label"] = st.session_state.get("jvcorr_mpl_y_label", "Current Density (mA/cm²)")
        jvcorr_mpl_settings["y_lim_set"] = st.session_state.get("jvcorr_mpl_y_lim_set", False)
        jvcorr_mpl_settings["y_min"] = st.session_state.get("jvcorr_mpl_y_min", 0.0)
        jvcorr_mpl_settings["y_max"] = st.session_state.get("jvcorr_mpl_y_max", 100.0)
        jvcorr_mpl_settings["x_lim_set"] = st.session_state.get("jvcorr_mpl_x_lim_set", False)
        jvcorr_mpl_settings["x_min"] = st.session_state.get("jvcorr_mpl_x_min", 0.0)
        jvcorr_mpl_settings["x_max"] = st.session_state.get("jvcorr_mpl_x_max", 1.0)
        jvcorr_mpl_settings["y_step_set"] = st.session_state.get("jvcorr_mpl_y_step_set", False)
        jvcorr_mpl_settings["y_step"] = st.session_state.get("jvcorr_mpl_y_step", 10.0)
        jvcorr_mpl_settings["x_step_set"] = st.session_state.get("jvcorr_mpl_x_step_set", False)
        jvcorr_mpl_settings["x_step"] = st.session_state.get("jvcorr_mpl_x_step", 0.2)
        jvcorr_mpl_settings["legend_mode"] = st.session_state.get("jvcorr_mpl_legend_mode", "Outside right")
        jvcorr_mpl_settings["legend_x"] = st.session_state.get("jvcorr_mpl_legend_x", 0.8)
        jvcorr_mpl_settings["legend_y"] = st.session_state.get("jvcorr_mpl_legend_y", 0.9)
        jvcorr_mpl_settings["legend_spacing"] = st.session_state.get("jvcorr_mpl_legend_spacing", 0.1)
        jvcorr_mpl_settings["legend_cols"] = st.session_state.get("jvcorr_mpl_legend_cols", 1)
        jvcorr_mpl_settings["legend_col_spacing"] = st.session_state.get("jvcorr_mpl_legend_col_spacing", 0.2)
        # Build a preview of parsed JV curves grouped by variation and device to allow the user
        # to select which curves to include.
        variation_map_preview: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for idx, (fbuf, df) in enumerate(zip(jv_files, curves)):
            # Skip empty dataframes
            if df.empty:
                continue
            file_name = getattr(fbuf, "name", "")
            vname, dname, pix = parse_variation_device_pixel(file_name)
            vname = vname or "Unknown"
            dname = dname or "NA"
            pix = pix or "NA"
            # Add to nested mapping
            if vname not in variation_map_preview:
                variation_map_preview[vname] = {}
            if dname not in variation_map_preview[vname]:
                variation_map_preview[vname][dname] = []
            # Determine scan number if present
            scan_num = parse_scan_number(file_name)
            variation_map_preview[vname][dname].append({"pixel": pix, "df": df, "index": idx, "scan": scan_num})
        # Display variation-level and pixel-level checkboxes
        index_to_var: Dict[int, str] = {}
        for vname, dev_dict in variation_map_preview.items():
            var_key = f"jvcorr_overlay_var_include_{vname.replace(' ', '_')}"
            st.checkbox(
                f"Include variation {vname}",
                value=True,
                key=var_key,
            )
            st.markdown(f"### Variation: {vname}")
            for dname, entries in dev_dict.items():
                st.markdown(f"**Device {dname}**")
                for entry in entries:
                    idx = entry["index"]
                    pixel = entry["pixel"]
                    scan_entry = entry.get("scan", "")
                    index_to_var[idx] = vname
                    # Compose a display label
                    if dname:
                        if scan_entry:
                            display_name = f"{scan_entry}_{vname}-{dname}-{pixel}"
                        else:
                            display_name = f"{vname}-{dname}-{pixel}"
                    else:
                        if scan_entry:
                            display_name = f"{scan_entry}_{vname}-{pixel}"
                        else:
                            display_name = f"{vname}-{pixel}"
                    pix_key = f"jvcorr_overlay_include_{idx}"
                    st.checkbox(
                        display_name,
                        value=True,
                        key=pix_key,
                    )
                    with st.expander(f"{display_name} data", expanded=False):
                        st.dataframe(entry["df"])
        # Store mapping in session state
        st.session_state['jvcorr_index_to_var'] = index_to_var
        # Forward/reverse scan toggles
        include_fw = st.checkbox(
            "Include Forward Scans (FW)",
            value=True,
            key="jvcorr_include_fw",
        )
        include_rv = st.checkbox(
            "Include Reverse Scans (RV)",
            value=True,
            key="jvcorr_include_rv",
        )
        # Determine include flags per curve
        include_flags: List[bool] = []
        index_to_var_map = st.session_state.get('jvcorr_index_to_var', {})
        for i in range(len(curves)):
            pix_flag = st.session_state.get(f"jvcorr_overlay_include_{i}", True)
            var_name = index_to_var_map.get(i, None)
            if var_name is not None:
                var_key = f"jvcorr_overlay_var_include_{var_name.replace(' ', '_')}"
                var_flag = st.session_state.get(var_key, True)
            else:
                var_flag = True
            include_flags.append(pix_flag and var_flag)
        # Plot overlay or separate
        if plot_mode == "Overlay all variations":
            # Create a Plotly figure for overlaying all selected curves
            fig = go.Figure()
            for i, df in enumerate(curves):
                # Skip excluded or empty curves
                if not include_flags[i] or df.empty:
                    continue
                # Forward scan
                if include_fw and "V_FW" in df.columns and "J_FW" in df.columns:
                    j_fw = df["J_FW"].to_numpy()
                    v_fw = df["V_FW"].to_numpy()
                    if smooth_toggle:
                        j_fw = pd.Series(j_fw).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                    if normalize_j and np.max(np.abs(j_fw)) != 0:
                        j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                    fig.add_trace(
                        go.Scatter(
                            x=v_fw,
                            y=j_fw,
                            name=f"{labels[i]} FW",
                            line=dict(color=colours[i], width=line_px),
                        )
                    )
                # Reverse scan
                if include_rv and "V_RV" in df.columns and "J_RV" in df.columns:
                    j_rv = df["J_RV"].to_numpy()
                    v_rv = df["V_RV"].to_numpy()
                    if smooth_toggle:
                        j_rv = pd.Series(j_rv).rolling(window=smooth_win, min_periods=1, center=True).mean().to_numpy()
                    if normalize_j and np.max(np.abs(j_rv)) != 0:
                        j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                    fig.add_trace(
                        go.Scatter(
                            x=v_rv,
                            y=j_rv,
                            name=f"{labels[i]} RV",
                            line=dict(color=colours[i], width=line_px, dash="dash"),
                        )
                    )
            # Apply legend settings
            legend_orientation_jv = "h" if jvcorr_plotly_legend_orient == "Horizontal" else "v"
            if jvcorr_plotly_legend_loc == "Outside right":
                legend_cfg = dict(
                    orientation=legend_orientation_jv,
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                )
            elif jvcorr_plotly_legend_loc == "Outside left":
                legend_cfg = dict(
                    orientation=legend_orientation_jv,
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=-0.05,
                )
            elif jvcorr_plotly_legend_loc == "Outside top":
                legend_cfg = dict(
                    orientation=legend_orientation_jv,
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                )
            elif jvcorr_plotly_legend_loc == "Outside bottom":
                legend_cfg = dict(
                    orientation=legend_orientation_jv,
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                )
            else:
                legend_cfg = dict(orientation=legend_orientation_jv)
            # Apply axis tick steps
            xaxis_kwargs: Dict[str, Any] = {}
            yaxis_kwargs: Dict[str, Any] = {}
            if jvcorr_plotly_x_lim_set:
                xaxis_kwargs["range"] = [jvcorr_plotly_x_min, jvcorr_plotly_x_max]
            if jvcorr_plotly_y_lim_set:
                yaxis_kwargs["range"] = [jvcorr_plotly_y_min, jvcorr_plotly_y_max]
            if jvcorr_plotly_x_step_set and jvcorr_plotly_x_step is not None:
                xaxis_kwargs["dtick"] = jvcorr_plotly_x_step
            if jvcorr_plotly_y_step_set and jvcorr_plotly_y_step is not None:
                yaxis_kwargs["dtick"] = jvcorr_plotly_y_step
            fig.update_layout(
                template=template_name,
                legend=legend_cfg,
                margin=dict(
                    l=jvcorr_plotly_margin_left,
                    r=jvcorr_plotly_margin_right,
                    b=jvcorr_plotly_margin_bottom,
                    t=jvcorr_plotly_margin_top,
                ),
                xaxis_title=jvcorr_mpl_settings["x_label"],
                yaxis_title=jvcorr_mpl_settings["y_label"],
                font=dict(
                    size=jvcorr_plotly_font_size,
                    family=jvcorr_plotly_font_family,
                ),
            )
            fig.update_xaxes(**xaxis_kwargs)
            fig.update_yaxes(**yaxis_kwargs)
            st.plotly_chart(fig, use_container_width=True)
            # Provide download button for Plotly figure
            buf = io.BytesIO()
            fig.write_image(buf, format=export_fmt, scale=3)
            st.download_button(
                label="Download JV plot (Plotly)",
                data=buf.getvalue(),
                file_name=f"{export_name}_plotly.{export_fmt}",
                mime=f"image/{export_fmt}",
                key="download_jvcorr_plot_plotly",
            )
            # Prepare Matplotlib overlay figure
            import matplotlib.ticker as mticker
            # Determine marker sequence for templates
            if jvcorr_mpl_settings["marker_template"] != "Custom (per dataset)":
                templ_map = {
                    "Triangles": ["^", "v"],
                    "Squares": ["s"],
                    "Diamonds": ["D"],
                    "Stars (glowing)": ["*"],
                    "Circles": ["o"],
                }
                templ_markers = templ_map.get(jvcorr_mpl_settings["marker_template"], ["o"])
                marker_cycle = itertools.cycle(templ_markers)
                markers_overlay = [next(marker_cycle) for _ in range(sum(include_flags))]
            else:
                markers_overlay = [m for m in markers]
            plt.style.use(jvcorr_mpl_settings["style"])
            # Setup figure and axes depending on legend mode
            jv_legend_mode = jvcorr_mpl_settings.get("legend_mode", "Outside right")
            jv_spacing = jvcorr_mpl_settings.get("legend_spacing", 0.2)
            legend_ax_m = None
            if jv_legend_mode == "Inside plot":
                fig_m, ax_m = plt.subplots(
                    figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"])
                )
            else:
                if jv_legend_mode in ["Outside right", "Outside left"]:
                    fig_m = plt.figure(figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"]))
                    if jv_legend_mode == "Outside right":
                        gs_jv = fig_m.add_gridspec(1, 2, width_ratios=[1 - jv_spacing, jv_spacing])
                        ax_m = fig_m.add_subplot(gs_jv[0])
                        legend_ax_m = fig_m.add_subplot(gs_jv[1])
                    else:
                        gs_jv = fig_m.add_gridspec(1, 2, width_ratios=[jv_spacing, 1 - jv_spacing])
                        legend_ax_m = fig_m.add_subplot(gs_jv[0])
                        ax_m = fig_m.add_subplot(gs_jv[1])
                    legend_ax_m.axis('off')
                elif jv_legend_mode in ["Outside top", "Outside bottom"]:
                    fig_m = plt.figure(figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"]))
                    if jv_legend_mode == "Outside top":
                        gs_jv = fig_m.add_gridspec(2, 1, height_ratios=[jv_spacing, 1 - jv_spacing])
                        legend_ax_m = fig_m.add_subplot(gs_jv[0])
                        ax_m = fig_m.add_subplot(gs_jv[1])
                    else:
                        gs_jv = fig_m.add_gridspec(2, 1, height_ratios=[1 - jv_spacing, jv_spacing])
                        ax_m = fig_m.add_subplot(gs_jv[0])
                        legend_ax_m = fig_m.add_subplot(gs_jv[1])
                    legend_ax_m.axis('off')
                else:
                    fig_m, ax_m = plt.subplots(
                        figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"])
                    )
            # Plot curves on Matplotlib overlay
            idx_marker_count = 0
            for i, df in enumerate(curves):
                if not include_flags[i] or df.empty:
                    continue
                # Determine marker
                marker_m = markers_overlay[idx_marker_count] if idx_marker_count < len(markers_overlay) else None
                # Forward scan
                if include_fw and "V_FW" in df.columns and "J_FW" in df.columns:
                    j_fw = df["J_FW"].to_numpy()
                    v_fw = df["V_FW"].to_numpy()
                    if smooth_toggle:
                        j_fw = (
                            pd.Series(j_fw)
                            .rolling(window=smooth_win, min_periods=1, center=True)
                            .mean()
                            .to_numpy()
                        )
                    if normalize_j and np.max(np.abs(j_fw)) != 0:
                        j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                    ax_m.plot(
                        v_fw,
                        j_fw,
                        label=f"{labels[i]} FW",
                        color=colours[i],
                        linewidth=jvcorr_mpl_settings["line_width"],
                        marker=marker_m if marker_m else None,
                        markersize=jvcorr_mpl_settings["marker_size"],
                    )
                # Reverse scan
                if include_rv and "V_RV" in df.columns and "J_RV" in df.columns:
                    j_rv = df["J_RV"].to_numpy()
                    v_rv = df["V_RV"].to_numpy()
                    if smooth_toggle:
                        j_rv = (
                            pd.Series(j_rv)
                            .rolling(window=smooth_win, min_periods=1, center=True)
                            .mean()
                            .to_numpy()
                        )
                    if normalize_j and np.max(np.abs(j_rv)) != 0:
                        j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                    ax_m.plot(
                        v_rv,
                        j_rv,
                        label=f"{labels[i]} RV",
                        color=colours[i],
                        linewidth=jvcorr_mpl_settings["line_width"],
                        linestyle="--",
                        marker=marker_m if marker_m else None,
                        markersize=jvcorr_mpl_settings["marker_size"],
                    )
                idx_marker_count += 1
            # Apply axis labels
            ax_m.set_xlabel(
                jvcorr_mpl_settings["x_label"],
                fontsize=jvcorr_mpl_settings["font_size"],
                fontname=jvcorr_mpl_settings["font_family"],
            )
            ax_m.set_ylabel(
                jvcorr_mpl_settings["y_label"],
                fontsize=jvcorr_mpl_settings["font_size"],
                fontname=jvcorr_mpl_settings["font_family"],
            )
            # Axis limits
            if jvcorr_mpl_settings.get("y_lim_set") and jvcorr_mpl_settings.get("y_min") is not None and jvcorr_mpl_settings.get("y_max") is not None:
                ax_m.set_ylim(jvcorr_mpl_settings["y_min"], jvcorr_mpl_settings["y_max"])
            if jvcorr_mpl_settings.get("x_lim_set") and jvcorr_mpl_settings.get("x_min") is not None and jvcorr_mpl_settings.get("x_max") is not None:
                ax_m.set_xlim(jvcorr_mpl_settings["x_min"], jvcorr_mpl_settings["x_max"])
            # Axis tick steps
            if jvcorr_mpl_settings.get("x_step_set") and jvcorr_mpl_settings.get("x_step") is not None:
                ax_m.xaxis.set_major_locator(mticker.MultipleLocator(jvcorr_mpl_settings["x_step"]))
            if jvcorr_mpl_settings.get("y_step_set") and jvcorr_mpl_settings.get("y_step") is not None:
                ax_m.yaxis.set_major_locator(mticker.MultipleLocator(jvcorr_mpl_settings["y_step"]))
            # Title
            ax_m.set_title(
                "JV Curves",
                fontsize=jvcorr_mpl_settings["font_size"] + 2,
                fontname=jvcorr_mpl_settings["font_family"],
                color="black",
            )
            # Legend
            legend_handles, legend_labels = ax_m.get_legend_handles_labels()
            if jv_legend_mode == "Inside plot" or legend_ax_m is None:
                lx = jvcorr_mpl_settings.get("legend_x", 0.8)
                ly = jvcorr_mpl_settings.get("legend_y", 0.9)
                ax_m.legend(
                    legend_handles,
                    legend_labels,
                    loc="center",
                    bbox_to_anchor=(lx, ly),
                    prop={
                        "size": jvcorr_mpl_settings["font_size"],
                        "family": jvcorr_mpl_settings["font_family"],
                    },
                )
                fig_m.tight_layout()
            else:
                if jv_legend_mode in ["Outside top", "Outside bottom"]:
                    ncols_jv = jvcorr_mpl_settings.get("legend_cols", 1)
                    col_sp_jv = jvcorr_mpl_settings.get("legend_col_spacing", 0.2)
                    legend_ax_m.legend(
                        legend_handles,
                        legend_labels,
                        loc="center",
                        ncol=ncols_jv,
                        columnspacing=col_sp_jv,
                        prop={
                            "size": jvcorr_mpl_settings["font_size"],
                            "family": jvcorr_mpl_settings["font_family"],
                        },
                    )
                else:
                    legend_ax_m.legend(
                        legend_handles,
                        legend_labels,
                        loc="center",
                        prop={
                            "size": jvcorr_mpl_settings["font_size"],
                            "family": jvcorr_mpl_settings["font_family"],
                        },
                    )
                fig_m.tight_layout()
            st.pyplot(fig_m)
            buf_m = io.BytesIO()
            fig_m.savefig(buf_m, format=export_fmt, dpi=300)
            st.download_button(
                label="Download JV plot (Matplotlib)",
                data=buf_m.getvalue(),
                file_name=f"{export_name}_mat.{export_fmt}",
                mime=f"image/{export_fmt}",
                key="download_jvcorr_overlay_mat",
            )
            plt.close(fig_m)
        else:
            # Separate plots by variation and device, similar to the default JV mode.
            import matplotlib.ticker as mticker
            variation_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for idx, (fbuf, df) in enumerate(zip(jv_files, curves)):
                if df.empty:
                    continue
                file_name = getattr(fbuf, "name", "")
                var, dev, pix = parse_variation_device_pixel(file_name)
                var = var or "Unknown"
                dev = dev or "NA"
                pix = pix or "NA"
                if var not in variation_map:
                    variation_map[var] = {}
                if dev not in variation_map[var]:
                    variation_map[var][dev] = []
                variation_map[var][dev].append({"pixel": pix, "df": df, "index": idx})
            # Loop through variations
            for vname, devs in variation_map.items():
                for dname, entries in devs.items():
                    # Identify indices to include for this variation/device
                    selected_indices = [
                        entry["index"]
                        for entry in entries
                        if st.session_state.get(f"jvcorr_overlay_include_{entry['index']}", True)
                        and st.session_state.get(
                            f"jvcorr_overlay_var_include_{vname.replace(' ', '_')}", True
                        )
                    ]
                    if not selected_indices:
                        continue
                    # Plotly figure for separate plot
                    fig_sep = go.Figure()
                    for idx in selected_indices:
                        dfp = curves[idx]
                        # Forward
                        if include_fw and "V_FW" in dfp.columns and "J_FW" in dfp.columns:
                            j_fw = dfp["J_FW"].to_numpy()
                            v_fw = dfp["V_FW"].to_numpy()
                            if smooth_toggle:
                                j_fw = (
                                    pd.Series(j_fw)
                                    .rolling(window=smooth_win, min_periods=1, center=True)
                                    .mean()
                                    .to_numpy()
                                )
                            if normalize_j and np.max(np.abs(j_fw)) != 0:
                                j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                            fig_sep.add_trace(
                                go.Scatter(
                                    x=v_fw,
                                    y=j_fw,
                                    name=f"{labels[idx]} FW",
                                    line=dict(color=colours[idx], width=line_px),
                                )
                            )
                        # Reverse
                        if include_rv and "V_RV" in dfp.columns and "J_RV" in dfp.columns:
                            j_rv = dfp["J_RV"].to_numpy()
                            v_rv = dfp["V_RV"].to_numpy()
                            if smooth_toggle:
                                j_rv = (
                                    pd.Series(j_rv)
                                    .rolling(window=smooth_win, min_periods=1, center=True)
                                    .mean()
                                    .to_numpy()
                                )
                            if normalize_j and np.max(np.abs(j_rv)) != 0:
                                j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                            fig_sep.add_trace(
                                go.Scatter(
                                    x=v_rv,
                                    y=j_rv,
                                    name=f"{labels[idx]} RV",
                                    line=dict(color=colours[idx], width=line_px, dash="dash"),
                                )
                            )
                    # Apply legend settings per separate plot
                    legend_orientation_jv = "h" if jvcorr_plotly_legend_orient == "Horizontal" else "v"
                    if jvcorr_plotly_legend_loc == "Outside right":
                        sep_legend_cfg = dict(
                            orientation=legend_orientation_jv,
                            yanchor="middle",
                            y=0.5,
                            xanchor="left",
                            x=1.02,
                        )
                    elif jvcorr_plotly_legend_loc == "Outside left":
                        sep_legend_cfg = dict(
                            orientation=legend_orientation_jv,
                            yanchor="middle",
                            y=0.5,
                            xanchor="right",
                            x=-0.05,
                        )
                    elif jvcorr_plotly_legend_loc == "Outside top":
                        sep_legend_cfg = dict(
                            orientation=legend_orientation_jv,
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                        )
                    elif jvcorr_plotly_legend_loc == "Outside bottom":
                        sep_legend_cfg = dict(
                            orientation=legend_orientation_jv,
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                        )
                    else:
                        sep_legend_cfg = dict(orientation=legend_orientation_jv)
                    fig_sep.update_layout(
                        template=template_name,
                        legend=sep_legend_cfg,
                        margin=dict(
                            l=jvcorr_plotly_margin_left,
                            r=jvcorr_plotly_margin_right,
                            b=jvcorr_plotly_margin_bottom,
                            t=jvcorr_plotly_margin_top,
                        ),
                        xaxis_title=jvcorr_mpl_settings["x_label"],
                        yaxis_title=jvcorr_mpl_settings["y_label"],
                        title=f"{vname} Device {dname} JV Curves",
                        font=dict(
                            size=jvcorr_plotly_font_size,
                            family=jvcorr_plotly_font_family,
                        ),
                    )
                    # Apply axis limits and steps
                    if jvcorr_plotly_x_lim_set:
                        fig_sep.update_xaxes(range=[jvcorr_plotly_x_min, jvcorr_plotly_x_max])
                    if jvcorr_plotly_y_lim_set:
                        fig_sep.update_yaxes(range=[jvcorr_plotly_y_min, jvcorr_plotly_y_max])
                    if jvcorr_plotly_x_step_set and jvcorr_plotly_x_step is not None:
                        fig_sep.update_xaxes(dtick=jvcorr_plotly_x_step)
                    if jvcorr_plotly_y_step_set and jvcorr_plotly_y_step is not None:
                        fig_sep.update_yaxes(dtick=jvcorr_plotly_y_step)
                    st.plotly_chart(fig_sep, use_container_width=True)
                    # Download button for Plotly separate
                    buf_sep = io.BytesIO()
                    fig_sep.write_image(buf_sep, format=export_fmt, scale=3)
                    st.download_button(
                        label="Download JV plot (Plotly)",
                        data=buf_sep.getvalue(),
                        file_name=f"{export_name}_{vname}_{dname}_plotly.{export_fmt}",
                        mime=f"image/{export_fmt}",
                        key=f"download_jvcorr_sep_plotly_{vname}_{dname}",
                    )
                    # Matplotlib separate figure
                    # Determine marker template
                    if jvcorr_mpl_settings["marker_template"] != "Custom (per dataset)":
                        templ_map = {
                            "Triangles": ["^", "v"],
                            "Squares": ["s"],
                            "Diamonds": ["D"],
                            "Stars (glowing)": ["*"],
                            "Circles": ["o"],
                        }
                        templ_markers = templ_map.get(jvcorr_mpl_settings["marker_template"], ["o"])
                        marker_cycle = itertools.cycle(templ_markers)
                        markers_dev = [next(marker_cycle) for _ in range(len(selected_indices))]
                    else:
                        markers_dev = [markers[i] for i in selected_indices]
                    # Setup Matplotlib figure based on legend position
                    jv_legend_mode = jvcorr_mpl_settings.get("legend_mode", "Outside right")
                    jv_spacing = jvcorr_mpl_settings.get("legend_spacing", 0.2)
                    legend_ax_md = None
                    if jv_legend_mode == "Inside plot":
                        fig_md, ax_md = plt.subplots(
                            figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"])
                        )
                    else:
                        if jv_legend_mode in ["Outside right", "Outside left"]:
                            fig_md = plt.figure(figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"]))
                            if jv_legend_mode == "Outside right":
                                gs_jv = fig_md.add_gridspec(1, 2, width_ratios=[1 - jv_spacing, jv_spacing])
                                ax_main_md = fig_md.add_subplot(gs_jv[0])
                                legend_ax_md = fig_md.add_subplot(gs_jv[1])
                            else:
                                gs_jv = fig_md.add_gridspec(1, 2, width_ratios=[jv_spacing, 1 - jv_spacing])
                                legend_ax_md = fig_md.add_subplot(gs_jv[0])
                                ax_main_md = fig_md.add_subplot(gs_jv[1])
                            ax_md = ax_main_md
                            legend_ax_md.axis('off')
                        elif jv_legend_mode in ["Outside top", "Outside bottom"]:
                            fig_md = plt.figure(figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"]))
                            if jv_legend_mode == "Outside top":
                                gs_jv = fig_md.add_gridspec(2, 1, height_ratios=[jv_spacing, 1 - jv_spacing])
                                legend_ax_md = fig_md.add_subplot(gs_jv[0])
                                ax_main_md = fig_md.add_subplot(gs_jv[1])
                            else:
                                gs_jv = fig_md.add_gridspec(2, 1, height_ratios=[1 - jv_spacing, jv_spacing])
                                ax_main_md = fig_md.add_subplot(gs_jv[0])
                                legend_ax_md = fig_md.add_subplot(gs_jv[1])
                            ax_md = ax_main_md
                            legend_ax_md.axis('off')
                        else:
                            fig_md, ax_md = plt.subplots(
                                figsize=(jvcorr_mpl_settings["fig_width"], jvcorr_mpl_settings["fig_height"])
                            )
                    # Plot curves for this variation/device
                    for cidx, idx in enumerate(selected_indices):
                        dfp = curves[idx]
                        colour = colours[idx]
                        marker_code = markers_dev[cidx]
                        # Forward scan
                        if include_fw and "V_FW" in dfp.columns and "J_FW" in dfp.columns:
                            j_fw = dfp["J_FW"].to_numpy()
                            v_fw = dfp["V_FW"].to_numpy()
                            if smooth_toggle:
                                j_fw = (
                                    pd.Series(j_fw)
                                    .rolling(window=smooth_win, min_periods=1, center=True)
                                    .mean()
                                    .to_numpy()
                                )
                            if normalize_j and np.max(np.abs(j_fw)) != 0:
                                j_fw = j_fw / np.max(np.abs(j_fw)) * 100.0
                            ax_md.plot(
                                v_fw,
                                j_fw,
                                label=f"{labels[idx]} FW",
                                color=colour,
                                linewidth=jvcorr_mpl_settings["line_width"],
                                marker=marker_code if marker_code != "" else None,
                                markersize=jvcorr_mpl_settings["marker_size"],
                            )
                        # Reverse scan
                        if include_rv and "V_RV" in dfp.columns and "J_RV" in dfp.columns:
                            j_rv = dfp["J_RV"].to_numpy()
                            v_rv = dfp["V_RV"].to_numpy()
                            if smooth_toggle:
                                j_rv = (
                                    pd.Series(j_rv)
                                    .rolling(window=smooth_win, min_periods=1, center=True)
                                    .mean()
                                    .to_numpy()
                                )
                            if normalize_j and np.max(np.abs(j_rv)) != 0:
                                j_rv = j_rv / np.max(np.abs(j_rv)) * 100.0
                            ax_md.plot(
                                v_rv,
                                j_rv,
                                label=f"{labels[idx]} RV",
                                color=colour,
                                linewidth=jvcorr_mpl_settings["line_width"],
                                linestyle="--",
                                marker=marker_code if marker_code != "" else None,
                                markersize=jvcorr_mpl_settings["marker_size"],
                            )
                    # Axis labels and limits
                    ax_md.set_xlabel(
                        jvcorr_mpl_settings["x_label"],
                        fontsize=jvcorr_mpl_settings["font_size"],
                        fontname=jvcorr_mpl_settings["font_family"],
                    )
                    ax_md.set_ylabel(
                        jvcorr_mpl_settings["y_label"],
                        fontsize=jvcorr_mpl_settings["font_size"],
                        fontname=jvcorr_mpl_settings["font_family"],
                    )
                    if jvcorr_mpl_settings.get("y_lim_set") and jvcorr_mpl_settings.get("y_min") is not None and jvcorr_mpl_settings.get("y_max") is not None:
                        ax_md.set_ylim(jvcorr_mpl_settings["y_min"], jvcorr_mpl_settings["y_max"])
                    if jvcorr_mpl_settings.get("x_lim_set") and jvcorr_mpl_settings.get("x_min") is not None and jvcorr_mpl_settings.get("x_max") is not None:
                        ax_md.set_xlim(jvcorr_mpl_settings["x_min"], jvcorr_mpl_settings["x_max"])
                    if jvcorr_mpl_settings.get("x_step_set") and jvcorr_mpl_settings.get("x_step") is not None:
                        ax_md.xaxis.set_major_locator(mticker.MultipleLocator(jvcorr_mpl_settings["x_step"]))
                    if jvcorr_mpl_settings.get("y_step_set") and jvcorr_mpl_settings.get("y_step") is not None:
                        ax_md.yaxis.set_major_locator(mticker.MultipleLocator(jvcorr_mpl_settings["y_step"]))
                    # Title and legend
                    ax_md.set_title(
                        f"{vname} Device {dname} JV Curves",
                        fontsize=jvcorr_mpl_settings["font_size"] + 2,
                        fontname=jvcorr_mpl_settings["font_family"],
                        color="black",
                    )
                    legend_handles_dev, legend_labels_dev = ax_md.get_legend_handles_labels()
                    if jv_legend_mode == "Inside plot" or legend_ax_md is None:
                        lx = jvcorr_mpl_settings.get("legend_x", 0.8)
                        ly = jvcorr_mpl_settings.get("legend_y", 0.9)
                        ax_md.legend(
                            legend_handles_dev,
                            legend_labels_dev,
                            loc="center",
                            bbox_to_anchor=(lx, ly),
                            prop={
                                "size": jvcorr_mpl_settings["font_size"],
                                "family": jvcorr_mpl_settings["font_family"],
                            },
                        )
                        fig_md.tight_layout()
                    else:
                        if jv_legend_mode in ["Outside top", "Outside bottom"]:
                            ncols_md = jvcorr_mpl_settings.get("legend_cols", 1)
                            col_sp_md = jvcorr_mpl_settings.get("legend_col_spacing", 0.2)
                            legend_ax_md.legend(
                                legend_handles_dev,
                                legend_labels_dev,
                                loc="center",
                                ncol=ncols_md,
                                columnspacing=col_sp_md,
                                prop={
                                    "size": jvcorr_mpl_settings["font_size"],
                                    "family": jvcorr_mpl_settings["font_family"],
                                },
                            )
                        else:
                            legend_ax_md.legend(
                                legend_handles_dev,
                                legend_labels_dev,
                                loc="center",
                                prop={
                                    "size": jvcorr_mpl_settings["font_size"],
                                    "family": jvcorr_mpl_settings["font_family"],
                                },
                            )
                        fig_md.tight_layout()
                    st.pyplot(fig_md)
                    # Save Matplotlib figure and provide download
                    buf_md = io.BytesIO()
                    fig_md.savefig(buf_md, format=export_fmt, dpi=300)
                    st.download_button(
                        label="Download JV plot (Matplotlib)",
                        data=buf_md.getvalue(),
                        file_name=f"{export_name}_{vname}_{dname}_mat.{export_fmt}",
                        mime=f"image/{export_fmt}",
                        key=f"download_jvcorr_plot_{vname}_{dname}_mat",
                    )
                    plt.close(fig_md)

if __name__ == "__main__":
    # Entry point for running this script.  When executed directly (for example by double‑clicking
    # the file), this block auto‑launches a new Streamlit process and then exits.  When executed
    # within Streamlit (or when relaunched via the auto‑launcher indicated by the
    # STREAMLIT_AUTORUN environment variable), it simply calls ``main()`` to render the app.
    import os
    import sys
    import subprocess
    # Detect whether Streamlit is currently running this script (i.e. launched via ``streamlit run``)
    # or if the environment variable STREAMLIT_AUTORUN is set.  In either of these cases we call
    # main() directly to avoid launching a new Streamlit process.
    is_streamlit = hasattr(st, "_is_running_with_streamlit") and st._is_running_with_streamlit
    if is_streamlit or os.environ.get("STREAMLIT_AUTORUN") == "1":
        main()
    else:
        # Mark that the next invocation originates from the auto‑launcher to prevent recursion.
        os.environ["STREAMLIT_AUTORUN"] = "1"
        # Launch a separate Streamlit process using the same Python executable.  Pass the updated
        # environment so that the child process detects STREAMLIT_AUTORUN and calls main().  We
        # immediately exit the parent process so that only one instance remains running.
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)], env=os.environ)
        sys.exit()
