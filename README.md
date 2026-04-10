# Stroke Prediction (ML)

## Requirements & setup

### Python (PDF build and scripts)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notebooks and Python helpers live under **`src/`** (e.g. `src/baseline_final.ipynb`, `src/stroke_data.py`). Open the **repository root** as the workspace. The first code cell in baseline notebooks adds `src` to `sys.path` so `from stroke_data import ...` works; `stroke_data` resolves CSV paths from the repo root. For `src/preprocessing.ipynb`, the first cell switches to the repo root when the kernel’s working directory is `src/`, so `data/...` paths stay valid.

**macOS + XGBoost:** if `import xgboost` still fails after `pip install`, install OpenMP once:

```bash
brew install libomp
```

### TeX (for building the proposal PDF with `pdflatex`)

LaTeX is not a pip package; it must be installed on your system. Use the option below for your OS (one-time setup).

| OS | Install |
|----|---------|
| **macOS** | `brew install --cask mactex` (or install [MacTeX](https://www.tug.org/mactex/) manually) |
| **Linux (Debian/Ubuntu)** | `sudo apt install texlive-base texlive-latex-extra` |
| **Windows** | Install [MiKTeX](https://miktex.org/), then ensure `pdflatex` is available in your terminal |

After installing TeX:

```bash
cd proposal/tex && pdflatex proposal.tex
```

To build the PDF without TeX (Python only):

```bash
cd proposal/tex && python3 build_pdf.py
# With venv: .venv/bin/python3 build_pdf.py
```
