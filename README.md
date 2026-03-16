# Stroke Prediction (ML)

## Requirements & setup

### Python (PDF build and scripts)

```bash
pip install -r requirements.txt
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
