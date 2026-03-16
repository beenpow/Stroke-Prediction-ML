# Proposal PDF

## Build PDF (no LaTeX needed)

```bash
cd proposal/tex
.venv/bin/python3 build_pdf.py
```

Output: `proposal.pdf`

- **LaTeX:** If you have `pdflatex`, you can instead run `pdflatex proposal.tex` for the LaTeX version.
- **Python PDF:** `build_pdf.py` uses ReportLab and the same content; run it when LaTeX is not installed.
