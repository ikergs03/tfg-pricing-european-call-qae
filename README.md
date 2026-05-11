# Estudio de la viabilidad de la computación cuántica en aplicaciones financieras

This repository contains the source code, LaTeX sources, and supporting scripts for the thesis *Estudio de la viabilidad de la computación cuántica en aplicaciones financieras*, with a focus on Quantum Hierarchical Risk Parity (QHRP).

## Repository layout

- `latex/`: full thesis source in LaTeX, bibliography, figures, and appendices.
- `src/`: Python code used for experiments, analysis, and figure generation.
- `downloads/`: scripts for downloading and preparing data.
- `data/`: input and derived datasets used in the experiments.
- `requirements_act_qiskit.txt`: Python dependencies for the Qiskit-based environment.
- `requirements_paper_qiskit.txt`: additional dependencies used for reproducing the thesis experiments.

## Requirements

- Python 3.10+ recommended.
- A LaTeX distribution with `pdflatex`, `bibtex`, `makeindex`, and `makeglossaries`.
- A working Qiskit-compatible Python environment.

## Reproducing the thesis

1. Create and activate a Python environment.
2. Install the required packages.
	```bash
	pip install -r requirements_act_qiskit.txt
	pip install -r requirements_paper_qiskit.txt
	```
3. Generate or download the data needed for the experiments.
4. Run the scripts in `src/` if you want to recompute figures or results.
5. Compile the thesis from the `latex/` directory.

## Compile the PDF

From `latex/`, run the standard LaTeX toolchain:

```bash
pdflatex tfgtfmthesisuam.tex
bibtex tfgtfmthesisuam
pdflatex tfgtfmthesisuam.tex
pdflatex tfgtfmthesisuam.tex
makeglossaries tfgtfmthesisuam
makeindex tfgtfmthesisuam
```

If your setup uses the `arara` workflow, the main `tfgtfmthesisuam.tex` file already includes the corresponding compilation directives.

## Data and reproducibility

Some intermediate artifacts are generated during the workflow and are not meant to be versioned. If you are using this repository as a public project, keep only the files needed to rebuild the thesis and rerun the experiments.

## License

This repository is provided for academic reference. Reuse, redistribution, or derivative works require permission from the author.
