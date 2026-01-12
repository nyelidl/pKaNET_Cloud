pKaNET Cloud
============

pKaNET Cloud is a web-based application for pH-dependent ligand preparation,
pKa prediction, and 2D/3D structure generation, built with Streamlit.

The server provides an easy-to-use graphical interface for generating
protonation-state–adjusted 3D structures suitable for molecular docking
and molecular dynamics simulations.


Features
--------
• Input support:
  - Single SMILES string
  - SMILES file (.smi)
  - Ligand structure file (.pdb, .mol2, .sdf)

• pKa prediction:
  - Machine-learning–based pKa prediction using pKaNET

• Protonation:
  - pH-adjusted protonation states using Dimorphite-DL

• Structure generation:
  - 2D structure visualization (RDKit)
  - 3D structure generation and energy minimization (ETKDG + MMFF/UFF)

• Output formats:
  - PDB (default)
  - SDF
  - MOL2 (automatically falls back to SDF if MOL2 is unavailable)

• Visualization:
  - Interactive 3D WebGL viewer (py3Dmol)
  - 2D chemical structure images

• Download:
  - ZIP archive of all generated outputs
  - ZIP archive of minimized structures only


Web Interface
-------------
The application is deployed using Streamlit Cloud and can be accessed
directly from a web browser. No local installation is required for users.


Repository Structure
--------------------
app.py
  Streamlit web interface (GUI)

core.py
  Core computational pipeline (pKa prediction, protonation, 3D generation)

requirements.txt
  Python dependencies for deployment

README.txt / README.md
  Project documentation


Local Development
-----------------
To run the app locally:

1. Create a Python environment (Python 3.10 recommended)

2. Install dependencies:
   pip install -r requirements.txt

3. Launch the app:
   streamlit run app.py


Important Notes
---------------
• Python 3.10 is recommended for compatibility with RDKit and chemistry libraries.
• MOL2 output availability depends on the RDKit build; automatic fallback to SDF
  is implemented when MOL2 writing is not supported.
• Large ligand files may be slow to visualize in the 3D viewer.


Citation
--------
If you use pKaNET Cloud in your research, please cite:

DFDD project: 
Hengphasatporn K et al., JCIM (2026)

Dimorphite-DL:
Ropp PJ et al., J Cheminform (2019)

RDKit:
RDKit: Open-source cheminformatics software

Acknowledgements
--------
This tool uses:

pKaPredict — for machine-learning based pKa prediction.
Machine-learning meets pKa (czodrowskilab) — for research and methodology inspirations in ML-based pKa estimation.
Thank you to the authors and maintainers of these open-source projects for providing valuable foundations for this tool.

Contact
-------
For questions, bug reports, or collaboration inquiries, please contact:
Kowit Hengphasatporn, CCS, University of Tsukuba
kowith@ccs.tsukuba.ac.jp
