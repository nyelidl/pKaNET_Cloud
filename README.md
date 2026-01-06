# 🧪 pKaNET Cloud — AI-Based Protonation Tool
### *Fast, lightweight pKa prediction & protonation directly in Google Colab & Webserver*

This notebook provides a **cloud-ready implementation** of a **pKaNET-style machine-learning model** for estimating pKa values and generating **pH-dependent protonation states**.

Designed for simplicity, portability, and reproducibility — **no docking, no MD, no enhanced sampling**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16oz3jR6gWOzaSaJImlVflPkgT8eTcfTi?usp=sharing)

[![Open In Web](https://pkanetcloud.streamlit.app/)

![pKaNET Workflow](https://github.com/nyelidl/pKaNET_Cloud/blob/main/Pka_WF.png)

---

## 🔍 **What This Tool Does**
- Predicts **acid/base pKa values** for any small molecule
- Determines the **major microspecies** at a user-defined pH
- Generates **pH-adjusted SMILES**
- Builds **3D structures** with RDKit (ETKDG)
- Minimizes the geometry (MMFF → UFF fallback)
- Outputs both **PDB** and **SDF** formats  
  (*SDF is used for visualization because hydrogens appear correctly in py3Dmol*)

---

## **Supported Inputs**
- SMILES (.smi)
- PDB, MOL2, SDF

---

## **Automatically generated outputs:**
- pH-adjusted SMILES
- predicted formal charge
- minimized_ligand.pdb
- minimized_ligand.sdf

---

## **Ideal for:**
- ligand preparation prior to **docking** (Vina, Glide, GOLD, etc.)
- **GAFF2/CGenFF** parameterization
- **QSAR** curation
- **cheminformatics** pipelines
- teaching **pKa / protonation** concepts

---

## **A focused, clean tool:**
**Just pKa prediction → protonation → 3D structure generation**, ready to run directly in the cloud. 🚀

---

### 🙏 Acknowledgements

This tool uses:

- **[pKaPredict](https://github.com/anastasiafloris/pKaPredict)** — for machine-learning based pKa prediction.  
- **[Machine-learning meets pKa (czodrowskilab)](https://github.com/czodrowskilab/Machine-learning-meets-pKa)** — for research and methodology inspirations in ML-based pKa estimation.  

Thank you to the authors and maintainers of these open-source projects for providing valuable foundations for this tool.

---

*This code is part of the DFDD project.*
