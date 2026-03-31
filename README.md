# SEA-AD-patient

A patient-metadata analysis repository for the **Seattle Alzheimer’s Disease Brain Cell Atlas (SEA-AD)**, focused on cleaning donor-level metadata, summarizing mixed clinical/pathology variables, and exploring associations using:

- **Factor Analysis of Mixed Data (FAMD)**
- **Normalized Mutual Information (NMI) network analysis**

This repository is **not** a GraphSAGE patient-level training project.  
Its current purpose is to build a clean and reproducible **SEA-AD donor metadata analysis workflow**.

---

## What this repository does

This project starts from the official SEA-AD donor metadata spreadsheet and produces:

1. cleaned donor metadata tables
2. one-hot encoded metadata tables
3. variable catalogs and summary statistics
4. decision-tree-based screening outputs for numeric and categorical variables
5. mixed-variable association analysis with FAMD
6. nonlinear dependency analysis with a mutual information network

---

## Data source

The repository uses the official **SEA-AD donor metadata** spreadsheet.

Current local data structure includes:

- `data/donor_metadata_Donor-metadata.xlsx`
- `output/raw/seaad_patient_metadata_original.xlsx`
- `output/processed/...`

---

## Repository structure

```text
SEA-AD-patient/
├─ .runtime/
├─ Mutual_Information_Network/
│  ├─ output/
│  ├─ results/
│  │  ├─ csv/
│  │  ├─ plots/
│  │  └─ tsv/
│  └─ scripts/
├─ check/
├─ data/
│  └─ donor_metadata_Donor-metadata.xlsx
├─ output/
│  ├─ raw/
│  └─ processed/
├─ results/
│  ├─ csv/
│  ├─ plots/
│  │  ├─ categorical/
│  │  ├─ categorical_tree/
│  │  ├─ dendrogram/
│  │  ├─ numeric/
│  │  └─ terminal_nodes/
│  └─ tsv/
├─ scripts/
│  ├─ 00_seaad_patient_metadata_analysis.py
│  ├─ 01_seaad_patient_association_method_selection.py
│  ├─ 02_seaad_patient_famd_association.py
│  ├─ 03_seaad_patient_mutual_information_network.py
│  └─ 04_run_seaad_patient_association.bat
├─ 04_run_seaad_patient_association.bat
├─ LICENSE
└─ README.md


# SEA-AD Patient Metadata Analysis

A reproducible pipeline for analyzing **SEA-AD donor-level metadata**, focusing on:

- metadata cleaning and structuring  
- mixed-variable statistical analysis  
- nonlinear association discovery  

Core methods:
- **FAMD (Factor Analysis of Mixed Data)**
- **Normalized Mutual Information (NMI) Network**

---

## Overview

This repository processes SEA-AD donor metadata to:

1. build clean, analysis-ready metadata tables  
2. summarize numeric and categorical variables  
3. explore relationships between clinical, demographic, and pathology variables  
4. identify nonlinear dependencies using information-theoretic methods  

---

## Data

Source:  
**Seattle Alzheimer’s Disease Brain Cell Atlas (SEA-AD)**  
https://brain-map.org/consortia/sea-ad

Input:
- donor metadata (`.xlsx`)

---

## Pipeline

### 1. Metadata preprocessing
`python scripts/00_seaad_patient_metadata_analysis.py`

- clean raw metadata  
- infer variable types  
- generate one-hot encoding  
- compute summary statistics  
- produce tree-based screening outputs  

---

### 2. Method selection
`python scripts/01_seaad_patient_association_method_selection.py`

- record selected analysis methods  
- save reproducibility logs  

---

### 3. FAMD analysis
`python scripts/02_seaad_patient_famd_association.py`

- model mixed variables in shared latent space  
- extract variable relationships  
- compute feature loadings and sample coordinates  

Outputs:
- relationship matrix  
- explained variance  
- feature loadings  
- visualization plots  

---

### 4. Mutual information network
`python scripts/03_seaad_patient_mutual_information_network.py`

- encode mixed variables  
- compute pairwise normalized mutual information  
- identify strong nonlinear associations  
- construct variable interaction network  

Outputs:
- NMI matrix  
- significant variable pairs  
- top association edges  
- network plots  

---

## Outputs

### Processed data
- cleaned metadata  
- one-hot encoded tables  

### Summaries
- variable catalog  
- numeric / categorical statistics  

### Analysis results
- FAMD relationship matrices  
- sample embeddings  
- feature loadings  
- mutual information network  

### Visualizations
- distribution plots  
- tree-based diagnostics  
- heatmaps  
- network graphs  

---

## Key Variables

The analysis includes mixed patient-level features such as:

- **Demographics**: age, sex, race, education  
- **Clinical**: cognitive status, diagnosis, neuroimaging  
- **Neuropathology**: Braak, CERAD, Thal, APOE  
- **Tissue quality**: PMI, brain pH, RIN  

---

## Methods

### FAMD
Captures global structure in mixed (numeric + categorical) metadata:
- interpretable latent space  
- variable contribution analysis  

### Mutual Information
Captures nonlinear dependencies:
- robust to non-monotonic relationships  
- builds association network between variables  

---

## Run Order
```
python scripts/00_seaad_patient_metadata_analysis.py
python scripts/01_seaad_patient_association_method_selection.py
python scripts/02_seaad_patient_famd_association.py
python scripts/03_seaad_patient_mutual_information_network.py
```

---

## Scope

This repository focuses on:

✔ patient metadata analysis  
✔ variable relationship discovery  
✔ reproducible preprocessing  

It does **not include**:
- graph neural networks  
- patient embedding models  
- subtype clustering  

---

## License

MIT License
