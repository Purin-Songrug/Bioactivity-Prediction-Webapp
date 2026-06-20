# Bioactivity Predictor & Cheminformatics ML Pipeline

An end-to-end drug discovery pipeline built with Python and Streamlit that leverages the ChEMBL API to retrieve bioactivity data, evaluates molecular properties, computes PubChem chemical fingerprints, and benchmarks machine learning regression models to predict $pIC_{50}$ values.

[Link](https://biopredict.streamlit.app/)

<img width="959" height="440" alt="image" src="https://github.com/user-attachments/assets/d0d3f602-8ee8-48be-bbe2-73e435158276" />

---

## Features

- **Universal ChEMBL ID Resolver:** Instantly look up individual compounds (e.g., `CHEMBL25`) or cross-reference target proteins and organisms (e.g., `CHEMBL346`) to feed data directly into your pipeline.
- **Automated Data Preparation:** Real-time extraction, duplicate removal, and threshold classification (`active`, `intermediate`, `inactive`) of high-throughput $IC_{50}$ bioactivity data.
- **Exploratory Data Analysis:** Computes and visualizes Lipinski’s Rule of Five chemical space descriptors with integrated Mann-Whitney U statistical testing.
- **High-Throughput Descriptor Extraction:** Seamless local execution of the Java-based **PaDEL-Descriptor** engine to extract PubChem fingerprints.
- **Multi-Regressor Benchmarking:** Evaluates performance profiles across dozens of ML model variants simultaneously using `LazyPredict`, alongside an optimized Random Forest architecture.

---

## Prerequisites & Installation

### 1. System Requirements
- **Python 3.8 - 3.11**
- **Java Runtime Environment (JRE):** Required locally by the PaDEL-Descriptor engine to parse chemical structures.

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/bioactivity-ml-pipeline.git](https://github.com/yourusername/bioactivity-ml-pipeline.git)
cd bioactivity-ml-pipeline
