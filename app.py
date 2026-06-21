import os
import sys
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cheminformatics and Machine Learning Imports
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from lazypredict.Supervised import LazyRegressor

# --- DEFENSIVE API INITIALIZATION ---
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except Exception as e:
    CHEMBL_AVAILABLE = False
    CHEMBL_ERROR_MSG = str(e)

# --- GLOBAL STYLING & CONFIGURATION ---
st.set_page_config(page_title="BioPredict", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PADEL_JAR = SCRIPT_DIR / "padel" / "PaDEL-Descriptor" / "PaDEL-Descriptor.jar"
DEFAULT_PADEL_XML = SCRIPT_DIR / "padel" / "PaDEL-Descriptor" / "PubchemFingerprinter.xml"

# --- CORE SESSION STATE INITIALIZATION ---
state_keys = {
    "search_results": None,
    "selected_target_id": "",
    "target_name": "",
    "raw_data": None,
    "curated_data": None,
    "fingerprint_data": None,
    "benchmark_results": None,
    "production_model": None,
    "feature_columns": None,
    "prediction_outcome": None,
    "performance_plot": None
}

for key, default_value in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- ALGORITHMIC BACKEND METHODOLOGIES ---
def calculate_lipinski(smiles_series):
    molecular_weight, log_p, h_donors, h_acceptors = [], [], [], []
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecular_weight.append(Descriptors.MolWt(mol))
            log_p.append(Descriptors.MolLogP(mol))
            h_donors.append(Descriptors.NumHDonors(mol))
            h_acceptors.append(Descriptors.NumHAcceptors(mol))
        else:
            molecular_weight.append(np.nan)
            log_p.append(np.nan)
            h_donors.append(np.nan)
            h_acceptors.append(np.nan)
    return pd.DataFrame({'MW': molecular_weight, 'LogP': log_p, 'NumHDonors': h_donors, 'NumHAcceptors': h_acceptors})

def convert_ic50_to_pic50(dataframe):
    pic50 = []
    for val in dataframe['standard_value']:
        molar_val = float(val) * (10**-9)
        pic50.append(-np.log10(molar_val))
    dataframe['pIC50'] = pic50
    return dataframe.drop(columns=['standard_value'])

# --- SIDEBAR CONTROL HUB ---
st.sidebar.title("Pipeline Controller")
st.sidebar.markdown("Navigate through processing layers smoothly:")

pipeline_stage = st.sidebar.radio(
    "Select Workflow Stage:",
    [
        "1. Target Data Mining",
        "2. Exploratory Data Analysis (EDA)",
        "3. Structural Fingerprinting",
        "4. Regressor Benchmarking",
        "5. Production Training & Inference"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Memory Pipeline Monitors")
st.sidebar.indicator = lambda label, status: st.sidebar.caption(f"{label}: {'🟢 Loaded' if status else '⚪ Empty'}")
st.sidebar.indicator("Stage 1 (Raw Data)", st.session_state.raw_data is not None)
st.sidebar.indicator("Stage 2 (Curated Data)", st.session_state.curated_data is not None)
st.sidebar.indicator("Stage 3 (Fingerprints)", st.session_state.fingerprint_data is not None)
st.sidebar.indicator("Stage 5 (Trained Model)", st.session_state.production_model is not None)

st.title("BioPredict")
st.subheader("A Computational Drug Discovery & Translational Modeling Dashboard")
st.markdown("---")

# ==========================================
# STAGE 1: TARGET DATA MINING
# ==========================================
if pipeline_stage == "1. Target Data Mining":
    st.header("Step 1: Data Mining Bioactivity Metrics via ChEMBL API")
    
    # Global outbound resource link integration
    st.markdown(" Explore cross-referenced metrics on the official [ChEMBL Database Platform](https://www.ebi.ac.uk/chembl/).")
    
    st.markdown("Local Data Fallback Workspace")
    uploaded_fallback = st.file_uploader(
        "If ChEMBL times out, upload a bioactivity CSV file (Requires columns: `molecule_chembl_id`, `canonical_smiles`, `standard_value`):", 
        type=["csv"]
    )
    
    if uploaded_fallback is not None:
        try:
            fb_df = pd.read_csv(uploaded_fallback)
            req_cols = {'molecule_chembl_id', 'canonical_smiles', 'standard_value'}
            if req_cols.issubset(fb_df.columns):
                if 'class' not in fb_df.columns:
                    bio_labels = []
                    for val in fb_df['standard_value']:
                        try:
                            f_val = float(val)
                            if f_val >= 10000: bio_labels.append("inactive")
                            elif f_val <= 1000: bio_labels.append("active")
                            else: bio_labels.append("intermediate")
                        except:
                            bio_labels.append("intermediate")
                    fb_df['class'] = bio_labels
                
                st.session_state.raw_data = fb_df.reset_index(drop=True)
                st.success(f"Data processed successfully! Injected {st.session_state.raw_data.shape[0]} compounds into Stage 1.")
            else:
                st.error(f"Incompatible schema mapping. The loaded CSV file must contain: {list(req_cols)}")
        except Exception as e:
            st.error(f"Error reading file structure: {e}")
            
    st.markdown("---")
    
    st.markdown("Universal ChEMBL ID Lookup & Resolver")
    id_input = st.text_input("Enter any ChEMBL ID (e.g., Molecule CHEMBL25, Target CHEMBL346):", value="CHEMBL25")
    
    if st.button("Fetch Entity Details", key="universal_lookup_btn"):
        if not CHEMBL_AVAILABLE:
            st.error(f"ChEMBL API Endpoint Client state is disconnected: {CHEMBL_ERROR_MSG}")
        elif not id_input.strip():
            st.warning("Please supply a valid alphanumeric ChEMBL lookup tag string.")
        else:
            with st.spinner("Resolving identifier across ChEMBL registries..."):
                target_id = id_input.strip().upper()
                is_resolved = False
                
                # Route A: Attempt to evaluate entry as a Small Molecule / Compound
                try:
                    mol_info = new_client.molecule.get(target_id)
                    if mol_info and 'molecule_chembl_id' in mol_info:
                        is_resolved = True
                        pref_name = mol_info.get('pref_name') or "Unnamed Compound Reference"
                        max_phase = mol_info.get('max_phase') or "N/A"
                        structures = mol_info.get('molecule_structures') or {}
                        canonical_smiles = structures.get('canonical_smiles') or "SMILES notation data missing"
                        
                        st.success(f"**Molecule Compound Identified:** {pref_name} ({target_id})")
                        st.markdown(f"- **Max Clinical Phase:** Stage {max_phase}")
                        st.markdown(f"**Resolved Canonical SMILES:**")
                        st.code(canonical_smiles, language="text")
                        
                        if canonical_smiles != "SMILES notation data missing":
                            single_df = pd.DataFrame([{
                                'molecule_chembl_id': target_id,
                                'canonical_smiles': canonical_smiles,
                                'standard_value': 1000.0,  # Operational baseline placeholder
                                'class': 'active'
                            }])
                            st.session_state.raw_data = single_df
                            st.success(f"Successfully loaded **{target_id}** directly into the active pipeline matrix!")
                        else:
                            st.error("Cannot feed pipeline: Canonical SMILES sequence notation is missing.")
                except Exception:
                    pass  # Gracefully drop through to evaluate under target registry
                
                # Route B: Fall back to evaluate entry as a Biological Target / Organism
                if not is_resolved:
                    try:
                        tgt_info = new_client.target.get(target_id)
                        if tgt_info and 'target_chembl_id' in tgt_info:
                            is_resolved = True
                            pref_name = tgt_info.get('pref_name') or "Unnamed Target Reference"
                            organism = tgt_info.get('organism') or "Unknown Organism"
                            target_type = tgt_info.get('target_type') or "N/A"
                            
                            st.success(f"**Target Entity Identified:** {pref_name} ({target_id})")
                            st.info(f"This is a **{target_type}** entry ({organism}), not an isolated molecule. "
                                    f"It has been automatically injected into the search context below! You can now hit "
                                    f"**'Extract & Filter IC50 Bioactivity Assays'** to pull its compound library data.")
                            
                            # Automatically cross-inject into downstream selection states
                            st.session_state.selected_target_id = target_id
                            st.session_state.target_name = pref_name
                            st.session_state.search_results = pd.DataFrame([{
                                'target_chembl_id': target_id,
                                'pref_name': pref_name,
                                'target_type': target_type,
                                'organism': organism
                            }])
                    except Exception:
                        pass
                
                if not is_resolved:
                    st.error(f"Identifier `{target_id}` could not be resolved as an active Molecule or Target entry.")
                    st.info("Verify your character syntax or look up the entry manually on the ChEMBL portal.")
                    
    st.markdown("---")
    st.markdown("Live API Target Lookup Panel")
    search_query = st.text_input("Enter Disease Target Protein Name:", value="acetylcholinesterase")
    
    if st.button("Query ChEMBL Database", type="primary"):
        with st.spinner("Reaching out to ChEMBL repository vectors..."):
            try:
                targets = pd.DataFrame.from_dict(new_client.target.search(search_query))
                if not targets.empty:
                    human_targets = targets[targets.organism == 'Homo sapiens']
                    if not human_targets.empty:
                        st.session_state.search_results = human_targets[['target_chembl_id', 'pref_name', 'target_type', 'organism']].reset_index(drop=True)
                    else:
                        st.session_state.search_results = targets[['target_chembl_id', 'pref_name', 'target_type', 'organism']].reset_index(drop=True)
                    
                    st.session_state.selected_target_id = st.session_state.search_results.iloc[0]['target_chembl_id']
                    st.session_state.target_name = st.session_state.search_results.iloc[0]['pref_name']
                else:
                    st.error("No valid bioactivity targets matched your structural lookup terms.")
            except Exception as e:
                st.error(f"Database Connectivity Failure: {e}")

    if st.session_state.search_results is not None:
        st.subheader("Matching Protein Target Subsets")
        st.dataframe(st.session_state.search_results, use_container_width=True)
        st.info(f"Target Locked: **{st.session_state.selected_target_id}** ({st.session_state.target_name})")
        
        if st.button("Extract & Filter IC50 Bioactivity Assays"):
            with st.spinner("Downloading high-throughput compound metrics..."):
                try:
                    query_results = new_client.activity.filter(
                        target_chembl_id=st.session_state.selected_target_id
                    ).filter(standard_type="IC50")
                    raw_df = pd.DataFrame.from_dict(query_results)
                    
                    if not raw_df.empty:
                        clean_df = raw_df.dropna(subset=['standard_value', 'canonical_smiles']).copy()
                        clean_df = clean_df.drop_duplicates(['canonical_smiles'])
                        subset_df = clean_df[['molecule_chembl_id', 'canonical_smiles', 'standard_value']].copy()
                        subset_df['standard_value'] = pd.to_numeric(subset_df['standard_value'])
                        
                        bio_labels = []
                        for val in subset_df['standard_value']:
                            if float(val) >= 10000: bio_labels.append("inactive")
                            elif float(val) <= 1000: bio_labels.append("active")
                            else: bio_labels.append("intermediate")
                        subset_df['class'] = bio_labels
                        
                        st.session_state.raw_data = subset_df.reset_index(drop=True)
                        st.success(f"Live download successful! Loaded {st.session_state.raw_data.shape[0]} unique compounds.")
                    else:
                        st.error("The selected target profile contains no valid quantitative IC50 metrics.")
                except Exception as api_err:
                    st.error("**ChEMBL Database Query Failed**")
                    st.info("The server rejected the heavy payload. Please use the Local Data Fallback Workspace section above to seed the matrix.")

    if st.session_state.raw_data is not None:
        st.markdown("### Cached Raw Structural Slices")
        st.dataframe(st.session_state.raw_data.head(10), use_container_width=True)

# ==========================================
# STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif pipeline_stage == "2. Exploratory Data Analysis (EDA)":
    st.header("Step 2: Calculate Lipinski Parameters and Normalize Values")
    
    if st.session_state.raw_data is None:
        st.warning("Execution Interrupted: Please seed your target dataset via Stage 1 before processing.")
    else:
        if st.button("Execute Physicochemical Parameter Extraction", type="primary"):
            with st.spinner("Extracting parameters with RDKit backend engine..."):
                working_df = st.session_state.raw_data.copy()
                working_df = working_df[working_df['standard_value'] > 0].copy()
                
                lipinski_df = calculate_lipinski(working_df['canonical_smiles'])
                combined_df = pd.concat([working_df, lipinski_df], axis=1).dropna(subset=['MW', 'LogP'])
                
                st.session_state.curated_data = convert_ic50_to_pic50(combined_df).reset_index(drop=True)

        if st.session_state.curated_data is not None:
            st.success("Physicochemical metrics calculated successfully.")
            st.dataframe(st.session_state.curated_data.head(10), use_container_width=True)
            
            st.markdown("### **Structural Chemistry Asset Distributions**")
            plot_df = st.session_state.curated_data[st.session_state.curated_data['class'] != 'intermediate']
            
            if not plot_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x='class', y='pIC50', data=plot_df, palette='Set2', ax=ax)
                    ax.set_title("Target Range Spread Profile (pIC50 Analysis)")
                    st.pyplot(fig)
                    plt.close(fig)
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.scatterplot(x='MW', y='LogP', hue='class', data=plot_df, alpha=0.6, palette='Set1', ax=ax)
                    ax.set_title("Lipinski Chemical Space Entry Map")
                    st.pyplot(fig)
                    plt.close(fig)

# ==========================================
# STAGE 3: STRUCTURAL FINGERPRINTING
# ==========================================
elif pipeline_stage == "3. Structural Fingerprinting":
    st.header("Step 3: Extract Molecular Fingerprints using PaDEL")
    
    st.markdown("#### Subprocess Executable Configuration Paths")
    padel_jar_input = st.text_input("Local Route to PaDEL-Descriptor.jar:", value=str(DEFAULT_PADEL_JAR))
    padel_xml_input = st.text_input("Local Route to Fingerprinter Settings (.xml):", value=str(DEFAULT_PADEL_XML))
    row_count_limit = st.slider("Select maximum compound cohort size for extraction slice:", 10, 1000, step=10, value=100)
    
    if st.session_state.curated_data is None:
        st.warning("Execution Interrupted: Please compute Lipinski filters inside Stage 2 first.")
    else:
        if st.button("Generate High-Throughput Descriptors Matrix", type="primary"):
            if not Path(padel_jar_input).exists() or not Path(padel_xml_input).exists():
                st.error("Path Error: Could not locate background Java files at the paths specified.")
            else:
                with st.spinner("Processing molecular codes into binary feature bit matrices..."):
                    working_df = st.session_state.curated_data.head(row_count_limit).copy()
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        smi_filepath = temp_path / "compounds.smi"
                        output_filepath = temp_path / "padel_features.csv"
                        
                        working_df[['canonical_smiles', 'molecule_chembl_id']].to_csv(smi_filepath, sep='\t', index=False, header=False)
                        
                        # FIXED: Changed option flag from '-standardizenitrole' to '-standardizenitro'
                        command_arguments = [
                            "java", "-Xmx2g", "-jar", padel_jar_input,
                            "-removesalt", "-standardizenitro", "-fingerprints",
                            "-descriptortypes", padel_xml_input,
                            "-dir", temp_dir, "-file", str(output_filepath)
                        ]
                        
                        proc = subprocess.run(command_arguments, capture_output=True, text=True)
                        
                        if output_filepath.exists():
                            bits_df = pd.read_csv(output_filepath)
                            features_x = bits_df.drop(columns=['Name'])
                            target_y = working_df['pIC50'].reset_index(drop=True)
                            st.session_state.fingerprint_data = pd.concat([features_x, target_y], axis=1)
                        else:
                            st.error("The background Java PaDEL engine failed to build output files.")
                            st.code(proc.stderr)

        if st.session_state.fingerprint_data is not None:
            st.success(f"Calculated {st.session_state.fingerprint_data.shape[1] - 1} PubChem attributes across compounds.")
            st.dataframe(st.session_state.fingerprint_data.head(10), use_container_width=True)

# ==========================================
# STAGE 4: REGRESSOR BENCHMARKING
# ==========================================
elif pipeline_stage == "4. Regressor Benchmarking":
    st.header("Step 4: Prune Low-Variance Features and Screen ML Models")
    
    variance_slider = st.slider("Select Variance Threshold Pruning Cutoff Point:", 0.0, 0.10, step=0.01, value=0.05)
    
    if st.session_state.fingerprint_data is None:
        st.warning("Execution Interrupted: Feature arrays must be built inside Stage 3 first.")
    else:
        if st.button("Run Automated Machine Learning Benchmarks", type="primary"):
            with st.spinner("Screening multiple regression models simultaneously via LazyPredict..."):
                master_set = st.session_state.fingerprint_data.copy()
                X_raw = master_set.drop(columns=['pIC50'])
                Y = master_set['pIC50']
                
                pruner = VarianceThreshold(threshold=variance_slider)
                try:
                    X_pruned = pd.DataFrame(pruner.fit_transform(X_raw))
                    X_train, X_test, Y_train, Y_test = train_test_split(X_pruned, Y, test_size=0.2, random_state=42)
                    
                    scripter = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                    
                    # FIXED: Correctly capture the performance metrics dataframe as the first returned element
                    models_performance, _ = scripter.fit(X_train, X_test, Y_train, Y_test)
                    
                    # Defensive polish: Strip any whitespace from the column strings to avoid parsing hits
                    models_performance.columns = models_performance.columns.str.strip()
                    
                    st.session_state.benchmark_results = models_performance
                except Exception as e:
                    st.error(f"Variance pruning error: {e}. Try lowering the cutoff threshold.")

        if st.session_state.benchmark_results is not None:
            st.success("Automated benchmarking complete.")
            st.dataframe(st.session_state.benchmark_results, use_container_width=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            top_models = st.session_state.benchmark_results.head(15)
            
            # Identify the exact R2 column header returned by your environment's LazyPredict version
            r2_col = "R-Square" if "R-Square" in top_models.columns else "R-Squared"
            
            # Render comparison plots safely using verified columns
            sns.barplot(y=top_models.index, x=r2_col, data=top_models, hue=top_models.index, legend=False, palette='viridis', ax=ax1)
            ax1.set_title("Model Benchmarking: R2 Scores (Higher is Better)")
            ax1.set_xlim(0, 1)
            
            sns.barplot(y=top_models.index, x="RMSE", data=top_models, hue=top_models.index, legend=False, palette='magma', ax=ax2)
            ax2.set_title("Model Benchmarking: RMSE Metrics (Lower is Better)")
            st.pyplot(fig)
            plt.close(fig)

# ==========================================
# STAGE 5: PRODUCTION MODELING & INFERENCE
# ==========================================
elif pipeline_stage == "5. Production Training & Inference":
    st.header("Step 5: Train Production Random Forest Model & Run Virtual Screen")
    
    if st.session_state.fingerprint_data is None:
        st.warning("Execution Interrupted: Fingerprint training arrays must be built inside Stage 3 first.")
    else:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Model Generation Panel")
            tree_count = st.number_input("Forest Trees Hyperparameter (n_estimators):", min_value=10, max_value=500, value=100)
            
            if st.button("Train Random Forest Classifier", type="primary"):
                with st.spinner("Fitting ensemble forests to descriptor features..."):
                    master_set = st.session_state.fingerprint_data.copy()
                    X = master_set.drop(columns=['pIC50'])
                    Y = master_set['pIC50']
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                    
                    production_forest = RandomForestRegressor(n_estimators=tree_count, random_state=42)
                    production_forest.fit(X_train, Y_train)
                    
                    r2_score = production_forest.score(X_test, Y_test)
                    Y_pred = production_forest.predict(X_test)
                    
                    st.session_state.production_model = production_forest
                    st.session_state.feature_columns = X.columns.tolist()
                    
                    fig, ax = plt.subplots(figsize=(5, 5))
                    sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha': 0.5}, color='teal', ax=ax)
                    ax.set_xlabel('Experimental pIC50 Values', fontweight='bold')
                    ax.set_ylabel('Model Predicted pIC50 Values', fontweight='bold')
                    ax.set_title(f'Evaluation Space (R2 = {r2_score:.4f})')
                    st.session_state.performance_plot = fig

            if st.session_state.production_model is not None:
                st.success("Production model trained successfully.")
                st.pyplot(st.session_state.performance_plot)
                plt.close()

        with col_right:
            st.subheader("In Silico Virtual Prediction Screening")
            st.markdown("Predict the target binding affinity of an unknown compound by pasting its unique SMILES notation.")
            
            input_smiles = st.text_area("Paste SMILES Code String:", value="O=C(Cc1ccccc1)NCCN1CCCCC1")
            
            if st.button("Predict Molecule Potency Score"):
                if st.session_state.production_model is None:
                    st.error("Please train your production model using the left panel before executing screening logic.")
                else:
                    test_molecule = Chem.MolFromSmiles(input_smiles)
                    if test_molecule is None:
                        st.error("Invalid SMILES input syntax. Could not decode molecular architecture.")
                    else:
                        with st.spinner("Extracting structural features via PaDEL engine..."):
                            with tempfile.TemporaryDirectory() as temp_dir:
                                temp_path = Path(temp_dir)
                                inf_smi = temp_path / "inference.smi"
                                inf_out = temp_path / "inference_out.csv"
                                
                                pd.DataFrame({'smiles': [input_smiles], 'id': ['UNKNOWN_COMPOUND']}).to_csv(inf_smi, sep='\t', index=False, header=False)
                                
                                # FIXED: Changed option flag from '-standardizenitrole' to '-standardizenitro'
                                command_arguments = [
                                    "java", "-Xmx2g", "-jar", str(DEFAULT_PADEL_JAR),
                                    "-removesalt", "-standardizenitro", "-fingerprints",
                                    "-descriptortypes", str(DEFAULT_PADEL_XML),
                                    "-dir", temp_dir, "-file", str(inf_out)
                                ]
                                subprocess.run(command_arguments, capture_output=True, text=True)
                                
                                if inf_out.exists():
                                    inf_features = pd.read_csv(inf_out).drop(columns=['Name'])
                                    inf_features = inf_features.reindex(columns=st.session_state.feature_columns, fill_value=0)
                                    
                                    predicted_pic50 = st.session_state.production_model.predict(inf_features)[0]
                                    calculated_ic50_nm = (10 ** (-predicted_pic50)) / (10 ** -9)
                                    
                                    st.session_state.prediction_outcome = {
                                        "pIC50": predicted_pic50,
                                        "IC50_nM": calculated_ic50_nm
                                    }
                                else:
                                    st.error("Descriptor compilation failed during virtual structure screening.")

            if st.session_state.prediction_outcome is not None:
                p_pIC50 = st.session_state.prediction_outcome["pIC50"]
                p_ic50 = st.session_state.prediction_outcome["IC50_nM"]
                
                st.markdown("---")
                st.markdown("### **Predicted Affinity Metrics Output**")
                st.metric(label="Predicted Potency (pIC50 Score)", value=f"{p_pIC50:.4f}")
                st.metric(label="Estimated Concentration (IC50 value)", value=f"{p_ic50:,.2f} nM")
                
                if p_ic50 <= 1000:
                    st.success("Classification Outcome: **ACTIVE** (Highly potent lead candidate)")
                elif p_ic50 >= 10000:
                    st.error("Classification Outcome: **INACTIVE** (Low target binding activity profile)")
                else:
                    st.warning("Classification Outcome: **INTERMEDIATE**")