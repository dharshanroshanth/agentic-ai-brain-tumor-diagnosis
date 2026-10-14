import streamlit as st
import os
import pandas as pd
from PIL import Image
from pipeline import DiagnosticPipeline
import time
import base64
import random

# Set page config for a premium look
st.set_page_config(
    page_title="Agentic AI Based Brain Tumor Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean Medical UI CSS
st.markdown("""
<style>
    /* Medical Variables */
    :root {
        --primary: #0d6efd;
        --secondary: #6c757d;
        --success: #198754;
        --warning: #ffc107;
        --danger: #dc3545;
        --bg: #F8F9FA;
        --card-bg: #FFFFFF;
        --text-dark: #212529;
        --text-muted: #6c757d;
    }
    
    .stApp {
        background-color: var(--bg);
        color: var(--text-dark);
        font-family: 'Inter', 'Roboto', 'Segoe UI', sans-serif;
    }

    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }

    /* Main Title */
    .clinical-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        color: var(--primary);
        margin-top: 1rem;
        margin-bottom: 0rem;
        letter-spacing: 1px;
    }
    
    .subtitle {
        text-align: center;
        color: var(--secondary);
        font-size: 1.1rem;
        letter-spacing: 2px;
        margin-bottom: 3rem;
        font-weight: 500;
        text-transform: uppercase;
    }

    /* Container styling */
    .hud-box {
        background: var(--card-bg);
        border-top: 4px solid var(--primary);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .scanner-container {
        position: relative;
        overflow: hidden;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 15px;
        background: #fff;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .scanner-line {
        position: absolute;
        width: 100%; height: 2px; background: rgba(13, 110, 253, 0.5);
        box-shadow: 0 0 10px rgba(13, 110, 253, 0.5);
        animation: scan 4s ease-in-out infinite;
        z-index: 10;
    }

    @keyframes scan { 0% { top: -10%; } 50% { top: 110%; } 100% { top: -10%; } }

    /* Terminal Report (Clinical Record) */
    .clinical-record {
        background: #fdfdfd;
        border-left: 4px solid var(--primary);
        padding: 25px;
        font-family: 'Inter', sans-serif;
        color: var(--text-dark);
        line-height: 1.7;
        font-size: 1.05rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-radius: 4px;
        margin-bottom: 10px;
        white-space: pre-wrap;
    }

    /* Buttons */
    .stButton>button {
        background: var(--primary) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        padding: 15px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(13, 110, 253, 0.2) !important;
    }
    
    .stButton>button:hover {
        background: #0b5ed7 !important;
        box-shadow: 0 6px 12px rgba(13, 110, 253, 0.3) !important;
        transform: translateY(-2px);
    }

    /* Progress bars */
    .prog-bg { width: 100%; background: #e9ecef; border-radius: 4px; height: 25px; margin-top: 5px; margin-bottom: 15px; position: relative; overflow: hidden; }
    .prog-fill { height: 100%; display: flex; align-items: center; padding-left: 15px; font-weight: 600; color: #fff; font-size: 0.9rem; transition: width 1s ease; border-radius: 4px;}
    
    .c-glioma { background: #e63946; }
    .c-meningioma { background: #f4a261; }
    .c-pituitary { background: #457b9d; }
    .c-notumor { background: #2a9d8f; }

    /* Alert / Warning */
    .critical-alert { border-left: 5px solid var(--danger); background: #fff; color: var(--danger); padding: 20px; font-weight: 600; font-size: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 20px 0; border-radius: 4px;}
    .safe-alert { border-left: 5px solid var(--success); background: #fff; color: var(--success); padding: 20px; font-weight: 600; font-size: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 20px 0; border-radius: 4px;}
    
    /* Metrics */
    [data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 700; color: var(--primary) !important; }
    [data-testid="stMetricLabel"] { color: var(--secondary) !important; font-size: 1rem !important; font-weight: 600; text-transform: uppercase; }
    
    /* Loading Sequence */
    .loading-text { color: var(--primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 10px; }
    
    h2, h3, h4 { color: var(--text-dark) !important; font-weight: 600 !important; }
    .section-title { font-size: 1.5rem; color: var(--primary) !important; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# Main Titles
st.markdown('<div class="clinical-title">AGENTIC AI BASED BRAIN TUMOR DIAGNOSIS</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clinical Diagnostic Assistant</div>', unsafe_allow_html=True)

# Model Loader
@st.cache_resource(show_spinner=False)
def load_model():
    return DiagnosticPipeline()

with st.spinner("Initializing Clinical Models..."):
    pipeline = load_model()

# Layout
c1, c2 = st.columns([1, 2.2], gap="large")

with c1:
    st.markdown("<div class='section-title'>PATIENT DEMOGRAPHICS</div>", unsafe_allow_html=True)
    
    patient_name = st.text_input("Patient Full Name", placeholder="e.g., John Doe")
    
    col_age, col_gen = st.columns(2)
    with col_age:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col_gen:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("<div class='section-title'>MEDICAL IMAGING</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        temp_path = "temp_uploaded_mri.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open(temp_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            
        st.markdown(f"""
        <div class="scanner-container">
            <div class="scanner-line"></div>
            <img src="data:image/jpeg;base64,{encoded_string}" style="width: 100%; display: block;" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="scanner-container" style="height: 300px; background: #e9ecef; border: 2px dashed #ced4da;">
            <p style='color:var(--secondary); font-size:1.1rem; font-weight:500;'>[ Awaiting Image Upload ]</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # System Info HUD
    st.markdown("""
    <div class="hud-box">
        <h4 style='color:var(--text-dark); margin-top:0;'>Module Status</h4>
        <div style="color:var(--secondary); font-size:0.95rem; line-height:2.0;">
            <span style="color:var(--success); font-weight:bold;">✓</span> Primary Perception (EfficientNet)<br>
            <span style="color:var(--success); font-weight:bold;">✓</span> Auxiliary Ensembles<br>
            <span style="color:var(--warning); font-weight:bold;">⟳</span> Explainability Engine (Standby)<br>
            <span style="color:var(--primary); font-weight:bold;">✓</span> Cognitive Orchestrator
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    if uploaded_file is None:
        st.markdown("""
        <div class="hud-box" style="text-align: center; padding: 150px 20px; border-top: none; background: transparent; box-shadow: none;">
            <div style="font-size: 5rem; color: #dee2e6; margin-bottom: 20px;">⚕️</div>
            <h2 style="color:var(--secondary); font-weight: 500;">Ready for Analysis</h2>
            <p style="color:var(--text-muted); font-size:1.1rem;">Please upload a patient MRI scan in the left panel.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("Run Diagnostic Analysis"):
            st.session_state["run_analysis"] = True
            st.session_state["analysis_result"] = None
            st.session_state["animate_report"] = True
            st.session_state["patient_info"] = {
                "id": f"MRN-{random.randint(100000, 999999)}",
                "name": patient_name if patient_name.strip() else "UNKNOWN PATIENT",
                "age": patient_age,
                "gender": patient_gender
            }
            
        if st.session_state.get("run_analysis", False):
            if st.session_state.get("analysis_result") is None:
                sequence_container = st.empty()
                
                with sequence_container.container():
                    st.markdown("<h3 style='color:var(--primary);'>Processing Patient Scan...</h3>", unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = [
                        (10, "Allocating resources..."),
                        (30, "Extracting primary radiological features..."),
                        (50, "Fusing multi-model embeddings..."),
                        (70, "Applying test-time augmentations..."),
                        (85, "Calculating predictive confidence intervals..."),
                        (95, "Executing safety and verification protocols..."),
                        (100, "Synthesizing final clinical report...")
                    ]
                    
                    for target_pct, text in steps:
                        status_text.markdown(f'<div class="loading-text">{text}</div>', unsafe_allow_html=True)
                        time.sleep(0.3)
                        progress_bar.progress(target_pct)
                        
                # Execute pipeline
                try:
                    p_info = st.session_state["patient_info"]
                    patient_info_str = f"PATIENT: {p_info['name'].upper()} | AGE: {p_info['age']} YRS | SEX: {p_info['gender'].upper()} | ID: {p_info['id']}"
                    
                    result = pipeline.run_case(patient_info_str, temp_path)
                    st.session_state["analysis_result"] = result
                    
                    # Clear loading sequence
                    sequence_container.empty()
                except Exception as e:
                    st.error(f"SYSTEM FAILURE: {e}")
                    st.session_state["run_analysis"] = False
                    st.stop()
            
            try:
                result = st.session_state["analysis_result"]
                p_info = st.session_state["patient_info"]
                p_name = p_info["name"]
                patient_age = p_info["age"]
                patient_gender = p_info["gender"]
                patient_id = p_info["id"]

                # --- RESULTS ---
                st.markdown("<div class='section-title'>DIAGNOSTIC RESULTS</div>", unsafe_allow_html=True)
                
                # Patient Demographics Banner
                st.markdown(f"""
                <div style="background: #fdfdfd; padding: 15px; border-radius: 6px; border-left: 4px solid var(--secondary); margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <span style="color: var(--secondary); font-size: 0.9rem;">PATIENT NAME</span><br>
                    <strong>{p_name.upper()}</strong>
                    <hr style="margin: 10px 0; border-color: #eee;">
                    <div style="display: flex; justify-content: space-between;">
                        <div><span style="color: var(--secondary); font-size: 0.8rem;">AGE</span><br><strong>{patient_age} YRS</strong></div>
                        <div><span style="color: var(--secondary); font-size: 0.8rem;">SEX</span><br><strong>{patient_gender.upper()}</strong></div>
                        <div><span style="color: var(--secondary); font-size: 0.8rem;">PATIENT ID</span><br><strong>{patient_id}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                m1, m2 = st.columns(2)
                diag = result['diagnosis'].upper()
                conf = result['confidence'] * 100
                    
                m1.metric("PRIMARY CLASSIFICATION", diag)
                m2.metric("CONFIDENCE PROBABILITY", f"{conf:.2f}%")
                
                # Safety Gate
                if result['decision'] == "ESCALATE_TO_HUMAN":
                    st.markdown('<div class="critical-alert">⚠️ CLINICAL ALERT: Low confidence or high-risk features detected. Human review required.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-alert">✓ SYSTEM VERIFIED: High confidence automated approval.</div>', unsafe_allow_html=True)
                    
                # Explainability Heatmaps
                if result.get('has_explainability'):
                    st.markdown("<div class='section-title'>EXPLAINABILITY HEATMAPS</div>", unsafe_allow_html=True)
                    h1, h2 = st.columns(2)
                    if result.get('gradcam_path') and os.path.exists(result['gradcam_path']):
                        h1.image(result['gradcam_path'], caption="Grad-CAM Activation")
                    if result.get('segmentation_path') and os.path.exists(result['segmentation_path']):
                        h2.image(result['segmentation_path'], caption="U-Net Segmentation")
                    
                # Clinical Guidelines & Next Steps
                from agents.report_generator import PROTOCOLS
                info = PROTOCOLS.get(result['diagnosis'], PROTOCOLS["no_tumor"])
                
                st.markdown("<div class='section-title'>RECOMMENDED CLINICAL ACTIONS</div>", unsafe_allow_html=True)
                
                st.info(f"**Urgency Level:** {info['urgency']}")
                st.markdown(f"**Standard Guidelines:** {info['guidelines']}")
                
                action_html = "<ul style='color: var(--text-dark); line-height: 1.8;'>"
                for measure in info['measures']:
                    action_html += f"<li>{measure.lstrip('- ')}</li>"
                action_html += "</ul>"
                st.markdown(action_html, unsafe_allow_html=True)
                
                # Probability Bars
                st.markdown("<div class='section-title'>DIFFERENTIAL PROBABILITIES</div>", unsafe_allow_html=True)
                
                probs = result['all_probs']
                class_colors = {
                    "glioma": "c-glioma",
                    "meningioma": "c-meningioma",
                    "pituitary": "c-pituitary",
                    "no_tumor": "c-notumor"
                }
                
                bars_html = ""
                for cls_name, prob in probs.items():
                    percent = prob * 100
                    css_class = class_colors.get(cls_name, "")
                    bars_html += f"""
                    <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-dark); margin-bottom: 5px;">
                        {cls_name.replace('_', ' ').title()} <span style="float:right; color:var(--text-muted);">{percent:.2f}%</span>
                    </div>
                    <div class="prog-bg">
                        <div class="prog-fill {css_class}" style="width: {percent}%;"></div>
                    </div>
                    """
                st.markdown(bars_html, unsafe_allow_html=True)
                
                # Clinical Record
                st.markdown("<div class='section-title'>CLINICAL SYNTHESIS</div>", unsafe_allow_html=True)
                
                terminal_box = st.empty()
                report_text = result["report"]
                
                if st.session_state.get("animate_report", True):
                    displayed_text = ""
                    for line in report_text.split('\\n'):
                        displayed_text += line + "\\n"
                        terminal_box.markdown(f'<div class="clinical-record">{displayed_text}</div>', unsafe_allow_html=True)
                        time.sleep(0.05) 
                    st.session_state["animate_report"] = False
                else:
                    terminal_box.markdown(f'<div class="clinical-record">{report_text}</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Export EHR Report",
                    data=report_text,
                    file_name=f"{patient_id}_Clinical_Report.txt",
                    mime="text/plain",
                )
                
            except Exception as e:
                st.error(f"RENDER FAILURE: {e}")
