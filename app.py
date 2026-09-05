import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Set up page configurations
st.set_page_config(page_title="Goat Toxemia AI Suite", page_icon="🐐", layout="centered")

st.title("🐐 Automated Goat Pregnancy Toxaemia Analyzer")
st.markdown("""
This production application performs pixel-level colorimetry on smartphone photos of your custom test pads to estimate 
**Glucose** and **Acetoacetate** levels, cross-referencing your custom veterinary threshold table.
""")

# --- 🧪 STEP 1: COMPLETED LABORATORY COLOR EXTRACTION MATRICES ---
# Mapped directly from your 8 glucose images (0 to 100 mg/dL scale)
GLUCOSE_PROFILES = [
    {"level": 0.0,   "rgb": [212, 214, 213], "desc": "Critical Depletion / Severe Risk"},
    {"level": 20.0,  "rgb": [203, 196, 187], "desc": "Hypoglycemic State"},
    {"level": 30.0,  "rgb": [188, 175, 168], "desc": "Borderline Subclinical Low"},
    {"level": 40.0,  "rgb": [198, 163, 145], "desc": "Minimum Normal Cutoff"},
    {"level": 60.0,  "rgb": [187, 146, 131], "desc": "Normal Range Baseline"},
    {"level": 70.0,  "rgb": [185, 137, 120], "desc": "Normal Range Baseline"},
    {"level": 80.0,  "rgb": [178, 118, 102], "desc": "Healthy Target Concentration"},
    {"level": 100.0, "rgb": [164, 94, 84],   "desc": "Optimum Energy Concentration"}
]

# Mapped sequentially from your 12 acetoacetate images (0 to 100 mg/dL scale)
ACETOACETATE_PROFILES = [
    {"level": 0.0,   "rgb": [214, 216, 215], "desc": "Negative / Normal"},
    {"level": 1.5,   "rgb": [200, 199, 194], "desc": "Normal Baseline"},
    {"level": 3.0,   "rgb": [174, 161, 184], "desc": "Normal Trace Range"},
    {"level": 7.0,   "rgb": [156, 134, 168], "desc": "SPT Lower Bound Alert"},
    {"level": 9.0,   "rgb": [146, 111, 154], "desc": "Active Subclinical State"},
    {"level": 11.5,  "rgb": [152, 105, 134], "desc": "CPT Transition Borderline"},
    {"level": 15.0,  "rgb": [142, 82, 126],  "desc": "Clinical Ketosis Range"},
    {"level": 20.0,  "rgb": [123, 62, 126],  "desc": "Severe Ketosis Range"},
    {"level": 40.0,  "rgb": [116, 44, 119],  "desc": "Critical Clinical Level"},
    {"level": 60.0,  "rgb": [114, 29, 108],  "desc": "Extreme Emergency Level"},
    {"level": 80.0,  "rgb": [103, 23, 95],   "desc": "Extreme Emergency Level"},
    {"level": 100.0, "rgb": [91, 18, 83],    "desc": "Extreme Emergency Level"}
]

# --- 🧠 STEP 2: MULTI-POINT SPECTRUM COMPUTATION ENGINE ---
def calculate_metric(sampled_rgb, profiles):
    """
    Computes mathematical Euclidean distance across the 3D RGB color spectrum 
    and uses inverse distance weighting to interpolate custom fractional concentrations.
    """
    distances = []
    for p in profiles:
        dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(sampled_rgb, p["rgb"])))
        distances.append((dist, p["level"], p["desc"]))
    
    # Sort profiles by shortest color distance match
    distances.sort(key=lambda x: x[0])
    
    d1, l1, desc1 = distances[0]
    d2, l2, desc2 = distances[1]
    
    if d1 == 0 or (d1 + d2) == 0:
        return l1, desc1
        
    # Inverse distance weighting interpolation formula
    w1 = 1.0 / (d1 + 1e-5)
    w2 = 1.0 / (d2 + 1e-5)
    
    interpolated_value = ((l1 * w1) + (l2 * w2)) / (w1 + w2)
    max_level = max(p["level"] for p in profiles)
    return min(max(interpolated_value, 0.0), max_level), desc1

# --- 📋 STEP 3: VETERINARY CLINICAL DECISION MATRIX ---
def evaluate_health_status(glucose, aceto):
    st.markdown("---")
    st.subheader("📋 AI Veterinary Health Assessment")
    
    # Clinical rules mapped perfectly to your diagnostic matrix table
    if glucose < 30.0 or aceto > 11.5:
        st.error("💥 **DIAGNOSIS: CPT (Clinical Pregnancy Toxaemia)**")
        st.markdown(f"""
        * **Estimated Glucose:** `{glucose:.1f} mg/dL` *(CPT Cutoff: < 30)*
        * **Estimated Acetoacetate:** `{aceto:.1f} mg/dL` *(CPT Cutoff: > 11.5)*
        
        **⚠️ HIGH RISK / METABOLIC CRISIS:** The doe's body is experiencing advanced clinical energy failure. 
        **Action Plan:** Contact a veterinarian immediately. Intravenous dextrose or immediate medical intervention is required to protect the mother and unborn kids.
        """)
        
    elif glucose < 40.0 or (6.9 <= aceto <= 11.5):
        st.warning("⚠️ **DIAGNOSIS: SPT (Subclinical Pregnancy Toxaemia)**")
        st.markdown(f"""
        * **Estimated Glucose:** `{glucose:.1f} mg/dL` *(SPT Cutoff: < 40)*
        * **Estimated Acetoacetate:** `{aceto:.1f} mg/dL` *(SPT Cutoff: 6.9 - 11.5)*
        
        **🔴 SUBCLINICAL ALERT PROFILE:** The animal is under severe metabolic stress without showing outward clinical indicators.
        **Action Plan:** Step up energy intake right away. Administer oral propylene glycol or high-energy drench, minimize nutritional stresses, and maximize high-quality feed access.
        """)
        
    else:
        st.success("✅ **DIAGNOSIS: Normal / Stable Metabolic Balance**")
        st.markdown(f"""
        * **Estimated Glucose:** `{glucose:.1f} mg/dL` *(Normal: 40 - 75)*
        * **Estimated Acetoacetate:** `{aceto:.1f} mg/dL` *(Normal: 0 - 6.9)*
        
        **🟢 HEALTHY Baseline:** The animal's parameters fall safely within safe baseline windows. Continue regular gestating feed management.
        """)

# --- 📱 STEP 4: APP WEB USER INTERFACE (UI) ---
st.subheader("1. Pad Image Extraction")
uploaded_file = st.file_uploader("Upload a clear photo of either your Glucose or Acetoacetate pad...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Process image matrix cleanly
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Automatically extract the average color from the center 30x30 pixels
    h, w, _ = img_rgb.shape
    cy, cx = h // 2, w // 2
    center_pixels = img_rgb[max(0, cy-15):min(h, cy+15), max(0, cx-15):min(w, cx+15)]
    extracted_rgb = [int(x) for x in np.mean(center_pixels, axis=(0, 1))]
    
    # Layout splits
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_rgb, caption="Uploaded Target Pad", use_container_width=True)
    with c2:
        st.markdown("**Extracted Electronic Color Footprint:**")
        st.code(f"R: {extracted_rgb[0]} | G: {extracted_rgb[1]} | B: {extracted_rgb[2]}")
        
        # Display matching colored chip
        chip = np.zeros((50, 180, 3), dtype=np.uint8) + extracted_rgb
        st.image(chip, caption="Sampled Target Color Chip")

    st.markdown("---")
    st.subheader("2. Run Multi-Biomarker Calibration")
    pad_choice = st.radio("Which test pad area did you upload above?", ["Glucose Pad", "Acetoacetate Pad"])

    # Track inputs safely inside user's session cache
    if "g_score" not in st.session_state: st.session_state.g_score = 55.0
    if "a_score" not in st.session_state: st.session_state.a_score = 1.0

    if pad_choice == "Glucose Pad":
        calculated_val, description = calculate_metric(extracted_rgb, GLUCOSE_PROFILES)
        st.session_state.g_score = float(calculated_val)
        st.metric("Analyzed Glucose Level", f"{calculated_val:.1f} mg/dL", help=description)
    else:
        calculated_val, description = calculate_metric(extracted_rgb, ACETOACETATE_PROFILES)
        st.session_state.a_score = float(calculated_val)
        st.metric("Analyzed Acetoacetate Level", f"{calculated_val:.1f} mg/dL", help=description)

    # Display real-time locked dashboard data matrix
    st.info(f"📊 Dashboard Values: Glucose = {st.session_state.g_score:.1f} mg/dL | Acetoacetate = {st.session_state.a_score:.1f} mg/dL")
    
    # Compute clinical decision output 
    evaluate_health_status(st.session_state.g_score, st.session_state.a_score)
else:
    st.info("💡 Please upload an image of a pad above to initialize the computer vision processing pipeline.")

st.markdown("\n\n***\n*Disclaimer: This tool is an automated screening assistant based on image metrics and colorimetry. It is intended for general informational support only and does not replace a definitive professional veterinary diagnosis or laboratory-grade blood analysis.*")
