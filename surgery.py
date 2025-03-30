# surgical_co_pilot.py
import streamlit as st
import google.generativeai as genai
import torch
import torch.nn as nn
import base64
import numpy as np
from PIL import Image
import io
import json
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Configuration
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

class TumorAnalyzer:
    def __init__(self):
        self.model = fcn_resnet50(pretrained=True).eval()
        self.structure_coords = {
            'optic_nerve': (100, 150),
            'carotid': (200, 180),
            'pituitary_gland': (150, 200)
        }

    def analyze(self, image):
        img_tensor = self._preprocess(image)
        with torch.no_grad():
            output = self.model(img_tensor)['out']
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        return {
            'size': self._calc_size(mask),
            'location': self._calc_location(mask),
            'distances': self._calc_distances(mask),
            'risk': self._assess_risk(mask)
        }

    def _preprocess(self, image):
        transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(image).unsqueeze(0)

    def _calc_size(self, mask):
        return np.sum(mask == 1) * 0.1  # 0.1mm/pixel

    def _calc_location(self, mask):
        y, x = np.where(mask == 1)
        return (np.mean(x), np.mean(y))

    def _calc_distances(self, mask):
        centroid = self._calc_location(mask)
        return {
            name: np.sqrt((centroid[0]-c[0])**2 + (centroid[1]-c[1])**2)
            for name, c in self.structure_coords.items()
        }

    def _assess_risk(self, mask):
        distances = self._calc_distances(mask)
        if any(d < 5 for d in distances.values()):
            return "High"
        return "Moderate" if self._calc_size(mask) > 15 else "Low"

class SurgicalVLM:
    def __init__(self):
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_plan(self, prompt, image):
        response = self.gemini.generate_content([prompt, image])
        return response.text

# Streamlit Interface
st.set_page_config(page_title="Surgical AI Co-Pilot", layout="wide")

# Initialize components
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = TumorAnalyzer()
if 'vlmodel' not in st.session_state:
    st.session_state.vlmodel = SurgicalVLM()

# Sidebar
with st.sidebar:
    st.header("Patient Context")
    patient_id = st.text_input("Patient ID")
    diagnosis = st.selectbox("Diagnosis", ["Pituitary Adenoma", "Meningioma", "Craniopharyngioma"])
    surgical_phase = st.select_slider("Surgical Phase", ["Nasal", "Sphenoid", "Sellar", "Closure"])

# Main Interface
st.title("🧠 Pituitary Surgery AI Assistant")

uploaded_image = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])
query = st.text_input("Surgical Query", placeholder="Ask about tumor details or surgical planning...")

if uploaded_image and query:
    img = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded MRI Scan", use_column_width=True)
        
        # Image analysis
        with st.spinner("Analyzing tumor..."):
            analysis = st.session_state.analyzer.analyze(img)
            
        st.subheader("Tumor Analysis")
        st.metric("Size", f"{analysis['size']:.1f}mm")
        st.metric("Risk Level", analysis['risk'])
        st.write("**Critical Structure Distances:**")
        for name, dist in analysis['distances'].items():
            st.write(f"- {name.replace('_', ' ').title()}: {dist:.1f}px")

    with col2:
        # Generate surgical plan
        prompt = f"""
        Generate surgical plan for:
        - Diagnosis: {diagnosis}
        - Surgical Phase: {surgical_phase}
        - Tumor Size: {analysis['size']:.1f}mm
        - Risk Level: {analysis['risk']}
        - Query: {query}

        Format response as JSON with:
        {{
            "next_steps": [],
            "instrument_recommendations": [],
            "critical_structures": [],
            "phase_specific_risks": ""
        }}
        """
        
        with st.spinner("Generating surgical plan..."):
            try:
                response = st.session_state.vlmodel.generate_plan(prompt, img)
                plan = json.loads(response.strip('```json\n').strip())
                
                st.subheader("Surgical Plan")
                st.json(plan)
                
                st.subheader("Safety Alerts")
                if analysis['risk'] == "High":
                    st.error("🚨 Immediate Attention Required! Tumor proximity to critical structures")
                elif analysis['risk'] == "Moderate":
                    st.warning("⚠️ Caution Advised: Monitor vital signs closely")
                else:
                    st.success("✅ Low Risk Profile: Proceed with standard protocol")
                    
            except Exception as e:
                st.error(f"Failed to generate plan: {str(e)}")
