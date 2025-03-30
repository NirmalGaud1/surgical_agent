# surgical_co_pilot_advanced.py
import streamlit as st
import google.generativeai as genai
import torch
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
from datetime import datetime
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Configuration
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
genai.configure(api_key=API_KEY)

class AdvancedTumorAnalyzer:
    def __init__(self):
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, pretrained=True)
        self.model.eval()
        self.anatomy_model = torch.hub.load('fepegar/resnet-3d', 'resnet18', pretrained=True)
        self.pixel_spacing = (1.0, 1.0)
        self.slice_thickness = 1.0
        self.structure_db = {
            'optic_nerve': {'position': (0.4, 0.6), 'safe_distance': 5.0},
            'carotid': {'position': (0.55, 0.5), 'safe_distance': 3.0}
        }

    def _preprocess(self, image):
        transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transforms(image).unsqueeze(0)

    def analyze(self, image_file):
        # Image loading and metadata extraction
        if image_file.type == "application/dicom":
            ds = pydicom.dcmread(image_file)
            img = self._process_dicom(ds)
            metadata = {
                'pixel_spacing': ds.PixelSpacing if 'PixelSpacing' in ds else (1.0, 1.0),
                'slice_thickness': ds.SliceThickness if 'SliceThickness' in ds else 1.0,
                'study_date': ds.StudyDate if 'StudyDate' in ds else datetime.today().strftime('%Y%m%d')
            }
        else:
            img = Image.open(image_file).convert('RGB')
            metadata = {
                'pixel_spacing': (1.0, 1.0),
                'slice_thickness': 1.0,
                'study_date': datetime.today().strftime('%Y%m%d')
            }

        # Tumor segmentation
        img_tensor = self._preprocess(img)
        with torch.no_grad():
            output = torch.sigmoid(self.model(img_tensor))
        tumor_mask = (output > 0.5).float().squeeze().numpy()

        # Anatomical analysis
        anatomy_features = self._analyze_anatomy(img)

        return {
            **metadata,
            'tumor_mask': tumor_mask,
            'anatomy_features': anatomy_features,
            'size_mm2': self._calc_size(tumor_mask, metadata['pixel_spacing']),
            'volume_mm3': self._calc_volume(tumor_mask, metadata),
            'risk_assessment': self._assess_risk(tumor_mask, metadata)
        }

    def _process_dicom(self, ds):
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return Image.fromarray(img.astype('uint8')).convert('RGB')

    def _calc_size(self, mask, pixel_spacing):
        return np.sum(mask) * pixel_spacing[0] * pixel_spacing[1]

    def _calc_volume(self, mask, metadata):
        return self._calc_size(mask, metadata['pixel_spacing']) * metadata['slice_thickness']

    def _analyze_anatomy(self, image):
        # Implement 3D anatomical analysis
        return {'structures_present': ['optic_nerve', 'carotid']}

    def _assess_risk(self, mask, metadata):
        centroid = self._calc_centroid(mask)
        risks = []
        for name, struct in self.structure_db.items():
            distance = np.sqrt(
                (centroid[0]-struct['position'][0])**2 +
                (centroid[1]-struct['position'][1])**2
            ) * metadata['pixel_spacing'][0]  # Convert to mm
            if distance < struct['safe_distance']:
                risks.append(f"{name.replace('_', ' ').title()} proximity ({distance:.1f}mm)")
        return risks if risks else ["No immediate risks detected"]

    def _calc_centroid(self, mask):
        y, x = np.where(mask == 1)
        return np.mean(x), np.mean(y)

class SurgicalAssistant:
    def __init__(self):
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        self.phase_templates = {
            "Nasal": ["Mucosal dissection", "Sphenoidotomy"],
            "Sphenoid": ["Sellar exposure", "Bone removal"],
            "Sellar": ["Dural opening", "Tumor resection"],
            "Closure": ["Hemostasis", "Nasal packing"]
        }

    def generate_plan(self, analysis, query, history):
        prompt = self._build_prompt(analysis, query, history)
        response = self.gemini.generate_content(prompt)
        return self._parse_response(response.text)

    def _build_prompt(self, analysis, query, history):
        return f"""
        **Pituitary Surgery Assistant Protocol**
        
        Patient Context:
        - Tumor Size: {analysis['size_mm2']:.1f} mm²
        - Tumor Volume: {analysis['volume_mm3']:.1f} mm³
        - Risk Factors: {analysis['risk_assessment']}
        - Surgical Phase: {history['phase']}
        - Previous Steps: {history['steps'][-3:]}
        
        Current Query: {query}
        
        Generate response with:
        1. Phase-appropriate next steps
        2. Instrument recommendations
        3. Risk mitigation strategies
        4. Anatomical considerations
        """

    def _parse_response(self, text):
        try:
            return json.loads(text.strip('```json\n').strip())
        except:
            return {"response": text}

# Streamlit Interface
st.set_page_config(page_title="Neuro Surgical AI", layout="wide")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = {
        'phase': 'Nasal',
        'steps': [],
        'instruments': [],
        'complications': []
    }

# Initialize components
analyzer = AdvancedTumorAnalyzer()
assistant = SurgicalAssistant()

# Sidebar - Surgical Timeline
with st.sidebar:
    st.header("⏳ Surgical Timeline")
    phase = st.selectbox("Current Phase", ["Nasal", "Sphenoid", "Sellar", "Closure"])
    st.session_state.history['phase'] = phase
    
    st.subheader("Completed Steps")
    for step in st.session_state.history['steps']:
        st.write(f"- {step}")
    
    if st.button("🔄 Reset Procedure"):
        st.session_state.history = {
            'phase': 'Nasal',
            'steps': [],
            'instruments': [],
            'complications': []
        }

# Main Interface
st.title("🧠 Advanced Pituitary Surgery AI")

# File Upload Section
col1, col2 = st.columns([2, 3])
with col1:
    uploaded_image = st.file_uploader("Upload Pre-op Imaging", 
                                    type=["jpg", "jpeg", "png", "dcm"],
                                    help="DICOM files preferred for accurate analysis")
    
    if uploaded_image:
        analysis = analyzer.analyze(uploaded_image)
        
        # Visualization
        fig, ax = plt.subplots()
        ax.imshow(analysis['tumor_mask'], cmap='jet', alpha=0.5)
        for name, struct in analyzer.structure_db.items():
            ax.scatter(*struct['position'], label=name.replace('_', ' ').title())
        ax.legend()
        st.pyplot(fig)
        
        # Metrics
        st.subheader("📊 Quantitative Analysis")
        cols = st.columns(2)
        cols[0].metric("Tumor Area", f"{analysis['size_mm2']:.1f} mm²")
        cols[1].metric("Tumor Volume", f"{analysis['volume_mm3']:.1f} mm³")
        
        st.subheader("⚠️ Risk Assessment")
        if analysis['risk_assessment']:
            for risk in analysis['risk_assessment']:
                st.error(risk)
        else:
            st.success("No critical risks detected")

with col2:
    query = st.text_input("Surgeon Input", placeholder="Enter surgical query or command...")
    
    if query and uploaded_image:
        with st.spinner("Generating AI-assisted response..."):
            try:
                response = assistant.generate_plan(analysis, query, st.session_state.history)
                
                st.subheader("🧭 Surgical Guidance")
                if 'steps' in response:
                    st.session_state.history['steps'].extend(response['steps'])
                    for step in response['steps']:
                        st.success(f"Next Step: {step}")
                
                if 'instruments' in response:
                    st.session_state.history['instruments'].extend(response['instruments'])
                    st.subheader("🛠️ Recommended Instruments")
                    st.write(", ".join(response['instruments']))
                
                if 'warnings' in response:
                    st.subheader("🚨 Critical Warnings")
                    for warning in response['warnings']:
                        st.error(warning)
                
                if 'anatomy_notes' in response:
                    st.subheader("🧬 Anatomical Considerations")
                    st.write(response['anatomy_notes'])
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Emergency Protocol
if st.sidebar.button("🚨 Activate Emergency Protocol"):
    st.session_state.history['complications'].append("Emergency protocol activated")
    st.sidebar.error("""
    EMERGENCY PROCEDURE:
    1. Maintain airway
    2. Control hemorrhage
    3. Notify senior staff
    4. Prepare for emergency imaging
    """)

# Historical Tracking
st.sidebar.subheader("Procedure Metrics")
st.sidebar.write(f"Steps Completed: {len(st.session_state.history['steps'])}")
st.sidebar.write(f"Instruments Used: {len(set(st.session_state.history['instruments']))}")
st.sidebar.write(f"Complications: {len(st.session_state.history['complications'])}")

if __name__ == "__main__":
    st.write("**Disclaimer:** AI recommendations require clinical validation")
