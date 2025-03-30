# surgical_co_pilot_final.py
import streamlit as st
import google.generativeai as genai
import torch
import torch.nn as nn
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
from datetime import datetime
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from monai.networks.nets import UNet  # Medical imaging-specific model

# Configuration
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c" 
genai.configure(api_key=API_KEY)

class RobustTumorAnalyzer:
    def __init__(self):
        # Initialize with MONAI's medical imaging model
        self.segmentation_model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2
        )
        
        # 3D analysis model using standard PyTorch
        self.anatomy_model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        
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
        try:
            if image_file.type == "application/dicom":
                ds = pydicom.dcmread(image_file)
                img = self._process_dicom(ds)
                metadata = self._extract_dicom_metadata(ds)
            else:
                img, metadata = self._process_standard_image(image_file)

            img_tensor = self._preprocess(img)
            with torch.no_grad():
                output = torch.sigmoid(self.segmentation_model(img_tensor))
            
            tumor_mask = (output > 0.5).float().squeeze().numpy()
            
            return {
                **metadata,
                'tumor_mask': tumor_mask,
                'size_mm2': self._calc_size(tumor_mask, metadata['pixel_spacing']),
                'volume_mm3': self._calc_volume(tumor_mask, metadata),
                'risk_assessment': self._assess_risk(tumor_mask, metadata)
            }
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

    def _process_dicom(self, ds):
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return Image.fromarray(img.astype('uint8')).convert('RGB')

    def _extract_dicom_metadata(self, ds):
        return {
            'pixel_spacing': ds.PixelSpacing if 'PixelSpacing' in ds else (1.0, 1.0),
            'slice_thickness': ds.SliceThickness if 'SliceThickness' in ds else 1.0,
            'study_date': ds.StudyDate if 'StudyDate' in ds else datetime.today().strftime('%Y%m%d')
        }

    def _process_standard_image(self, image_file):
        return Image.open(image_file).convert('RGB'), {
            'pixel_spacing': (1.0, 1.0),
            'slice_thickness': 1.0,
            'study_date': datetime.today().strftime('%Y%m%d')
        }

    def _calc_size(self, mask, pixel_spacing):
        return np.sum(mask) * pixel_spacing[0] * pixel_spacing[1]

    def _calc_volume(self, mask, metadata):
        return self._calc_size(mask, metadata['pixel_spacing']) * metadata['slice_thickness']

    def _assess_risk(self, mask, metadata):
        centroid = self._calc_centroid(mask)
        risks = []
        for name, struct in self.structure_db.items():
            distance = np.sqrt(
                (centroid[0]-struct['position'][0])**2 +
                (centroid[1]-struct['position'][1])**2
            ) * metadata['pixel_spacing'][0]
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
        try:
            prompt = self._build_prompt(analysis, query, history)
            response = self.gemini.generate_content(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            st.error(f"Planning failed: {str(e)}")
            return {"error": str(e)}

    def _build_prompt(self, analysis, query, history):
        return f"""
        **Surgical Planning Request**
        
        Patient Context:
        - Tumor Area: {analysis['size_mm2']:.1f} mm²
        - Tumor Volume: {analysis['volume_mm3']:.1f} mm³
        - Identified Risks: {', '.join(analysis['risk_assessment'])}
        - Current Phase: {history['phase']}
        - Recent Steps: {', '.join(history['steps'][-3:])}
        
        Query: {query}

        Required Response Format:
        {{
            "steps": ["step1", "step2"],
            "instruments": ["instrument1", "instrument2"],
            "warnings": ["warning1", "warning2"],
            "anatomy_notes": "Notes about anatomical considerations"
        }}
        """

    def _parse_response(self, text):
        try:
            return json.loads(text.strip('```json\n').strip())
        except json.JSONDecodeError:
            return {"response": text}

# Streamlit Interface
st.set_page_config(page_title="Medical Surgical AI", layout="wide")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = {
        'phase': 'Nasal',
        'steps': [],
        'instruments': [],
        'complications': []
    }

# Initialize components
analyzer = RobustTumorAnalyzer()
assistant = SurgicalAssistant()

# Sidebar
with st.sidebar:
    st.header("🕒 Surgical Progress")
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
st.title("🏥 AI-Powered Surgical Assistant")

# File Upload
uploaded_image = st.file_uploader("Upload Medical Imaging", 
                                type=["jpg", "jpeg", "png", "dcm"],
                                help="DICOM files recommended for precise analysis")

if uploaded_image:
    analysis = analyzer.analyze(uploaded_image)
    
    if analysis:  # Only proceed if analysis succeeded
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Visualization
            fig, ax = plt.subplots()
            ax.imshow(analysis['tumor_mask'], cmap='jet', alpha=0.5)
            for name, struct in analyzer.structure_db.items():
                ax.scatter(*struct['position'], label=name.replace('_', ' ').title())
            ax.legend()
            st.pyplot(fig)
            
            # Metrics
            st.subheader("📈 Quantitative Analysis")
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
            
            if query:
                response = assistant.generate_plan(analysis, query, st.session_state.history)
                
                if 'error' not in response:
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

# Emergency Protocol
if st.sidebar.button("🚨 Activate Emergency Protocol"):
    st.session_state.history['complications'].append("Emergency protocol activated")
    st.sidebar.error("""
    EMERGENCY PROCEDURE:
    1. Maintain airway and breathing
    2. Control active hemorrhage
    3. Notify senior surgical team
    4. Initiate emergency imaging protocol
    5. Prepare for potential conversion to open procedure
    """)

# Track Procedure Metrics
st.sidebar.subheader("Procedure Statistics")
st.sidebar.metric("Steps Completed", len(st.session_state.history['steps']))
st.sidebar.metric("Unique Instruments Used", len(set(st.session_state.history['instruments'])))
st.sidebar.metric("Complications Recorded", len(st.session_state.history['complications']))

if __name__ == "__main__":
    st.write("**Important:** AI recommendations should be verified by qualified medical professionals")
