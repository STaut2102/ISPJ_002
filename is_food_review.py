import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- 1. โหลดโมเดลและตัวแปลงข้อมูล ---
@st.cache_resource
def load_assets():
    m_ml = joblib.load('ensemble_model.pkl')
    p_ml = joblib.load('preprocessor1.pkl')
    m_nn = load_model('nn_model.h5')
    s_nn = joblib.load('scaler2.pkl')
    return m_ml, p_ml, m_nn, s_nn

model_ml, prep_ml, model_nn, scaler_nn = load_assets()

# --- 2. ส่วนเมนูข้าง (Sidebar) ---
st.sidebar.title("เมนูการใช้งาน")
page = st.sidebar.radio("เลือกหน้าเว็บ", [
    "ข้อมูลโมเดล ML", 
    "ข้อมูลโมเดล NN", 
    "ทดสอบทำนาย (ML)", 
    "ทดสอบทำนาย (NN)"
])

# --- 3. เนื้อหาแต่ละหน้า ---

# หน้าที่ 1: อธิบาย ML
if page == "ข้อมูลโมเดล ML":
    st.title("📊 รายละเอียดโมเดล Machine Learning")
    st.markdown("""
    ### แนวทางการพัฒนา
    โมเดลนี้พัฒนาขึ้นเพื่อทำนายโอกาสความสำเร็จของร้านอาหารโดยอิงจากปัจจัยภายนอก เช่น ประเภทอาหาร ราคา และทำเล
    
    ### การเตรียมข้อมูล (Data Preparation)
    - **One-Hot Encoding:** แปลงประเภทอาหาร 40 แบบ และทำเล 4 แห่ง ให้เป็นตัวเลข Matrix
    - **Standard Scaling:** ปรับสเกลราคา (Price) ให้มีความสมดุล
    
    ### ทฤษฎีอัลกอริทึม (Ensemble Learning)
    เราใช้เทคนิค **Voting Classifier** โดยรวมโมเดล 3 ประเภท:
    1. **Random Forest:** ใช้การตัดสินใจแบบกลุ่มต้นไม้
    2. **XGBoost:** ใช้วิธี Gradient Boosting เพื่อความแม่นยำสูง
    3. **Logistic Regression:** ช่วยในการคำนวณความน่าจะเป็นเชิงเส้น
    
    **แหล่งอ้างอิง:** Scikit-learn Documentation, Kaggle Dataset
    """)

# หน้าที่ 2: อธิบาย NN
elif page == "ข้อมูลโมเดล NN":
    st.title("🧠 รายละเอียดโมเดล Neural Network")
    st.markdown("""
    ### แนวทางการพัฒนา
    เน้นการวิเคราะห์ปัจจัยภายในร้าน เช่น การบริการ ความสะอาด และความพึงพอใจของลูกค้า
    
    ### การเตรียมข้อมูล (Data Preparation)
    - **Min-Max Scaling:** ปรับค่า Features ทั้งหมด (คะแนน 1-10) ให้อยู่ในช่วง 0 ถึง 1 เพื่อให้ประสาทเทียมเรียนรู้ได้ดีที่สุด
    
    ### โครงสร้างโมเดล (ANN Structure)
    โมเดลถูกออกแบบโครงสร้างเองประกอบด้วย:
    - **Input Layer:** รับ 5 ปัจจัย (Service, Rating, Time, Clean, Delivery)
    - **Hidden Layers:** 2 ชั้น (16 และ 8 Nodes) พร้อมฟังก์ชัน **ReLU**
    - **Output Layer:** 1 Node พร้อมฟังก์ชัน **Sigmoid** เพื่อส่งผลเป็นเปอร์เซ็นต์
    
    **แหล่งอ้างอิง:** TensorFlow Keras Guide
    """)

# หน้าที่ 3: ทดสอบ ML
elif page == "ทดสอบทำนาย (ML)":
    st.title("🚀 ทดสอบทำนาย (ปัจจัยภายนอก)")
    t = st.selectbox("เลือกประเภทอาหาร", ['ข้าวมันไก่', 'Fine Dining', 'ชาบูหม้อไฟ', 'Specialty Coffee', 'หมูกระทะ']) # ใส่ให้ครบ 40
    l = st.selectbox("เลือกทำเล", ['ห้างสรรพสินค้า', 'ย่านออฟฟิศ', 'ย่านที่พักอาศัย', 'แหล่งท่องเที่ยว'])
    p = st.number_input("ราคาต่อหัว (บาท)", 40, 5000, 150)
    
    if st.button("คำนวณโอกาสรอด"):
        in_data = pd.DataFrame([[t, l, p]], columns=['Type', 'Location', 'Price'])
        in_cleaned = prep_ml.transform(in_data)
        res = model_ml.predict_proba(in_cleaned)[0][1]
        st.success(f"โอกาสรอด: {res*100:.2f}%")
        st.error(f"โอกาสร่วง: {(1-res)*100:.2f}%")

# หน้าที่ 4: ทดสอบ NN
elif page == "ทดสอบทำนาย (NN)":
    st.title("🎯 ทดสอบทำนาย (ปัจจัยคุณภาพ)")
    s1 = st.slider("คะแนนบริการ", 1, 10, 5)
    s2 = st.slider("ดาวรีวิวออนไลน์", 1, 5, 3)
    s3 = st.slider("เวลารอ (นาที)", 5, 60, 20)
    s4 = st.slider("ความสะอาด", 1, 10, 5)
    s5 = st.slider("สัดส่วน Delivery", 0.0, 1.0, 0.5)
    
    if st.button("วิเคราะห์ผล"):
        in_nn = scaler_nn.transform([[s1, s2, s3, s4, s5]])
        res_nn = model_nn.predict(in_nn)[0][0]
        st.metric("โอกาสรอด", f"{res_nn*100:.2f}%")

# เครดิต
st.sidebar.markdown("---")
st.sidebar.caption("🤖 **AI Collaboration Credit**")
st.sidebar.write("Datasets & Model Architecture designed with support from **Gemini AI (Google)**")
