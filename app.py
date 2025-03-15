import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def main():
    st.set_page_config(page_title="AI Model Web App")
    
    # CSS ที่ปรับปรุงใหม่ โดยลดระยะห่างของแท็บ
    st.markdown(
        """
        <style>
        /* พื้นหลังและสีตัวอักษรทั่วไป */
        body {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* การจัดแต่งแท็บ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;  /* ปรับจาก 40px เป็น 10px เพื่อให้แท็บชิดกันมากขึ้น */
            justify-content: center;
            background: rgba(255, 255, 255, 0.05);
            padding: 10px 0;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 600;
            color: #e0e0e0;
            padding: 12px 25px;
            border-radius: 12px;
            background: linear-gradient(145deg, #2c3e50, #4ca1af);
            transition: all 0.4s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(145deg, #3498db, #5dade2);
            color: #ffffff;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.25);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(145deg, #27ae60, #2ecc71) !important;
            color: #ffffff !important;
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
        }
        
        /* กล่องเนื้อหา */
        .content-box {
            background: rgba(255, 255, 255, 0.08);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            text-align: center;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
        }
        
        /* หัวข้อ */
        h1, h2, h3 {
            color: #f1c40f;
            text-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        /* อินพุตและปุ่ม */
        .stNumberInput > div > div, .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
        }
        .stNumberInput input, .stSelectbox select {
            color: #ffffff;
        }
        
        /* ตาราง */
        .stDataFrame {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 10px;
        }
        
        /* Animation */
        .stMarkdown {
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #3498db;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #2980b9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # สร้างแท็บเมนูหลัก
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📊 Model Machine Learning", "📊 Model Neural Network", "📝 Theory Of ML & NN"])

    with tab1:
        show_overview()

    with tab2:
        show_demo1()

    with tab3:
        show_demo2()

    with tab4:
        show_new_page()

def load_data():
    df = pd.read_csv("wine_data.csv")  # ปรับ path ตามที่คุณกำหนด
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def show_overview():
    st.title("Detail of Dataset")
    page = st.selectbox("🔍 Select Category", ["แหล่งที่มาของข้อมูล DataSet (ML)", 
                                               "Feature Engineering (ML)", 
                                               "Fix missing Value DataSet",
                                               "แหล่งที่มาของข้อมูล DataSet (NN)",
                                               "Feature Engineering (NN)",                                           
                                               ])
    
    if page == "แหล่งที่มาของข้อมูล DataSet (ML)":
        st.subheader("แหล่งที่มาของข้อมูล Data Set")
        st.write("**ที่มา:** Kaggle Website Platform Dataset")
        st.write(
            "ชุดข้อมูล Wine Dataset มาจากการวิเคราะห์ทางเคมีของไวน์ที่ปลูกในภูมิภาคเดียวกันในอิตาลี "
            "โดยเริ่มต้นจากงานวิจัยในยุค 1990s และเผยแพร่ผ่าน UCI Machine Learning Repository "
            "ก่อนถูกนำมาใช้อย่างแพร่หลายใน Kaggle กลุ่มของเราสนใจนำข้อมูลนี้มาฝึกโมเดล Machine Learning "
            "เพื่อทำนาย **Proline** ซึ่งเป็นสารต้านอนุมูลอิสระที่สำคัญในไวน์ และช่วยบ่งบอกถึงคุณภาพและความสมบูรณ์ของไวน์ "
            "นอกจากนี้ยังสามารถใช้ในการจำแนกหรือจัดกลุ่มประเภทของไวน์ได้อีกด้วย"
        )
        st.write("ข้อมูลใน Dataset Wine")
        df = pd.read_csv(r"dataset/wine-clustering.csv")
        st.dataframe(df)

    elif page == "แหล่งที่มาของข้อมูล DataSet (NN)" :
        st.subheader("แหล่งที่มาของข้อมูล Data Set")
        st.write("**ที่มา:** Kaggle Website Platform Dataset")
        st.write(
            "ชุดข้อมูล accident มาจากการวิเคราะห์อัตราการการรอดและไม่รอดโดย จะมีเพศ ความเร็วในการขับขี่"
            "ใส่หมวกหรือไม่ ช่วงอายุ และเพศ โดยใช้ Model Neural Network ทำนายอัตรารอดและอัตราไม่รอด โดยให้"
            "ทำนายจาก Dataset โดยฝึกโมเดลเป็น Supervised Learning"
        )
        df = pd.read_csv(r"dataset/accident.csv")
        st.dataframe(df)

    elif page == "Feature Engineering (NN)":
        st.subheader("Feature Engineering (Neural Network)")
        st.write("ชุดข้อมูลที่ใช้ในการพัฒนาโมเดล Neural Network มาจาก `accident.csv` ซึ่งประกอบด้วยข้อมูลเกี่ยวกับอุบัติเหตุ "
                "โดยเราเลือกใช้ 5 ฟีเจอร์หลักเพื่อทำนายโอกาสรอดชีวิต (Survived) ดังนี้:")
        
        st.write("""
        - **Seatbelt_Used**: การสวมเข็มขัดนิรภัย (Yes/No)
          - **ที่มา**: คอลัมน์ใน dataset บ่งบอกว่าผู้ประสบเหตุสวมเข็มขัดนิรภัยหรือไม่
          - **การแปลง**: ใช้ LabelEncoder แปลง 'Yes' เป็น 1 และ 'No' เป็น 0 เพื่อให้โมเดลเข้าใจได้
          - **ความสำคัญ**: เข็มขัดนิรภัยช่วยลดความรุนแรงของการบาดเจ็บในอุบัติเหตุ
        
        - **Speed_of_Impact**: ความเร็วขณะเกิดการชน (หน่วย: km/h)
          - **ที่มา**: ค่าความเร็วที่บันทึกจากเหตุการณ์
          - **การแปลง**: เติม missing values ด้วยค่าเฉลี่ย (mean) ของคอลัมน์นี้
          - **ความสำคัญ**: ความเร็วสูงสัมพันธ์กับโอกาสรอดชีวิตที่ลดลง
        
        - **Helmet_Used**: การสวมหมวกกันน็อก (Yes/No)
          - **ที่มา**: คอลัมน์ใน dataset บ่งบอกว่าผู้ประสบเหตุสวมหมวกกันน็อกหรือไม่
          - **การแปลง**: ใช้ LabelEncoder แปลง 'Yes' เป็น 1 และ 'No' เป็น 0
          - **ความสำคัญ**: หมวกกันน็อกลดความเสี่ยงการบาดเจ็บที่ศีรษะ
        
        - **Age**: อายุของผู้ประสบเหตุ (หน่วย: ปี)
          - **ที่มา**: คอลัมน์ใน dataset แสดงอายุของบุคคล
          - **การแปลง**: เติม missing values ด้วยค่าเฉลี่ย (mean) และใช้เป็นตัวเลขต่อเนื่อง
          - **ความสำคัญ**: อายุอาจสัมพันธ์กับความแข็งแรงทางร่างกายและการตอบสนองต่ออุบัติเหตุ
        
        - **Gender**: เพศของผู้ประสบเหตุ (Male/Female/Unknown)
          - **ที่มา**: คอลัมน์ใน dataset บ่งบอกเพศ
          - **การแปลง**: เติม missing values ด้วย 'Unknown' และใช้ LabelEncoder แปลงเป็น 0 (Female), 1 (Male), 2 (Unknown)
          - **ความสำคัญ**: เพศอาจมีผลต่อความเสี่ยงหรือพฤติกรรมการขับขี่
        """)


    elif page == "Feature Engineering (ML)":
        st.subheader("Feature Engineering (ML)")
        st.write("ชุดข้อมูลประกอบด้วย **13 ฟีเจอร์** ดังนี้:")
        st.write(
            "- **Alcohol**: ปริมาณแอลกอฮอล์ (% โดยปริมาตร) ส่งผลต่อรสชาติและความเข้มข้น\n"
            "- **Malic Acid**: กรดมาลิก เกี่ยวข้องกับความเปรี้ยวและกลิ่นผลไม้\n"
            "- **Ash**: เถ้าที่เหลือจากการเผาไหม้ บ่งบอกแร่ธาตุ\n"
            "- **Ash Alcanity**: ความเป็นด่างของเถ้า สะท้อนสมดุลกรด-ด่าง\n"
            "- **Magnesium**: ปริมาณแมกนีเซียม แร่ธาตุสำคัญในกระบวนการหมัก\n"
            "- **Total Phenols**: ฟีนอลทั้งหมด เกี่ยวข้องกับสีและความฝาด\n"
            "- **Flavanoids**: ฟลาโวนอยด์ สารต้านอนุมูลอิสระและรสชาติ\n"
            "- **Nonflavanoid Phenols**: ฟีนอลที่ไม่ใช่ฟลาโวนอยด์ ปรับสมดุลรสชาติ\n"
            "- **Proanthocyanins**: โปรแอนโธไซยานิน สารต้านอนุมูลอิสระที่ส่งผลต่อสี\n"
            "- **Color Intensity**: ความเข้มของสีไวน์\n"
            "- **Hue**: โทนสีของไวน์ เช่น แดงเข้มหรืออ่อน\n"
            "- **OD280/OD315 of Diluted Wines**: ค่าการดูดกลืนแสง วัดโปรตีนและฟีนอล\n"
            "- **Proline**: กรดอะมิโนเป้าหมาย สารต้านอนุมูลอิสระบ่งบอกคุณภาพ"
        )
        st.write("ฟีเจอร์เหล่านี้ถูกใช้ในการทำนาย Proline และจัดกลุ่มไวน์")

    elif page == "Fix missing Value DataSet":
        st.subheader("Fix Missing Value DataSet")
        st.write(
            "ในชุดข้อมูลดั้งเดิมจาก UCI ไม่มี Missing Values แต่ในทางปฏิบัติอาจพบข้อมูลขาดหายได้ "
            "เราเลือกจัดการโดย**เติมค่าที่ขาดหายด้วยค่าเฉลี่ย (Mean)** ของแต่ละคอลัมน์ เพราะ:\n"
            "- รักษาคุณสมบัติโดยรวมของข้อมูล\n"
            "- ลดผลกระทบจากค่าที่หายไปโดยไม่ต้องลบข้อมูล\n"
            "- เหมาะกับโมเดล SVR และ KNN ที่ต้องการข้อมูลครบถ้วน"
        )
        st.write("ตัวอย่าง: ถ้า Magnesium ขาดหาย จะเติมด้วยค่าเฉลี่ยของคอลัมน์นั้น")
        st.subheader("การแก้ไข Missing Data ให้มีความสมบูรณ์ให้กับการ Train Model")
        st.write("ต้องใช้ค่าเฉลี่ย ? : ไม่ต้องคำนวณอะไรซับซ้อนมากมาย และหาค่าเฉลี่ยของคอลัมน์นั้น ๆ แล้วใส่ข้อมูลใหม่ทดแทนเข้าไปที่ขาดหายไป")
        st.write("การแก้ไขให้ Missing Data มีความสมบูรณ์ให้กับการ Train Mode")


def show_demo1():
    st.title("📊 Model Machine Learning: Proline Wine Prediction with SVR and KNN")
    
    svr_model_path = "modelml/svr_model_new.pkl"  # ปรับ path
    knn_model_path = "modelml/knn_model_new.pkl"  # ปรับ path
    scaler_path = "modelml/scaler.pkl"  # ปรับ path
    
    try:
        with open(svr_model_path, 'rb') as f:
            svr_model = pickle.load(f)
        with open(knn_model_path, 'rb') as f:
            knn_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ .pkl กรุณาตรวจสอบ path!")
        return

    # Input: ป้อนข้อมูลด้วยมือ
    st.subheader("Input Data")
    st.write("กรุณาป้อนข้อมูล 12 ฟีเจอร์และ True Proline สำหรับทำนายและเปรียบเทียบ:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alcohol = st.number_input("Alcohol", min_value=0.0, value=14.23)
        malic_acid = st.number_input("Malic Acid", min_value=0.0, value=1.71)
        ash = st.number_input("Ash", min_value=0.0, value=2.43)
    with col2:
        ash_alcanity = st.number_input("Ash Alcanity", min_value=0.0, value=15.6)
        magnesium = st.number_input("Magnesium", min_value=0.0, value=127.0)
        total_phenols = st.number_input("Total Phenols", min_value=0.0, value=2.8)
    with col3:
        flavanoids = st.number_input("Flavanoids", min_value=0.0, value=3.06)
        nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", min_value=0.0, value=0.28)
        proanthocyanins = st.number_input("Proanthocyanins", min_value=0.0, value=2.29)
    with col4:
        color_intensity = st.number_input("Color Intensity", min_value=0.0, value=5.64)
        hue = st.number_input("Hue", min_value=0.0, value=1.04)
        od280 = st.number_input("OD280", min_value=0.0, value=3.92)
        true_proline = st.number_input("True Proline", min_value=0.0, value=1065.0)  # เพิ่ม True Proline

    # สร้าง DataFrame จาก Input
    input_data = pd.DataFrame({
        'Alcohol': [alcohol], 'Malic_Acid': [malic_acid], 'Ash': [ash], 'Ash_Alcanity': [ash_alcanity],
        'Magnesium': [magnesium], 'Total_Phenols': [total_phenols], 'Flavanoids': [flavanoids],
        'Nonflavanoid_Phenols': [nonflavanoid_phenols], 'Proanthocyanins': [proanthocyanins],
        'Color_Intensity': [color_intensity], 'Hue': [hue], 'OD280': [od280]
    })

    # ปรับสเกลข้อมูล Input
    input_scaled = scaler.transform(input_data)

    # ทำนายด้วย SVR และ KNN
    svr_pred = svr_model.predict(input_scaled)
    knn_pred = knn_model.predict(input_scaled)

    # Output: แสดงผลการทำนาย
    st.subheader("Output: Predicted Proline")
    st.write("**True Proline (ป้อนด้วยมือ):**", true_proline)
    st.write("**SVR Prediction:**", svr_pred[0])
    st.write("**KNN Prediction:**", knn_pred[0])

    # คำนวณเมตริก
    svr_mae = mean_absolute_error([true_proline], svr_pred)
    svr_r2 = r2_score([true_proline], svr_pred)  # R² อาจไม่สมเหตุสมผลกับข้อมูลเดี่ยว แต่คำนวณไว้
    svr_rmse = np.sqrt(mean_squared_error([true_proline], svr_pred))

    knn_mae = mean_absolute_error([true_proline], knn_pred)
    knn_r2 = r2_score([true_proline], knn_pred)  # R² อาจไม่สมเหตุสมผลกับข้อมูลเดี่ยว แต่คำนวณไว้
    knn_rmse = np.sqrt(mean_squared_error([true_proline], knn_pred))

    # แสดงตารางเปรียบเทียบ
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['SVR', 'KNN'],
        'MAE': [svr_mae, knn_mae],
        'R²': [svr_r2, knn_r2],
        'RMSE': [svr_rmse, knn_rmse]
    })
    st.write("**เปรียบเทียบความแม่นยำ:**")
    st.dataframe(comparison_df.style.format({"MAE": "{:.4f}", "R²": "{:.4f}", "RMSE": "{:.4f}"}))

    # กราฟเปรียบเทียบ True vs Predicted
    st.subheader("Visualization: True vs Predicted Proline")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # กราฟ SVR
    ax1.scatter([true_proline], svr_pred, c='blue', edgecolor='k', s=100, label='SVR Prediction')
    ax1.plot([true_proline, true_proline], [true_proline, svr_pred[0]], 'b--', lw=2)
    ax1.set_xlabel('True Proline')
    ax1.set_ylabel('Predicted Proline (SVR)')
    ax1.set_title(f'SVR: True vs Predicted\nMAE: {svr_mae:.2f}, RMSE: {svr_rmse:.2f}')
    ax1.legend()

    # กราฟ KNN
    ax2.scatter([true_proline], knn_pred, c='green', edgecolor='k', s=100, label='KNN Prediction')
    ax2.plot([true_proline, true_proline], [true_proline, knn_pred[0]], 'g--', lw=2)
    ax2.set_xlabel('True Proline')
    ax2.set_ylabel('Predicted Proline (KNN)')
    ax2.set_title(f'KNN: True vs Predicted\nMAE: {knn_mae:.2f}, RMSE: {knn_rmse:.2f}')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig1)

    # กราฟแท่งเปรียบเทียบเมตริก
    st.subheader("Visualization: Metric Comparison")
    fig2, ax = plt.subplots(figsize=(10, 6))
    comparison_df_melted = comparison_df.melt(id_vars=['Model'], value_vars=['MAE', 'R²', 'RMSE'], var_name='Metric', value_name='Value')
    sns.barplot(x='Metric', y='Value', hue='Model', data=comparison_df_melted, palette='viridis', ax=ax)
    ax.set_title('Comparison of SVR and KNN Metrics')
    ax.set_ylabel('Value')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

def show_demo2():
    st.title("📈 Neural Network Model: Survival Prediction")
    st.write("โมเดลนี้ใช้ Neural Network เพื่อทำนายโอกาสรอดชีวิตจากข้อมูล Seatbelt_Used, Speed_of_Impact, Helmet_Used, Age, และ Gender")

    # โหลดโมเดล Neural Network
    nn_model_path = "modelml/neural_network_model.keras"  # ปรับ path ตามที่เซฟไว้
    try:
        nn_model = keras.models.load_model(nn_model_path)
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ neural_network_model.keras กรุณาตรวจสอบ path!")
        return

    # Input: ป้อนข้อมูลด้วยมือ (5 inputs)
    st.subheader("Input Data")
    st.write("กรุณาป้อนข้อมูล 5 ฟีเจอร์สำหรับทำนายโอกาสรอดชีวิต:")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        seatbelt_used = st.selectbox("Seatbelt Used", ["Yes", "No"])
    with col2:
        speed_of_impact = st.number_input("Speed of Impact (km/h)", min_value=0.0, value=50.0)
    with col3:
        helmet_used = st.selectbox("Helmet Used", ["Yes", "No"])
    with col4:
        age = st.number_input("Age", min_value=0, value=30)
    with col5:
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])

    # แปลงข้อมูล input เป็น numerical
    seatbelt_used_num = 1 if seatbelt_used == "Yes" else 0
    helmet_used_num = 1 if helmet_used == "Yes" else 0
    gender_num = {"Male": 1, "Female": 0, "Unknown": 2}.get(gender)  # แปลง Gender ให้ตรงกับ LabelEncoder
    input_data = np.array([[seatbelt_used_num, speed_of_impact, helmet_used_num, age, gender_num]])

    # ทำนายด้วย Neural Network
    nn_pred = nn_model.predict(input_data)[0][0]

    st.subheader("Output: Survival Prediction")
    st.write(f"**Probability of Not Surviving (1 = Not Survived, 0 = Survived):** {nn_pred:.4f}")
    survival_text = "Survived" if nn_pred > 0.50 else "Not Survived" 
    st.write(f"**Prediction:** {survival_text}")

    # Visualization
    st.subheader("Visualization: Prediction Probability")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Probability"], [nn_pred], color='orange' if nn_pred > 0.50 else 'red', alpha=0.7) 
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Neural Network Prediction Probability")
    ax.axhline(0.50, color='gray', linestyle='--', label='Threshold (0.500)')  
    ax.text(0, nn_pred + 0.05, f'{nn_pred:.4f}', ha='center', va='bottom')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.write("""
    **การตีความผลลัพธ์:**
    - ค่าที่ทำนายคือความน่าจะเป็น (0 ถึง 1)
    - ถ้า > 0.50: คาดการณ์ว่ารอดชีวิต (Survived)
    - ถ้า <= 0.50: คาดการณ์ว่าไม่รอดชีวิต (Not Survived)
    """)

def show_new_page():
    st.title("📊 Theory of ML and How to Devlopment Model")
    page = st.selectbox("🔍 Select Category", ["Algorithm ML",
                                               "Algorithm NN", 
                                               "Devlopment ML", 
                                               "Devlopment NN"])
    
    if page == "Algorithm ML":
        st.subheader("SVR (Support Vector Regression)")
        st.write("SVR เป็นหนึ่งในเทคนิคของ Support Vector Machine (SVM) ที่ถูกปรับให้เหมาะกับงาน Regression (ทำนายค่าตัวเลข) "
        "แทนที่จะเป็น Classification (จำแนกประเภท) ซึ่ง SVM เดิมถูกออกแบบมา SVM จะพยายามหาเส้นหรือระนาบ (Hyperplane) ที่แยกข้อมูลได้ดีที่สุด "
        "แต่ใน SVR จะหาเส้น (หรือระนาบในมิติสูง) ที่สามารถทำนายค่าได้ใกล้เคียงกับข้อมูลจริงมากที่สุด โดยมีข้อจำกัดบางอย่างที่ทำให้โมเดลมีความยืดหยุ่นและทนต่อความคลาดเคลื่อน")
        st.image("img/SVR.png",caption="SVR")
        st.subheader("KNN (K-Nearest Neighbors)")
        st.write("KNN เป็นอัลกอริทึม Machine Learning ที่เรียบง่ายแต่ทรงพลัง ทำงานโดยใช้หลักการ 'เพื่อนบ้านที่ใกล้ที่สุด'" 
        "โดยปกติ KNN ถูกใช้ในงาน Classification (จำแนกประเภท) แต่ในกรณีของคุณ "
        "เราใช้ KNN Regression (KNeighborsRegressor) เพื่อทำนายค่าเชิงตัวเลข เช่น Proline")
        st.image("img/KNN.png",caption="KNN")

    elif page == "Algorithm NN":
        st.subheader("Neural Network")
        st.write(
        "Neural Network (NN) เป็นโมเดลที่ได้รับแรงบันดาลใจจากโครงสร้างของสมองมนุษย์ โดยใช้หน่วยที่เรียกว่า "
        "Neuron ในการประมวลผลข้อมูล ซึ่งประกอบไปด้วย 3 ส่วนหลัก ได้แก่:")

        st.markdown(
        """
        - **Input Layer (ชั้นอินพุต):** รับข้อมูลที่เป็น Feature เข้ามาในโมเดล
        - **Hidden Layers (ชั้นซ่อน):** ประมวลผลข้อมูลโดยใช้ Weights และ Activation Function เช่น ReLU หรือ Sigmoid
        - **Output Layer (ชั้นเอาต์พุต):** ให้ผลลัพธ์สุดท้ายของโมเดล เช่น การทำนายว่า "รอดชีวิต" หรือ "ไม่รอดชีวิต"
        """)

        st.write(
        "Neural Network ทำงานโดยใช้กระบวนการ Forward Propagation และ Backward Propagation "
        "เพื่อปรับค่าพารามิเตอร์ให้เหมาะสมกับข้อมูลที่ต้องการเรียนรู้ ซึ่งสามารถนำไปใช้ได้กับปัญหาทั้ง Classification และ Regression")
        st.image("img/NN.png",caption="Neural Network Layer")

    elif page == "Devlopment ML":
        st.subheader("ขั้นตอนการพัฒนา Model ML")
        st.write("1.นำเข้าไลบรารี่ที่ใช้สำหรับการเทรน")
        code = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("2.โหลดข้อมูลและจัดการ Missing Values")
        code = '''data = pd.read_csv('wine-clustering.csv')

# จัดการ Missing Values ด้วยค่าเฉลี่ยของแต่ละคอลัมน์ (เฉพาะตัวเลข)
data.fillna(data.mean(numeric_only=True), inplace=True)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("3. แยก Input (X) และ Output (y)")
        code = '''X = data.drop(columns=['Proline'])  # คุณสมบัติทั้งหมด ยกเว้น Proline
y = data['Proline']  # ตัวแปรเป้าหมาย
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("4. แบ่งข้อมูล Train/Test")
        code = '''np.random.seed(42)  # ทำให้ผลลัพธ์ซ้ำกันได้
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("5.ปรับสเกลข้อมูลด้วย StandardScaler")
        code = '''scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ปรับ y ด้วย StandardScaler เช่นกัน
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("6. สร้างและปรับแต่ง SVR ด้วย GridSearchCV")
        code = '''svr = SVR(kernel='rbf')

param_grid_svr = {
    'C': [50, 100, 200],
    'epsilon': [0.01, 0.1],
    'gamma': ['scale', 0.01, 0.1]
}

grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr.fit(X_train_scaled, y_train_scaled)

best_svr = grid_search_svr.best_estimator_
y_pred_svr_scaled = best_svr.predict(X_test_scaled)
y_pred_svr = y_scaler.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()

svr_mse = mean_squared_error(y_test, y_pred_svr)
svr_r2 = r2_score(y_test, y_pred_svr)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("7. สร้างและปรับแต่ง KNN ด้วย GridSearchCV")
        code = '''knn = KNeighborsRegressor()

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_knn.fit(X_train_scaled, y_train)

best_knn = grid_search_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)

knn_mse = mean_squared_error(y_test, y_pred_knn)
knn_r2 = r2_score(y_test, y_pred_knn)

'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("8. ใช้ KMeans จัดกลุ่มผลลัพธ์")
        code = '''y_combined = np.column_stack((y_test, y_pred_svr, y_pred_knn))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(y_combined)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("8. ใช้ KMeans จัดกลุ่มผลลัพธ์")
        code = '''y_combined = np.column_stack((y_test, y_pred_svr, y_pred_knn))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(y_combined)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("9. แสดงผลลัพธ์เป็นกราฟ")
        code = '''plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(y_test, y_pred_svr, c=clusters, cmap='viridis', edgecolor='k', s=100, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Proline')
plt.ylabel('Predicted Proline (SVR)')
plt.title('SVR: True vs Predicted Proline')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 2, 2)
scatter = plt.scatter(y_test, y_pred_knn, c=clusters, cmap='viridis', edgecolor='k', s=100, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Proline')
plt.ylabel('Predicted Proline (KNN)')
plt.title('KNN: True vs Predicted Proline')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.show()
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("9. แสดงผลลัพธ์เป็นกราฟ")
        code = '''with open('svr_model.pkl', 'wb') as f:
    pickle.dump(best_svr, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('y_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
        st.image("img/download.png",caption="Result training")

    elif page == "Devlopment NN":
        st.subheader("ขั้นตอนการพัฒนา Model NN")

        st.write("1.Import Libraries")
        code = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import joblib
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("2: อ่านข้อมูลและจัดการ Missing Values")
        code = '''# อ่านข้อมูล
data = pd.read_csv('/content/accident.csv')  # แทนที่ด้วยชื่อไฟล์จริงถ้ามี

# จัดการ missing data
data['Speed_of_Impact'] = data['Speed_of_Impact'].fillna(data['Speed_of_Impact'].mean())  # เติมค่าเฉลี่ย
data['Age'] = data['Age'].fillna(data['Age'].mean())  # เติมค่าเฉลี่ย
data['Gender'] = data['Gender'].fillna('Unknown')  # เติมค่าเป็น 'Unknown'
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("3: แปลงข้อมูล Categorical เป็น Numerical")
        code = '''# แปลง categorical variables เป็น numerical
le = LabelEncoder()
data['Seatbelt_Used'] = le.fit_transform(data['Seatbelt_Used'])
data['Helmet_Used'] = le.fit_transform(data['Helmet_Used'])
data['Gender'] = le.fit_transform(data['Gender'])  # แปลง Gender (Male/Female/Unknown) เป็นตัวเลข

# ตรวจสอบข้อมูลหลังจัดการ missing values
print("Missing values after imputation:")
print(data.isnull().sum())
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("4: เตรียม Features และ Target")
        code = '''# เตรียม features และ target
X = data[['Seatbelt_Used', 'Speed_of_Impact', 'Helmet_Used', 'Age', 'Gender']].values
y = data['Survived'].values

# แปลง output: 1 (รอด) -> 0 และ 0 (ไม่รอด) -> 1
y = 1 - y
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("5: แบ่งข้อมูล Train/Test")
        code = '''# แบ่ง train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("6: สร้าง Neural Network Mode")
        code = '''# สร้างโมเดล (รับ 5 inputs)
model = keras.Sequential([
    layers.Input(shape=(5,)),  # ปรับให้รับ 5 inputs
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("7: Compile และ Train โมเดล")
        code = '''# Compile โมเดล
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train โมเดล
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=1,
    validation_split=0.2,
    verbose=1
)
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("8: Evaluate โมเดล")
        code = '''# Evaluate โมเดล
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("9: เซฟโมเดล (2 วิธี)")
        code = '''# วิธีที่ 1: เซฟเป็น .pkl ด้วย joblib
joblib.dump(model, 'neural_network_model.pkl')
print("Model saved as neural_network_model.pkl using joblib")

# วิธีที่ 2: เซฟแบบ Keras native (แนะนำ)
model.save('neural_network_model.keras')
print("Model saved as neural_network_model.keras using Keras native format")
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)

        st.write("10: โหลดโมเดลและทดสอบ Predict")
        code = '''# โหลดโมเดลที่เซฟไว้
loaded_model_keras = keras.models.load_model('neural_network_model.keras')

# ทดสอบ predict
sample = np.array([[1, 50, 1, 30, 1]])  # Seatbelt_Yes, Speed_50, Helmet_Yes, Age_30, Gender_Male(1)
prediction = loaded_model_keras.predict(sample)
print(f"Prediction (0 = survived, 1 = not survived): {prediction[0][0]:.4f}")
'''
        st.code(code, language="python", line_numbers=False, wrap_lines=False, height=False)
        
if __name__ == "__main__":
    main()