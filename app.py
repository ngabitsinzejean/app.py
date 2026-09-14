import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------
# 1. GENERATE SYNTHETIC DATASET (Mimicking NISR SAS/CPI)
# ----------------------------------------------------
@st.cache_data
def load_synthetic_nisr_data():
    np.random.seed(42)
    districts = ['Musanze', 'Nyagatare', 'Bugesera', 'Huye', 'Rubavu', 'Gatsibo']
    crops = ['Irish Potato', 'Maize', 'Beans', 'Rice', 'Sorghum']
    
    data = []
    for _ in range(1000):
        district = np.random.choice(districts)
        crop = np.random.choice(crops)
        
        # Add realistic rules based on Rwandan geography
        if district == 'Musanze' and crop == 'Irish Potato':
            rainfall = np.random.uniform(1200, 1600)
            yield_tonnes = np.random.uniform(18, 25)
            price_per_kg = np.random.uniform(300, 450)
        elif district == 'Nyagatare' and crop == 'Maize':
            rainfall = np.random.uniform(800, 1100)
            yield_tonnes = np.random.uniform(4, 7)
            price_per_kg = np.random.uniform(350, 500)
        else:
            rainfall = np.random.uniform(700, 1400)
            yield_tonnes = np.random.uniform(2, 12)
            price_per_kg = np.random.uniform(400, 800)
            
        data.append([district, crop, round(rainfall, 1), round(yield_tonnes, 2), int(price_per_kg)])
        
    return pd.DataFrame(data, columns=['District', 'Crop_Type', 'Avg_Rainfall_mm', 'Yield_Tonnes_Per_HA', 'Market_Price_FRW'])

df = load_synthetic_nisr_data()

# ----------------------------------------------------
# 2. TRAIN MACHINE LEARNING MODEL (AI Engine)
# ----------------------------------------------------
@st.cache_resource
def train_ai_model(data):
    le_dist = LabelEncoder()
    le_crop = LabelEncoder()
    
    data_encoded = data.copy()
    data_encoded['District'] = le_dist.fit_transform(data['District'])
    data_encoded['Crop_Type'] = le_crop.fit_transform(data['Crop_Type'])
    
    X = data_encoded[['District', 'Crop_Type', 'Avg_Rainfall_mm']]
    y_yield = data_encoded['Yield_Tonnes_Per_HA']
    y_price = data_encoded['Market_Price_FRW']
    
    model_yield = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y_yield)
    model_price = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y_price)
    
    return model_yield, model_price, le_dist, le_crop

model_yield, model_price, le_dist, le_crop = train_ai_model(df)

# ----------------------------------------------------
# 3. STREAMLIT USER INTERFACE (Dashboard Presentation)
# ----------------------------------------------------
st.set_page_config(page_title="Umuhinzi AI Dashboard", layout="wide")

st.title("🇷🇼 Umuhinzi AI: Smart Agriculture Matchmaking Platform")
st.markdown("### NISR 2026 Big Data Hackathon Project Proposal — *Aligned with NST2 Pillars*")
st.write("Iyi sisitemu ishingiye ku makuru ya **NISR (Seasonal Agricultural Survey & CPI)** ikokesha AI guhanura umusaruro w'ubuhinzi n'ibiciro ku masoko kugira ngo ifashe abahinzi n'abafata ibyemezo.")

st.divider() # Hahinduwe hano kuva kuri st.hr()

col1, col2 = st.columns(2)

with col1:
    st.header("⚙️ Injiza Imiterere y'Ubutaka")
    input_district = st.selectbox("Hitamo Akarere (District):", df['District'].unique())
    input_crop = st.selectbox("Hitamo Igihingwa (Crop Type):", df['Crop_Type'].unique())
    input_rainfall = st.slider("Ikigereranyo cy'Imvura (Expected Rainfall in mm):", 500, 1800, 1000)
    
    if st.button("Hana / Predict Insights", type="primary"):
        dist_enc = le_dist.transform([input_district])
        crop_enc = le_crop.transform([input_crop])
        
        pred_input = [[dist_enc[0], crop_enc[0], input_rainfall]]
        pred_yield = model_yield.predict(pred_input)[0]
        pred_price = model_price.predict(pred_input)[0]
        
        st.success("### Results Generated Successfully!")
        st.metric(label="Umusaruro uteganyijwe (Predicted Yield)", value=f"{pred_yield:.2f} Tons / Hectare")
        st.metric(label="Igiciro ku Isoko (Estimated Market Price)", value=f"{int(pred_price)} FRW / KG")

with col2:
    st.header("📊 Imiterere n'Isesengura ry'Amakuru (NISR Datasets Data Exploration)")
    
    tab1, tab2 = st.tabs(["Market Price Distribution", "Yield Insights per District"])
    
    with tab1:
        st.subheader(f"Ibiciro bya {input_crop} mu Turere Twose")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        crop_filtered = df[df['Crop_Type'] == input_crop]
        crop_filtered.groupby('District')['Market_Price_FRW'].mean().plot(kind='bar', color='#ff7f0e', ax=ax)
        ax.set_ylabel("Average Price (FRW / KG)")
        ax.set_xlabel("Districts")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Ugereranyo rw'Umusaruro Uteye Imbere bitewe n'Igihingwa")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        df.groupby('Crop_Type')['Yield_Tonnes_Per_HA'].mean().sort_values().plot(kind='barh', color='#2ca02c', ax=ax2)
        ax2.set_xlabel("Yield (Tonnes per HA)")
        ax2.set_ylabel("Crop Variant")
        st.pyplot(fig2)

st.divider() # Hahinduwe hano kuva kuri st.hr()
st.dataframe(df.head(10), use_container_width=True)
st.caption("Aya ni amakuru y'icyitegererezo ashobora guhindurwa na Data nyonyo mukura muri NISR.")
