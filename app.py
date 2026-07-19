import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG (must be the first Streamlit command)
# ==========================================================
st.set_page_config(
    page_title="Zelbytes Agritech | Yield Forecast",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CACHED MODEL + SCALER LOADING
# ==========================================================
@st.cache_resource
def load_model():
    model = joblib.load("models/champion.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler


model, scaler = load_model()


# ==========================================================
# PREDICTION FUNCTION — the ONLY definition of predict_yield
# in this project. Nothing else should redefine this name.
# ==========================================================
def predict_yield(temperature, humidity, co2, light):
    temp_humidity = temperature * humidity

    data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "CO2": [co2],
        "Light": [light],
        "Temp_Humidity": [temp_humidity]
    })

    scaled = scaler.transform(data)
    scaled_df = pd.DataFrame(
        scaled,
        columns=["Temperature", "Humidity", "CO2", "Light", "Temp_Humidity"]
    )

    prediction = model.predict(scaled_df)
    return prediction[0]


# ==========================================================
# DESIGN SYSTEM — modern light theme, agritech green/earth palette
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* ---- Solid black base, red kept only as a faint accent glow ---- */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(139,0,0,0.14) 0%, rgba(0,0,0,0) 40%),
        radial-gradient(circle at 85% 85%, rgba(180,20,20,0.10) 0%, rgba(0,0,0,0) 45%),
        #000000;
    background-attachment: fixed;
}

/* ---- Sidebar (kept green, untouched) ---- */
section[data-testid="stSidebar"] {
    background: #1B3A2B;
}
section[data-testid="stSidebar"] * {
    color: #EAF3EC !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div div div {
    background: #5FBF7A !important;
}

/* ---- Headings ---- */
h1 {
    color: #F5F5F5;
    font-weight: 800;
    letter-spacing: -0.5px;
}
h2, h3, h4 {
    color: #F0E0E0;
    font-weight: 700;
}
p, li, span, label {
    color: #E8E8E8;
}

/* ---- Hero banner (glass) ---- */
.hero {
    background: rgba(20, 20, 20, 0.85);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(220,60,60,0.4);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(178,34,34,0.15), 0 8px 32px rgba(0,0,0,0.6);
}
.hero h1 {
    color: #FFFFFF !important;
    margin-bottom: 4px;
}
.hero p {
    color: #F0F0F0 !important;
    font-size: 16px;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(178,34,34,0.85);
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ---- Metric cards (glass) ---- */
[data-testid="stMetric"] {
    background: rgba(22,22,22,0.85);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(220,60,60,0.3);
    padding: 18px 20px;
    border-radius: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.5);
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #E89999 !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800;
}

/* ---- Card containers (glass) ---- */
.panel {
    background: rgba(24, 24, 24, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(220,60,60,0.35);
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 4px 28px rgba(178,34,34,0.12), 0 4px 24px rgba(0,0,0,0.6);
    margin-bottom: 20px;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #B22222 0%, #7A1515 100%);
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    height: 50px;
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #D02E2E 0%, #8C1A1A 100%);
    color: #FFFFFF;
    box-shadow: 0 0 16px rgba(210,40,40,0.5);
}

/* ---- Download button ---- */
.stDownloadButton > button {
    background: rgba(20,8,8,0.5);
    color: #F0E0E0;
    border: 1.5px solid rgba(220,60,60,0.4);
    border-radius: 10px;
    font-weight: 600;
}
.stDownloadButton > button:hover {
    border-color: #D02E2E;
    color: #FFFFFF;
}

/* ---- Expanders (glass) ---- */
.streamlit-expanderHeader {
    background: rgba(22,22,22,0.85);
    backdrop-filter: blur(12px);
    border-radius: 10px;
    font-weight: 600;
    color: #F5F5F5;
    border: 1px solid rgba(220,60,60,0.3);
}
.streamlit-expanderContent {
    background: rgba(18,18,18,0.8);
    backdrop-filter: blur(12px);
    border-radius: 0 0 10px 10px;
    border: 1px solid rgba(220,60,60,0.15);
    border-top: none;
}
.streamlit-expanderContent p,
.streamlit-expanderContent li,
.streamlit-expanderContent td,
.streamlit-expanderContent th {
    color: #E8E8E8 !important;
}

/* ---- Alerts (warning/success/info/error) glassy tint ---- */
[data-testid="stAlert"] {
    backdrop-filter: blur(10px);
    border-radius: 12px;
}

/* ---- Table / dataframe ---- */
[data-testid="stDataFrame"] {
    background: rgba(20,8,8,0.4);
    border-radius: 10px;
}

hr {
    border-color: rgba(220,60,60,0.2);
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HERO HEADER
# ==========================================================
st.markdown("""
<div class="hero">
    <span class="badge">🍄 ZELBYTES AGRITECH</span>
    <h1>Mushroom Yield Forecast</h1>
    <p>AI-powered daily yield prediction from live polyhouse sensor readings.
    Adjust values in the sidebar and generate an instant forecast.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR — SENSOR INPUTS
# ==========================================================
with st.sidebar:
    st.markdown("### 🌱 Sensor Controls")
    st.caption("Set current polyhouse readings")

    temperature = st.slider("Temperature (°C)", 10.0, 35.0, 22.0, 0.1)
    humidity = st.slider("Humidity (%)", 50.0, 100.0, 88.0, 0.5)
    co2 = st.slider("CO₂ (ppm)", 400, 2000, 900, 10)
    light = st.slider("Light Intensity (lux)", 100, 1000, 500, 10)

    st.markdown("---")
    st.markdown("""
    **Recommended Range**
    - 🌡 Temperature: 18–26 °C
    - 💧 Humidity: 80–95 %
    - 🌬 CO₂: 700–1200 ppm
    - 💡 Light: 300–700 lux
    """)

# ==========================================================
# INPUT VALIDATION
# ==========================================================
warnings = []
if temperature < 15 or temperature > 30:
    warnings.append("Temperature is outside the model's recommended range.")
if humidity < 70 or humidity > 98:
    warnings.append("Humidity is outside the model's recommended range.")
if co2 < 500 or co2 > 1500:
    warnings.append("CO₂ is outside the model's recommended range.")
if light < 200 or light > 800:
    warnings.append("Light intensity is outside the model's recommended range.")

if warnings:
    st.warning("⚠️ " + "  •  ".join(warnings))

# ==========================================================
# LIVE SENSOR READOUT
# ==========================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("🌡 Temperature", f"{temperature:.1f} °C")
kpi2.metric("💧 Humidity", f"{humidity:.1f} %")
kpi3.metric("🌬 CO₂", f"{co2} ppm")
kpi4.metric("💡 Light", f"{light} lux")

st.write("")

# ==========================================================
# MAIN DASHBOARD
# ==========================================================
left_col, right_col = st.columns([1, 1.1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🍄 Yield Prediction")

    predict_clicked = st.button("Predict Yield", use_container_width=True)

    if predict_clicked:
        prediction = predict_yield(temperature, humidity, co2, light)

        st.metric(label="Estimated Daily Yield", value=f"{prediction:.2f} kg")

        confidence = max(70, min(98, 100 - abs(humidity - 88) / 2))
        st.progress(confidence / 100)
        st.caption(f"Estimated prediction confidence: {confidence:.0f}%")

        if prediction >= 8:
            st.success("✅ Excellent growing conditions.")
        elif prediction >= 6:
            st.info("ℹ️ Good growing conditions.")
        elif prediction >= 4:
            st.warning("⚠️ Average production expected.")
        else:
            st.error("❌ Low expected yield — adjust environmental conditions.")

        result = pd.DataFrame({
            "Temperature": [temperature],
            "Humidity": [humidity],
            "CO2": [co2],
            "Light": [light],
            "Predicted_Yield_kg": [prediction]
        })
        st.download_button(
            "⬇ Download Prediction (CSV)",
            result.to_csv(index=False),
            file_name="prediction.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Set sensor values in the sidebar, then click **Predict Yield**.")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 📈 What-if Analysis — Humidity Sweep")
    st.caption("Predicted yield as humidity varies, other sensors held fixed.")

    humidity_values = np.linspace(70, 98, 29)
    predictions = [predict_yield(temperature, h, co2, light) for h in humidity_values]

    chart_df = pd.DataFrame({
        "Humidity (%)": humidity_values,
        "Predicted Yield (kg)": predictions
    }).set_index("Humidity (%)")

    st.line_chart(chart_df, use_container_width=True, color="#D02E2E")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# INFO SECTION
# ==========================================================
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    with st.expander("📊 Model Information"):
        st.markdown("""
        | Property | Value |
        |---|---|
        | Champion Model | **Linear Regression** |
        | Models Compared | Linear Regression, Random Forest, Tuned Random Forest (GridSearchCV) |
        | Features | Temperature, Humidity, CO₂, Light, Temp × Humidity |
        | Target | Mushroom Yield (kg) |
        | Split | 80% train / 20% test (chronological) |
        | Preprocessing | MinMaxScaler, fit on training data only |
        """)

with info_col2:
    with st.expander("📈 Model Performance"):
        st.markdown("""
        Test-set results for all candidates evaluated:

        | Model | Test MAE | RMSE | R² |
        |---|---|---|---|
        | **Linear Regression** ⭐ | **0.188** | **0.207** | **0.330** |
        | Random Forest (Default) | 0.188 | 0.226 | 0.201 |
        | Random Forest (Tuned) | 0.189 | 0.228 | 0.189 |

        Linear Regression was selected: lowest RMSE, highest R²
        (explains the most variance in yield), tied for lowest
        Test MAE, trains instantly, and is fully interpretable.
        """)

with info_col3:
    with st.expander("📖 Methodology"):
        st.markdown("""
        1. Collect & clean sensor readings
        2. Engineer `Temp × Humidity` feature
        3. Scale features (train-only fit)
        4. Train Linear Regression, Random Forest, and Tuned Random Forest
        5. Compare on Test MAE, RMSE, R²
        6. Select champion by best overall metrics
        7. Deploy via this dashboard
        """)

st.markdown("---")
st.caption("Zelbytes Agritech · Built with Streamlit, Scikit-learn, Pandas & NumPy")