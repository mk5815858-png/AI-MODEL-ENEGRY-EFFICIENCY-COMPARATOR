import streamlit as st
import time
import psutil
import pandas as pd
import plotly.express as px

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Green AI Comparator",
    page_icon="🌱",
    layout="wide"
)

# ---------------- Custom Dark Styling ----------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Model Selection")

selected_models = st.sidebar.multiselect(
    "Choose Models",
    ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"],
    default=["Linear Regression", "Random Forest", "XGBoost", "Neural Network"]
)

st.sidebar.markdown("---")
st.sidebar.info("Green AI Benchmarking Tool")

# ---------------- Title ----------------
st.title("🌱 AI Model Energy Efficiency Comparator")
st.markdown("### Sustainable & Carbon-Aware AI Benchmarking Dashboard")

# ---------------- Dataset ----------------
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Model Dictionary ----------------
model_dict = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "Neural Network": MLPRegressor(max_iter=500)
}

# ---------------- Evaluation Function ----------------
def evaluate_model(model, name):
    process = psutil.Process()
    
    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()
    start_memory = process.memory_info().rss

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    end_time = time.time()
    end_memory = process.memory_info().rss
    end_cpu = psutil.cpu_percent(interval=None)

    training_time = end_time - start_time
    memory_used = (end_memory - start_memory) / (1024 * 1024)
    cpu_usage = (start_cpu + end_cpu) / 2

    r2 = r2_score(y_test, predictions)

    # Carbon emission estimation (simplified)
    carbon_emission = training_time * cpu_usage * 0.0001

    efficiency_score = r2 / (training_time * (memory_used + 1))

    return {
        "Model": name,
        "R2 Score": round(r2, 4),
        "Training Time (s)": round(training_time, 4),
        "Memory (MB)": round(memory_used, 2),
        "CPU Usage (%)": round(cpu_usage, 2),
        "CO₂ Emission (kg)": round(carbon_emission, 6),
        "Efficiency Score": round(efficiency_score, 6)
    }

# ---------------- Run Button ----------------
if st.button("🚀 Run Green AI Comparison"):

    progress = st.progress(0)
    results = []

    for i, model_name in enumerate(selected_models):
        result = evaluate_model(model_dict[model_name], model_name)
        results.append(result)
        progress.progress((i + 1) / len(selected_models))

    results_df = pd.DataFrame(results)

    st.markdown("## 📊 Results Table")
    st.dataframe(results_df, use_container_width=True)

    # ---------------- Best Model ----------------
    best = results_df.sort_values(by="Efficiency Score", ascending=False).iloc[0]

    # Eco Badge System
    if best["Efficiency Score"] > 0.01:
        badge = "🥇 GOLD Eco Model"
    elif best["Efficiency Score"] > 0.005:
        badge = "🥈 SILVER Eco Model"
    else:
        badge = "🥉 BRONZE Eco Model"

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Most Efficient", best["Model"])
    col2.metric("🌍 Lowest CO₂ Model",
                results_df.sort_values(by="CO₂ Emission (kg)").iloc[0]["Model"])
    col3.metric("⭐ Eco Badge", badge)

    # ---------------- Charts ----------------
    st.markdown("## 📈 Performance Visualization")

    fig1 = px.bar(results_df, x="Model", y="R2 Score", title="Model Accuracy")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(results_df, x="Model", y="CO₂ Emission (kg)", title="Carbon Emission Comparison")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(results_df, x="Model", y="Efficiency Score", title="Energy Efficiency Score")
    st.plotly_chart(fig3, use_container_width=True)

    st.success("✅ Green AI Benchmarking Completed!")