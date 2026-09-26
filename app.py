import streamlit as st
import pandas as pd
import joblib

# 1. PERSIAPAN LOAD ARTEFAK DENGAN CACHE
@st.cache_resource
def load_artefak():
    scaler = joblib.load('scaler_marketing.joblib')
    model_kmeans = joblib.load('kmeans_marketing.joblib')
    return scaler, model_kmeans

scaler, model_kmeans = load_artefak()

# 2. UI STREAMLIT
st.title("🎯 Prediksi Segmen Pelanggan")
st.write("Aplikasi interaktif untuk memprediksi kelompok pelanggan berdasarkan model K-Means Clustering.")

# Deskripsi cluster
cluster_descriptions = {
    0: {
        "label": "Cluster 0 - Low-Income Bargain Hunter",
        "desc": "Pendapatan rendah, pengeluaran wine rendah, cukup sering memanfaatkan diskon."
    },
    1: {
        "label": "Cluster 1 - High-Income Premium Buyer",
        "desc": "Pendapatan tinggi, pengeluaran wine tinggi, jarang mengandalkan diskon."
    },
    2: {
        "label": "Cluster 2 - Mid-Income Moderate Spender",
        "desc": "Pendapatan menengah, pengeluaran wine sedang, pembelian diskon moderat."
    }
}

# Input widgets dengan layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Income (Pendapatan Tahunan)", value=50000.0, step=1000.0)
    mnt_wines = st.number_input("MntWines (Pengeluaran Wine)", value=300.0, step=10.0)

with col2:
    num_deals = st.number_input("NumDealsPurchases (Pembelian via Diskon)", value=2.0, step=1.0)


# 3. LOGIKA PREDIKSI
if st.button("Prediksi Cluster"):
    input_data = pd.DataFrame(
        [[income, mnt_wines, num_deals]],
        columns=['Income', 'MntWines', 'NumDealsPurchases']
    )
    
    scaled_input = scaler.transform(input_data)
    cluster_result = int(model_kmeans.predict(scaled_input)[0])
    
    cluster_info = cluster_descriptions.get(cluster_result, {
        "label": f"Cluster {cluster_result}",
        "desc": "Deskripsi tidak tersedia."
    })
    
    st.success(f"Pelanggan ini masuk ke dalam: **{cluster_info['label']}**")
    st.info(f"**Deskripsi:** {cluster_info['desc']}")
    st.metric(label="Status Prediksi", value="Berhasil", delta=f"Cluster {cluster_result}")
