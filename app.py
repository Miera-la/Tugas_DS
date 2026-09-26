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
st.title("Prediksi Segmen Pelanggan")
st.write("Aplikasi interaktif untuk memprediksi kelompok pelanggan berdasarkan model K-Means Clustering.")

# Keterangan format mata uang (Gunakan \\$ agar tidak dianggap formula matematika LaTeX/KaTeX oleh Streamlit)
st.info(
    "💡 **Informasi Format Mata Uang:**\n\n"
    "Data pada model ini menggunakan standar mata uang **Dolar AS (USD)** sesuai dataset asli (*Marketing Campaign*). "
)

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

# Input widgets dengan format integer murni (tanpa ,00) dan live preview format USD
col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Income (Pendapatan Tahunan dalam USD)",
        min_value=0,
        max_value=200000,
        value=50000,
        step=1000,
        format="%d"
    )
    st.caption(f"💵 Nominal: **\\${income:,} USD**")

    mnt_wines = st.number_input(
        "MntWines (Pengeluaran Wine dalam USD)",
        min_value=0,
        max_value=5000,
        value=300,
        step=10,
        format="%d"
    )
    st.caption(f"🍷 Nominal: **\\${mnt_wines:,} USD**")

with col2:
    num_deals = st.number_input(
        "NumDealsPurchases (Jumlah Pembelian via Diskon)",
        min_value=0,
        max_value=50,
        value=2,
        step=1,
        format="%d"
    )
    st.caption(f"🏷️ Frekuensi: **{num_deals} kali transaksi**")

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
    st.info(f"**Deskripsi Karakteristik:** {cluster_info['desc']}")
    st.write(
        f"📊 **Ringkasan Input:** "
        f"Pendapatan: **\\${income:,} USD** | "
        f"Belanja Wine: **\\${mnt_wines:,} USD** | "
        f"Pembelian Diskon: **{num_deals} kali**"
    )
    st.metric(label="Status Prediksi", value="Berhasil", delta=f"Cluster {cluster_result}")
