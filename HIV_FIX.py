import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Membuat variabel untuk menyimpan nilai berupa file csv
df = pd.read_csv('HIV1.csv')
HIV = pd.read_csv('HIV.csv', delimiter=';')

st.title("Aplikasi Data Mining HIV K-Means Clustering\n")
st.header("Isi Dataset")
st.write(HIV)

#--------------------------------------------------------------------------------------------------------------------------------#
# K-Means Clustering
# 1. Elbow Method -> untuk menentukan jumlah cluster optimal.

# Data numerik untuk clustering
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
X = df[numerical_cols]

# Tentukan jumlah cluster yang diuji (misalnya, 2-10)
inertia = []
k_range = range(2, 8)

# Iterasi melalui jumlah cluster
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Membuat grafik menggunakan Matplotlib
fig, ax = plt.subplots()
ax.plot(k_range, inertia, marker='o', linestyle='-', color='orange')
ax.set_title('Metode Elbow')
ax.set_xlabel('Number of clusters k')
ax.set_ylabel('Total Within Sum of Square')
ax.set_xticks(k_range)
ax.grid(True)

# Menandai "elbow" point (secara visual, mungkin tidak otomatis)
# Berdasarkan grafik contoh, elbow terlihat di k=4
elbow_point = 4
plt.axvline(x=elbow_point, color='r', linestyle='--', label=f'Optimal k = {elbow_point}')
plt.legend()

# Menampilkan Grafik di Streamlit
st.pyplot(fig)
st.write(f'Berdasarkan grafik dari metode Elbow, perkiraan jumlah cluster optimal adalah **{elbow_point}**.')

#--------------------------------------------------------------------------------------------------------------#
# 2. Silhouette Score -> sebagai ukuran kualitas cluster.

# Preprocessing (Scaling)
X, y = make_blobs(n_samples=40, cluster_std=1.00, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduksi dimensi dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering dan evaluasi silhouette score untuk berbagai jumlah cluster
silhouette_scores_rounded = []
k_values = range(2, 8) # Dimulai dari 2 karena 1 cluster tidak memiliki makna
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores_rounded.append(round(score, 2))  # Bulatkan skor di sini

# Visualisasi Plot hasil
fig, ax = plt.subplots()
ax.plot(k_values, silhouette_scores_rounded, marker='o', color='orange')
ax.set_title('Metode Silhouette Coefficient')
ax.set_xlabel('Number of clusters k')
ax.set_ylabel('Average Silhouette Width Score')
ax.grid(True)

# Menandai titik optimal (berdasarkan nilai tertinggi)
optimal_k_index = silhouette_scores_rounded.index(max(silhouette_scores_rounded))
optimal_k = k_values[optimal_k_index]
ax.axvline(x=optimal_k, color='brown', linestyle='--', label=f'Optimal k = {optimal_k}')
ax.legend()
fig.tight_layout()

# Menampilkan Grafik di Streamlit
st.pyplot(fig)
st.write(f'Berdasarkan grafik dari metode Silhouette Coefficient, perkiraan ukuran kualitas jumlah cluster adalah **{optimal_k}**.')

#---------------------------------------------------------------------------------------------------#
# Menampilkan K-Means Clustering dalam bentuk Sliding
st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,8,3,1) #Memasukan Slider

#--------------------------------------------------------------------------------------------------------------#
# Proses Clustering 
def k_means(n_clust): # Memasukkan fungsi clusternya berdasarkan Slider
    kmeans = KMeans(n_clusters=n_clust).fit(df) # Memanggil fungsi K-Means ke dataset
    df['Labels']=kmeans.labels_
 
    # Buat objek Figure dan Axes
    fig, ax = plt.subplots(figsize=(10, 8)) # Masukkan Plotnya
    ax.set_title('Hasil K-Means Clustering')

    # Visualisasi scatter plot dengan Seaborn
    sns.scatterplot(x='Umur', y='Kelurahan', hue='Labels', data=df, markers=True, 
                   size='Labels', palette=sns.color_palette('hls', n_clust))
    
    # Tambahkan anotasi di tengah cluster
    for label in df['Labels'].unique():
        ax.annotate(str(label),
            (df[df['Labels']==label]['Umur'].mean(),
            df[df['Labels']==label]['Kelurahan'].mean()),
            horizontalalignment ='center',
            verticalalignment ='center',
            size = 20, weight='bold',
            color ='black')
    
    st.header('Cluster Plot') # Memanggil Label
    st.pyplot(fig)
    st.write(df)

k_means(clust) #dimana clust ini ditentukan oleh slider
