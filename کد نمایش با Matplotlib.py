
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. بارگذاری داده‌ها با استفاده از csv
def load_csv_data(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        data = [row for row in csv_reader]
    return headers, data

# مسیر فایل CSV شما
file_path = 'G:\kaleg\Book2.csv'
headers, raw_data = load_csv_data(file_path)

# 2. تبدیل به دیتافریم pandas
df = pd.DataFrame(raw_data, columns=headers)

# 3. پیش‌پردازش داده‌ها
# تبدیل مقادیر غیرعددی به عددی
for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.factorize(df[col])[0]

# تبدیل به آرایه numpy و نرمال‌سازی
data_array = df.values.astype(float)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_array)

# 4. خوشه‌بندی با K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

# 5. کاهش ابعاد برای نمایش (با استفاده از PCA از sklearn)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

# 6. نمایش نتایج با matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title('نتایج خوشه‌بندی K-Means')
plt.xlabel('مولفه اصلی ۱')
plt.ylabel('مولفه اصلی ۲')
plt.colorbar(label='خوشه')
plt.grid(True)
plt.show()

# نمایش اطلاعات خوشه‌ها
print("توزیع داده‌ها در خوشه‌ها:")
print(df['Cluster'].value_counts())