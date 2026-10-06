import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

df = pd.read_csv('X_Train.csv')
df_Y = pd.read_csv('Y_Train.csv')

otv=[]

mean=df_Y['mean_Ke1'].values.mean()
std=df_Y['mean_Ke1'].values.std()
otv.append([round(mean, 4), round(std, 4)])
#otv.append([mean,std])
otv1 = pd.DataFrame(otv, columns=['median', 'std'])
otv1.to_csv('otvety.csv', index=False)


#Загрузка данных

#values = df_new['sum_cur'].dropna().values
values = df['sum_cur'].dropna().values
# 1. Оценка плотности распределения для поиска локальных максимумов
kde = gaussian_kde(values)
x_grid = np.linspace(values.min(), values.max(), 25)
density = kde(x_grid)

# 2. Поиск 4 наиболее высоких локальных пиков
peaks, properties = find_peaks(density, height=0)
# Сортируем пики по высоте (убывание) и берем 4 самых высоких
peak_heights = properties['peak_heights']
sorted_indices = np.argsort(peak_heights)[::-1]
top4_indices = sorted_indices[:4]
peak_x = x_grid[peaks[top4_indices]]
peak_x_sorted = np.sort(peak_x)  # сортируем для порядка (необязательно)

print("Найденные локальные максимумы (4 самых высоких):")
for i, p in enumerate(peak_x_sorted):
    print(f"  Пик {i+1}: {p:.4f}")

# 3. K-means с инициализацией центрами = найденные максимумы
#    Приводим центры к нужной форме (n_clusters, n_features)
initial_centers = peak_x_sorted.reshape(-1, 1)
kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=1, random_state=42)
labels = kmeans.fit_predict(values.reshape(-1, 1))
values=df_Y['mean_Ke1'].values
df2 = pd.DataFrame({'value': values, 'cluster': labels})

# 4. Построение гистограммы по кластерам
plt.figure(figsize=(12, 6))
otv=[]
colors = ['red', 'blue', 'green', 'orange']
for clust in sorted(df2['cluster'].unique()):
    cluster_data = df2[df2['cluster'] == clust]['value']
    plt.hist(cluster_data, bins=30, alpha=0.6, color=colors[clust],
             label=f'Кластер {clust} (n={len(cluster_data)})')
    median_val = cluster_data.median()
    std_val = cluster_data.std()
    otv.append([round(median_val, 4), round(std_val, 4)])
    plt.axvline(median_val, color=colors[clust], linestyle='dashed', linewidth=2,
                label=f'Медиана кл.{clust} = {median_val:.2f}')

plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Гистограмма с кластеризацией (центры = локальные максимумы)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Дополнительно: ящик с усами по кластерам
plt.figure(figsize=(8, 5))
data_by_cluster = [df2[df2['cluster'] == clust]['value'].values for clust in sorted(df2['cluster'].unique())]
plt.boxplot(data_by_cluster, labels=[f'Кластер {c}' for c in sorted(df2['cluster'].unique())])
plt.title('Разброс значений по кластерам')
plt.ylabel('Значения')
plt.grid(alpha=0.3)
plt.show()

# Вывод статистики по кластерам
print("\nСтатистика по кластерам:")
for clust in sorted(df2['cluster'].unique()):
    cluster_data = df2[df2['cluster'] == clust]['value']
    print(f"Кластер {clust}: размер={len(cluster_data)}, медиана={cluster_data.median():.4f}, "
          f"ср.кв.откл.={cluster_data.std():.4f}, мин={cluster_data.min():.4f}, макс={cluster_data.max():.4f}")

otv2 = pd.DataFrame(otv, columns=['median', 'std'])
otv2.to_csv('otvety.csv', index=False)
import numpy as np
import pandas as pd

otv = pd.read_csv('otvety.csv')
ref_data = {
    'median': [0.000, 0.004, -0.004, 0.000],
    'std':    [0.6206, 0.5823, 0.5252, 0.4948]
}

ref_df = pd.DataFrame(ref_data)

n = min(len(otv), len(ref_df))
print(n)
if n == 0:
    print("Нет данных для сравнения (df пуст).")
else:

    df_n = otv.iloc[:n].reset_index(drop=True)
    ref_n = ref_df.iloc[:n].reset_index(drop=True)


    def metric(actual, ref):
        if ref == 0.0:
            return 1.0 if actual == 0 else 0.0
        rel_diff = abs(actual - ref) / abs(ref)
        return max(0.0, 1.0 - rel_diff)

    metrics_median = []
    metrics_std = []

    for i in range(n):
        m_med = metric(df_n.loc[i, 'median'], ref_n.loc[i, 'median'])
        m_std = metric(df_n.loc[i, 'std'], ref_n.loc[i, 'std'])
        metrics_median.append(m_med)
        metrics_std.append(m_std)

    metrics_row_avg = [(m_med + m_std) / 2 for m_med, m_std in zip(metrics_median, metrics_std)]
    overall_metric = sum(metrics_row_avg) / 4

print(overall_metric)
