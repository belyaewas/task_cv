import numpy as np
import pandas as pd

otv = pd.read_csv('otvety.csv')
ref_data = {
    'median': [1.7099, 3.0353, 5.328, 5.9776],
    'std':    [0.2132, 0.3539 , 0.4751, 0.613]
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