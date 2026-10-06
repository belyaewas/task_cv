import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
data1 = pd.read_csv('Y_Test.csv')
data2 = pd.read_csv('otvety.csv')
from sklearn.metrics import r2_score
r2 = r2_score(data1['Y1'], data2['Y1'])
print(r2)