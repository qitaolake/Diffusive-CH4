import sys
from sklearn import preprocessing
import sklearn.metrics
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

csv_path = sys.argv[1]
data = pd.read_csv(csv_path, sep=',', encoding='ISO-8859-1')
features_saved = ['Chla_center', 'Tw_center', 'Kd_center', 'PAR_center', 'Chla*Tw', 'Log(Chla*Tw)', 'Chla*Kd', 'TW*Kd', 'log10(Tw*Kd)', 'Kd*Par', 'WS_m/s']
target = 'Log10Fm_¦Ìmol/m2/d'

pd_target = data[target]
pd_features = data.drop(columns=[target])
pd_features_cols = list(pd_features.columns.values)
for col in pd_features_cols:
    if col not in features_saved:
        pd_features = pd_features.drop(columns=[col])
        continue
    
training_features, testing_features, training_target, testing_target = train_test_split(pd_features, pd_target, random_state=42)
    
# Average CV score on the training set was: -0.3909964334173784
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    KernelRidge(alpha=0.9)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print("MSE：", round(sklearn.metrics.mean_squared_error(testing_target, results), 4))
print("RMSE：", round(sklearn.metrics.mean_squared_error(testing_target, results, squared=False), 4))
print("MAE：", round(sklearn.metrics.mean_absolute_error(testing_target, results), 4))
print("R2：", round(sklearn.metrics.r2_score(testing_target, results), 4))
print("The difference between the true value and the predicted value：")
print(testing_target.values - results)
