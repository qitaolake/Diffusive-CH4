import sys
from sklearn import preprocessing
import sklearn.metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from tpot.builtins import ZeroCount
from xgboost import XGBRegressor

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
    
# Average CV score on the training set was: -0.39024249672712225
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    Normalizer(norm="l2"),
    ZeroCount(),
    XGBRegressor(learning_rate=0.1, max_depth=1, min_child_weight=11, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.6000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print("MSE：", round(sklearn.metrics.mean_squared_error(testing_target, results), 4))
print("RMSE：", round(sklearn.metrics.mean_squared_error(testing_target, results, squared=False), 4))
print("MAE：", round(sklearn.metrics.mean_absolute_error(testing_target, results), 4))
print("R2：", round(sklearn.metrics.r2_score(testing_target, results), 4))
print("The difference between the true value and the predicted value：")
print(testing_target.values - results)
