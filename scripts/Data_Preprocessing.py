import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter

def load_data(path):
    return pd.read_csv(path)

def scale_data(data, scaling_method='minmax'):
    scaler_dict = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
    }

    scaler = scaler_dict[scaling_method]
    return scaler.fit_transform(data)

def handle_imbalanced(data, under_sample=0.5, over_sample=0.5, random_state=42):
    X = data.drop(columns=['Class'])
    y = data['Class']

    over = SMOTE(random_state=random_state, sampling_strategy=over_sample)
    under = RandomUnderSampler(random_state=random_state, sampling_strategy=under_sample)

    pipeline = Pipeline(steps=[('over', over), ('under', under)])
    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    print(f"Resampling: {Counter(y_resampled)}")
    return X_resampled, y_resampled
