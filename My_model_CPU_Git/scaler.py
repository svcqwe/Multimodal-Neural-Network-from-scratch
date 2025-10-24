import numpy as np
import json 

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = self.feature_range[0]
        self.max_ = self.feature_range[1]
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        
    def fit(self, X):
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_        # Вычисляем размах данных
        
        self.scale_ = np.ones_like(data_range)      # Создаем массив scale_, изначально заполненный единицами
        
        non_constant_mask = data_range != 0         # Маска для непостоянных признаков (где размах НЕ равен 0)
        
        self.scale_[non_constant_mask] = (self.max_ - self.min_) / data_range[non_constant_mask]    # Вычисляем scale_ только для непостоянных признаков
        
        return self
    
    def transform(self, X):
        if self.scale_ is None:
            raise ValueError("Сначала вызовите fit() или fit_transform()")
        
        X = np.array(X)
        X_std = (X - self.data_min_) * self.scale_ + self.min_
        return X_std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        if self.scale_ is None:
            raise ValueError("Сначала вызовите fit() или fit_transform()")
            
        X = np.array(X)
        X_original = (X - self.min_) / self.scale_ + self.data_min_
        return X_original
    
    def save(self, path:str = "scaler_params.json"):
        params = {
            "feature_range": self.feature_range,
            "data_min": self.data_min_.tolist(),
            "data_max": self.data_max_.tolist(),
            "scale": self.scale_.tolist()
        }
        with open(path, "a") as f:
            f.write(json.dumps(params, indent=4))
        return
    
    def load_scaler(self, path:str = "scaler_params.json"):
        with open(path, "r") as file:
            data = json.load(file)
        
        self.feature_range = data["feature_range"]
        self.data_min_ = np.array(data["data_min"])
        self.data_max_ = np.array(data["data_max"])
        self.scale_ = np.array(data["scale"])