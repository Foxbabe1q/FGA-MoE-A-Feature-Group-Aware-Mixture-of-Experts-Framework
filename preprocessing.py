"""
Data preprocessing pipeline for ICU mortality prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """Preprocess features grouped by medical domain"""
    
    def __init__(self, feature_groups, categorical_features, 
                 binary_features, log_transform_features):
        self.feature_groups = feature_groups
        self.categorical_features = categorical_features
        self.binary_features = binary_features
        self.log_transform_features = log_transform_features
        
        self.scalers = {}
        self.label_encoders = {}
        self.fitted = False
    
    def _clean_age(self, df):
        """Clean age column (convert '>89' to numeric)"""
        df = df.copy()
        if 'age' in df.columns:
            df['age'] = df['age'].replace('>89', '90')
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
        return df
    
    def _encode_categorical(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def _log_transform(self, df):
        """Apply log transformation to specified features"""
        df = df.copy()
        
        for col in self.log_transform_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        
        return df
    
    def _impute_missing(self, df, fit=True):
        """Impute missing values with median"""
        df = df.copy()
        
        for group_name, features in self.feature_groups.items():
            for col in features:
                if col not in df.columns:
                    continue
                
                # Skip binary features (keep NaN as 0)
                if col in self.binary_features:
                    df[col] = df[col].fillna(0)
                    continue
                
                # Numerical features: use median
                if fit:
                    median_val = df[col].median()
                    self.scalers[f'{col}_median'] = median_val
                else:
                    median_val = self.scalers.get(f'{col}_median', df[col].median())
                
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def _normalize(self, df, fit=True):
        """Normalize features by group"""
        df = df.copy()
        
        for group_name, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            
            if not group_features:
                continue
            
            if fit:
                scaler = StandardScaler()
                df[group_features] = scaler.fit_transform(df[group_features])
                self.scalers[group_name] = scaler
            else:
                scaler = self.scalers[group_name]
                df[group_features] = scaler.transform(df[group_features])
        
        return df
    
    def _group_features(self, df):
        """Group features by medical domain"""
        grouped = {}
        
        for group_name, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            if group_features:
                grouped[group_name] = df[group_features].values.astype(np.float32)
        
        return grouped
    
    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        df = self._clean_age(df)
        df = self._encode_categorical(df, fit=True)
        df = self._log_transform(df)
        df = self._impute_missing(df, fit=True)
        df = self._normalize(df, fit=True)
        
        self.fitted = True
        return self._group_features(df)
    
    def transform(self, df):
        """Transform data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df = self._clean_age(df)
        df = self._encode_categorical(df, fit=False)
        df = self._log_transform(df)
        df = self._impute_missing(df, fit=False)
        df = self._normalize(df, fit=False)
        
        return self._group_features(df)
