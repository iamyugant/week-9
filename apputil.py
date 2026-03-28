import pandas as pd
import numpy as np

class GroupEstimate:
    
    def __init__(self, estimate='mean'):
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.columns = None
        self.default_category = None
        self.default_estimates = None
    
    def fit(self, X, y, default_category=None):
        df = X.copy()
        df['_target'] = y
        
        self.columns = list(X.columns)
        
        if self.estimate == 'mean':
            self.group_estimates = df.groupby(self.columns)['_target'].mean()
        else:
            self.group_estimates = df.groupby(self.columns)['_target'].median()
        
        if default_category is not None:
            self.default_category = default_category
            if self.estimate == 'mean':
                self.default_estimates = df.groupby(default_category)['_target'].mean()
            else:
                self.default_estimates = df.groupby(default_category)['_target'].median()
    
    def predict(self, X_):
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.columns)
        
        results = []
        missing_count = 0
        
        for idx, row in X_.iterrows():
            if len(self.columns) == 1:
                key = row[self.columns[0]]
            else:
                key = tuple(row[self.columns])
            
            try:
                value = self.group_estimates.loc[key]
                results.append(value)
            except KeyError:
                if self.default_category is not None and self.default_estimates is not None:
                    try:
                        default_key = row[self.default_category]
                        value = self.default_estimates.loc[default_key]
                        results.append(value)
                    except KeyError:
                        results.append(np.nan)
                        missing_count += 1
                else:
                    results.append(np.nan)
                    missing_count += 1
        
        if missing_count > 0:
            print(f"{missing_count} observation(s) had missing groups")
        
        return np.array(results)