import sys

def p(msg):
    print(msg, flush=True)

p(f"Python: {sys.version}")

try:
    import numpy as np
    p(f"NumPy OK: {np.__version__}")
except Exception as e:
    p(f"NumPy Error: {e}")

try:
    import pandas as pd
    p(f"Pandas OK: {pd.__version__}")
except Exception as e:
    p(f"Pandas Error: {e}")

try:
    import xgboost as xgb
    p(f"XGBoost OK: {xgb.__version__}")
except Exception as e:
    p(f"XGBoost Error: {e}")

try:
    import lightgbm as lgb
    p(f"LightGBM OK: {lgb.__version__}")
except Exception as e:
    p(f"LightGBM Error: {e}")

try:
    import flask
    p(f"Flask OK: {flask.__version__}")
except Exception as e:
    p(f"Flask Error: {e}")

try:
    import sklearn
    p(f"Sklearn OK: {sklearn.__version__}")
except Exception as e:
    p(f"Sklearn Error: {e}")
