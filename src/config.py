# src/config.py
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, 'src')
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
FEATURES_PATH = os.path.join(DATA_PATH, 'processed', 'selected_features.csv')

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
