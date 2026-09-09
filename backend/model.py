import joblib
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ml"
    / "models"
    / "predictive_maintenance_model.pkl"
)


model = joblib.load(MODEL_PATH)
