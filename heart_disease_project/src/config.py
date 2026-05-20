"""Project paths, constants, and shared settings."""
from pathlib import Path

# --- Project paths --------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"

for _d in [RAW_DIR, PROC_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TOP_K_FEATURES = 8  # how many features each FS method picks

# --- Data source ----------------------------------------------------------
# Birincil yontem: ucimlrepo paketi (id=45). Yedek: asagidaki legacy URL.
UCI_DATASET_ID = 45
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Otomatik indirme calismazsa manuel indirilen .data dosyasi buraya konur.
MANUAL_DATA_FILE = RAW_DIR / "processed.cleveland.data"
# Yerel cache (basliklı CSV) yolu — load_raw bunu save_to olarak kullanir.
RAW_CSV = RAW_DIR / "heart_cleveland.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Continuous numeric features — IQR outlier removal & StandardScaler apply here
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical features — One-Hot Encoding applies here
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

TARGET = "target"  # binary target after num > 0 conversion
