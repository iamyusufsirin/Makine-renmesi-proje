# Heart Disease Prediction Project

UCI Cleveland Heart Disease veri seti üzerinde ikili sınıflandırma. Görev tanımı:
ön işleme → özellik seçimi → modelleme → değerlendirme & SHAP → Streamlit deploy.

## Klasör yapısı
```
.
├── data/
│   ├── raw/             # ham veri (UCI)
│   └── processed/       # train/test (encoded, imputed, outlier-cleaned)
├── models/              # joblib ile kaydedilmiş model & scaler
├── outputs/
│   ├── figures/         # tüm grafikler (.png)
│   └── metrics/         # metrik tabloları (.csv ve .md)
├── src/
│   ├── config.py        # yollar ve sabitler
│   └── utils.py         # yardımcı fonksiyonlar
├── 01_eda_preprocessing.ipynb
├── 02_feature_selection.ipynb
├── 03_modeling.ipynb
├── requirements.txt
└── README.md
```

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Veriyi indirme

UCI'de `.csv` dosyası yoktur; veri `processed.cleveland.data` adlı başlıksız
bir dosyadadır. `01` notebook'unu çalıştırınca `load_raw` veriyi otomatik bulur:
`ucimlrepo` paketi → legacy URL → manuel dosya. `ucimlrepo` `requirements.txt`'te
yüklü olduğu için ilk çalıştırmada genelde otomatik iner.

**Otomatik inmezse manuel yöntem:**
1. https://archive.ics.uci.edu/dataset/45/heart+disease adresine git
2. "Download" butonu → `heart+disease.zip` iner
3. Zip'ten `processed.cleveland.data` dosyasını çıkar
4. `data/raw/processed.cleveland.data` yoluna koy
5. `01` notebook'undaki yükleme hücresini tekrar çalıştır

## Çalıştırma sırası
1. `01_eda_preprocessing.ipynb` — EDA, train/test split, IQR outlier removal,
   one-hot encoding, eksik değer doldurma. `data/processed/` altına yazar.
2. `02_feature_selection.ipynb` — StandardScaler + 3 özellik seçim yöntemi
   karşılaştırması. `models/scaler.joblib` ve seçili özellik listesi.
3. `03_modeling.ipynb` — LogReg, RandomForest, XGBoost (GridSearchCV + 5-fold CV).
   Confusion matrix ve ilk metrik tabloları.
4. `04_evaluation_shap.ipynb` — ROC (CV-mean ve test), recall-threshold, SHAP.
5. `app/streamlit_app.py` — canlı tahmin + canlı SHAP açıklaması.

## Notlar
- `RANDOM_STATE=42` her yerde sabit.
- IQR ve scaler yalnızca train üzerinde fit edilir, test'e transform uygulanır.
- Tıbbi vakada **Recall** öncelikli metrik (yanlış-negatif maliyeti yüksek).
