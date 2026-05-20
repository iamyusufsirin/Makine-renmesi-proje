"""
Kalp Hastalığı Risk Tahmin Sistemi — Streamlit deployment (Görev 5)
====================================================================

Form input → canlı risk tahmini → tahminin altında o hastaya ait canlı SHAP
force plot ile "model neden bu kararı verdi" açıklaması.

ÇALIŞTIRMA (proje kökünden):
    streamlit run app/streamlit_app.py

ÖN KOŞUL:
    01–04 notebook'ları çalıştırılmış olmalı. Uygulama şu dosyaları kullanır:
    models/deploy_info.joblib, models/scaler.joblib, models/<en_iyi_model>.joblib,
    models/shap_background.csv  (LinearExplainer kullanılırsa).

STREAMLIT CLOUD'A DEPLOY:
    1. Tüm proje klasörünü (models/ dahil) bir GitHub reposuna push et.
    2. share.streamlit.io → New app → repo seç → ana dosya: app/streamlit_app.py
    3. requirements.txt otomatik kullanılır.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # streamlit icin headless backend
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# ----------------------------------------------------------------------
# Yollar ve sabitler
# ----------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Kategorik kod -> Türkçe etiket (gerçek UCI Cleveland kodlaması)
LABELS = {
    "sex":     {1: "Erkek", 0: "Kadın"},
    "cp":      {1: "Tipik anjina", 2: "Atipik anjina",
                3: "Anjinal olmayan ağrı", 4: "Asemptomatik"},
    "fbs":     {1: "Evet (>120 mg/dl)", 0: "Hayır"},
    "restecg": {0: "Normal", 1: "ST-T dalga anormalliği",
                2: "Sol ventrikül hipertrofisi"},
    "exang":   {1: "Evet", 0: "Hayır"},
    "slope":   {1: "Yukarı eğimli", 2: "Düz", 3: "Aşağı eğimli"},
    "ca":      {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal":    {3: "Normal (3)", 6: "Sabit defekt (6)",
                7: "Geri dönüşümlü defekt (7)"},
}

HELP = {
    "age": "Hastanın yaşı (yıl)",
    "trestbps": "Dinlenme halinde sistolik kan basıncı (mm Hg)",
    "chol": "Serum kolesterol seviyesi (mg/dl)",
    "thalach": "Egzersiz testinde ulaşılan maksimum kalp atış hızı",
    "oldpeak": "Egzersizin dinlenmeye göre yarattığı ST depresyonu",
    "cp": "Göğüs ağrısının klinik tipi",
    "restecg": "Dinlenme elektrokardiyografi sonucu",
    "slope": "Pik egzersiz ST segmentinin eğimi",
    "ca": "Floroskopiyle renklenen ana damar sayısı (0–3)",
    "thal": "Talasemi test sonucu",
}


# ----------------------------------------------------------------------
# Artefakt kontrolü ve yükleme
# ----------------------------------------------------------------------
def missing_artifacts():
    needed = ["deploy_info.joblib", "scaler.joblib"]
    return [f for f in needed if not (MODELS_DIR / f).exists()]


@st.cache_resource
def load_artifacts():
    """Model, scaler ve SHAP explainer'ı yükler (oturum boyunca cache'lenir)."""
    info = joblib.load(MODELS_DIR / "deploy_info.joblib")
    model = joblib.load(MODELS_DIR / info["best_model_file"])
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    selected = info["selected_features"]
    threshold = float(info["decision_threshold"])
    best_name = info["best_model_name"]

    # SHAP explainer — model tipine göre
    if best_name in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    else:  # LogisticRegression -> LinearExplainer (arka plan verisi gerekir)
        background = pd.read_csv(MODELS_DIR / "shap_background.csv")[selected]
        explainer = shap.LinearExplainer(model, background)

    return {
        "model": model, "scaler": scaler, "explainer": explainer,
        "selected": selected, "threshold": threshold, "best_name": best_name,
    }


@st.cache_data
def load_best_model_metrics(best_name):
    """outputs/metrics/evaluation_metrics.csv'den en iyi modelin satırını okur."""
    path = METRICS_DIR / "evaluation_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["model"] == best_name]
    return row.iloc[0].to_dict() if len(row) else None


# ----------------------------------------------------------------------
# Çekirdek mantık (Streamlit'ten bağımsız — test edilebilir)
# ----------------------------------------------------------------------
def build_feature_row(input_dict, scaler, selected, scale=True):
    """Ham hasta girdisini modelin beklediği özellik satırına dönüştürür.

    Eğitimdeki pipeline ile birebir aynı: sürekli değişkenler StandardScaler ile
    ölçeklenir, kategorikler one-hot kodlamasına çevrilir. One-hot dummy kolon
    adları '{orijinal}_{deger}' biçiminde olduğu için tek satır girdiden doğrudan
    yeniden inşa edilir (get_dummies tek satırda bozulduğu için bu yol tercih edilir).

    scale=True  -> model girişi (ölçeklenmiş sürekli değişkenler)
    scale=False -> SHAP grafiğinde gösterim (ham, okunabilir değerler)
    """
    raw = pd.DataFrame([input_dict])
    if scale:
        raw[CONTINUOUS] = scaler.transform(raw[CONTINUOUS])

    row = {}
    for feat in selected:
        if feat in raw.columns:                 # sürekli değişken
            row[feat] = float(raw[feat].iloc[0])
        elif "_" in feat:                       # one-hot dummy
            original, value_str = feat.rsplit("_", 1)
            row[feat] = 1 if float(input_dict[original]) == float(value_str) else 0
        else:
            row[feat] = 0
    return pd.DataFrame([row])[selected]


def predict_risk(model, X_input, threshold):
    proba = float(model.predict_proba(X_input)[:, 1][0])
    label = int(proba >= threshold)
    return proba, label


def compute_shap(explainer, X_input, X_display):
    """Tek hasta için SHAP Explanation üretir; gösterim verisini ham değerlerle değiştirir."""
    shap_exp = explainer(X_input)
    if shap_exp.values.ndim == 3:               # ikili sınıflandırma -> pozitif sınıf
        shap_exp = shap_exp[:, :, 1]
    shap_exp.data = X_display.values            # grafiklerde gerçek değerleri göster
    return shap_exp


def force_figure(shap_exp, X_display):
    """SHAP force plot matplotlib figürü."""
    plt.close("all")
    shap.force_plot(
        base_value=shap_exp[0].base_values,
        shap_values=shap_exp[0].values,
        features=X_display.iloc[0],
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(11, 3)
    return fig


def waterfall_figure(shap_exp):
    """SHAP waterfall plot matplotlib figürü (force plot'a göre daha okunaklı)."""
    plt.close("all")
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_exp[0], show=False)
    return plt.gcf()


# ----------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------
def render_sidebar(art):
    with st.sidebar:
        st.header("Model Bilgisi")
        st.write(f"**En iyi model:** {art['best_name']}")
        st.write(f"**Karar eşiği:** {art['threshold']:.3f}")
        st.write(f"**Kullanılan özellik:** {len(art['selected'])}")

        m = load_best_model_metrics(art["best_name"])
        if m:
            st.divider()
            st.caption("Test seti performansı (Görev 4)")
            st.metric("Recall (duyarlılık)", f"{m['test_recall']*100:.1f}%")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{m['test_accuracy']*100:.1f}%")
            c2.metric("Precision", f"{m['test_precision']*100:.1f}%")
            st.caption("Tıbbi vaka olduğu için recall öncelikli metriktir.")

        st.divider()
        st.caption("UCI Cleveland Heart Disease veri seti üzerinde eğitilmiştir. "
                   "Karar destek aracıdır, tıbbi teşhis yerine geçmez.")


def patient_form():
    """Hasta bilgilerini toplayan form; gönderildiğinde (submitted, input_dict) döner."""
    with st.form("patient_form"):
        st.subheader("Hasta Bilgileri")
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Yaş", 29, 77, 54, help=HELP["age"])
            trestbps = st.slider("Dinlenme kan basıncı (mm Hg)", 94, 200, 130,
                                 help=HELP["trestbps"])
            chol = st.slider("Serum kolesterol (mg/dl)", 126, 564, 240,
                             help=HELP["chol"])
            thalach = st.slider("Maksimum kalp atış hızı", 71, 202, 150,
                                help=HELP["thalach"])
            oldpeak = st.slider("ST depresyonu (oldpeak)", 0.0, 6.2, 1.0, 0.1,
                                help=HELP["oldpeak"])

        with c2:
            sex = st.selectbox("Cinsiyet", [1, 0],
                               format_func=lambda v: LABELS["sex"][v])
            cp = st.selectbox("Göğüs ağrısı tipi", [1, 2, 3, 4],
                              format_func=lambda v: LABELS["cp"][v], help=HELP["cp"])
            fbs = st.selectbox("Açlık kan şekeri >120 mg/dl", [0, 1],
                               format_func=lambda v: LABELS["fbs"][v])
            restecg = st.selectbox("Dinlenme EKG sonucu", [0, 1, 2],
                                   format_func=lambda v: LABELS["restecg"][v],
                                   help=HELP["restecg"])

        with c3:
            exang = st.selectbox("Egzersize bağlı anjina", [0, 1],
                                 format_func=lambda v: LABELS["exang"][v])
            slope = st.selectbox("ST segment eğimi", [1, 2, 3],
                                 format_func=lambda v: LABELS["slope"][v],
                                 help=HELP["slope"])
            ca = st.selectbox("Renklenen ana damar sayısı (ca)", [0, 1, 2, 3],
                              format_func=lambda v: LABELS["ca"][v], help=HELP["ca"])
            thal = st.selectbox("Talasemi (thal)", [3, 6, 7],
                                format_func=lambda v: LABELS["thal"][v],
                                help=HELP["thal"])

        submitted = st.form_submit_button("🔍 Risk Hesapla", use_container_width=True)

    input_dict = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                      fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                      oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
    return submitted, input_dict


def render_result(proba, threshold):
    st.subheader("Tahmin Sonucu")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Kalp Hastalığı Risk Skoru", f"{proba * 100:.1f}%")
    with c2:
        if proba >= threshold:
            st.error(f"⚠️ **Yüksek risk** — model bu hastayı 'hasta' sınıfına "
                     f"yerleştiriyor (karar eşiği: %{threshold * 100:.1f}).")
        else:
            st.success(f"✅ **Düşük risk** — model bu hastayı 'sağlıklı' sınıfına "
                       f"yerleştiriyor (karar eşiği: %{threshold * 100:.1f}).")
        st.progress(min(proba, 1.0))
    st.caption("Not: Bu sonuç bir karar destek çıktısıdır, kesin tıbbi teşhis değildir.")


def main():
    st.set_page_config(page_title="Kalp Hastalığı Risk Tahmini",
                       page_icon="🫀", layout="wide")

    missing = missing_artifacts()
    if missing:
        st.error(
            "Gerekli model dosyaları bulunamadı: "
            + ", ".join(missing)
            + "\n\nÖnce `01`–`04` notebook'larını sırayla çalıştırın."
        )
        st.stop()

    art = load_artifacts()
    render_sidebar(art)

    st.title("🫀 Kalp Hastalığı Risk Tahmin Sistemi")
    st.markdown(
        "Hastanın klinik bilgilerini girin ve **Risk Hesapla**'ya basın. "
        "Sonuç ile birlikte modelin kararını açıklayan SHAP grafiği gösterilir."
    )

    submitted, input_dict = patient_form()

    if submitted:
        X_input = build_feature_row(input_dict, art["scaler"], art["selected"],
                                    scale=True)
        X_display = build_feature_row(input_dict, art["scaler"], art["selected"],
                                      scale=False)
        proba, _ = predict_risk(art["model"], X_input, art["threshold"])

        st.divider()
        render_result(proba, art["threshold"])

        st.divider()
        st.subheader("Model Neden Bu Sonucu Verdi? (SHAP Açıklaması)")
        st.markdown(
            "Aşağıdaki grafik bu hastaya özel bir açıklamadır. "
            "**Kırmızı** oklar riski **artıran**, **mavi** oklar **azaltan** "
            "özelliklerdir; ok uzunluğu katkının büyüklüğünü gösterir."
        )

        shap_exp = compute_shap(art["explainer"], X_input, X_display)

        fig_force = force_figure(shap_exp, X_display)
        st.pyplot(fig_force)
        plt.close(fig_force)

        with st.expander("Detaylı katkı grafiği (waterfall)"):
            st.markdown(
                "Her özelliğin tahmini ortalama değerden (E[f(x)]) bu hastanın "
                "skoruna (f(x)) nasıl taşıdığını adım adım gösterir."
            )
            fig_wf = waterfall_figure(shap_exp)
            st.pyplot(fig_wf)
            plt.close(fig_wf)


if __name__ == "__main__":
    main()
