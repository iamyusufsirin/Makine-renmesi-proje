"""Reusable helpers for the heart-disease project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- I/O ------------------------------------------------------------------
def load_raw(
    columns: list[str],
    save_to: Path | None = None,
    url: str | None = None,
    manual_data_file: Path | None = None,
    dataset_id: int = 45,
) -> pd.DataFrame:
    """Cleveland Heart Disease verisini yukler.

    Sirayla denenir:
      1. Yerel cache CSV (save_to) -> varsa direkt okunur.
      2. Manuel indirilen baslıksiz .data dosyasi (manual_data_file).
      3. ucimlrepo paketi (resmi UCI yukleyici) -> EN GUVENILIR.
      4. Legacy ham URL (url).
    Hepsi basarisiz olursa manuel indirme talimatiyla hata firlatir.
    """
    save_to = Path(save_to) if save_to else None
    manual_data_file = Path(manual_data_file) if manual_data_file else None

    # 1) Yerel cache (basliklı CSV) ---------------------------------------
    if save_to and save_to.exists():
        return pd.read_csv(save_to)

    df = None

    # 2) Manuel indirilen .data dosyasi (basliksiz) -----------------------
    if manual_data_file and manual_data_file.exists():
        df = pd.read_csv(manual_data_file, header=None, names=columns, na_values="?")
        print(f"Manuel dosyadan okundu: {manual_data_file.name}")

    # 3) ucimlrepo paketi -------------------------------------------------
    if df is None:
        try:
            from ucimlrepo import fetch_ucirepo

            ds = fetch_ucirepo(id=dataset_id)
            df = pd.concat([ds.data.features, ds.data.targets], axis=1)
            df.columns = columns  # kolon adlarini sabit semaya hizala
            print("ucimlrepo ile indirildi.")
        except Exception as e:  # paket yok ya da ag erisimi yok
            print(f"ucimlrepo basarisiz ({type(e).__name__}); URL deneniyor...")

    # 4) Legacy URL -------------------------------------------------------
    if df is None and url:
        try:
            df = pd.read_csv(url, header=None, names=columns, na_values="?")
            print("Legacy URL ile indirildi.")
        except Exception as e:
            print(f"URL de basarisiz ({type(e).__name__}).")

    if df is None:
        raise RuntimeError(
            "\n--- Veri otomatik indirilemedi. MANUEL YONTEM ---\n"
            "  1. Tarayicida ac: https://archive.ics.uci.edu/dataset/45/heart+disease\n"
            "  2. Sag ustteki 'Download' butonuna bas -> heart+disease.zip iner\n"
            "  3. Zip'i ac, icinden SADECE 'processed.cleveland.data' dosyasini al\n"
            f"  4. O dosyayi su yola kopyala: {manual_data_file}\n"
            "  5. Bu hucreyi tekrar calistir (kod baslıksiz .data dosyasini otomatik okur).\n"
        )

    # Cache'e basliklı CSV olarak yaz ------------------------------------
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
        print(f"Cache'lendi: {save_to}")
    return df


def savefig(fig, path: Path, dpi: int = 120) -> None:
    """Save a matplotlib figure consistently and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, stem: Path) -> None:
    """Persist a DataFrame to both .csv and .md for easy report copy/paste."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(df.to_markdown(index=False))


# --- Outlier removal ------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def remove_outliers_iqr(
    df: pd.DataFrame, columns: Iterable[str], k: float = 1.5
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Drop rows where any of `columns` falls outside its IQR fences.

    Returns the filtered DataFrame and the bounds (for documentation / reuse).
    Bounds are computed from the input df only — call this on TRAIN ONLY to
    avoid data leakage; do NOT apply the same row-drop logic to the test set.
    """
    bounds = {c: iqr_bounds(df[c], k) for c in columns}
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in bounds.items():
        mask &= df[col].between(lo, hi)
    return df[mask].copy(), bounds


# --- Encoding alignment ---------------------------------------------------
def align_dummies(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure train and test have the same dummy columns after get_dummies."""
    all_cols = train.columns.union(test.columns)
    train = train.reindex(columns=all_cols, fill_value=0)
    test = test.reindex(columns=all_cols, fill_value=0)
    return train, test
