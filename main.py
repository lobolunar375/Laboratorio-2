"""
main.py
=======
Pipeline principal de deteccion de DDoS usando Machine Learning.
Replica la metodologia de:
  "DDoS Detection using Machine Learning Approach"
   Janivasya & Rachmawati, 2024, Procedia Computer Science

Dataset: UNSW-NB15
  Busca automaticamente en:
    data/Training and Testing Sets/UNSW_NB15_training-set.csv
    data/Training and Testing Sets/UNSW_NB15_testing-set.csv
  Si no los encuentra, genera datos sinteticos.

Salidas regeneradas en cada ejecucion (carpeta outputs/):
  - feature_importance.png
  - confusion_matrices.png
  - metrics_comparison.png
  - results_summary.csv

Uso:
  python main.py
  python main.py --data-dir ruta/al/dataset
  python main.py --k-features 20
  python main.py --test-size 0.20
"""

import os
import sys
import time
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd

# Forzar salida UTF-8 en Windows para evitar UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split

# Modulos locales
from preprocessing import preprocess_pipeline, scale_features
from feature_selection import select_features
from models import train_all_models
from evaluate import evaluate_all

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20     # 80/20 split
K_FEATURES   = 20       # Features seleccionadas con chi-cuadrado
OUTPUT_DIR   = "outputs"
DATA_DIR     = "data"


# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    sep = "=" * 68
    print("\n" + sep)
    print("  DDoS Detection using Machine Learning Approach")
    print("  Janivasya & Rachmawati, 2024 - Procedia Computer Science")
    print("  Dataset: UNSW-NB15 (real)")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# LIMPIAR Y PREPARAR OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def prepare_output_dir(output_dir):
    """
    Elimina los archivos previos en outputs/ y recrea la carpeta limpia.
    Garantiza que todos los archivos se regeneren en cada ejecucion.
    """
    if os.path.exists(output_dir):
        # Eliminar solo archivos (no subcarpetas) del directorio
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        print(f"[INFO] Carpeta '{output_dir}/' limpiada (outputs previos eliminados).")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Carpeta '{output_dir}/' creada.")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DDoS Detection ML Pipeline -- UNSW-NB15"
    )
    parser.add_argument("--data-dir",    type=str,   default=DATA_DIR,
                        help=f"Directorio con los CSV del UNSW-NB15 (default: '{DATA_DIR}')")
    parser.add_argument("--k-features",  type=int,   default=K_FEATURES,
                        help=f"Numero de features chi2 (default: {K_FEATURES})")
    parser.add_argument("--test-size",   type=float, default=TEST_SIZE,
                        help=f"Proporcion del test set (default: {TEST_SIZE})")
    parser.add_argument("--output-dir",  type=str,   default=OUTPUT_DIR,
                        help=f"Directorio de salida (default: '{OUTPUT_DIR}')")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(data_dir=DATA_DIR, k_features=K_FEATURES,
                 test_size=TEST_SIZE, output_dir=OUTPUT_DIR):
    """
    Ejecuta el pipeline completo de deteccion de DDoS:
      1. Preprocesamiento (carga, limpieza, filtrado, encoding)
      2. Seleccion de caracteristicas (Chi-cuadrado, k=20)
      3. Division Train/Test (80/20 estratificado)
      4. Escalado (StandardScaler)
      5. Entrenamiento de 8 modelos
      6. Evaluacion + generacion de todos los archivos en outputs/

    Retorna DataFrame con metricas comparativas.
    """
    t_total = time.time()

    # Limpiar outputs previos y crear directorio fresco
    prepare_output_dir(output_dir)

    sep = "-" * 60

    # ── PASO 1: PREPROCESAMIENTO ─────────────────────────────────────────────
    print("\n" + sep)
    print("  PASO 1/5 - Preprocesamiento")
    print(sep)
    X, y, feature_names = preprocess_pipeline(data_dir)
    X_arr = X.values if hasattr(X, "values") else X

    # ── PASO 2: SELECCION DE CARACTERISTICAS ─────────────────────────────────
    print("\n" + sep)
    print(f"  PASO 2/5 - Seleccion de caracteristicas (Chi-cuadrado, k={k_features})")
    print(sep)
    X_selected, selected_names, selector = select_features(
        X_arr, y, feature_names, k=k_features, output_dir=output_dir
    )
    print(f"  [OK] feature_importance.png generado")

    # ── PASO 3: SPLIT TRAIN/TEST ──────────────────────────────────────────────
    train_pct = int((1 - test_size) * 100)
    test_pct  = int(test_size * 100)
    print("\n" + sep)
    print(f"  PASO 3/5 - Division Train/Test ({train_pct}/{test_pct})")
    print(sep)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train : {X_train.shape[0]:,} muestras  "
          f"(Normal: {(y_train==0).sum():,} | DoS: {(y_train==1).sum():,})")
    print(f"  Test  : {X_test.shape[0]:,} muestras  "
          f"(Normal: {(y_test==0).sum():,}  | DoS: {(y_test==1).sum():,})")

    # ── PASO 4: ESCALADO ──────────────────────────────────────────────────────
    print("\n" + sep)
    print("  PASO 4/5 - Escalado (StandardScaler)")
    print(sep)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # ── PASO 5a: ENTRENAMIENTO ────────────────────────────────────────────────
    print("\n" + sep)
    print("  PASO 5a/5 - Entrenamiento de 8 modelos")
    print(sep)
    trained_models = train_all_models(X_train_sc, y_train)

    # ── PASO 5b: EVALUACION ───────────────────────────────────────────────────
    print("\n" + sep)
    print("  PASO 5b/5 - Evaluacion y generacion de resultados")
    print(sep)
    results_df = evaluate_all(trained_models, X_test_sc, y_test, output_dir)

    # ── RESUMEN FINAL ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    sep2 = "=" * 60
    print("\n" + sep2)
    print(f"  PIPELINE COMPLETADO en {elapsed:.1f}s")
    print(f"\n  Archivos generados en '{output_dir}/':")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size  = os.path.getsize(fpath)
        ext   = os.path.splitext(fname)[1].upper().lstrip(".")
        print(f"    [{ext:<3}] {fname:<35} {size/1024:>7.1f} KB")
    print(sep2 + "\n")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()
    args = parse_args()

    results = run_pipeline(
        data_dir   = args.data_dir,
        k_features = args.k_features,
        test_size  = args.test_size,
        output_dir = args.output_dir
    )

    print("\nRESUMEN FINAL:")
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(results.to_string(index=False))
    print()
