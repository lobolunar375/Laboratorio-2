"""
preprocessing.py
================
Modulo de carga y preprocesamiento de datos para deteccion de DDoS.
Dataset: UNSW-NB15 (archivos reales de training y testing)
Paper: "DDoS Detection using Machine Learning Approach"
       Janivasya & Rachmawati, 2024, Procedia Computer Science

Columnas reales del dataset (45 total):
  id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate,
  sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit,
  swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean,
  trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm,
  ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login,
  ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports,
  attack_cat, label

Categorias de ataque en el dataset:
  Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance,
  Analysis, Backdoor, Shellcode, Worms

Estrategia de etiquetado:
  - DoS   -> label = 1 (ataque)
  - Normal-> label = 0 (normal)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DEL DATASET REAL
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FILENAME = "UNSW_NB15_training-set.csv"
TEST_FILENAME  = "UNSW_NB15_testing-set.csv"
SUBFOLDER      = "Training and Testing Sets"   # subcarpeta donde estan los CSV

# Columnas categoricas a codificar
CATEGORICAL_COLS = ["proto", "service", "state"]

# Columna objetivo y categoria de ataque
TARGET_COL     = "label"
ATTACK_CAT_COL = "attack_cat"

# Columnas a descartar (id no aporta informacion)
DROP_COLS = ["id"]


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE DATOS SINTETICOS (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_synthetic_data(n_samples=50_000, random_state=42):
    """
    Genera un dataset sintetico con la estructura real del UNSW-NB15.
    Se usa solo si no se encuentran los archivos CSV.
    """
    print("[INFO] Generando datos sinteticos con estructura UNSW-NB15...")
    rng = np.random.default_rng(random_state)

    n_normal = int(n_samples * 0.60)
    n_dos    = n_samples - n_normal

    def make_block(n, is_attack):
        scale = 2.5 if is_attack else 1.0
        return {
            "id":                rng.integers(1, 999999, n),
            "dur":               np.abs(rng.normal(0.5 * scale, 0.5, n)),
            "proto":             rng.choice(["tcp", "udp", "icmp", "arp"], n, p=[0.5,0.3,0.15,0.05]),
            "service":           rng.choice(["-", "http", "ftp", "smtp", "ssh", "dns"], n, p=[0.4,0.3,0.1,0.05,0.05,0.1]),
            "state":             rng.choice(["FIN", "CON", "REQ", "INT", "RST"], n),
            "spkts":             rng.integers(1, 500 * int(scale), n),
            "dpkts":             rng.integers(0, 400 * int(scale), n),
            "sbytes":            rng.integers(100, 10_000 * int(scale + 1), n),
            "dbytes":            rng.integers(0, 8_000 * int(scale + 1), n),
            "rate":              np.abs(rng.normal(100 * scale, 50, n)),
            "sttl":              rng.integers(32, 128, n),
            "dttl":              rng.integers(32, 128, n),
            "sload":             np.abs(rng.normal(50_000 * scale, 20_000, n)),
            "dload":             np.abs(rng.normal(30_000, 15_000, n)),
            "sloss":             rng.integers(0, 20 * int(scale), n),
            "dloss":             rng.integers(0, 20 * int(scale), n),
            "sinpkt":            np.abs(rng.normal(0.1 * scale, 0.05, n)),
            "dinpkt":            np.abs(rng.normal(0.2, 0.1, n)),
            "sjit":              np.abs(rng.normal(5 * scale, 5, n)),
            "djit":              np.abs(rng.normal(3, 4, n)),
            "swin":              rng.integers(0, 65535, n),
            "stcpb":             rng.integers(0, 2**31, n),
            "dtcpb":             rng.integers(0, 2**31, n),
            "dwin":              rng.integers(0, 65535, n),
            "tcprtt":            np.abs(rng.normal(0.05, 0.02, n)),
            "synack":            np.abs(rng.normal(0.02, 0.01, n)),
            "ackdat":            np.abs(rng.normal(0.01, 0.005, n)),
            "smean":             rng.integers(28, 1500, n),
            "dmean":             rng.integers(0, 1500, n),
            "trans_depth":       rng.integers(0, 5, n),
            "response_body_len": rng.integers(0, 5000, n),
            "ct_srv_src":        rng.integers(1, 100, n),
            "ct_state_ttl":      rng.integers(0, 6, n),
            "ct_dst_ltm":        rng.integers(1, 100, n),
            "ct_src_dport_ltm":  rng.integers(1, 100, n),
            "ct_dst_sport_ltm":  rng.integers(1, 100, n),
            "ct_dst_src_ltm":    rng.integers(1, 100, n),
            "is_ftp_login":      rng.integers(0, 2, n),
            "ct_ftp_cmd":        rng.integers(0, 5, n),
            "ct_flw_http_mthd":  rng.integers(0, 5, n),
            "ct_src_ltm":        rng.integers(1, 100, n),
            "ct_srv_dst":        rng.integers(1, 100, n),
            "is_sm_ips_ports":   rng.integers(0, 2, n),
            "attack_cat":        ["DoS"] * n if is_attack else ["Normal"] * n,
            "label":             [1] * n if is_attack else [0] * n,
        }

    df = pd.concat([
        pd.DataFrame(make_block(n_normal, False)),
        pd.DataFrame(make_block(n_dos,    True))
    ], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[INFO] Dataset sintetico generado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir="data"):
    """
    Carga los archivos CSV del UNSW-NB15.
    Busca en:
      1. data_dir/Training and Testing Sets/
      2. data_dir/  (directamente)
    Si no los encuentra, genera datos sinteticos.
    """
    # Posibles rutas donde pueden estar los archivos
    search_paths = [
        os.path.join(data_dir, SUBFOLDER),   # subcarpeta oficial
        data_dir,                             # raiz de data/
    ]

    train_path, test_path = None, None
    for folder in search_paths:
        tp = os.path.join(folder, TRAIN_FILENAME)
        sp = os.path.join(folder, TEST_FILENAME)
        if os.path.exists(tp) and os.path.exists(sp):
            train_path, test_path = tp, sp
            break

    if train_path is None:
        print(f"[WARN] Archivos UNSW-NB15 no encontrados en '{data_dir}'.")
        print("[WARN] Usando datos sinteticos. Para usar datos reales coloca:")
        print(f"[WARN]   {os.path.join(data_dir, SUBFOLDER, TRAIN_FILENAME)}")
        print(f"[WARN]   {os.path.join(data_dir, SUBFOLDER, TEST_FILENAME)}")
        return _generate_synthetic_data()

    print(f"[INFO] Cargando dataset real...")
    print(f"  Training : {train_path}")
    print(f"  Testing  : {test_path}")

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test  = pd.read_csv(test_path,  low_memory=False)

    print(f"  Training shape : {df_train.shape[0]:,} x {df_train.shape[1]}")
    print(f"  Testing  shape : {df_test.shape[0]:,} x {df_test.shape[1]}")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  Total combinado: {df.shape[0]:,} x {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXPLORACION INICIAL
# ─────────────────────────────────────────────────────────────────────────────
def explore_data(df):
    """Imprime estadisticas generales del dataset."""
    print("\n" + "=" * 60)
    print("  EXPLORACION DEL DATASET")
    print("=" * 60)
    print(f"  Shape            : {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"  Valores nulos    : {df.isnull().sum().sum():,}")
    print(f"  Duplicados       : {df.duplicated().sum():,}")

    if TARGET_COL in df.columns:
        counts = df[TARGET_COL].value_counts()
        print(f"\n  Balance de clases ('{TARGET_COL}'):")
        for cls, cnt in counts.items():
            print(f"    Clase {cls}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")

    if ATTACK_CAT_COL in df.columns:
        print(f"\n  Distribucion de '{ATTACK_CAT_COL}':")
        for cat, cnt in df[ATTACK_CAT_COL].value_counts().items():
            print(f"    {str(cat):<22}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df):
    """
    Limpia el dataset:
    - Elimina duplicados
    - Rellena nulos (mediana para numericos, moda para categoricos)
    - Reemplaza infinitos con NaN y los elimina
    - Elimina columna 'id' (no informativa)
    """
    print("\n[INFO] Iniciando limpieza de datos...")
    original_rows = len(df)

    df = df.drop_duplicates()
    print(f"  Duplicados eliminados: {original_rows - len(df):,}")

    # Infinitos -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Rellenar nulos
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Eliminar columnas no informativas
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Columnas eliminadas: {cols_to_drop}")

    print(f"  Shape tras limpieza: {df.shape[0]:,} x {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FILTRADO DoS / Normal
# ─────────────────────────────────────────────────────────────────────────────
def filter_ddos_normal(df):
    """
    Filtra unicamente registros Normal y DoS.
    Crea etiqueta binaria: 0 = Normal, 1 = DoS.

    En el UNSW-NB15 real las categorias son:
      Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance,
      Analysis, Backdoor, Shellcode, Worms
    """
    if ATTACK_CAT_COL not in df.columns:
        print(f"[WARN] Columna '{ATTACK_CAT_COL}' no encontrada. Usando '{TARGET_COL}' existente.")
        return df

    # Normalizar texto
    df[ATTACK_CAT_COL] = df[ATTACK_CAT_COL].astype(str).str.strip().str.lower()

    dos_mask    = df[ATTACK_CAT_COL].str.contains(r"\bdos\b|\bddos\b", case=False, na=False)
    normal_mask = df[ATTACK_CAT_COL].str.contains(r"\bnormal\b",       case=False, na=False)

    df_filtered = df[dos_mask | normal_mask].copy()
    df_filtered[TARGET_COL] = dos_mask[dos_mask | normal_mask].astype(int).values

    print(f"\n[INFO] Filtrado DoS/Normal:")
    print(f"  Normal : {(df_filtered[TARGET_COL] == 0).sum():,}")
    print(f"  DoS    : {(df_filtered[TARGET_COL] == 1).sum():,}")
    print(f"  Total  : {len(df_filtered):,}")

    return df_filtered


# ─────────────────────────────────────────────────────────────────────────────
# CODIFICACION CATEGORICA
# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df):
    """
    Label Encoding para proto, service, state.
    Elimina attack_cat (ya se uso para crear label).
    """
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    if ATTACK_CAT_COL in df.columns:
        df = df.drop(columns=[ATTACK_CAT_COL])

    print(f"[INFO] Codificacion categorica: {list(encoders.keys())}")
    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# ESCALADO
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    """
    StandardScaler ajustado SOLO en train para evitar data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("[INFO] Escalado con StandardScaler completado.")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_pipeline(data_dir="data"):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    Retorna (X, y, feature_names).
    """
    df = load_data(data_dir)
    explore_data(df)
    df = clean_data(df)
    df = filter_ddos_normal(df)
    df, _ = encode_categoricals(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[TARGET_COL].values

    print(f"\n[INFO] Preprocesamiento finalizado:")
    print(f"  Features : {X.shape[1]} columnas")
    print(f"  Muestras : {X.shape[0]:,}")
    print(f"  Columnas : {list(X.columns)}")

    return X, y, list(X.columns)


if __name__ == "__main__":
    X, y, feat_names = preprocess_pipeline()
    print(f"\nListo: X={X.shape}, y={y.shape}")
