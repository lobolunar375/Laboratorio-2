"""
evaluate.py
===========
Evaluación de modelos para detección de DDoS.
Paper: "DDoS Detection using Machine Learning Approach"
       Janivasya & Rachmawati, 2024, Procedia Computer Science

Genera:
  - Métricas por modelo: Accuracy, Precision, Recall, F1-score
  - Matrices de confusión (grid 2×4)
  - Gráfico comparativo de métricas (bar chart agrupado)
  - Tabla CSV de resultados (Tabla 1 del paper)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG      = "#0D1117"
CARD_BG      = "#161B22"
BORDER_CLR   = "#30363D"
TEXT_CLR     = "#C9D1D9"
ACCENT_CLR   = "#58A6FF"
SUCCESS_CLR  = "#3FB950"
WARN_CLR     = "#D29922"
DANGER_CLR   = "#F85149"

# Colores para las métricas en el gráfico comparativo
METRIC_COLORS = {
    "Accuracy" : "#58A6FF",
    "Precision": "#3FB950",
    "Recall"   : "#D29922",
    "F1-Score" : "#F85149",
}

# Resultados de referencia del paper (Tabla 1)
PAPER_RESULTS = {
    "Logistic Regression": {"Accuracy": 92.5, "Precision": 91.8, "Recall": 93.2, "F1-Score": 92.5},
    "KNN"                : {"Accuracy": 95.1, "Precision": 94.7, "Recall": 95.6, "F1-Score": 95.1},
    "SVM"                : {"Accuracy": 93.2, "Precision": 92.6, "Recall": 93.8, "F1-Score": 93.2},
    "Random Forest"      : {"Accuracy": 97.7, "Precision": 97.5, "Recall": 97.9, "F1-Score": 97.7},
    "Decision Tree"      : {"Accuracy": 96.3, "Precision": 96.1, "Recall": 96.5, "F1-Score": 96.3},
    "MLP"                : {"Accuracy": 96.1, "Precision": 95.8, "Recall": 96.4, "F1-Score": 96.1},
    "LSTM"               : {"Accuracy": 96.5, "Precision": 96.2, "Recall": 96.8, "F1-Score": 96.5},
    "GRU"                : {"Accuracy": 96.4, "Precision": 96.1, "Recall": 96.7, "F1-Score": 96.4},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CALCULAR MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    model_dict: dict,
    y_pred:     np.ndarray,
    y_test:     np.ndarray
) -> dict:
    """
    Calcula Accuracy, Precision, Recall y F1-score para un modelo.
    Retorna diccionario con el nombre del modelo y sus métricas.
    """
    name = model_dict["name"]
    acc  = accuracy_score(y_test, y_pred)  * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec  = recall_score(y_test, y_pred,    zero_division=0) * 100
    f1   = f1_score(y_test, y_pred,        zero_division=0) * 100

    return {
        "Model"    : name,
        "Accuracy" : round(acc,  2),
        "Precision": round(prec, 2),
        "Recall"   : round(rec,  2),
        "F1-Score" : round(f1,   2),
        "Train Time (s)": round(model_dict.get("train_time", 0), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(
    models_results: list[tuple],   # [(model_dict, y_pred), ...]
    y_test:         np.ndarray,
    output_dir:     str = "outputs"
) -> None:
    """
    Genera un grid 2×4 con las matrices de confusión de los 8 modelos.
    Cada subgráfico tiene un heatmap con anotaciones de porcentaje.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(models_results)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Matrices de Confusión — 8 Modelos\n"
        "DDoS Detection · UNSW-NB15 · Janivasya & Rachmawati 2024",
        fontsize=14, color=TEXT_CLR, fontweight="bold", y=1.01
    )

    axes_flat = axes.flatten()

    for idx, (model_dict, y_pred) in enumerate(models_results):
        ax = axes_flat[idx]
        ax.set_facecolor(CARD_BG)
        ax.patch.set_facecolor(CARD_BG)

        cm = confusion_matrix(y_test, y_pred)

        # Normalizar para porcentajes
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        # Anotaciones: valor absoluto + porcentaje
        annot = np.array([
            [f"{cm[i,j]}\n({cm_norm[i,j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])

        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=annot, fmt="",
            cmap="Blues",
            linewidths=0.5,
            linecolor=BORDER_CLR,
            cbar=False,
            annot_kws={"size": 9, "color": "white", "weight": "bold"}
        )

        ax.set_title(model_dict["name"], fontsize=10, color=ACCENT_CLR, pad=6, fontweight="bold")
        ax.set_xlabel("Predicho", fontsize=8, color=TEXT_CLR)
        ax.set_ylabel("Real",     fontsize=8, color=TEXT_CLR)
        ax.set_xticklabels(["Normal", "DoS"], fontsize=8, color=TEXT_CLR)
        ax.set_yticklabels(["Normal", "DoS"], fontsize=8, color=TEXT_CLR, rotation=0)

        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_CLR)

    # Ocultar subplots vacíos
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(pad=2.0)
    save_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close()
    print(f"[INFO] Matrices de confusión guardadas: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRÁFICO COMPARATIVO DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Genera un bar chart agrupado comparando las 4 métricas entre los 8 modelos.
    Incluye líneas de referencia del paper.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models    = results_df["Model"].tolist()
    n_models  = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    bar_width = 0.18
    offsets   = np.linspace(-(n_metrics-1)/2, (n_metrics-1)/2, n_metrics) * bar_width

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    for i, metric in enumerate(metrics):
        vals = results_df[metric].values
        bars = ax.bar(
            x + offsets[i], vals,
            width=bar_width,
            label=metric,
            color=METRIC_COLORS[metric],
            alpha=0.85,
            edgecolor=DARK_BG,
            linewidth=0.5
        )
        # Etiquetas sobre barras
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=6.5, color=TEXT_CLR, rotation=90
            )

    # Línea de referencia al 95%
    ax.axhline(y=95, color=WARN_CLR, linestyle="--", linewidth=1.0, alpha=0.7,
               label="Referencia 95%")

    # Configuración de ejes
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10, color=TEXT_CLR)
    ax.set_yticks(np.arange(80, 101, 5))
    ax.set_yticklabels([f"{v}%" for v in np.arange(80, 101, 5)],
                        fontsize=9, color=TEXT_CLR)
    ax.set_ylim(80, 102)

    ax.set_xlabel("Modelo", fontsize=12, color=TEXT_CLR, labelpad=10)
    ax.set_ylabel("Score (%)", fontsize=12, color=TEXT_CLR, labelpad=10)
    ax.set_title(
        "Comparación de Métricas — 8 Modelos de ML/DL\n"
        "DDoS Detection · UNSW-NB15 · Janivasya & Rachmawati 2024",
        fontsize=13, color=TEXT_CLR, fontweight="bold", pad=14
    )

    ax.legend(
        loc="lower right", fontsize=9,
        facecolor=CARD_BG, edgecolor=BORDER_CLR,
        labelcolor=TEXT_CLR
    )
    ax.grid(axis="y", color=BORDER_CLR, linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_CLR)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[INFO] Gráfico de métricas guardado: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GUARDAR CSV Y TABLA COMPARATIVA (Tabla 1 del paper)
# ─────────────────────────────────────────────────────────────────────────────
def save_results_csv(results_df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """
    Guarda la tabla de métricas como CSV.
    Retorna el path del archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "results_summary.csv")
    results_df.to_csv(save_path, index=False, float_format="%.2f")
    print(f"[INFO] Tabla de resultados guardada: {save_path}")
    return save_path


def print_results_table(results_df: pd.DataFrame) -> None:
    """
    Imprime la tabla comparativa de métricas en consola
    (replica la Tabla 1 del paper).
    """
    print("\n" + "="*75)
    print("  TABLA DE RESULTADOS — DDoS Detection (Tabla 1 del paper)")
    print("="*75)
    print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time(s)':>9}")
    print(f"  {'-'*70}")

    best_acc = results_df["Accuracy"].max()

    for _, row in results_df.iterrows():
        marker = "[*]" if row["Accuracy"] == best_acc else "   "
        print(
            f"  {marker} {row['Model']:<20} "
            f"{row['Accuracy']:>9.2f}% "
            f"{row['Precision']:>9.2f}% "
            f"{row['Recall']:>9.2f}% "
            f"{row['F1-Score']:>9.2f}% "
            f"{row['Train Time (s)']:>8.2f}s"
        )

    print("="*75)
    print("  [*] = Mejor modelo   (Referencia paper: Random Forest ~97.68%)")
    print("="*75)


def compare_with_paper(results_df: pd.DataFrame) -> None:
    """
    Compara los resultados obtenidos con los del paper.
    Muestra diferencias por modelo.
    """
    print("\n" + "="*65)
    print("  COMPARACIÓN CON RESULTADOS DEL PAPER")
    print("="*65)
    print(f"  {'Model':<22} {'Obtenido':>10} {'Paper':>10} {'Diff':>8}")
    print(f"  {'-'*55}")

    for _, row in results_df.iterrows():
        model_name = row["Model"]
        obtained   = row["Accuracy"]

        # Buscar nombre compatible con PAPER_RESULTS
        paper_ref = None
        for k in PAPER_RESULTS:
            if k.lower() in model_name.lower() or model_name.lower() in k.lower():
                paper_ref = PAPER_RESULTS[k]["Accuracy"]
                break

        if paper_ref is not None:
            diff = obtained - paper_ref
            sign = "+" if diff >= 0 else ""
            print(f"  {model_name:<22} {obtained:>9.2f}% {paper_ref:>9.1f}% {sign}{diff:>6.2f}%")
        else:
            print(f"  {model_name:<22} {obtained:>9.2f}%  {'N/A':>10}")

    print("="*65)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE EVALUACIÓN COMPLETO
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all(
    trained_models: list[dict],
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    output_dir:     str = "outputs"
) -> pd.DataFrame:
    """
    Evalúa todos los modelos, genera gráficos y guarda resultados.

    Parámetros
    ----------
    trained_models : Lista de dicts retornada por models.train_all_models()
    X_test         : Features de prueba (escaladas)
    y_test         : Etiquetas reales
    output_dir     : Directorio de salida

    Retorna
    -------
    results_df     : DataFrame con métricas por modelo
    """
    from models import predict

    print("\n" + "="*60)
    print("  EVALUACIÓN DE MODELOS")
    print("="*60)

    all_metrics      = []
    models_results   = []   # Para matrices de confusión

    for model_dict in trained_models:
        if model_dict["model"] is None:
            print(f"[SKIP] {model_dict['name']} — TF no disponible.")
            continue

        print(f"\n[EVAL] {model_dict['name']}...")
        y_pred = predict(model_dict, X_test)

        metrics = compute_metrics(model_dict, y_pred, y_test)
        all_metrics.append(metrics)
        models_results.append((model_dict, y_pred))

        # Imprimir reporte de clasificación detallado
        print(classification_report(
            y_test, y_pred,
            target_names=["Normal", "DoS"],
            zero_division=0
        ))

    results_df = pd.DataFrame(all_metrics)

    # Gráficos
    plot_confusion_matrices(models_results, y_test, output_dir)
    plot_metrics_comparison(results_df, output_dir)

    # Tabla y CSV
    print_results_table(results_df)
    compare_with_paper(results_df)
    save_results_csv(results_df, output_dir)

    return results_df


if __name__ == "__main__":
    print("[INFO] evaluate.py cargado correctamente.")
