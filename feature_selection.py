"""
feature_selection.py
====================
Seleccion de caracteristicas mediante Chi-cuadrado (SelectKBest).
Replica la Figura 1 del paper de Janivasya & Rachmawati (2024).

Pasos:
  1. Convertir features a valores no negativos (necesario para chi2)
  2. Aplicar SelectKBest con chi2, k=20
  3. Graficar importancia (barh chart horizontal)
  4. Retornar X transformado con las 20 mejores features
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para entornos sin pantalla
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Paleta de colores para el grafico (inspirada en el paper)
PALETTE_BLUE = "#1565C0"
PALETTE_LIGHT = "#42A5F5"

# Features de referencia del paper (Fig. 1) -- sirven para comparar el ranking obtenido
PAPER_TOP_FEATURES = [
    "dmean", "sload", "sjit", "rate", "dpkts",
    "dload", "djit", "sinpkt", "spkts", "ct_state_ttl",
    "ct_srv_dst", "ct_srv_src", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "dur", "sbytes", "dbytes"
]


def select_features(
    X,
    y,
    feature_names,
    k=20,
    output_dir="outputs"
):
    """
    Aplica SelectKBest con chi-cuadrado para seleccionar las k mejores features.

    Parametros
    ----------
    X             : Array de features (puede tener valores negativos)
    y             : Etiquetas binarias
    feature_names : Nombres de las columnas de X
    k             : Numero de features a seleccionar (default: 20)
    output_dir    : Directorio donde guardar el grafico

    Retorna
    -------
    X_selected    : Array con las k features seleccionadas
    selected_names: Nombres de las features seleccionadas
    selector      : Objeto SelectKBest ajustado
    """
    print(f"\n[INFO] Seleccion de caracteristicas con Chi-cuadrado (k={k})...")

    # Chi-cuadrado requiere valores no negativos --> escalar a [0, 1]
    mms = MinMaxScaler()
    X_nonneg = mms.fit_transform(X)

    # Ajustar selector
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X_nonneg, y)

    # Obtener scores e indices seleccionados
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_names   = [feature_names[i] for i in selected_indices]
    selected_scores  = [scores[i] for i in selected_indices]

    # Ordenar por score descendente
    sorted_pairs = sorted(zip(selected_scores, selected_names), reverse=True)
    sorted_scores, sorted_names = zip(*sorted_pairs)

    print(f"\n  Top {k} features seleccionadas (Chi2 score):")
    print(f"  {'Feature':<25} {'Chi2 Score':>15}")
    print(f"  {'-'*40}")
    for name, score in zip(sorted_names, sorted_scores):
        marker = "[OK]" if name.lower() in [p.lower() for p in PAPER_TOP_FEATURES] else "    "
        print(f"  {marker} {name:<23} {score:>15.2f}")

    # Comparar con features del paper
    paper_found = [n for n in sorted_names if n.lower() in [p.lower() for p in PAPER_TOP_FEATURES]]
    print(f"\n  Coincidencias con features del paper (Fig.1): {len(paper_found)}/{k}")

    # Graficar
    _plot_feature_importance(sorted_names, sorted_scores, k, output_dir)

    return X_selected, list(sorted_names), selector


def _plot_feature_importance(feature_names, scores, k, output_dir):
    """
    Genera un grafico de barras horizontal con la importancia de las features,
    similar a la Figura 1 del paper.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    # Normalizar scores para color degradado
    norm_scores = np.array(scores) / max(scores)
    colors = [plt.cm.Blues(0.4 + 0.6 * s) for s in norm_scores]

    # Invertir para que la mejor feature este arriba
    bars = ax.barh(
        range(k),
        list(reversed(list(scores))),
        color=list(reversed(colors)),
        edgecolor="#30363D",
        linewidth=0.5,
        height=0.75
    )

    # Etiquetas
    ax.set_yticks(range(k))
    ax.set_yticklabels(list(reversed(list(feature_names))),
                       fontsize=10, color="#C9D1D9")

    # Valores en cada barra
    for i, (bar, score) in enumerate(zip(bars, reversed(list(scores)))):
        ax.text(
            bar.get_width() + max(scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center", ha="left",
            fontsize=8.5, color="#8B949E"
        )

    # Linea vertical de referencia
    ax.axvline(x=0, color="#30363D", linewidth=0.8)

    # Titulos y ejes
    ax.set_xlabel("Chi2 Score", fontsize=12, color="#C9D1D9", labelpad=10)
    ax.set_title(
        f"Feature Importance -- Top {k} Features (Chi-Squared)\n"
        "Janivasya & Rachmawati 2024 - UNSW-NB15",
        fontsize=13, color="#E6EDF3", pad=14, fontweight="bold"
    )
    ax.tick_params(colors="#8B949E", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    # Leyenda de referencia
    legend_patch = mpatches.Patch(color=PALETTE_BLUE, label="Chi2 Score (SelectKBest)")
    ax.legend(handles=[legend_patch], loc="lower right",
              facecolor="#161B22", edgecolor="#30363D",
              labelcolor="#C9D1D9", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[INFO] Grafico guardado: {save_path}")


if __name__ == "__main__":
    # Prueba rapida con datos aleatorios
    from preprocessing import preprocess_pipeline
    X, y, feature_names = preprocess_pipeline()
    X_sel, sel_names, selector = select_features(X.values, y, feature_names)
    print(f"\nShape despues de seleccion: {X_sel.shape}")
