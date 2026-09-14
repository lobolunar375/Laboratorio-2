# DDoS Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-green)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![Paper](https://img.shields.io/badge/Paper-Procedia%20CS%202024-red)](https://doi.org/10.1016/j.procs.2024)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Reproduccion completa de la metodologia del paper:

> **"DDoS Detection using Machine Learning Approach"**
> Janivasya & Rachmawati, 2024 — *Procedia Computer Science*, Elsevier

Implementa y compara **8 algoritmos de Machine Learning y Deep Learning** para clasificacion binaria de trafico de red como **Normal** o **Ataque DDoS**, usando el dataset benchmark **UNSW-NB15**.

---

## Resultados Obtenidos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 94.21% | 79.35% | 82.82% | 81.05% |
| KNN (k=5) | 97.90% | 96.77% | 88.93% | 92.69% |
| SVM (Linear) | 94.33% | 79.60% | 83.52% | 81.52% |
| **Random Forest** ✅ | **98.89%** | **98.00%** | **94.53%** | **96.23%** |
| Decision Tree | 98.32% | 94.32% | 94.44% | 94.38% |
| MLP | 98.64% | 97.94% | 92.88% | 95.34% |
| LSTM | 98.56% | 97.83% | 92.42% | 95.05% |
| GRU | 98.55% | 98.30% | 91.87% | 94.97% |

> **Random Forest** es el mejor modelo, replicando la conclusion principal del paper original.
> Resultados obtenidos sobre el dataset real UNSW-NB15 (109,353 muestras tras filtrado DoS/Normal).

---

## Estructura del Proyecto

```
Ia2/
├── preprocessing.py        # Carga, limpieza, filtrado DoS/Normal, encoding, escalado
├── feature_selection.py    # Seleccion de 20 features con Chi-cuadrado (SelectKBest)
├── models.py               # Definicion y entrenamiento de los 8 modelos
├── evaluate.py             # Metricas, matrices de confusion, graficos, CSV
├── main.py                 # Pipeline orquestador con argumentos CLI
├── requirements.txt        # Dependencias del proyecto
├── informe_ddos_ml.tex     # Informe completo en LaTeX
├── data/                   # Dataset UNSW-NB15 (descargar manualmente)
│   └── Training and Testing Sets/
│       ├── UNSW_NB15_training-set.csv
│       └── UNSW_NB15_testing-set.csv
└── outputs/                # Generado automaticamente al ejecutar
    ├── feature_importance.png
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    └── results_summary.csv
```

---

## Metodologia

El pipeline replica exactamente la metodologia del paper:

```
Dataset UNSW-NB15
      │
      ▼
1. Carga y Exploracion    →  257,673 registros x 45 columnas
      │
      ▼
2. Preprocesamiento       →  Limpieza, Label Encoding, filtrado DoS/Normal
      │                      109,353 muestras resultantes
      ▼
3. Seleccion de Features  →  Chi-cuadrado (SelectKBest, k=20)
      │
      ▼
4. Split Train/Test       →  80% entrenamiento / 20% prueba (estratificado)
      │
      ▼
5. Entrenamiento          →  8 modelos: LR, KNN, SVM, RF, DT, MLP, LSTM, GRU
      │
      ▼
6. Evaluacion             →  Accuracy, Precision, Recall, F1-Score
                             Matrices de confusion + graficos comparativos
```

---

## Instalacion y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/lobolunar375/Laboratorio-2.git
cd Laboratorio-2
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset UNSW-NB15

Descarga los archivos desde la pagina oficial:
**https://research.unsw.edu.au/projects/unsw-nb15-dataset**

Copia los siguientes archivos a la carpeta `data/Training and Testing Sets/`:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

> **Nota:** Si no tienes el dataset, el programa genera automaticamente datos sinteticos
> con la misma estructura para probar el pipeline completo.

### 4. Ejecutar el pipeline

```bash
# Ejecucion basica
python main.py

# Con parametros personalizados
python main.py --k-features 20 --test-size 0.20 --output-dir outputs
```

#### Parametros disponibles

| Parametro | Default | Descripcion |
|---|---|---|
| `--data-dir` | `data` | Directorio con los CSV del UNSW-NB15 |
| `--k-features` | `20` | Numero de features a seleccionar con Chi2 |
| `--test-size` | `0.20` | Proporcion del conjunto de prueba |
| `--output-dir` | `outputs` | Directorio donde guardar los resultados |

### 5. Ver resultados

Los archivos se generan automaticamente en `outputs/`:

| Archivo | Descripcion |
|---|---|
| `feature_importance.png` | Top 20 features por Chi-cuadrado (replica Fig. 1 del paper) |
| `confusion_matrices.png` | Grid 2x4 con matrices de confusion de los 8 modelos |
| `metrics_comparison.png` | Bar chart comparativo de las 4 metricas |
| `results_summary.csv` | Tabla completa exportada en CSV |

---

## Modulos

### `preprocessing.py`
- Carga los CSV del UNSW-NB15 o genera datos sinteticos como fallback
- Limpia el dataset (duplicados, nulos, infinitos)
- Filtra solo clases **Normal** y **DoS** para clasificacion binaria
- Aplica Label Encoding a variables categoricas (`proto`, `service`, `state`)

### `feature_selection.py`
- Aplica `SelectKBest` con funcion de puntuacion `chi2` de scikit-learn
- Escala a `[0,1]` previamente (requisito de chi-cuadrado: valores no negativos)
- Genera grafico de importancia de features (barh horizontal)
- Muestra coincidencias con las features reportadas en el paper (Fig. 1)

### `models.py`
- Implementa los 8 modelos con sus hiperparametros del paper
- Detecta automaticamente si TensorFlow esta disponible (fallback a sklearn MLP)
- Interfaz unificada `predict()` para todos los tipos de modelos

### `evaluate.py`
- Calcula Accuracy, Precision, Recall y F1-Score para cada modelo
- Genera grid de matrices de confusion (2x4, 8 subplots)
- Genera bar chart agrupado comparativo
- Exporta tabla de metricas a CSV
- Compara resultados con los reportados en el paper original

### `main.py`
- Orquesta el pipeline completo de principio a fin
- Limpia `outputs/` antes de cada ejecucion (resultados siempre frescos)
- Acepta argumentos de linea de comandos via `argparse`

---

## Requisitos del Sistema

- **Python**: 3.9 o superior
- **RAM**: minimo 4 GB (recomendado 8 GB para el dataset completo)
- **Almacenamiento**: ~500 MB para los archivos CSV del dataset
- **SO**: Windows, Linux o macOS

### Dependencias principales

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tensorflow>=2.12.0   # Opcional
```

---

## Sobre el Dataset UNSW-NB15

El **UNSW-NB15** fue creado por el Cyber Range Lab de la Universidad de New South Wales (Australia).
Contiene trafico de red real mezclado con ataques sinteticos generados con IXIA PerfectStorm.

| Caracteristica | Valor |
|---|---|
| Total de registros | 257,673 |
| Numero de columnas | 45 |
| Clases de ataque | 9 (Generic, Exploits, Fuzzers, DoS, etc.) |
| Registros Normal | 93,000 (36.1%) |
| Registros DoS | 16,353 (6.3%) |
| Muestras usadas (filtro) | 109,353 |

---

## Referencia del Paper

```bibtex
@article{janivasya2024ddos,
  title   = {DDoS Detection using Machine Learning Approach},
  author  = {Janivasya, S. and Rachmawati, D.},
  journal = {Procedia Computer Science},
  year    = {2024},
  publisher = {Elsevier}
}
```

---

## Licencia

Este proyecto es de uso academico. Ver [LICENSE](LICENSE) para mas detalles.
