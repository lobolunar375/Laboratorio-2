# Reproducción Experimental: Comparison of Classification Methods

Este repositorio contiene el código fuente, los datos generados y los resultados de la reproducción experimental del artículo científico:

> **Comparison of Classification Methods Based on the Type of Attributes and Sample Size**
> Waleed Aloraini, Mohammed Mahdi Alenezi (Journal of Big Data, 2022)

## 📌 Descripción de la Implementación

El código implementa y evalúa el impacto que tienen los atributos continuos frente a los discretos y el tamaño de la muestra en el rendimiento de 7 clasificadores clásicos de Machine Learning.

La implementación se realizó empleando **Programación Orientada a Objetos (POO)** en Python y utiliza las bibliotecas estándar de análisis de datos (`numpy`, `pandas`, `scikit-learn`, `matplotlib`).

### ⚙️ Componentes Principales (`reproduccion.py`)

El script está dividido en 5 clases para garantizar modularidad y mantenibilidad:

1. **`DatasetGenerator`**: Genera dinámicamente conjuntos de datos sintéticos controlados.
   - Implementa las **Ecuaciones 1 y 2** propuestas por los autores originales para definir la variable objetivo combinando atributos continuos y discretos.
2. **`ClassifierEvaluator`**: Configura y evalúa los clasificadores mediante Validación Cruzada Estratificada de 10 pliegues (*10-fold Stratified CV*), utilizando el **AUC** (Área Bajo la Curva ROC) como métrica principal.
3. **`ResultsVisualizer`**: Genera gráficas comparativas de rendimiento (AUC) por clasificador, dataset y tamaño de muestra.
4. **`ResultsManager`**: Exporta las métricas cuantitativas a archivos CSV.
5. **`ExperimentRunner`**: Orquesta el flujo completo de evaluación, integrando todas las piezas (patrón *Facade*).

## 🔬 Diseño Experimental

- **Clasificadores evaluados:** Decision Tree (CART), C4.5 (Aproximado con criterio de entropía), K-Nearest Neighbors ($k=5$), Regresión Logística, Naive Bayes (Gaussiano), SVM (RBF) y Clasificador Lineal (Ridge).
- **Datasets:** 11 configuraciones (DS19 a DS29), que varían desde 10 atributos continuos y 0 discretos (DS19) hasta 0 atributos continuos y 10 discretos (DS29).
- **Tamaños de Muestra:** 200, 500 y 1000 registros.

## 🚀 Cómo Ejecutar

### Prerrequisitos
Tener instalado Python 3.x y las siguientes dependencias:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Ejecución
El programa interactivo evaluará los modelos y guardará las métricas automáticamente. Simplemente ejecuta el script en la terminal:
```bash
python reproduccion.py
```
El script pedirá seleccionar la cantidad de registros a simular, calculará los AUC empleando validación cruzada y exportará las gráficas PNG y los archivos CSV correspondientes.

## 📊 Resultados Generados
Este repositorio incluye las gráficas resultantes para las muestras de 200, 500 y 1000 registros (`auc_*_records.png`), así como una gráfica resumen del efecto del tamaño de la muestra (`efecto_tamano_muestra.png`), la comparación con el paper y las tablas CSV en bruto.
