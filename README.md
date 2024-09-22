# PROYECTOANS2023MIAD

# Optimización de la Dosificación de Coagulantes en PTAP Tibitoc

## Descripción del Proyecto

Este proyecto tiene como objetivo realizar un análisis de como se podria mejorar la eficiencia en la dosificación de coagulantes en la planta de tratamiento de agua potable Tibitoc (PTAP Tibitoc), que abastece alrededor del 30% del agua consumida en la ciudad de Bogotá, teniendo en cuenta datos historicos. Utilizando técnicas de aprendizaje no supervisado y análisis multivariado, se identifican patrones en las variables fisicoquímicas del agua cruda y tratada, que influyen en la cantidad de productos químicos necesarios en el proceso de coagulación y floculación.

Se aplican metodologías como el **Análisis de Componentes Principales (PCA)** para la reducción de dimensiones, **K-Means Clustering**, **Agglomerative Clustering**, y **DBSCAN** para segmentar los datos, así como modelos de regresión para predecir la dosis óptima de coagulantes.

## Estructura del Proyecto

El proyecto está organizado en las siguientes secciones:

- **Revisión de Datos**: Procesamiento y limpieza de los datos obtenidos de las muestras de agua en diferentes etapas del tratamiento.
- **Análisis Estadístico**: Identificación de las principales variables fisicoquímicas que afectan la dosificación de coagulantes.
- **Reducción de Dimensionalidad (PCA)**: Reducción de dimensiones para identificar las variables más influyentes en el proceso.
- **Clustering**: Segmentación de los datos mediante K-Means, DBSCAN y Agglomerative Clustering.
- **Predicción de Dosis**: Modelos de regresión que predicen la dosis óptima de coagulantes utilizando las variables seleccionadas.

## Requisitos del Proyecto

Este proyecto utiliza Python 3 y las siguientes bibliotecas:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Organización Documentos

Se encontrara la carpeta Data, la cual contiene los datos. La carpeta Figs, con las imagenes del .IPYN y finalmente el documento IPYN final, dado que existe una versión preiliminar presentada para la semana 3, esta no debe ser tenida en cuenta para esta entrega.
