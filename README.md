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

## Datos

Acontinuación se presenta el diccionario de los datos:

-FECHA: Fecha de registro de la muestra.
-TEMPERATURA ⁰C BN: Temperatura en grados Celsius en Bocatoma Norte.
-OXÍGENO DISUELTO mg/L O₂ BN: Concentración de oxígeno disuelto en Bocatoma Norte (mg/L).
-TURBIEDAD UNT BN: Turbidez en unidades NTU en Bocatoma Norte.
-COLOR UPC BN: Color en unidades UPC en Bocatoma Norte.
-CONDUCTIVIDAD µS/cm BN: Conductividad en microsiemens por centímetro en Bocatoma Norte.
-pH BN: pH en Bocatoma Norte.
-MATERIA ORGÁNICA mg/L BN: Concentración de materia orgánica en Bocatoma Norte (mg/L).
-NITRÓGENO AMONIACAL µg/L BN: Concentración de nitrógeno amoniacal en Bocatoma Norte (µg/L).
-MANGANESO mg/L BN: Concentración de manganeso en Bocatoma Norte (mg/L).
-ALCALINIDAD TOTAL mg/L CaCO₃ BN: Alcalinidad total en Bocatoma Norte (mg/L de CaCO₃).
-CLORUROS mg/L Cl⁻ BN: Concentración de cloruros en Bocatoma Norte (mg/L).
-DUREZA TOTAL mg/L CaCO₃ BN: Dureza total en Bocatoma Norte (mg/L de CaCO₃).
-DUREZA CÁLCICA mg/L CaCO₃ BN: Dureza cálcica en Bocatoma Norte (mg/L de CaCO₃).
-HIERRO TOTAL mg/L Fe³⁺ BN: Concentración de hierro total en Bocatoma Norte (mg/L de Fe³⁺).
-ALUMINIO RESIDUAL mg/L Al BN: Concentración de aluminio residual en Bocatoma Norte (mg/L).
-OXÍGENO DISUELTO mg/L O₂ BN: Concentración de oxígeno disuelto en Bocatoma Norte (mg/L).
-POTENCIAL REDOX mV BN: Potencial redox en milivoltios en Bocatoma Norte.
-NITRITOS mg/L NO₂ BN: Concentración de nitritos en Bocatoma Norte (mg/L).
-FOSFATOS mg/L BN: Concentración de fosfatos en Bocatoma Norte (mg/L).
-NITRATOS mg/L NO₃ BN: Concentración de nitratos en Bocatoma Norte (mg/L).
-SULFATOS mg/L SO₄ BN: Concentración de sulfatos en Bocatoma Norte (mg/L).
-COT mg/L BN: Carbono orgánico total en Bocatoma Norte (mg/L).
** Agua Cruda:
-DÍA CRU: Día de muestreo en agua cruda.
-TEMPERATURA ⁰C CRU: Temperatura en grados Celsius en agua cruda.
-OXÍGENO DISUELTO mg/L O₂ CRU: Concentración de oxígeno disuelto en agua cruda (mg/L).
-TURBIEDAD UNT CRU: Turbidez en unidades NTU en agua cruda.
-COLOR UPC CRU: Color en unidades UPC en agua cruda.
-CONDUCTIVIDAD µS/cm CRU: Conductividad en microsiemens por centímetro en agua cruda.
-pH CRU: pH en agua cruda.
-MATERIA ORGÁNICA mg/L CRU: Concentración de materia orgánica en agua cruda (mg/L).
-NITRÓGENO AMONIACAL µg/L CRU: Concentración de nitrógeno amoniacal en agua cruda (µg/L).
-MANGANESO mg/L CRU: Concentración de manganeso en agua cruda (mg/L).
-ALCALINIDAD TOTAL mg/L CaCO₃ CRU: Alcalinidad total en agua cruda (mg/L de CaCO₃).
-CLORUROS mg/L Cl⁻ CRU: Concentración de cloruros en agua cruda (mg/L).
-DUREZA TOTAL mg/L CaCO₃ CRU: Dureza total en agua cruda (mg/L de CaCO₃).
-DUREZA CÁLCICA mg/L CaCO₃ CRU: Dureza cálcica en agua cruda (mg/L de CaCO₃).
-HIERRO TOTAL mg/L Fe³⁺ CRU: Concentración de hierro total en agua cruda (mg/L de Fe³⁺).
-ALUMINIO RESIDUAL mg/L Al CRU: Concentración de aluminio residual en agua cruda (mg/L).
-POTENCIAL REDOX mV CRU: Potencial redox en milivoltios en agua cruda.
-NITRITOS mg/L NO₂ CRU: Concentración de nitritos en agua cruda (mg/L).
-NITRATOS mg/L NO₃ CRU: Concentración de nitratos en agua cruda (mg/L).
-FOSFATOS mg/L CRU: Concentración de fosfatos en agua cruda (mg/L).
-SULFATOS mg/L SO₄ CRU: Concentración de sulfatos en agua cruda (mg/L).
-COT mg/L CRU: Carbono orgánico total en agua cruda (mg/L).
-SÓLIDOS SUSPENDIDOS mg/L CRU: Sólidos suspendidos en agua cruda (mg/L).
** Agua Mezclada:
-DÍA MEZ: Día de muestreo en agua mezclada.
-OXÍGENO DISUELTO mg/L O₂ MEZ: Concentración de oxígeno disuelto en agua mezclada (mg/L).
-TEMPERATURA ⁰C MEZ: Temperatura en grados Celsius en agua mezclada.
-pH MEZ: pH en agua mezclada.
-CLORO LIBRE mg/L Cl₂ MEZ: Concentración de cloro libre en agua mezclada (mg/L).
-CLORO COMBINADO mg/L Cl₂ MEZ: Concentración de cloro combinado en agua mezclada (mg/L).
-CLORO TOTAL mg/L Cl₂ MEZ: Concentración de cloro total en agua mezclada (mg/L).
-POTENCIAL REDOX mV MEZ: Potencial redox en milivoltios en agua mezclada.
**Productos Químicos:
-Al₂(SO₄) ppm: Concentración de Al₂(SO₄) en ppm.

Al₂(SO₄) SÓLIDO ppm: Concentración de Al₂(SO₄) sólido en ppm.

PAC ppm: Concentración de PAC en ppm.

FeCl₃ ppm: Concentración de FeCl₃ en ppm.

MES: Mes de la toma de muestra.
