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

"FECHA": "Fecha de registro de la muestra",
    "TEMPERATURA   ⁰C BN": "Temperatura en grados Celsius en Bocatoma Norte",
    "OXIGENO DISUELTO  mg/l O2 BN": "Concentración de oxígeno disuelto en Bocatoma Norte (mg/l)",
    "TURBIEDAD  UNT BN": "Turbidez en unidades NTU en Bocatoma Norte",
    "COLOR  UPC BN": "Color en unidades UPC en Bocatoma Norte",
    "CONDUCTIVIDAD   uS/cm BN": "Conductividad en microsiemens por centímetro en Bocatoma Norte",
    "PH BN": "pH en Bocatoma Norte",
    "MATERIA ORGANICA mg/L BN": "Concentración de materia orgánica en Bocatoma Norte (mg/L)",
    "NITROGENO AMONIACAL  µg/l BN": "Concentración de nitrógeno amoniacal en Bocatoma Norte (µg/l)",
    "MANGANESOS  mg/L Mn. BN": "Concentración de manganeso en Bocatoma Norte (mg/L)",
    "ALCALINIDAD TOTAL  mg/L CaCO3 BN": "Alcalinidad total en Bocatoma Norte (mg/L de CaCO3)",
    "CLORUROS  mg/L Cl- BN": "Concentración de cloruros en Bocatoma Norte (mg/L)",
    "DUREZA TOTAL  mg/L CaCO3 BN": "Dureza total en Bocatoma Norte (mg/L de CaCO3)",
    "DUREZA CALCICA  mg/L CaCO3 BN": "Dureza cálcica en Bocatoma Norte (mg/L de CaCO3)",
    "HIERRO TOTAL  mg/L Fe+3 BN": "Concentración de hierro total en Bocatoma Norte (mg/L de Fe+3)",
    "ALUMINIO RESIDUAL  mg/L Al BN": "Concentración de aluminio residual en Bocatoma Norte (mg/L)",
    "OXIGENO DISUELTO  mg/L O2 BN": "Concentración de oxígeno disuelto en Bocatoma Norte (mg/L)",
    "POTENCIAL REDOX  mV BN": "Potencial redox en milivoltios en Bocatoma Norte",
    "TEMPERATURA    ⁰C BN": "Temperatura en grados Celsius en Bocatoma Norte",
    "NITRITOS  mg/L NO2 BN": "Concentración de nitritos en Bocatoma Norte (mg/L)",
    "FOSFATOS mg/l BN": "Concentración de fosfatos en Bocatoma Norte (mg/L)",
    "NITRATOS  mg/L NO3 BN": "Concentración de nitratos en Bocatoma Norte (mg/L)",
    "SULFATOS  mg/L SO4 BN": "Concentración de sulfatos en Bocatoma Norte (mg/L)",
    "COT  mg/L  BN": "Carbono orgánico total en Bocatoma Norte (mg/L)",
    "DIA CRU": "Día de muestreo en agua cruda",
    "TEMPERATURA   ⁰C CRU": "Temperatura en grados Celsius en agua cruda",
    "OXIGENO DISUELTO  mg/l O2 CRU": "Concentración de oxígeno disuelto en agua cruda (mg/l)",
    "TURBIEDAD  UNT CRU": "Turbidez en unidades NTU en agua cruda",
    "COLOR  UPC CRU": "Color en unidades UPC en agua cruda",
    "CONDUCTIVIDAD   uS/cm CRU": "Conductividad en microsiemens por centímetro en agua cruda",
    "PH CRU": "pH en agua cruda",
    "MATERIA ORGANICA mg/L CRU": "Concentración de materia orgánica en agua cruda (mg/L)",
    "NITROGENO AMONIACAL  µg/l CRU": "Concentración de nitrógeno amoniacal en agua cruda (µg/l)",
    "MANGANESOS  mg/L Mn. CRU": "Concentración de manganeso en agua cruda (mg/L)",
    "ALCALINIDAD TOTAL  mg/L CaCO3 CRU": "Alcalinidad total en agua cruda (mg/L de CaCO3)",
    "CLORUROS  mg/L Cl- CRU": "Concentración de cloruros en agua cruda (mg/L)",
    "DUREZA TOTAL  mg/L CaCO3 CRU": "Dureza total en agua cruda (mg/L de CaCO3)",
    "DUREZA CALCICA  mg/L CaCO3 CRU": "Dureza cálcica en agua cruda (mg/L de CaCO3)",
    "HIERRO TOTAL  mg/L Fe+3 CRU": "Concentración de hierro total en agua cruda (mg/L de Fe+3)",
    "ALUMINIO RESIDUAL  mg/L Al CRU": "Concentración de aluminio residual en agua cruda (mg/L)",
    "OXIGENO DISUELTO  mg/L O2 CRU": "Concentración de oxígeno disuelto en agua cruda (mg/L)",
    "POTENCIAL REDOX  mV CRU": "Potencial redox en milivoltios en agua cruda",
    "NITRITOS  mg/L NO2 CRU": "Concentración de nitritos en agua cruda (mg/L)",
    "NITRATOS  mg/L NO3 CRU": "Concentración de nitratos en agua cruda (mg/L)",
    "FOSFATOS mg/l CRU": "Concentración de fosfatos en agua cruda (mg/L)",
    "SULFATOS  mg/L SO4 CRU": "Concentración de sulfatos en agua cruda (mg/L)",
    "COT  mg/L  CRU": "Carbono orgánico total en agua cruda (mg/L)",
    "SOLIDOS SUSPENDIDOS  mg/L  CRU": "Sólidos suspendidos en agua cruda (mg/L)",
    "DIA MEZ": "Día de muestreo en agua mezclada",
    "OXIGENO DISUELTO mg/L O2 MEZ": "Concentración de oxígeno disuelto en agua mezclada (mg/L)",
    "TEMPERATURA   ⁰C MEZ": "Temperatura en grados Celsius en agua mezclada",
    "PH MEZ": "pH en agua mezclada",
    "CLORO LIBRE mg/L Cl2 MEZ": "Concentración de cloro libre en agua mezclada (mg/L)",
    "CLORO COMBINADO mg/L Cl2 MEZ": "Concentración de cloro combinado en agua mezclada (mg/L)",
    "CLORO TOTAL mg/L Cl2 MEZ": "Concentración de cloro total en agua mezclada (mg/L)",
    "POTENCIAL REDOX  Mv MEZ": "Potencial redox en milivoltios en agua mezclada",
    "Al₂(SO)₄ ppm": "Concentración de Al₂(SO)₄ en ppm",
    "Al₂(SO)₄  SOLIDO ppm": "Concentración de Al₂(SO)₄ sólido en ppm",
    "PAC ppm": "Concentración de PAC en ppm",
    "FeCl3 ppm": "Concentración de FeCl3 en ppm",
    "MES": "Mes de la toma de muestra"
