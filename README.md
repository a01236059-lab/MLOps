# MLOps

Actividades de la materia Operaciones de aprendizaje automático

Actividad Individual | Código de experimentación base con MLFlow



\--------------------------------------------------------------

El siguiente proyecto implementa el proceso de entrenamiento y testeo utilizando la base de datos Airline Satisfaction disponible en Keggel mediante un modelo de clasificación de satisfacción de pasajeros de aerolíneas utilizando Logistic Regression.

Los datos son cargados desde un archivo CSV y posteriormente se someten a unan etapa de preprocesamiento. Durante esa fase, se eliminan columnas que no aportan valor al modelo, se imputan valores faltantes y se transforman las variables categóricas mediante codificación One-Hot Encoding para que puedan ser utilizadas por el algoritmo de aprendizaje automático.

El modelo de clasificación se implementa utilizando Logistic Regression de Scikit-learn. Los principales hiperparámetros del modelo, como el valor de regularización (C), el tipo de penalización (penalty), el número máximo de iteraciones (max\_iter) y la semilla aleatoria (random\_state), son configurables desde la línea de comandos mediante la biblioteca argparse, permitiendo ejecutar múltiples experimentos sin modificar el código fuente.

Durante la etapa de evaluación se calculan métricas de desempeño como accuracy, precision y recall, además de generar una matriz de confusión que permite analizar con mayor detalle el comportamiento del clasificador.

Para garantizar la reproducibilidad y facilitar la comparación de resultados, el proyecto incorpora MLflow Tracking. En cada ejecución se registran automáticamente los hiperparámetros utilizados, las métricas obtenidas, la matriz de confusión como artefacto y el modelo entrenado. Esto permite comparar múltiples corridas dentro de un mismo experimento y seleccionar la configuración con mejor desempeño.

Finalmente, se incluye un archivo requirements.txt que documenta las dependencias y versiones utilizadas, así como la fijación de una semilla aleatoria para asegurar que ejecuciones realizadas con los mismos parámetros produzcan resultados consistentes y reproducibles.



\-----------------------------------------------------------------------------

Instalación y ejecución del proyecto

Requisitos previos

Antes de ejecutar el proyecto, es necesario contar con:

* Python 3.10 o superior.
* Un entorno virtual o entorno Conda configurado.



Instalación de dependencias:

1. Clonar o descargar el repositorio
2. Crear y activar el entorno de trabajo 
3. Instalar dependencias del proyecto utilizando archivo requirements.txt



Ejecución:

El script puede ejecutarse utilizando los valores por defecto definidos para los hiperparámetros --> Python train.py

En posible modificar  los hiperparámetros del modelo mediante argumentos de línea de comandos:

ej. Python train.py --C 0.1 --max\_iter 3000 --penatly l2 --random\_state 42



Argumentos disponibles

\--C              Intensidad de la regularización.

\--max\_iter       Número máximo de iteraciones.

\--penalty        Tipo de penalización (l1, l2, elasticnet o None).

\--random\_state   Semilla aleatoria para reproducibilidad.





