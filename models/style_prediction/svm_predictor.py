import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import glob
import time
import datetime

from models.style_prediction.feature_enginering import load_data, clean_data, create_features, calcular_indices_basicos, calcular_indices_avanzados, calcular_indices_especializados, categorizar_estilos

def cargar_modelos(ruta_directorio: str = "models/style_prediction/modelos_entrenados") -> Dict[str, Dict[str, Any]]:
    """
    Carga modelos previamente entrenados desde archivos.
    
    Args:
        ruta_directorio (str): Ruta al directorio con modelos guardados
    
    Returns:
        dict: Diccionario con modelos cargados por categoría
    """
    # Verificar que el directorio exista
    if not os.path.exists(ruta_directorio):
        print(f"ERROR: Directorio no encontrado: {ruta_directorio}")
        print(f"Intentando crear directorio: {ruta_directorio}")
        try:
            os.makedirs(ruta_directorio, exist_ok=True)
        except Exception as e:
            print(f"Error al crear directorio: {e}")
        return {}
    
    # Diccionario para almacenar modelos
    modelos = {}
    
    # Categorías de estilo
    categorias = [
        'orientacion_general', 'fase_ofensiva', 'patron_ataque',
        'intensidad_defensiva', 'altura_bloque', 'tipo_transicion',
        'estilo_posesion'
    ]
    
    # Buscar archivos de modelo para cada categoría
    for categoria in categorias:
        # Patrón para buscar archivos
        patron = f"modelo_svm_{categoria}_*.joblib"
        
        # Buscar archivos que coincidan
        archivos = glob.glob(os.path.join(ruta_directorio, patron))
        
        if archivos:
            # Ordenar por fecha (más reciente primero)
            archivos.sort(reverse=True)
            
            # Cargar el modelo más reciente
            try:
                modelo_info = joblib.load(archivos[0])
                modelos[categoria] = modelo_info
                
            except Exception as e:
                print(f"ERROR al cargar modelo para '{categoria}': {e}")
        else:
            print(f"No se encontró modelo para '{categoria}'")

    if len(modelos) == 0:
        print("\nNo se encontraron modelos. Iniciando proceso de entrenamiento...")
        
        try:
            # Cargar datos
            datos = load_data(
                master_path="data/raw/master/master_liga_vert_atlmed.csv",
                stats_path="data/raw/stats_atm_por_partido_pag4.csv",
                eventos_dir="data/raw/parquet/"
            )
            
            # Procesar datos
            datos_limpios = clean_data(datos)

            datos_procesados = create_features(datos_limpios)
            
            if isinstance(datos_procesados, pd.DataFrame) and not datos_procesados.empty:
                df_combined = datos_procesados
                
                # Categorizar estilos
                df_categorizado = categorizar_estilos(df_combined)
                
                # Entrenar modelos
                # Importar LabelEncoder y SimpleImputer para manejar valores faltantes

                # Extraer las categorías de las columnas del DataFrame
                categorias = [col for col in df_categorizado.columns if col in [
                    'orientacion_general', 'fase_ofensiva', 'patron_ataque', 'intensidad_defensiva', 
                    'altura_bloque', 'tipo_transicion', 'estilo_posesion'
                ]]

                # Imprimir información de diagnóstico
                for col in df_categorizado.columns:
                    print(f"{col}: {df_categorizado[col].dtype} - Valores NaN: {df_categorizado[col].isna().sum()}")

                # Seleccionar explícitamente solo columnas numéricas conocidas
                columnas_numericas = ['IIJ', 'IVJO', 'EC', 'EF', 'IJD', 'IER', 'IV', 'IPA', 'IA', 'IDD', 'ICJ', 
                                    'IEO', 'IPT', 'IAB', 'ICT', 'ICP', 'rival_categoria_num', 'posesion']

                # Filtrar para asegurarnos de que sólo usamos columnas que existen en el DataFrame
                columnas_a_usar = [col for col in columnas_numericas if col in df_categorizado.columns]
                
                X_procesado = df_categorizado[columnas_a_usar].copy()

                # Verificar que todo es numérico y manejar valores NaN
                for col in X_procesado.columns:
                    if X_procesado[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
                        
                        X_procesado[col] = pd.to_numeric(X_procesado[col], errors='coerce')
                    
                    # Detectar y reportar NaN
                    nan_count = X_procesado[col].isna().sum()
                    if nan_count > 0:
                        
                        # Imputar valores NaN con la mediana
                        X_procesado[col] = X_procesado[col].fillna(X_procesado[col].median())

                y = df_categorizado[categorias]

                # Convertir también las etiquetas si es necesario
                y_procesado = y.copy()
                for col in y_procesado.columns:
                    if y_procesado[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
                        
                        label_encoder = LabelEncoder()
                        y_procesado[col] = label_encoder.fit_transform(y_procesado[col])
                        # Mostrar el mapeo
                        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                                           
                    # Manejar NaN en etiquetas si existen
                    nan_count = y_procesado[col].isna().sum()
                    if nan_count > 0:
                        
                        # Para etiquetas, suele ser mejor usar el modo (valor más frecuente)
                        y_procesado[col] = y_procesado[col].fillna(y_procesado[col].mode()[0])

                # Obtener lista de características
                features = X_procesado.columns.tolist()

                # Dividir datos en conjuntos de entrenamiento y prueba
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_procesado, y_procesado, test_size=0.25, random_state=42)

                # Crear diccionarios de etiquetas por categoría
                y_train_dict = {}
                y_test_dict = {}
                for categoria in categorias:
                    y_train_dict[categoria] = y_train[categoria]
                    y_test_dict[categoria] = y_test[categoria]

                modelos = entrenar_todos_modelos({
                    "datos": df_categorizado,
                    "categorias": categorias,
                    "X_train": X_train,
                    "y_train": y_train,
                    "y_train_dict": y_train_dict,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_test_dict": y_test_dict,
                    "features": features
                }, ruta_directorio)

        except Exception as e:
            import traceback
            print(f"ERROR en entrenamiento: {e}")
            print(traceback.format_exc()) 

    return modelos

# Configuración de visualización
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

def preparar_datos_modelado(df, normalizar=True):
    """
    Prepara los datos para su uso en modelos SVM de predicción de estilo.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        normalizar (bool): Si se deben normalizar los datos
    
    Returns:
        dict: Diccionario con datos preparados
    """
        
    # Columnas que esperamos (versión original)
    columnas_esperadas = [
        'IIJ', 'IVJO', 'EC', 'EF', 'IJD', 'IER', 'IV', 'IPA', 'IA', 'IDD', 'ICJ', 
        'IEO', 'IPT', 'IAB', 'ICT', 'ICP'
    ]
    
    # Verificar qué columnas están disponibles
    columnas_disponibles = [col for col in columnas_esperadas if col in df.columns]
    
    # Si ninguna columna esperada está disponible, usar todas las columnas numéricas
    if not columnas_disponibles:
        print("No se encontraron columnas esperadas. Usando todas las columnas numéricas.")
        columnas_disponibles = df.select_dtypes(include=['number']).columns.tolist()
        # Excluir columnas que no son características
        columnas_a_excluir = ['jornada', 'orientacion_general', 'fase_ofensiva', 'patron_ataque', 
                             'intensidad_defensiva', 'altura_bloque', 'tipo_transicion', 'estilo_posesion']
        columnas_disponibles = [col for col in columnas_disponibles if col not in columnas_a_excluir]
        
    # Filtrar datos para tener sólo las columnas disponibles
    X = df[columnas_disponibles].copy()
    
    # Verificar y manejar valores NaN
    for col in X.columns:
        nan_count = X[col].isna().sum()
        if nan_count > 0:
            print(f"Columna {col} tiene {nan_count} valores NaN. Imputando con la mediana.")
            X[col] = X[col].fillna(X[col].median())
    
    # Asegurar que todas las columnas son numéricas
    for col in X.columns:
        if X[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            print(f"Convirtiendo columna {col} a numérica")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())  # Rellenar NaN resultantes
        
    # Si no hay características disponibles, no intentar normalizar
    if len(columnas_disponibles) == 0:
        print("Error: No hay características disponibles para el modelado")
        # Retornar un diccionario vacío pero con las claves necesarias para evitar errores
        return {
            "X": pd.DataFrame(),
            "features": [],
            "X_scaled": np.array([]).reshape(0, 0)
        }
    
    # Normalizar si se solicita
    if normalizar and len(X) > 0:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
        
    # Devolver diccionario con datos procesados
    return {
        "X": X,
        "features": columnas_disponibles,
        "X_scaled": X_scaled
    }

def entrenar_modelo_svm(X_train: pd.DataFrame, y_train: pd.Series, 
                       optimizar_hiperparametros: bool = True, 
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Entrena un modelo SVM para clasificación de estilos de juego.
    
    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        y_train (pd.Series): Etiquetas de entrenamiento
        optimizar_hiperparametros (bool): Si True, realiza optimización de hiperparámetros
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        dict: Diccionario con modelo entrenado y metadatos:
            - 'modelo': Modelo SVM entrenado
            - 'mejores_parametros': Mejores hiperparámetros (si se optimizó)
            - 'clases': Clases únicas en los datos
    """
        
    resultado = {}
    
    # Registrar tiempo de inicio del entrenamiento
    tiempo_inicio = time.time()
    
    # Optimización de hiperparámetros
    if optimizar_hiperparametros:
        
        # Definir parámetros a probar
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        # Crear y entrenar GridSearchCV
        grid = GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
                
        modelo = grid.best_estimator_
        resultado['mejores_parametros'] = grid.best_params_
    else:
        # Entrenar modelo con parámetros predeterminados
        modelo = SVC(kernel='rbf', probability=True, random_state=random_state)
        modelo.fit(X_train, y_train)
    
    # Calcular tiempo de entrenamiento
    tiempo_fin = time.time()
    tiempo_entrenamiento = tiempo_fin - tiempo_inicio

    # Guardar modelo y clases
    resultado['modelo'] = modelo
    resultado['clases'] = list(y_train.unique())
    resultado['tiempo_entrenamiento'] = tiempo_entrenamiento
    
    return resultado

def seleccionar_caracteristicas(X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame, k: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Selecciona las k características más importantes usando ANOVA F-value.
    
    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        y_train (pd.Series): Etiquetas de entrenamiento
        X_test (pd.DataFrame): Características de prueba
        k (int): Número de características a seleccionar
    
    Returns:
        tuple: (X_train_reducido, X_test_reducido, caracteristicas_seleccionadas)
    """
    
    # Ajustar el valor k al máximo disponible
    k = min(k, X_train.shape[1])
    
    # Crear y ajustar selector
    selector = SelectKBest(f_classif, k=k)
    X_train_reducido = selector.fit_transform(X_train, y_train)
    X_test_reducido = selector.transform(X_test)
    
    # Obtener nombres de características seleccionadas
    caracteristicas_seleccionadas = [nombre for i, nombre in enumerate(X_train.columns) if selector.get_support()[i]]
        
    # Convertir a DataFrame para facilitar su uso
    X_train_reducido_df = pd.DataFrame(X_train_reducido, 
                                     columns=[f'feature_{i}' for i in range(k)],
                                     index=X_train.index)
    X_test_reducido_df = pd.DataFrame(X_test_reducido, 
                                    columns=[f'feature_{i}' for i in range(k)],
                                    index=X_test.index)
    
    return X_train_reducido_df, X_test_reducido_df, caracteristicas_seleccionadas, selector

def evaluar_modelo(modelo: Any, X: pd.DataFrame, y: pd.Series, 
                 nombre_modelo: str = "") -> Tuple[float, float]:
    """
    Evalúa el rendimiento de un modelo de clasificación.
    
    Args:
        modelo: Modelo entrenado con método predict
        X (pd.DataFrame): Características para evaluación
        y (pd.Series): Etiquetas verdaderas
        nombre_modelo (str): Nombre del modelo para mostrar en resultados
    
    Returns:
        tuple: (precision, f1_score)
    """
    # Realizar predicciones
    y_pred = modelo.predict(X)
    
    # Calcular métricas
    precision = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
        
    return precision, f1

def visualizar_matriz_confusion(modelo: Any, X: pd.DataFrame, y: pd.Series, 
                              clases: List[str], titulo: str = "Matriz de Confusión") -> None:
    """
    Visualiza la matriz de confusión para el modelo.
    
    Args:
        modelo: Modelo entrenado con método predict
        X (pd.DataFrame): Características
        y (pd.Series): Etiquetas verdaderas
        clases (list): Lista de clases únicas
        titulo (str): Título para la gráfica
    """
    # Generar predicciones
    y_pred = modelo.predict(X)
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y, y_pred)
    
    # Normalizar matriz
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=clases, yticklabels=clases)
    plt.title(titulo, fontsize=16)
    plt.ylabel('Etiqueta Real', fontsize=14)
    plt.xlabel('Etiqueta Predicha', fontsize=14)
    plt.tight_layout()
    plt.show()

def crear_visualizacion_pca(X: pd.DataFrame, y: pd.Series, titulo: str = "", 
                          random_state: int = 42) -> Tuple[plt.Figure, PCA]:
    """
    Crea una visualización 2D de los datos usando PCA.
    
    Args:
        X (pd.DataFrame): Características
        y (pd.Series): Etiquetas para colorear puntos
        titulo (str): Título para la gráfica
        random_state (int): Semilla para reproducibilidad
    
    Returns:
        tuple: (figura, objeto_pca)
    """
    # Crear y ajustar PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convertir etiquetas a numéricos si es necesario
    if y.dtype == 'object':
        classes = y.unique()
        class_colors = {clase: plt.cm.viridis(i/len(classes)) for i, clase in enumerate(classes)}
        colors = [class_colors[label] for label in y]
        
        # Crear scatter plot
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.8, s=80)
        
        # Añadir leyenda
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=clase)
                          for clase, color in class_colors.items()]
        ax.legend(handles=legend_elements, title="Estilos de juego", loc="best")
    else:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.8, s=80, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Etiqueta')
    
    # Configurar gráfico
    ax.set_title(titulo if titulo else "Visualización PCA de los datos", fontsize=16)
    ax.set_xlabel(f"Dimensión táctica 1 (Varianza: {pca.explained_variance_ratio_[0]:.2f})", fontsize=14)
    ax.set_ylabel(f"Dimensión táctica 2 (Varianza: {pca.explained_variance_ratio_[1]:.2f})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return fig, pca

def entrenar_todos_modelos(datos_modelo: Dict[str, Any], 
                         reducir_dimensionalidad: bool = True, 
                         k: int = 7, 
                         optimizar_hiperparametros: bool = True,
                         guardar_modelos: bool = True,
                         ruta_guardado: str = "models/style_prediction/modelos_entrenados") -> Dict[str, Dict[str, Any]]:
    """
    Entrena modelos SVM para todas las categorías de estilo.
    
    Args:
        datos_modelo (dict): Datos preparados para modelado
        reducir_dimensionalidad (bool): Si True, aplica selección de características
        k (int): Número de características a seleccionar si se reduce dimensionalidad
        optimizar_hiperparametros (bool): Si True, optimiza hiperparámetros
        guardar_modelos (bool): Si True, guarda los modelos entrenados
        ruta_guardado (str): Ruta para guardar los modelos
    
    Returns:
        dict: Diccionario con resultados para cada categoría
    """
            
    # Resultados para cada categoría
    resultados_modelos = {}
    
    # Crear carpeta para guardar modelos si no existe
    if guardar_modelos:

        if not os.path.exists(ruta_guardado):
            try:
                os.makedirs(ruta_guardado)
            except Exception as e:
                print(f"Error al crear directorio: {e}")
    
    # Iterar sobre cada categoría
    for categoria in datos_modelo['categorias']:
        
        # Obtener datos para esta categoría
        X_train = datos_modelo['X_train']
        y_train = datos_modelo['y_train_dict'][categoria]
        X_test = datos_modelo['X_test']
        y_test = datos_modelo['y_test_dict'][categoria]
        
        # Información de las características originales
        caracteristicas_originales = datos_modelo['features']
        
        # Aplicar selección de características
        if reducir_dimensionalidad:

            X_train_reducido, X_test_reducido, caracteristicas_seleccionadas, selector = seleccionar_caracteristicas(
                X_train, y_train, X_test, k=k
            )
            
            # Entrenar modelo con características reducidas
            resultado = entrenar_modelo_svm(
                X_train_reducido, y_train, 
                optimizar_hiperparametros=optimizar_hiperparametros
            )
            
            # Evaluar modelo
            modelo = resultado['modelo']
            precision, f1 = evaluar_modelo(modelo, X_test_reducido, y_test, f"reducido para {categoria}")
            
            # Guardar selector y características seleccionadas
            resultado['selector'] = selector
            resultado['caracteristicas'] = caracteristicas_seleccionadas
            resultado['caracteristicas_originales'] = caracteristicas_originales
            
            # Crear visualización PCA
            fig, pca = crear_visualizacion_pca(
                pd.concat([X_train_reducido, X_test_reducido]), 
                pd.concat([y_train, y_test]), 
                titulo=f"Visualización PCA para categoría: {categoria}"
            )
            plt.close(fig)  # Cerrar figura para liberar memoria
            
            # Guardar PCA
            resultado['pca'] = pca
        else:
            # Entrenar modelo con todas las características
            resultado = entrenar_modelo_svm(
                X_train, y_train, 
                optimizar_hiperparametros=optimizar_hiperparametros
            )
            
            # Evaluar modelo
            modelo = resultado['modelo']
            precision, f1 = evaluar_modelo(modelo, X_test, y_test, f"completo para {categoria}")
            
            # Guardar características
            resultado['caracteristicas'] = caracteristicas_originales
            resultado['caracteristicas_originales'] = caracteristicas_originales
            
            # Crear visualización PCA
            fig, pca = crear_visualizacion_pca(
                pd.concat([X_train, X_test]), 
                pd.concat([y_train, y_test]), 
                titulo=f"Visualización PCA para categoría: {categoria}"
            )
            plt.close(fig)  # Cerrar figura para liberar memoria
            
            # Guardar PCA
            resultado['pca'] = pca
        
        # Guardar métricas
        resultado['precision'] = precision
        resultado['f1'] = f1
        
        # Guardar conjuntos de datos
        if reducir_dimensionalidad:
            resultado['X_train'] = X_train_reducido
            resultado['X_test'] = X_test_reducido
        else:
            resultado['X_train'] = X_train
            resultado['X_test'] = X_test
        
        resultado['y_train'] = y_train
        resultado['y_test'] = y_test
        
        # Guardar modelo
        if guardar_modelos:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            nombre_archivo = os.path.join(ruta_guardado, f"modelo_svm_{categoria}_{timestamp}.joblib")
            joblib.dump(resultado, nombre_archivo)
        
        # Añadir a resultados
        resultados_modelos[categoria] = resultado
    
    # Resumen de rendimiento
    for categoria, info in resultados_modelos.items():
        print(f"{categoria}: Precisión={info['precision']:.4f}, F1-score={info['f1']:.4f}")
        print(f"  Características: {info['caracteristicas']}")
        print(f"  Clases: {info['clases']}")
        print("  " + "-"*50)
    
    return resultados_modelos

def predecir_estilo(datos_partido: pd.DataFrame, modelos: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Predice el estilo de juego para un partido específico.
    
    Args:
        datos_partido (pd.DataFrame): DataFrame con características de un partido
        modelos (dict): Diccionario con modelos entrenados para cada categoría
    
    Returns:
        dict: Predicciones para cada categoría de estilo
    """

    # Al inicio de predecir_estilo, después de verificar que datos_partido no está vacío
    
    for categoria, info_modelo in modelos.items():
        print(f"Modelo {categoria} espera: {info_modelo['caracteristicas']}")
        
    # Verificar que haya datos
    if datos_partido.empty:
        print("ERROR: No hay datos de partido para predecir")
        return {}
    
    # Calcular los índices tácticos necesarios
    try:        
        # Asumimos que datos_partido contiene las estadísticas básicas
        # Calcular índices básicos
        indices_basicos = calcular_indices_basicos(datos_partido, datos_partido)
        
        # Calcular índices avanzados (si es posible)
        indices_avanzados = calcular_indices_avanzados(pd.DataFrame(), pd.DataFrame(), datos_partido)
        
        # Calcular índices especializados combinando estadísticas y otros índices
        df_combinado = datos_partido.copy()
        
        # Añadir índices básicos al combinado
        if not indices_basicos.empty:
            for col in indices_basicos.columns:
                if col != 'jornada' and col in indices_basicos.columns:
                    df_combinado[col] = indices_basicos[col].values
        
        # Añadir índices avanzados al combinado
        if not indices_avanzados.empty:
            for col in indices_avanzados.columns:
                if col != 'jornada' and col in indices_avanzados.columns:
                    df_combinado[col] = indices_avanzados[col].values
        
        # Calcular índices especializados
        indices_especializados = calcular_indices_especializados(df_combinado)
        
        # Añadir índices especializados al combinado
        if not indices_especializados.empty:
            for col in indices_especializados.columns:
                if col != 'jornada' and col in indices_especializados.columns:
                    df_combinado[col] = indices_especializados[col].values
        
        # Verificar qué índices se han calculado
        indices_calculados = [col for col in df_combinado.columns if col.startswith('I')]
                
        # Usar el DataFrame combinado que ahora contiene los índices
        datos_con_indices = df_combinado
        
    except Exception as e:
        print(f"Error al calcular índices: {e}")
        datos_con_indices = datos_partido
    
    # Diccionario para almacenar predicciones
    predicciones = {}
    
    # Iterar por cada categoría/modelo
    for categoria, info_modelo in modelos.items():
        modelo = info_modelo['modelo']
        caracteristicas = info_modelo['caracteristicas']
        
        # Verificar si las características están disponibles
        caract_disponibles = [c for c in caracteristicas if c in datos_con_indices.columns]
        if len(caract_disponibles) != len(caracteristicas):
            print(f"ADVERTENCIA: Faltan características para {categoria}:")
            print(f"  Requeridas: {caracteristicas}")
            print(f"  Disponibles: {caract_disponibles}")
            print(f"  Faltantes: {[c for c in caracteristicas if c not in caract_disponibles]}")
            continue
        
        # Verificar si hay selector (reducción de dimensionalidad)
        if 'selector' in info_modelo:
            selector = info_modelo['selector']
            caracteristicas_originales = info_modelo['caracteristicas_originales']
            
            # Extraer características requeridas
            X = datos_con_indices[caracteristicas_originales].values.reshape(1, -1)
            
            # Aplicar selector
            X_reducido = selector.transform(X)
            
            # Predecir
            try:
                prediccion = modelo.predict(X_reducido)[0]
                predicciones[categoria] = prediccion
            except Exception as e:
                print(f"Error al predecir {categoria}: {e}")
        else:
            # Extraer características requeridas
            X = datos_con_indices[caracteristicas].values.reshape(1, -1)
            
            # Predecir
            try:
                prediccion = modelo.predict(X)[0]
                predicciones[categoria] = prediccion
            except Exception as e:
                print(f"Error al predecir {categoria}: {e}")
    
    return predicciones

def analizar_importancia_caracteristicas(X: pd.DataFrame, y: pd.Series, modelo_tipo: str = "svm") -> pd.DataFrame:
    """
    Analiza la importancia de las características para la clasificación.
    
    Args:
        X (pd.DataFrame): Características
        y (pd.Series): Etiquetas
        modelo_tipo (str): Tipo de modelo a usar ('svm' o 'tree')
    
    Returns:
        pd.DataFrame: DataFrame con importancia de características
    """
    from sklearn.tree import DecisionTreeClassifier
    
    if modelo_tipo.lower() == "svm":
        # Para SVM, usar valores F de ANOVA
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X, y)
        
        # Obtener puntuaciones
        importancia = selector.scores_
    elif modelo_tipo.lower() == "tree":
        # Para árboles, usar importancia de características
        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X, y)
        
        # Obtener importancia
        importancia = tree.feature_importances_
    else:
        raise ValueError(f"Tipo de modelo no soportado: {modelo_tipo}")
    
    # Crear DataFrame con importancia
    df_importancia = pd.DataFrame({
        'caracteristica': X.columns,
        'importancia': importancia
    })
    
    # Ordenar por importancia
    df_importancia = df_importancia.sort_values('importancia', ascending=False)
    
    return df_importancia

