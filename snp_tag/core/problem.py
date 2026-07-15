"""
Módulo de Definición del Problema (problem.py)
---------------------------------------------
Define la clase TagSNPProblem compatible con PyMoo, implementando funciones 
de evaluación vectorizadas para optimizar el rendimiento computacional.
"""

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np # Importa la librería NumPy para operaciones matriciales vectorizadas
from pymoo.core.problem import Problem # Importa la clase base Problem de PyMoo para definir problemas de optimización


def transformar_objetivos_a_minimizacion(
    k: np.ndarray,
    min_cobertura: np.ndarray,
    hamming_med: np.ndarray,
    varianza: np.ndarray,
    modo_transformacion: str = 'neg',
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Convierte objetivos de maximización a minimización según el modo configurado.
    """
    modo = str(modo_transformacion or 'neg').strip().lower() # Normaliza el modo de transformación a minúsculas eliminando espacios
    if modo == 'inverse': # Comprueba si el modo de transformación es el inverso
        # En inverse, las divisiones por cero dan 0.0. Al usar constraints formales,
        # pymoo sabrá que son infactibles mediante out['G'].
        f2 = np.divide(1.0, min_cobertura, out=np.zeros_like(min_cobertura, dtype=float), where=(min_cobertura > 0)) # Calcula el inverso de la cobertura mínima protegiendo contra división por cero
        f3 = np.divide(1.0, hamming_med, out=np.zeros_like(hamming_med, dtype=float), where=(hamming_med > 0)) # Calcula el inverso de la distancia de Hamming media protegiendo contra división por cero
    else: # Si el modo no es inverso, se asume transformación por negación
        f2 = -min_cobertura # Niega la cobertura mínima para transformarla en un objetivo de minimización
        f3 = -hamming_med # Niega la distancia de Hamming media para transformarla en un objetivo de minimización

    F = np.column_stack([ # Apila las matrices de objetivos en columnas para formar la matriz de evaluación final
        k, # El primer objetivo es minimizar el número de Tag SNPs seleccionados
        f2, # El segundo objetivo transformado (cobertura mínima)
        f3, # El tercer objetivo transformado (distancia de Hamming media)
        varianza, # El cuarto objetivo es minimizar la varianza de la cobertura
    ]).astype(float) # Convierte la matriz apilada al tipo de dato flotante para PyMoo

    return F # Retorna la matriz de objetivos transformados F


def evaluar_poblacion_vectorizado(
    X_bool: np.ndarray,
    matriz_discrepancia: np.ndarray,
    modo_transformacion: str = 'neg',
    devolver_min_cobertura: bool = False,
    modo_evaluacion: str = 'absoluta',
    cap_tolerancia: float = 3.0,
) -> np.ndarray:
    """
    Evaluación vectorizada de la población sobre los cuatro objetivos del problema.
    """
    # k: número de SNPs seleccionados por cada individuo
    k = X_bool.sum(axis=1).astype(float) # Calcula el número total de SNPs seleccionados (k) sumando los valores booleanos por individuo
    
    # Salvaguarda de emergencia: penalizar masivamente individuos vacíos.
    # El operador de reparación ('Repair') de PyMoo evita que esto ocurra en la práctica.
    sin_seleccion = (k == 0) # Identifica con un array booleano qué individuos no tienen ningún SNP seleccionado
    if sin_seleccion.any(): # Comprueba si existe al menos un individuo sin SNPs seleccionados
        k[sin_seleccion] = 1e9 # Aplica una penalización enorme en k para descartar soluciones vacías

    # Distancias de Hamming por par mediante producto matricial
    # D shape: (tam_poblacion, n_pares)
    D = (matriz_discrepancia.astype(np.int32) @ X_bool.T.astype(np.int32)).T.astype(float) # Calcula la distancia de Hamming por par usando multiplicación de matrices optimizada

    # f2: Cobertura mínima entre pares (base robusta para transformación)
    min_cobertura = D.min(axis=1) # Extrae la cobertura mínima (distancia de Hamming más baja) de cada individuo
    
    # Aplicar el tope de tolerancia biológica
    cobertura_efectiva = np.minimum(min_cobertura, cap_tolerancia) # Limita la cobertura evaluada a un máximo biológicamente útil (cap_tolerancia)
    
    if modo_evaluacion == 'proportional': # Comprueba si se está utilizando el modo de evaluación proporcional
        # NORMALIZACIÓN PROPORCIONAL (Ting et al.)
        # La tolerancia se mantiene en valor absoluto para coincidir con la implementación original de Ting,
        # donde la cobertura mínima no se divide por k.
        tolerancia_eval = cobertura_efectiva # Asigna la cobertura efectiva como tolerancia a evaluar
        hamming_med = D.mean(axis=1) / k # Calcula la distancia de Hamming media proporcional dividiendo por la cardinalidad k
        varianza = D.var(axis=1) / (k ** 2) # Calcula la varianza de la distancia de Hamming proporcional dividiendo por k al cuadrado
    else: # Si el modo de evaluación es absoluto
        # MÉTRICA ABSOLUTA
        tolerancia_eval = cobertura_efectiva # Asigna la cobertura efectiva como tolerancia a evaluar
        # f3: Distancia media
        hamming_med = D.mean(axis=1) # Calcula la distancia media de Hamming absoluta sin normalizar por k
        # f4: Varianza (Balance)
        varianza = D.var(axis=1) # Calcula la varianza absoluta de las distancias de Hamming

    # Retorno en formato de minimización (PyMoo standard)
    F = transformar_objetivos_a_minimizacion( # Llama a la función de transformación para preparar la matriz F
        k, # Pasa el array de cardinalidades
        tolerancia_eval, # Pasa el array de tolerancias calculadas
        hamming_med, # Pasa el array de distancias medias de Hamming
        varianza, # Pasa el array de varianzas
        modo_transformacion=modo_transformacion, # Especifica el modo de transformación a usar
    ) # Cierra los parámetros de la llamada a la función de transformación
    if devolver_min_cobertura: # Comprueba si la función debe retornar también la cobertura mínima en crudo
        return F, min_cobertura # Retorna la tupla con la matriz objetivo F y las coberturas mínimas
    return F # Retorna únicamente la matriz de evaluación de objetivos F

class ProblemaTagSNP(Problem):
    """
    Formulación multiobjetivo del problema de selección de Tag SNPs.
    """
    def __init__(
        self,
        H: np.ndarray,
        pair_idx: np.ndarray,
        normalizar_busqueda: bool = False,
        modo_transformacion_objetivos: str = 'neg',
        modo_evaluacion: str = 'absoluta',
        cap_tolerancia: float = 3.0,
    ):
        """
        Inicializa el problema de selección de Tag SNPs.

        Args:
            H (np.ndarray): Matriz de haplotipos.
            pair_idx (np.ndarray): Índices de pares de haplotipos.
            normalizar_busqueda (bool): Normalizar la búsqueda.
            modo_transformacion_objetivos (str): Modo de transformación de objetivos.
            modo_evaluacion (str): Modo de evaluación.
            cap_tolerancia (float): Tope de tolerancia.
        """
        self.H = H # Almacena la matriz de haplotipos en la instancia del problema
        self.pair_idx = pair_idx # Almacena los índices de los pares de haplotipos a comparar
        self.normalizar_busqueda = bool(normalizar_busqueda) # Asegura que la bandera de normalización es un booleano
        self.modo_transformacion_objetivos = str(modo_transformacion_objetivos or 'neg').strip().lower() # Normaliza el texto de transformación
        self.modo_evaluacion = str(modo_evaluacion or 'absoluta').strip().lower() # Normaliza el texto del modo de evaluación
        self.cap_tolerancia = float(cap_tolerancia) # Asegura que el tope de tolerancia sea un número de punto flotante
        
        # Precomputación de la matriz de discrepancia (diferencias bit a bit)
        self.matriz_discrepancia = (H[pair_idx[:, 0], :] != H[pair_idx[:, 1], :]).astype(np.int16) # Crea la matriz de diferencias lógicas transformando el resultado a entero corto
        
        n_var = H.shape[1] # Extrae el número de variables de decisión basándose en el número de SNPs
        D_completa = self.matriz_discrepancia.sum(axis=1).astype(float) # Calcula el máximo teórico de discrepancias si se usaran todos los SNPs
        
        self._escala_f1 = max(1.0, float(n_var)) # Define la escala máxima para el objetivo 1 basándose en el número total de SNPs
        
        if self.modo_evaluacion == 'proportional': # Comprueba si la métrica de evaluación está en modo proporcional
            self._escala_f4 = 0.25 # Fija el límite superior de la varianza proporcional (la varianza de una variable en [0,1] no supera 0.25)
            if self.modo_transformacion_objetivos == 'inverse': # Comprueba el tipo de transformación en modo proporcional
                self._escala_f2 = max(1.0, float(self.cap_tolerancia)) # Escala el objetivo 2 por el valor de la tolerancia requerida
                self._escala_f3 = 10.0 # Define empíricamente el valor nadir del objetivo 3 en el espacio inverso
            else: # 'neg'
                self._escala_f2 = max(1.0, float(self.cap_tolerancia)) # Aplica la escala del objetivo 2 con la misma tolerancia biológica
                self._escala_f3 = 1.0 # Establece la escala a 1.0 porque el valor máximo de la proporción invertida es |-1.0| = 1.0
        else: # Si el modo de evaluación es absoluto
            if self.modo_transformacion_objetivos == 'inverse': # Comprueba si se invierten los objetivos en evaluación absoluta
                # En inverse, f2 y f3 ya viven en escala ~[0, 1]; usar 1 evita
                # sobrecomprimir objetivos en MOEA/D durante la búsqueda.
                self._escala_f2 = 1.0 # Deja la escala en 1.0 para mantener los valores en un rango saludable
                self._escala_f3 = 1.0 # Deja la escala en 1.0 por la misma razón
            else: # Si los objetivos se transforman negándolos
                self._escala_f2 = max(1.0, float(self.cap_tolerancia)) # Normaliza el objetivo 2 con base en la máxima tolerancia admisible
                self._escala_f3 = max(1.0, float(D_completa.mean())) # Ajusta el objetivo 3 utilizando la distancia media hipotética de un set completo
            self._escala_f4 = max(1.0, float(D_completa.var())) # Escala el objetivo de la varianza con la máxima varianza posible observada
        
        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=1, xl=0, xu=1, vtype=bool) # Llama al constructor de PyMoo definiendo el tipo de variables, el número de objetivos y las restricciones

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evalúa la población.

        Args:
            X (np.ndarray): Población de soluciones.
            out (dict): Diccionario para almacenar los resultados.
            *args: Argumentos adicionales.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población evaluada.
        """
        X_bool = X.astype(bool) # Convierte el array continuo entregado por el optimizador al formato lógico necesario
        F_crudo, min_cobertura = evaluar_poblacion_vectorizado( # Extrae los objetivos brutos y las coberturas individuales mínimas mediante la función auxiliar
            X_bool, # Envía el vector genotípico binario procesado
            self.matriz_discrepancia, # Envía la matriz precalculada con las diferencias estáticas
            modo_transformacion=self.modo_transformacion_objetivos, # Pasa el argumento de transformación del modelo
            devolver_min_cobertura=True, # Obliga a retornar el cálculo interno de las coberturas mínimas
            modo_evaluacion=self.modo_evaluacion, # Determina si se aplica normalización a nivel de función
            cap_tolerancia=self.cap_tolerancia, # Dicta el tope en la valoración de coberturas extra
        ) # Finaliza el envío de argumentos para la obtención del rendimiento de la población
        
        # Exportar restricción: g(x) <= 0.
        # Es decir, 1.0 - min_cobertura <= 0
        out['G'] = (1.0 - min_cobertura).reshape(-1, 1) # Evalúa y almacena la métrica de penalización, obligando a PyMoo a respetar que min_cobertura sea mayor a 1

        if self.normalizar_busqueda: # Revisa el flag para saber si debe escalar el espacio de los objetivos de cara al motor
            F_escalado = F_crudo.copy() # Crea un duplicado del arreglo que protege los valores iniciales computados
            F_escalado[:, 0] /= self._escala_f1 # Normaliza dividiendo la cardinalidad (primer objetivo) con el escalar preparado
            F_escalado[:, 1] /= self._escala_f2 # Realiza la compresión del valor de cobertura al marco normalizado
            F_escalado[:, 2] /= self._escala_f3 # Ajusta el eje correspondiente a las distancias de Hamming
            F_escalado[:, 3] /= self._escala_f4 # Reubica la escala de la dispersión/varianza con la cota teórica
            out['F'] = F_escalado # Transfiere la matriz de evaluación adaptada al registro maestro
        else: # Entra por esta vía si se pidió emplear los registros no modificados
            out['F'] = F_crudo # Vuelca directamente las magnitudes absolutas obtenidas

def calcular_distinguibilidad_snps(H, pair_idx):
    """
    Cuantifica la capacidad de cada SNP para discriminar entre pares de haplotipos.
    """
    a = H[pair_idx[:, 0], :] # Extrae los haplotipos de la primera parte del grupo de pares
    b = H[pair_idx[:, 1], :] # Separa la otra mitad constituyente del catálogo de pares
    discrepancia = (a != b).astype(np.int8) # Ejecuta el chequeo lógico para crear una matriz de uno y cero
    return discrepancia.sum(axis=0).astype(float) # Obtiene el recuento discriminativo total operando el sumatorio por eje y emite el vector resultante
