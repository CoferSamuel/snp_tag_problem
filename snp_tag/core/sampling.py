"""
Módulo de Estrategias de Muestreo (sampling.py)
----------------------------------------------
Implementa diversas tácticas de inicialización de la población, incluyendo
métodos aleatorios dispersos y construcciones heurísticas tipo Greedy.
"""

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import numpy as np # Importa NumPy para optimización y cálculo de matrices en la inicialización
from pymoo.core.sampling import Sampling # Importa la interfaz Sampling de PyMoo para crear clases de muestreo personalizadas

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.core.problem import calcular_distinguibilidad_snps # Importa la función para evaluar la capacidad de discriminación de SNPs
from snp_tag.engine.diagnostics_logic import detectar_bloques_ld # Importa la función de diagnóstico de Desequilibrio de Ligamiento (LD)


def _construir_tabla_cobertura(H, pair_idx):
    """
    Construye la matriz de cobertura por SNP (pares x SNPs).

    Ejemplo
    -------
    >>> H = np.array([[0, 1, 0],
    ...               [1, 1, 0],
    ...               [0, 0, 1]])
    >>> pair_idx = np.array([[0, 1],
    ...                      [0, 2]])
    >>> _construir_tabla_cobertura(H, pair_idx)
    array([[ True, False, False],
           [False,  True,  True]])
    """
    if pair_idx.size == 0 or H.size == 0: # Comprueba si los índices de pares o la matriz de haplotipos están vacíos
        return np.zeros((0, H.shape[1]), dtype=bool) # Retorna una matriz booleana vacía con el número correcto de columnas (SNPs)
    a = H[pair_idx[:, 0], :] # Extrae los haplotipos de los primeros elementos de cada par
    b = H[pair_idx[:, 1], :] # Extrae los haplotipos de los segundos elementos de cada par
    return (a != b) # Compara ambos grupos de haplotipos y devuelve la matriz booleana de discrepancia

def _agrupar_por_cobertura(cover_table):
    """
    Agrupa SNPs con identica cobertura sobre pares.
    """
    if cover_table.size == 0: # Verifica si la tabla de cobertura provista está vacía
        return [] # Retorna una lista vacía de grupos si no hay datos que procesar
    packed = np.packbits(cover_table.astype(np.uint8), axis=0) # Comprime las columnas booleanas en bytes para agilizar la comparación
    grupos_dict = {} # Inicializa un diccionario para almacenar los índices agrupados por su firma en bytes
    for s in range(cover_table.shape[1]): # Itera a lo largo de cada SNP (columna) en la tabla de cobertura
        key = packed[:, s].tobytes() # Convierte la columna comprimida del SNP en una cadena inmutable de bytes para usar como clave
        grupos_dict.setdefault(key, []).append(s) # Añade el índice del SNP al grupo correspondiente creando la clave si no existe
    return [np.array(v, dtype=int) for v in grupos_dict.values()] # Retorna una lista de arrays con los índices agrupados de SNPs idénticos

def _orden_greedy_ting(H, rng):
    """
    Ordena SNPs por balance de alelos, filtrando duplicados exactos vectorizadamente.
    """
    if H.size == 0: # Comprueba si la matriz de haplotipos de entrada está vacía
        return np.array([], dtype=int) # Retorna un array vacío de tipo entero si no hay datos
        
    # 1. Filtrar columnas idénticas (SNPs redundantes) de forma 100% vectorizada
    _, indices_unicos = np.unique(H, axis=1, return_index=True) # Extrae los índices de los SNPs únicos eliminando duplicados absolutos
    
    n_hap = H.shape[0] # Obtiene el número total de haplotipos presentes en el dataset
    
    # 2. Calcular la suma sólo para los representantes únicos
    suma = H[:, indices_unicos].sum(axis=0).astype(float) # Suma las activaciones por columna (SNP) utilizando solo los índices únicos seleccionados
    
    # 3. Calcular balance (cercanía a n_hap / 2)
    balance = (n_hap / 2.0) - np.abs(suma - (n_hap / 2.0)) # Evalúa qué tan cerca está la frecuencia del alelo menor de 0.5 (ideal para distinguir)
    
    # 4. Añadir ruido mínimo para desempates estocásticos
    ruido = rng.normal(0.0, 1e-6, size=balance.shape[0]) # Genera una pequeña perturbación aleatoria para cada SNP único
    
    # 5. Ordenar los índices únicos basándose en el balance perturbado
    orden_relativo = np.argsort(balance + ruido) # Ordena los SNPs de menor a mayor balance (los mejores quedarán al final)
    
    # Devolver los índices originales ordenados
    return indices_unicos[orden_relativo].astype(int) # Retorna los identificadores originales ordenados y garantizados de tipo entero

def _greedy_por_orden(orden, cover_table):
    """
    Selecciona SNPs segun el orden dado hasta cubrir todos los pares.
    """
    n_snps = cover_table.shape[1] # Averigua el número total de SNPs disponibles en la tabla de cobertura
    seleccionados = np.zeros(n_snps, dtype=bool) # Crea un array booleano de ceros indicando que inicialmente no hay SNPs seleccionados
    if cover_table.size == 0: # Si la matriz de cobertura carece de elementos válidos
        if n_snps > 0: # Siempre y cuando haya al menos un SNP elegible
            idx = int(orden[0]) if orden.size > 0 else 0 # Elige el primer SNP del orden o el índice 0 como solución de contingencia
            seleccionados[idx] = True # Marca el SNP de contingencia como activado
        return seleccionados # Retorna la solución temprana

    cubiertos = np.zeros(cover_table.shape[0], dtype=bool) # Lleva un registro booleano de los pares de haplotipos que ya están cubiertos
    for s in orden[::-1]: # Itera sobre el array de orden invertido para evaluar primero los mejores candidatos
        if cubiertos.all(): # Verifica prematuramente si todos los pares ya fueron resueltos
            break # Detiene la selección si ya se alcanzó cobertura universal
        contrib = cover_table[:, s] # Extrae el vector de pares que este SNP candidato es capaz de discriminar
        if np.any((~cubiertos) & contrib): # Comprueba si el candidato aporta cobertura a algún par aún no resuelto
            seleccionados[s] = True # Incorpora el SNP candidato a la solución final
            cubiertos |= contrib # Actualiza el registro de pares cubiertos sumando la contribución del nuevo SNP
    if not seleccionados.any() and n_snps > 0: # Control de fallos por si no se seleccionó absolutamente ningún SNP
        idx = int(orden[0]) if orden.size > 0 else 0 # Selecciona el de mayor prioridad por defecto para no devolver k=0
        seleccionados[idx] = True # Realiza la activación del elemento comodín
    return seleccionados # Entrega el vector genotípico construido de forma codiciosa

def _greedy_grupos(cover_table, grupos, rng):
    """
    Selecciona 1 SNP por grupo hasta cubrir todos los pares.
    """
    n_snps = cover_table.shape[1] # Dimensiona el espacio de la solución basándose en el conteo de columnas
    seleccionados = np.zeros(n_snps, dtype=bool) # Asienta el genotipo de partida completamente vacío (ceros lógicos)
    if cover_table.size == 0: # Examina si la matriz evaluativa está despoblada
        if n_snps > 0: # Confirma la existencia de genes posibles para evitar accesos nulos
            seleccionados[rng.integers(0, n_snps)] = True # Marca un SNP al azar porque no hay criterios de selección válidos
        return seleccionados # Finaliza tempranamente enviando la respuesta
    cubiertos = np.zeros(cover_table.shape[0], dtype=bool) # Genera la bitácora de cumplimiento de pares
    grupos_orden = grupos.copy() # Duplica la estructura de grupos para no mutar el parámetro de entrada
    rng.shuffle(grupos_orden) # Desordena la lista de grupos para inyectar diversidad en cada ejecución
    for grupo in grupos_orden: # Inicia el paso a través de cada grupo de SNPs equivalentes
        if cubiertos.all(): # Valida el objetivo de cubrimiento total en cada paso
            break # Corta el bucle si ya se cumplió la meta de resolución
        idx = int(rng.integers(0, len(grupo))) # Escoge un índice aleatorio dentro del rango del grupo actual
        s = int(grupo[idx]) # Recupera el identificador global del SNP seleccionado
        seleccionados[s] = True # Integra el SNP al conjunto resolutivo
        cubiertos |= cover_table[:, s] # Unifica las nuevas distinciones conseguidas por este genotipo
    if not seleccionados.any() and n_snps > 0: # Revisa si el bucle concluyó sin agregar genes a la fórmula
        seleccionados[rng.integers(0, n_snps)] = True # Forzar la inclusión de una variante estocástica
    return seleccionados # Retorna la configuración obtenida de los grupos

def _unique_grupos(grupos, n_snps, rng):
    """
    Selecciona exactamente 1 SNP por grupo.
    """
    seleccionados = np.zeros(n_snps, dtype=bool) # Plantea el vector de diseño en blanco
    for grupo in grupos: # Procesa correlativamente cada agrupación detectada
        idx = int(rng.integers(0, len(grupo))) # Rifar internamente qué SNP representará a su bloque
        seleccionados[int(grupo[idx])] = True # Suscribe el gen ganador
    if not seleccionados.any() and n_snps > 0: # Previene cromosomas nulos a toda costa
        seleccionados[rng.integers(0, n_snps)] = True # Asignación de resguardo
    return seleccionados # Otorga la matriz construida

def construir_solucion_greedy(H, pair_idx, indices_ordenados=None):
    """
    Construye una solución mediante una aproximación voraz.
    """
    n_snps = H.shape[1] # Determina la dimensión final del vector binario
    n_pares = pair_idx.shape[0] # Contabiliza las metas de discriminación
    
    if indices_ordenados is None: # Si el flujo no suministró un orden previo calculado
        puntuacion = calcular_distinguibilidad_snps(H, pair_idx) # Demanda el costo/beneficio actual para los marcadores
        indices_ordenados = np.argsort(-puntuacion) # Clasifica en formato descendente para atacar primero a los más prometedores
        
    seleccionados = np.zeros(n_snps, dtype=bool) # Dispone la cadena genómica inicializada en estado inactivo
    cubiertos = np.zeros(n_pares, dtype=bool) # Configura la estructura para rastrear el grado de éxito
    
    a = H[pair_idx[:, 0], :] # Acopia la cara A de los pares a comparar
    b = H[pair_idx[:, 1], :] # Acopia la cara B de los binomios
    discrepancia = (a != b) # Confecciona el mapa binario de divergencias útiles
    
    for s in indices_ordenados: # Repasa linealmente el escalafón de SNPs sugerido
        if np.all(cubiertos): # Testea si hay necesidad de continuar buscando
            break # Suspende el ciclo for si la tarea está concluida
        contribucion = discrepancia[:, s] # Aísla el vector de utilidad de la posición en curso
        if np.any((~cubiertos) & contribucion): # Criba el vector buscando si ofrece aportes a áreas inexploradas
            seleccionados[s] = True # Consagra este marcador en la compilación
            cubiertos = cubiertos | contribucion # Empalma el avance actual con el progreso consolidado
            
    if not seleccionados.any(): # Sistema anti-bloqueo para resultados sin aportaciones
        seleccionados[indices_ordenados[0]] = True # Inyecta el candidato estadísticamente más apto
    return seleccionados # Resuelve entregando la alineación formulada

def _agrupar_por_distinguibilidad(indices_desc, puntuaciones):
    """Agrupa SNPs con idéntica capacidad de discriminación."""
    grupos = [] # Arreglo principal contenedor de divisiones
    if len(indices_desc) == 0: # Chequea si el inventario de elementos base está vacío
        return grupos # Reacciona rápidamente con respuesta hueca
    actual = [int(indices_desc[0])] # Inicia el primer conjunto provisional depositando el elemento líder
    puntuacion_actual = float(puntuaciones[indices_desc[0]]) # Fija la marca métrica a cotejar en el transcurso
    for idx in indices_desc[1:]: # Baraja iterativamente los elementos restantes tras el primero
        s = float(puntuaciones[idx]) # Rescata la métrica para el factor actual en la lupa
        if s == puntuacion_actual: # Contrastación contra el líder para ver si hay empate
            actual.append(int(idx)) # Expande el conjunto si coexisten el mismo desempeño
        else: # Ante quiebre de tendencia en la tabla
            grupos.append(np.array(actual, dtype=int)) # Solidifica el grupo anterior en un bloque NumPy y lo adjunta
            actual = [int(idx)] # Rebota declarando un nuevo bloque liderado por el actual candidato
            puntuacion_actual = s # Ajusta la nueva pauta de métrica a seguir
    grupos.append(np.array(actual, dtype=int)) # Atrapa el último conjunto rezagado en el ciclo
    return grupos # Emite los empaquetamientos finales

def _ordenar_con_desempate_aleatorio(grupos_puntuacion, rng):
    """Mantiene el orden greedy pero introduce estocasticidad en los empates."""
    partes = [] # Recipiente provisorio de listados permutados
    for grupo in grupos_puntuacion: # Examina los colectivos de igual puntuación uno a uno
        partes.append(rng.permutation(grupo) if len(grupo) > 1 else grupo) # Mezcla si hay competencia interna; de lo contrario inserta intacto
    return np.concatenate(partes).astype(int) # Funde los mini-lotes en una secuencia definitiva de tipo entero

class MuestreoAleatorioDisperso(Sampling):
    """Genera soluciones binarias con baja densidad de activos."""
    def __init__(self, prob: float = 0.05, semilla=42):
        """
        Inicializa el muestreo aleatorio disperso.

        Args:
            prob (float): Probabilidad de que un SNP sea seleccionado.
            semilla (int): Semilla para el generador de números aleatorios.
        """
        super().__init__() # Interpela a la arquitectura padre para configuraciones estándar de muestreo
        self.prob = prob # Resguarda el índice de probabilidad base
        self.rng = np.random.default_rng(semilla) # Activa una semilla local robusta de generación estocástica
    def _do(self, problem, n_samples, **kwargs):
        """
        Genera una población de soluciones aleatorias dispersas.

        Args:
            problem: Problema de optimización.
            n_samples (int): Número de soluciones a generar.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población de soluciones binarias.
        """
        X = self.rng.random((n_samples, problem.n_var)) < self.prob # Imprime un retículo booleano comparando el flotante aleatorio frente al umbral
        vacíos = ~X.any(axis=1) # Destaca a los individuos que se han quedado estériles
        if vacíos.any(): # Escudriña si hay pacientes a reparar
            for i in np.where(vacíos)[0]: # Pasa lista a los índices defectuosos
                X[i, self.rng.integers(0, problem.n_var)] = True # Enciende arbitrariamente un genoma en los individuos apagados
        return X # Restituye la demografía configurada

def construir_solucion_multicobertura(H, pair_idx, target_k, rng):
    """
    Construye una solución voraz dinámica exigiendo que cada par de haplotipos
    se distinga 'target_k' veces (si es biológicamente posible).
    """
    n_snps = H.shape[1] # Censa los marcadores
    n_pares = pair_idx.shape[0] # Modula las responsabilidades
    
    seleccionados = np.zeros(n_snps, dtype=bool) # Moldea la tira genética virgen
    cubiertos = np.zeros(n_pares, dtype=int) # Habilita un contador integrador y no un flag booleano
    
    a = H[pair_idx[:, 0], :] # Fractura la matriz a buscar el flanco A
    b = H[pair_idx[:, 1], :] # Fractura la matriz hacia la contraparte B
    discrepancia = (a != b).astype(int) # Formaliza la topología de divergencias como array numérico
    
    # Límite biológico para prevenir bucles infinitos
    cobertura_maxima_biologica = discrepancia.sum(axis=1) # Sondea los márgenes fisiológicos permisibles por el conjunto
    # Salvaguarda: el objetivo para cada par no puede superar lo que el dataset permite
    objetivo_real = np.minimum(target_k, cobertura_maxima_biologica) # Poda las ambiciones frente a la realidad estructural
    
    while True: # Engrana la iteración persistente
        # Encontrar pares que aún no han alcanzado su cobertura objetivo real
        necesitan_cobertura = cubiertos < objetivo_real # Produce máscara con los pares insuficientes
        if not np.any(necesitan_cobertura): # Si nadie demanda más cobertura
            break # Escapa la atadura del bucle
            
        # Puntuación: número de pares insatisfechos que cada SNP puede distinguir
        # Sólo sumamos las filas de discrepancia donde necesitan_cobertura es True
        puntuacion_dinamica = discrepancia[necesitan_cobertura].sum(axis=0) # Evalúa la fuerza resolutiva frente a la bolsa de no cubiertos
        
        # Excluir SNPs que ya han sido seleccionados
        puntuacion_dinamica[seleccionados] = -1 # Neutraliza a los ya partícipes
        
        max_score = np.max(puntuacion_dinamica) # Pesquisa la contribución cenital
        if max_score <= 0: # Si la pujanza cesó de ser fructífera
            break # Declara el agotamiento de opciones válidas
            
        candidatos = np.where(puntuacion_dinamica == max_score)[0] # Congrega a los líderes de utilidad
        
        # Desempate estocástico
        mejor_snp = rng.choice(candidatos) # Decide la contienda interna aleatoriamente
        
        seleccionados[mejor_snp] = True # Integra al vencedor
        cubiertos += discrepancia[:, mejor_snp] # Concede los frutos al inventario
        
    # Garantizar que al menos un SNP es seleccionado en caso de objetivos degenerados
    if not seleccionados.any() and n_snps > 0: # Escudo terminal de vacío
        seleccionados[rng.integers(0, n_snps)] = True # Coloca un cimiento salvavidas
        
    return seleccionados # Finaliza

class MuestreoGreedyMultiCobertura(Sampling):
    """
    Inicialización basada en una heurística voraz de cobertura múltiple progresiva.
    Fuerza al algoritmo a seleccionar SNPs redundantes distribuyendo un objetivo
    de cobertura desde 1 hasta max_cobertura_objetivo entre los individuos.
    """
    def __init__(self, H, pair_idx, max_cobertura_objetivo=5, semilla=42):
        """
        Inicializa el muestreo voraz de cobertura múltiple.

        Args:
            H (np.ndarray): Matriz de haplotipos.
            pair_idx (np.ndarray): Índices de pares de haplotipos.
            max_cobertura_objetivo (int): Cobertura máxima objetivo.
            semilla (int): Semilla para el generador de números aleatorios.
        """
        super().__init__() # Invoca el setup clásico de las fábricas de muestreo
        self.H = H # Anida los genotipos puros
        self.pair_idx = pair_idx # Asegura las alianzas de análisis
        self.max_cobertura_objetivo = int(max_cobertura_objetivo) # Certifica la restricción superior como entera
        self.rng = np.random.default_rng(semilla) # Levanta el mecanismo oráculo
        
    def _do(self, problem, n_samples, **kwargs):
        """
        Genera una población de soluciones voraces de cobertura múltiple.

        Args:
            problem: Problema de optimización.
            n_samples (int): Número de soluciones a generar.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población de soluciones binarias.
        """
        X = np.zeros((n_samples, problem.n_var), dtype=bool) # Proyecta la tabla vacía de habitantes
        
        # Distribución lineal de los objetivos de cobertura en la población.
        # Va desde 1 hasta max_cobertura_objetivo (ej. 1, 1, 2, 2, 3, 3...)
        k_targets = np.linspace(1, self.max_cobertura_objetivo, n_samples).astype(int) # Propaga la pendiente de exigencia entre el grueso
        
        for i in range(n_samples): # Cursa sobre cada espécimen requerido
            target_k = k_targets[i] # Aisla su umbral de tolerancia prescrito
            X[i] = construir_solucion_multicobertura(self.H, self.pair_idx, target_k, self.rng) # Deposita la conformación elaborada ad hoc
            
        return X # Retorna la metrópolis

class MuestreoGreedyTing(Sampling):
    """
    Inicializacion mixta inspirada en Ting (GreedyInit + Greedy_init + Unique_init).
    """
    def __init__(self, H, pair_idx, ratio_greedy=0.5, semilla=42):
        """
        Inicializa el muestreo voraz de cobertura múltiple.

        Args:
            H (np.ndarray): Matriz de haplotipos.
            pair_idx (np.ndarray): Índices de pares de haplotipos.
            max_cobertura_objetivo (int): Cobertura máxima objetivo.
            semilla (int): Semilla para el generador de números aleatorios.
        """
        super().__init__() # Arraiga las constantes nucleares
        self.H = H # Almacena el reservorio biológico
        self.pair_idx = pair_idx # Resguarda las referencias duales
        self.ratio_greedy = float(ratio_greedy) # Impone el peso en crudo
        self.rng = np.random.default_rng(semilla) # Habilita el estocástico

        self.cover_table = _construir_tabla_cobertura(H, pair_idx) # Prefabrica la topografía discriminante
        self.grupos_cobertura = _agrupar_por_cobertura(self.cover_table) # Apila los elementos funcionalmente indistinguibles
        self.orden_ting = _orden_greedy_ting(H, self.rng) # Despliega la cadena de prioridades métricas

        self.semilla_greedy = _greedy_por_orden(self.orden_ting, self.cover_table) # Concibe la unidad maestra de eficiencia
        self.semilla_densa = np.zeros(H.shape[1], dtype=bool) # Abre paso a la plantilla masiva
        self.semilla_densa[self.orden_ting] = True # Rellena la matriz en las coordenadas de prioridad

    def _do(self, problem, n_samples, **kwargs):
        """
        Genera una población de soluciones voraces de cobertura múltiple.

        Args:
            problem: Problema de optimización.
            n_samples (int): Número de soluciones a generar.
            **kwargs: Argumentos adicionales.

        Returns:
            np.ndarray: Población de soluciones binarias.
        """
        n_samples = int(n_samples) # Obliga a tratar a las unidades en bloque
        X = np.zeros((n_samples, problem.n_var), dtype=bool) # Elabora el cuaderno en blanco poblacional

        n_greedy_total = int(np.ceil(n_samples * self.ratio_greedy)) # Modula la porción elitista de base
        n_anchor = int(np.ceil(self.ratio_greedy * 10.0)) # Determina el puñado de vanguardia

        pos = 0 # Inicializa el marcador posicional
        for _ in range(0, n_anchor, 2): # Trota en saltos de a dos para distribuir moldes
            if pos >= n_samples: # Si rebasó los confines
                break # Interrumpe abruptamente
            X[pos] = self.semilla_greedy # Siembra la pepita primaria
            pos += 1 # Recorre a la derecha
            if pos >= n_samples: # Doble comprobante de bordes
                break # Frenazo
            X[pos] = self.semilla_densa # Inserta la contraparte hipernutrida
            pos += 1 # Avanza casillero

        while pos < min(n_greedy_total, n_samples): # Criba la zona de mezcla
            X[pos] = _greedy_grupos(self.cover_table, self.grupos_cobertura, self.rng) # Fija aportes comunales
            pos += 1 # Añade al censo
            if pos >= min(n_greedy_total, n_samples): # Chequea contención
                break # Trunca el ciclo
            X[pos] = _unique_grupos(self.grupos_cobertura, problem.n_var, self.rng) # Agrega las anomalías segregadas
            pos += 1 # Mueve marcador

        if pos < n_samples: # En caso de déficit final
            X[pos:] = self.rng.random((n_samples - pos, problem.n_var)) < 0.5 # Completa con puro desorden estocástico ecuánime

        return X # Finaliza

# Helpers para MuestreoGreedyHolistico

def _ancla_max_hamming_medio(discrepancia, distinguibilidad, n_snps, n_pares, rng):
    """
    Construye una solución que maximiza la distancia media de Hamming.
    Selecciona SNPs en orden descendente de contribución media, continuando
    más allá de la cobertura mínima hasta que la mejora marginal sea despreciable.
    """
    seleccionados = np.zeros(n_snps, dtype=bool) # Crea pizarra genómica en blanco
    D_acum = np.zeros(n_pares, dtype=float) # Abre contador para acumular las distancias Hamming

    ruido = rng.normal(0.0, 1e-6, size=n_snps) # Forja rugosidades matemáticas mínimas
    orden = np.argsort(-(distinguibilidad + ruido)) # Promueve el escrutinio partiendo de los óptimos

    cobertura_alcanzada = False # Declara el estatus de completitud en espera
    for s in orden: # Escruta sobre el lineamiento trazado
        contrib = float(discrepancia[:, s].sum()) # Condensa el peso aportado por el participante
        if contrib <= 0: # Si la rentabilidad es hueca
            continue # Desprecia al sujeto y prosigue

        seleccionados[s] = True # Adhiere el marcador productivo
        D_acum += discrepancia[:, s].astype(float) # Eleva las métricas con la contribución incorporada

        if not cobertura_alcanzada: # Observa si hay pendientes globales
            if D_acum.min() >= 1: # Indaga si el techo base fue perforado
                cobertura_alcanzada = True # Valida la cobertura elemental
            continue # Permite escalar al próximo sujeto velozmente

        # Tras cobertura: parar cuando la mejora marginal < 0.5%
        media_actual = D_acum.mean() # Extrae el núcleo ponderado del acopio
        if media_actual > 0 and (contrib / n_pares) / media_actual < 0.005: # Sopesa si los dividendos siguen mereciendo el gasto (0.5% de mejora)
            break # Estabiliza la recolección

    if not seleccionados.any(): # Cortafuegos de inanición
        seleccionados[int(orden[0])] = True # Engancha la mejor pieza solitaria
    return seleccionados # Expide la solución max-Hamming


def _ancla_min_varianza(discrepancia, H, pair_idx, n_snps, n_pares, rng):
    """
    Construye una solución con mínima varianza en distancias de Hamming.
    Parte de una cobertura mínima y añade SNPs que equilibran las distancias
    entre pares.
    """
    distinguibilidad = discrepancia.sum(axis=0).astype(float) # Obtiene suma discriminatoria global
    indices_base = np.argsort(-distinguibilidad) # Formula ranking general sin aleatoriedades
    seleccionados = construir_solucion_greedy(H, pair_idx, indices_base).copy() # Arranca con la plataforma clásica
    D_acum = discrepancia[:, seleccionados].sum(axis=1).astype(float) # Registra la línea basal de distancias logradas

    k_inicial = int(seleccionados.sum()) # Mide cuántos genes han sido puestos en marcha
    max_extras = min(k_inicial * 2, n_snps - k_inicial) # Impone cota de iteración compensatoria

    for _ in range(max_extras): # Trilla buscando tapar huecos estadísticos
        candidatos = np.where(~seleccionados)[0] # Caza marcadores durmientes
        if len(candidatos) == 0: # Detecta clausura material
            break # Cancela las diligencias

        var_actual = float(D_acum.var()) # Pulsa la agitación estadística presente
        if var_actual < 1e-9: # Observa estabilizaciones absolutas
            break # Da por zanjada la optimización

        # Varianza resultante tras añadir cada candidato
        D_cand = discrepancia[:, candidatos].astype(float) # Extrae el lote matricial potencial
        D_nuevas = D_acum[:, None] + D_cand # Simula adiciones colosales de un paso
        varianzas = D_nuevas.var(axis=0) # Saca las varianzas emergentes

        mejor_idx = int(np.argmin(varianzas)) # Identifica al héroe estabilizador
        if varianzas[mejor_idx] >= var_actual * 0.99: # Descarta incorporaciones si la mejora ronda lo nulo (<1%)
            break # Remata prematuramente

        seleccionados[candidatos[mejor_idx]] = True # Contrata al nivelador estelar
        D_acum = D_nuevas[:, mejor_idx].copy() # Anexa la simulación ganadora a la realidad material

    return seleccionados # Provee la fisonomía optimizada


def _construir_solucion_bloques(discrepancia, bloques, n_snps, n_pares,
                                 indices_bloques, rng):
    """
    Construye una solución seleccionando representantes de un subconjunto de
    bloques LD mediante mini-greedy restringido a los SNPs de cada bloque.
    Incorpora un mecanismo de reparación para garantizar cobertura total.
    """
    seleccionados = np.zeros(n_snps, dtype=bool) # Alista cimiento en negativo absoluto
    cubiertos = np.zeros(n_pares, dtype=bool) # Prologa cimiento paralelo de metas de cubrimiento

    for b_idx in indices_bloques: # Escudriña cada bloque convocado
        snps_bloque = bloques[b_idx] # Aísla residentes del distrito LD actual
        if len(snps_bloque) == 0: # Obvia vecindarios fantasma
            continue # Sigue el rumbo

        # Puntuación: pares aún no cubiertos que cada SNP del bloque distingue
        pendientes = ~cubiertos # Extrae retrato de la insatisfacción latente
        scores = discrepancia[pendientes][:, snps_bloque].sum(axis=0) if pendientes.any() \
            else discrepancia[:, snps_bloque].sum(axis=0) # Cuantifica el poder de la bolsa activa frente a las lagunas

        orden_local = np.argsort(-scores.astype(float)) # Escalafona la asamblea distrital
        for idx_local in orden_local: # Inspecciona los líderes vecinales
            s_global = snps_bloque[idx_local] # Transfiere referencia del microcosmos al macro
            contrib = discrepancia[:, s_global].astype(bool) # Pondera utilidades
            if np.any((~cubiertos) & contrib): # Evalúa pertinencia
                seleccionados[s_global] = True # Ficha al marcador en cuestión
                cubiertos |= contrib # Anota progreso
            if cubiertos.all(): # Coteja saciedad general
                break # Rompe inspección vecinal
        if cubiertos.all(): # Coteja saciedad matriz
            break # Rompe bucle de bloques

    # Reparación greedy si los bloques seleccionados fueron insuficientes
    if not cubiertos.all(): # Vigila resquicios insalvables
        pendientes = ~cubiertos # Delimita abismos
        scores_globales = discrepancia[pendientes].sum(axis=0).astype(float) # Ejerce fuerza bruta sumatoria global
        scores_globales[seleccionados] = -1 # Aísla a los ya trabajadores
        
        ruido = rng.normal(0.0, 1e-6, size=n_snps) # Emerge ruido difuminador
        orden_reparacion = np.argsort(-(scores_globales + ruido)) # Ordena fuerza laboral libre
        
        for s_global in orden_reparacion: # Convoca obreros transitorios
            if scores_globales[s_global] <= 0: # Exonera a los inútiles puros
                continue # Descarta
            contrib = discrepancia[:, s_global].astype(bool) # Mide pericia específica
            if np.any((~cubiertos) & contrib): # Si el aporte sirve para parchear
                seleccionados[s_global] = True # Activa rescate
                cubiertos |= contrib # Aúna esfuerzo
            if cubiertos.all(): # Evalúa final de parcheo
                break # Desvanece operativo

    if not seleccionados.any() and n_snps > 0: # Guardaespaldas de fallos catastróficos
        seleccionados[rng.integers(0, n_snps)] = True # Encienda baliza aleatoria
    return seleccionados # Finaliza


def _construir_complemento(discrepancia, solucion_base, n_snps, n_pares, rng):
    """
    Construye el complemento de una solución existente: apunta a los pares
    peor cubiertos por la base y, tras satisfacerlos, continúa la selección
    para garantizar una solución 100% factible.
    """
    D_base = discrepancia[:, solucion_base].sum(axis=1).astype(float) # Obtiene topografía de aciertos del ancestro

    if D_base.sum() == 0: # Si la matriz original era hueca
        sol = np.zeros(n_snps, dtype=bool) # Resetea tabla
        sol[rng.integers(0, n_snps)] = True # Emite mutación al azar
        return sol # Corta la subrutina tempranamente

    mediana = float(np.median(D_base)) # Extrae meridiano del esfuerzo
    mal_cubiertos = D_base <= mediana # Etiqueta al espectro de baja atención

    # Puntuar por contribución a los pares mal cubiertos, excluyendo SNPs de la base
    scores = discrepancia[mal_cubiertos].sum(axis=0).astype(float) # Evalúa pericia frente al déficit
    scores[solucion_base] = -1 # Apaga a la plantilla nativa de la carrera

    ruido = rng.normal(0.0, 1e-6, size=n_snps) # Modula empate con aleatoriedad fina
    orden = np.argsort(-(scores + ruido)) # Desciende por rangos de salvación

    seleccionados = np.zeros(n_snps, dtype=bool) # Acuña un prospecto prístino
    cubiertos_obj = np.zeros(n_pares, dtype=bool) # Inicializa medidor de paliativos

    # Primera fase: cubrir los pares débiles (mal_cubiertos)
    for s in orden: # Filtra a los bomberos idóneos
        if scores[s] <= 0: # Cuando el talento marginal cesa
            break # Cierra alistamiento de urgencia
        contrib = discrepancia[:, s].astype(bool) # Obtiene el área de actuación
        if np.any((~cubiertos_obj) & contrib): # Analiza sinergias
            seleccionados[s] = True # Dota de herramientas
            cubiertos_obj |= contrib # Consolida extinción de faltas
        if cubiertos_obj[mal_cubiertos].all(): # Observa erradicación de urgencias
            break # Mitiga asedio

    # Segunda fase: completar la cobertura total para evitar soluciones infactibles
    if not cubiertos_obj.all(): # Verifica rezagos de segunda ola
        pendientes = ~cubiertos_obj # Demarca carencias
        scores_resto = discrepancia[pendientes].sum(axis=0).astype(float) # Acota talentos para rezagados
        scores_resto[seleccionados] = -1 # Separa voluntarios vigentes
        
        orden_resto = np.argsort(-(scores_resto + rng.normal(0.0, 1e-6, size=n_snps))) # Organiza batida final
        for s in orden_resto: # Trilla brigada de remate
            if scores_resto[s] <= 0: # Descarta bultos sin talento
                continue # Evade interrupción
            contrib = discrepancia[:, s].astype(bool) # Comprueba aptitudes postreras
            if np.any((~cubiertos_obj) & contrib): # Si el toque ayuda
                seleccionados[s] = True # Ficha
                cubiertos_obj |= contrib # Cierra herida
            if cubiertos_obj.all(): # Supervisa alta médica total
                break # Cese de emergencia

    if not seleccionados.any(): # Monitoreo de desastres totales
        candidatos = np.where(~solucion_base)[0] # Acopia reserva inactiva
        if len(candidatos) > 0: # Si hay reservas
            seleccionados[rng.choice(candidatos)] = True # Recluta a discreción
        else: # Si se secó la bolsa
            seleccionados[rng.integers(0, n_snps)] = True # Toma lo que hay ciegamente
    return seleccionados # Despacha


def _muestreo_guiado_disperso(distinguibilidad, discrepancia, n_snps, n_pares, k_objetivo, rng):
    """
    Muestreo ponderado por distinguibilidad con cardinalidad objetivo,
    seguido de una reparación greedy para asegurar cobertura total.
    """
    total = distinguibilidad.sum() # Condensa pesos teóricos
    if total <= 0: # Si hay planicie total
        p = np.full(n_snps, 1.0 / n_snps) # Distribuye chances isométricas
    else: # Con topografía normal
        p = distinguibilidad / total # Esculpe probabilidades normadas

    p = p * k_objetivo # Propaga la intención cardinal
    p = np.clip(p, 0.01, 0.8) # Encapsula la aleatoriedad en márgenes de cordura

    sol = rng.random(n_snps) < p # Decanta estocásticamente el vector de chances
    
    # Evaluar cobertura
    cubiertos = discrepancia[:, sol].any(axis=1) if sol.any() else np.zeros(n_pares, dtype=bool) # Audita impacto real del lance
    
    # Reparación greedy
    if not cubiertos.all(): # Advierte lagunas de servicio
        pendientes = ~cubiertos # Plasma mapa de necesidades
        scores = discrepancia[pendientes].sum(axis=0).astype(float) # Mide pericias sobre necesidades
        scores[sol] = -1 # Aísla a los ya ganadores
        
        ruido = rng.normal(0.0, 1e-6, size=n_snps) # Ensucia con azar micrométrico
        orden = np.argsort(-(scores + ruido)) # Ordena batallón de salvamento
        
        for s in orden: # Despliega rescate
            if scores[s] <= 0: # Obvia rezagados puros
                continue # Pasa
            contrib = discrepancia[:, s].astype(bool) # Confirma efectividad puntual
            if np.any((~cubiertos) & contrib): # Revisa si remedia deficiencias
                sol[s] = True # Adhiere al salvador
                cubiertos |= contrib # Subsanado
            if cubiertos.all(): # Vigila estado de calma general
                break # Cese de alertas

    if not sol.any(): # Paracaídas de fallos de cero total
        sol[rng.integers(0, n_snps)] = True # Abre cúpula ciega
    return sol # Rinde el ejemplar


class MuestreoGreedyHolistico(Sampling):
    """
    Inicialización de cinco niveles para optimización multiobjetivo de Tag SNPs.

    Siembra la población cubriendo sistemáticamente las cuatro dimensiones del
    frente de Pareto con soluciones estructuralmente diversas:

    Tier 1: Anclas de Pareto           (~5-10%)  - Extremos de cada objetivo
    Tier 2: Barrido k-Cover            (~25%)    - Multicobertura progresiva
    Tier 3: Ensamblaje por bloques LD  (~25%)    - Diversidad estructural genómica
    Tier 4: Inyección de complementos  (~20%)    - Soluciones que parchean debilidades
    Tier 5: Exploración guiada dispersa(~20-25%) - Muestreo ponderado por importancia

    Post-procesado: deduplicación fenotípica (siempre activo).
    """

    def __init__(self, H, pair_idx, max_k=5, semilla=42):
        super().__init__() # Echa a rodar la maquinaria base del Sampling
        self.H = H # Deposita matriz maestra
        self.pair_idx = pair_idx # Resguarda atlas de cruces
        self.max_k = int(max_k) # Fija límite operativo estricto
        self.rng = np.random.default_rng(semilla) # Activa motor oracular interno

        # Precomputación
        self.discrepancia = (H[pair_idx[:, 0]] != H[pair_idx[:, 1]]).astype(np.int16) # Genera huellas dactilares discrepantes
        self.distinguibilidad = self.discrepancia.sum(axis=0).astype(float) # Saca pesos gravitacionales de base
        self.n_snps = H.shape[1] # Dimensiona longitud espacial
        self.n_pares = pair_idx.shape[0] # Dimensiona carga operativa

        # Detección de bloques LD para Tier 3 (basada en datos,
        # funciona con cualquier dataset).
        segmentos = detectar_bloques_ld(H) # Infiere vecindarios LD desde la biología
        # Fallback: si la estructura LD es demasiado uniforme y produce
        # muy pocos bloques, dividir posicionalmente para garantizar diversidad.
        if len(segmentos) < 5: # Si el terreno es demasiado llano
            n_bloques_min = min(10, max(5, self.n_snps // 50)) # Computa bloques artificiales sanos
            segmentos_pos = [(i * self.n_snps // n_bloques_min, # Interpola divisor izquierdo
                              (i + 1) * self.n_snps // n_bloques_min) # Interpola divisor derecho
                             for i in range(n_bloques_min)] # Expande la interpolación en listado
            segmentos = segmentos_pos # Remplaza el atlas con el sintético
        self.bloques = [np.arange(s, e) for s, e in segmentos] # Convierte tuplas de aristas en matrices lineales concretas

    def _construir_anclas(self):
        """Tier 1: construye una solución extrema por cada objetivo."""
        anclas = [] # Declara recipiente de estandartes

        # Ancla 1: Min-k (mínima cardinalidad con cobertura)
        indices = np.argsort(-self.distinguibilidad) # Formula ruta clásica de abordaje
        anclas.append(construir_solucion_greedy(self.H, self.pair_idx, indices)) # Consolida base mínima histórica

        # Ancla 2: Max-tolerancia (k-cover al máximo)
        anclas.append(construir_solucion_multicobertura( # Adosa guerrero hipercubridor
            self.H, self.pair_idx, self.max_k, self.rng # Provee municiones pesadas
        )) # Finja de acople

        # Ancla 3: Max distancia media de Hamming
        anclas.append(_ancla_max_hamming_medio( # Ensambla coloso del Hamming
            self.discrepancia, self.distinguibilidad, # Atributos basales
            self.n_snps, self.n_pares, self.rng # Atributos auxiliares
        )) # Culmina

        # Ancla 4: Min varianza
        anclas.append(_ancla_min_varianza( # Moldea arquitecto de la equidad
            self.discrepancia, self.H, self.pair_idx, # Suministros
            self.n_snps, self.n_pares, self.rng # Suministros espaciales
        )) # Completa

        return anclas # Entrega lote élite

    def _sweep_k_cover(self, n):
        """Tier 2: barrido de cobertura progresiva con spacing geométrico."""
        soluciones = [] # Cubículo general de soluciones barridas
        k_vals = np.unique(np.geomspace(1, self.max_k, max(n, 2)).astype(int)) # Traza arco curvo de metas de cobertura
        # Distribuir n individuos entre los valores de k
        repeticiones = max(1, n // len(k_vals)) # Dosifica cupos por tramo
        for k in k_vals: # Cabalga sobre umbrales
            for _ in range(repeticiones): # Reprocesa para satisfacer la cuota
                if len(soluciones) >= n: # Contención antidesbordes
                    break # Frena si rebosa
                soluciones.append(construir_solucion_multicobertura( # Pide construcción específica
                    self.H, self.pair_idx, int(k), self.rng # Alimenta máquina de cobertura
                )) # Cierra petición
        # Rellenar si faltan
        while len(soluciones) < n: # Verifica déficit post-reparto
            k = int(self.rng.integers(1, self.max_k + 1)) # Extrae meta residual flotante
            soluciones.append(construir_solucion_multicobertura( # Tira de la palanca otra vez
                self.H, self.pair_idx, k, self.rng # Manda insumos
            )) # Consuma
        return soluciones[:n] # Poda excesos sutiles

    def _bloques_assembly(self, n):
        """Tier 3: ensamblaje de soluciones por subconjuntos de bloques LD."""
        soluciones = [] # Reserva de híbridos
        n_bloques = len(self.bloques) # Contabiliza provincias LD
        if n_bloques == 0: # Si hay cero distritos
            return soluciones # Devuelve nada

        for _ in range(n): # Itera por encargo
            # Subconjunto aleatorio de bloques (entre 40% y 100% de los bloques)
            n_sel = self.rng.integers(max(1, n_bloques * 2 // 5), n_bloques + 1) # Baraja cupo de asamblea
            indices = self.rng.choice(n_bloques, size=n_sel, replace=False) # Convoca distritos sin reposición
            indices.sort() # Ordena distritos convocados
            soluciones.append(_construir_solucion_bloques( # Fabrica asamblea vecinal
                self.discrepancia, self.bloques, self.n_snps, # Inserta materia prima
                self.n_pares, indices, self.rng # Inserta actas oraculares
            )) # Ensambla al repositorio
        return soluciones # Vuelve con canasto lleno

    def _complementos(self, soluciones_existentes, n):
        """Tier 4: construye complementos de soluciones existentes."""
        resultados = [] # Matriz de salvavidas
        n_base = len(soluciones_existentes) # Cuenta flota primaria
        if n_base == 0: # Si la flota es nula
            return resultados # Revoca labor

        for i in range(n): # Ejerce por solicitud de magnitud
            base = soluciones_existentes[i % n_base] # Entresaca espécimen cíclico
            resultados.append(_construir_complemento( # Diseña sombra correctora
                self.discrepancia, base, self.n_snps, self.n_pares, self.rng # Atribuye elementos
            )) # Incorpora
        return resultados # Termina

    def _guided_sparse(self):
        """Tier 5: un individuo con muestreo disperso ponderado y reparación."""
        # Cardinalidad objetivo: entre el min-k observado y algo moderado
        k_obj = self.rng.integers( # Gira ruleta cardinal
            max(1, int(self.distinguibilidad.size * 0.01)), # Piso estricto del uno por ciento
            max(2, int(self.distinguibilidad.size * 0.15)) # Techo blando del quince por ciento
        ) # Concreta objetivo numérico
        return _muestreo_guiado_disperso( # Manda a la fragua estocástica
            self.distinguibilidad, self.discrepancia, self.n_snps, self.n_pares, k_obj, self.rng # Adjunta requerimientos
        ) # Resuelve

    def _deduplicar(self, X):
        """Post-procesado: deduplicación fenotípica mediante mutaciones monótonas."""
        huellas = set() # Crea diccionario inmutable de espectros genéticos
        for i in range(len(X)): # Camina sobre la metrópolis resultante
            fp = X[i].tobytes() # Fija estampa criptográfica del habitante
            if fp in huellas: # Confirma colisiones fatales de identidad
                # Mutar de forma segura: convertir de 1 a 3 bits 'False' a 'True'.
                # Añadir SNPs matemáticamente preserva la cobertura existente.
                candidatos = np.where(~X[i])[0] # Busca nichos ociosos del cromosoma
                if len(candidatos) > 0: # Avala disponibilidad
                    n_flips = min(len(candidatos), int(self.rng.integers(1, 4))) # Pondera cuántos empujes dar (1-3)
                    bits = self.rng.choice(candidatos, n_flips, replace=False) # Lotería sin repetición
                    X[i, bits] = True # Enciende genes inactivos
            huellas.add(X[i].tobytes()) # Sella nueva identidad post-quirúrgica (o nativa)
        return X # Extiende la colonia higienizada

    def _do(self, problem, n_samples, **kwargs):
        n_samples = int(n_samples) # Endurece tipo de cupos globales
        X = np.zeros((n_samples, problem.n_var), dtype=bool) # Alza edificio habitacional en cero
        pos = 0 # Inicializa el ascensor poblacional

        # === Tier 1: Anclas de Pareto (~5-10%) ===
        anclas = self._construir_anclas() # Convoca a la junta de arquitectos
        n_anclas = min(len(anclas), max(2, n_samples // 10)) # Asigna cuotas de élite restringidas
        for i in range(n_anclas): # Empieza vaciado
            if pos >= n_samples: # Si rebosa la pileta
                break # Frenos de emergencia
            X[pos] = anclas[i] # Instala prócer
            pos += 1 # Eleva el conteo

        restantes = n_samples - pos # Contabiliza cupos vacantes
        if restantes <= 0: # Si ya llenó
            return self._deduplicar(X) # Lava y entrega

        # === Tier 2: Barrido k-Cover (~25%) ===
        n_t2 = max(1, int(round(restantes * 0.30))) # Dosifica segundo estrato (30% de los restos)
        for sol in self._sweep_k_cover(n_t2): # Destila individuos barridos
            if pos >= n_samples: # Vigila fronteras
                break # Detén
            X[pos] = sol # Adosa
            pos += 1 # Sube

        # === Tier 3: Ensamblaje por bloques LD (~25%) ===
        n_t3 = max(1, int(round(restantes * 0.30))) # Prepara tercer peldaño (30% de los restos)
        for sol in self._bloques_assembly(n_t3): # Recupera forjados del LD
            if pos >= n_samples: # Chequeos cautelares
                break # Frena
            X[pos] = sol # Pega
            pos += 1 # Asciende

        # === Tier 4: Inyección de complementos (~20%) ===
        n_t4 = max(1, int(round(restantes * 0.20))) # Acomoda parches (20% de lo que sobró al inicio)
        existentes = X[:pos].copy() # Fotocopia el legado actual
        for sol in self._complementos(existentes, n_t4): # Ejecuta compensadores
            if pos >= n_samples: # Evade fugas de memoria
                break # Seca surtidor
            X[pos] = sol # Pega tapiz
            pos += 1 # Cuenta

        # === Tier 5: Exploración guiada dispersa (resto) ===
        while pos < n_samples: # El saldo va de relleno estocástico fino
            X[pos] = self._guided_sparse() # Rellena capilaridad suelta
            pos += 1 # Incrementa

        # === Post-procesado: deduplicación fenotípica ===
        return self._deduplicar(X) # Enjuaga el total, purga clones y despacha
