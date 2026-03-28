import numpy as np
from pymoo.core.problem import Problem
from point import Point 

class TspProblem(Problem):
    """
    Esta classe define o nosso problema TSP (Etapa 0) para o Pymoo.
    Ela usa uma representação baseada em "DNA de floats" (Ranking-Based).
    """
    
    def __init__(self, depot: Point, clients: list[Point]):
        """
        Inicializa o problema.
        
        :param depot: O ponto de Depósito (onde a rota começa e termina)
        :param clients: A lista de Pontos (Hospitais, etc.) a serem visitados
        """
        
        self.depot = depot
        self.clients = clients
        
        n_var = len(self.clients) # n_var: Número de variáveis (clientes a visitar)
        n_obj = 1                 # n_obj: Número de objetivos (minimizar distância total)
        n_constr = 0              # n_constr: Número de restrições
        
        # xl / xu: Limites inferiores (lower) e superiores (upper) das variáveis
        xl = np.zeros(n_var)  # Limite inferior = 0.0
        xu = np.ones(n_var)   # Limite superior = 1.0
        
        # Chama o construtor da classe base Problem
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Esta é a função de fitness (o "Decodificador").
        O Pymoo chama esta função para avaliar cada indivíduo (DNA).
        
        :param x: Um array 2D do NumPy com os "DNAs" da população atual.
                  Formato: (tamanho_populacao, n_var)
        :param out: Um dicionário onde salvaremos os resultados (fitness)
        """
        
        fitness_values = []
        
        # O Pymoo avalia a população inteira de uma vez.
        # Precisamos iterar em cada "DNA" (cada linha de 'x')
        for dna in x:

            # --- Etapa A: Decodificar o "DNA" em uma Rota ---
            # dna é: [0.85, 0.12, 0.99, ...]
            # Usamos argsort() para obter os ÍNDICES em ordem ordenada
            # Ex: [0.85, 0.12, 0.99] -> argsort dá [1, 0, 2] (ordem crescente)
            # Nós queremos a ordem decrescente (do maior para o menor)
            
            sorted_indices = np.argsort(dna)[::-1] 
            # Ex: [0.85, 0.12, 0.99] -> argsort[::-1] dá [2, 0, 1]
            
            # Crie a lista de pontos do cliente na ordem decodificada
            route_clients = [self.clients[i] for i in sorted_indices]
            
            # Crie a rota completa do TSP (começa e termina no depósito)
            full_route = [self.depot] + route_clients + [self.depot]

            # --- Etapa B: Calcular o Fitness (Distância Total) ---
            total_distance = 0.0
            for i in range(len(full_route) - 1):
                ponto_A = full_route[i]
                ponto_B = full_route[i+1]
                total_distance += ponto_A.distance_to(ponto_B)
                
            # Adiciona o fitness deste indivíduo à lista
            fitness_values.append(total_distance)
            
        # --- Etapa C: Devolver os resultados para o Pymoo ---
        # O Pymoo espera os resultados no dicionário 'out'
        # 'F' é a chave para o fitness (Objetivos)
        out["F"] = np.array(fitness_values)