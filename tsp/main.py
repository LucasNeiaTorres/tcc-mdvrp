import sys
import csv
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from point import Point 
from problem import TspProblem 

NUM_GENERATIONS = 100
POPULATION_SIZE = 50

def read_csv(file_path):
    print("Lendo arquivo CSV:", file_path) 
    
    points = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            point = Point( 
                point_id=int(row['id_ponto']),
                name=row['nome_ponto'],
                x=float(row['coord_x']),
                y=float(row['coord_y']),
                location_type=row['tipo_local'],
                demand=int(row['demanda_pessoas']),
                tw_start=int(row['janela_inicio']),
                tw_end=int(row['janela_fim']),
                service_time=int(row['tempo_servico']),
                risk=float(row['risco_local'])
            )
            points.append(point)
    return points

def main():
    if len(sys.argv) != 2:
        print("Uso: python tsp.py arquivo_in.csv")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    points = read_csv(input_file)
    
    depots = []      # Lista de pontos de partida (onde os ônibus começam)
    clients = []     # Lista de pontos de coleta (quem precisa ser salvo)
    shelters = []    # Lista de pontos de destino (para onde vão)
    
    for point in points:
        if point.location_type == 'DEPOSITO':
            depots.append(point)
        elif point.location_type in ['HOSPITAL', 'ASILO', 'ESCOLA', 'CLINICA']:
            clients.append(point)
        elif point.location_type == 'ABRIGO':
            shelters.append(point)
        
    if not depots:
        print("Erro: Nenhum DEPOSITO encontrado no arquivo.")
        sys.exit(1)
        
    if not clients:
        print("Erro: Nenhum CLIENTE (Hospital, Asilo, etc.) encontrado.")
        sys.exit(1)

    if not shelters:
        print("Erro: Nenhum ABRIGO encontrado.")
        sys.exit(1)

    print("Cenário carregado com sucesso:")
    print(f"  - {len(depots)} Depósitos (Origens)")
    print(f"  - {len(clients)} Clientes (Pontos de Coleta)")
    print(f"  - {len(shelters)} Abrigos (Destinos)")
    
    depot = depots[0]
    
    problem = TspProblem(
        depot=depot,
        clients=clients
    )
    
    algorithm = GA(
        pop_size=POPULATION_SIZE,
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", NUM_GENERATIONS)
    
    print(f"Iniciando otimização do TSP com {NUM_GENERATIONS} gerações e tamanho de população {POPULATION_SIZE}...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True
    )
    
    print("\n--- Otimização Concluída ---")
    print(f"Melhor distância encontrada: {res.F[0]:.2f}")
    
    # 'res.X' contém o "DNA" da melhor solução
    best_dna = res.X
    
    # Decodifica o melhor DNA para ver a rota
    sorted_indices = np.argsort(best_dna)[::-1]
    best_route_clients = [clients[i] for i in sorted_indices]
    
    print("Melhor Rota Encontrada:")
    print(f"  {depot.name}")
    for point in best_route_clients:
        print(f"  -> {point.name} (id: {point.point_id})")
    print(f"  -> {depot.name}")
    
    
if __name__ == "__main__":
    main()