# tcc-mdvrp

## Estrutura do projeto

tcc_mdvrp/
│
├── data/                  # Arquivos de entrada (instâncias)
│   ├── raw/               # Bases de dados originais
│   └── processed/         # Bases de dados adaptadas
│
├── notebooks/             # Exclusivo para Jupyter Notebooks
│
├── src/
│   ├── __init__.py
│   ├── core/              # Entidades do problema
│   │   ├── __init__.py
│   │   ├── entities.py    # Classes: Cliente, Deposito, Veiculo, Rota
│   │   └── solution.py    # Classe que representa uma Solução inteira e calcula Fitness
│   │
│   ├── algorithms/        # Os algoritmos
│   │   ├── __init__.py
│   │   ├── pso.py         # Lógica do enxame (partículas, velocidade)
│   │   └── split.py       # Algoritmo de divisão da rota gigante
│   │
│   ├── utils/             # Ferramentas auxiliares
│   │   ├── __init__.py
│   │   ├── data_loader.py # Lógica para ler os .txt do diretório /data
│   │   └── metrics.py     # Funções para calcular distância euclidiana, etc
│   │
│   └── main.py
│
├── tests/
│   ├── __init__.py
│
├── config.yaml            # Parâmetros do algoritmo (inércia, max_iter, capacidade)
├── requirements.txt       # Dependências (numpy, matplotlib, etc)
└── README.md              # Como rodar o seu projeto


## Descrição


## Pré-requisitos
Certifique-se de ter o seguinte instalado em seu sistema:
- Python 3.8 ou superior
- `pip` para gerenciar pacotes Python

## Configuração do Ambiente

1. **Clone o repositório**:
   ```sh
   git clone <URL_DO_REPOSITORIO>
   cd tcc-mdvrp
    ```
2. **Crie o ambiente virtual**:
   ```sh
   python3 -m venv venv
   ```

3. **Ative o ambiente virtual**:
    - No Windows:
      ```sh
      venv\Scripts\activate
      ```
    - No Linux/Mac:
      ```sh
      source venv/bin/activate
      ```

4. **Instale as dependências**:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

```sh
python3 main.py <arquivo_in>
```