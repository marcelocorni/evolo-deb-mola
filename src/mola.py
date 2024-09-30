import time
import numpy as np
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO
from mealpy import FloatVar
import plotly.express as px
import pandas as pd
import streamlit as st

class CustomPenaltyOptimization:
    def __init__(self, number_of_constraints, model_class=GA.BaseGA, num_executions=30, num_evaluations=10000):
        """
        Construtor.
        Parameters:
        - number_of_constraints: número de restrições do problema.
        - model_class: classe do modelo de otimização (ex: BaseGA da biblioteca Mealpy)
        - num_executions: número de execuções independentes para a otimização.
        - num_evaluations: número total de avaliações da função objetivo.
        """
        self.number_of_constraints = number_of_constraints
        self.model_class = model_class
        self.num_executions = num_executions
        self.num_evaluations = num_evaluations

    def funcao_objetivo(self, solution):
        x1, x2, x3 = solution

        def g1(x):
            return 1 - (x3**2 * x1) / (71785 * x3**4)
        def g2(x):
            return (4 * x2**2 - x3 * x2) / (12566 * (x2 * x3**3 - x3**4)) + (1 / (5108 * x3**2)) - 1
        def g3(x):
            return 1 - (140.45 * x3 / (x2**2 * x1))
        def g4(x):
            return (x2 + x3) / 1.5

        violations = [g1(solution), g2(solution), g3(solution), g4(solution)]
        violations = [violation if violation > 0 else 0 for violation in violations]

        V = (x1 + 2) * x2 * x3**2

        return V, violations  # Certifique-se de que o valor de V é o primeiro item retornado

    # Função de penalização personalizada
    def penalizacao(self, x, objetivo, restricoes, populacao_atual):
        # Verifica se a solução é viável (se todas as restrições são satisfeitas)
        viavel = all([g(x) >= 0 for g in restricoes])
       
        if viavel:
            return objetivo(x)
        else:
            # Obtendo fmax, que é o pior valor objetivo das soluções viáveis
            solucoes_viaveis = [sol for sol in populacao_atual if all(g(sol) >= 0 for g in restricoes)]
            if solucoes_viaveis:
                f_max = max([objetivo(sol) for sol in solucoes_viaveis])
            else:
                f_max = 10000  # Um valor alto padrão se nenhuma solução viável estiver presente
            
            # Soma das penalidades das violações das restrições
            penalidade = sum([abs(g(x)) if g(x) < 0 else 0 for g in restricoes])
            return f_max + penalidade

    def penalized_objective_function(self, solution, population):
        """
        Função objetiva penalizada que calcula o fitness usando a penalização personalizada.
        Parameters:
        - solution: solução para avaliar
        - population: população para cálculo dos coeficientes de penalidade
        
        Returns:
        - fitness penalizado (valor escalar)
        """
        # Definir restrições para a penalização
        restricoes = [
            lambda x: 1 - (x[2]**2 * x[0]) / (71785 * x[2]**4),
            lambda x: (4 * x[1]**2 - x[2] * x[1]) / (12566 * (x[1] * x[2]**3 - x[2]**4)) + (1 / (5108 * x[2]**2)) - 1,
            lambda x: 1 - (140.45 * x[2] / (x[1]**2 * x[0])),
            lambda x: (x[1] + x[2]) / 1.5
        ]

        # Aplicar a função de penalização personalizada
        fitness = self.penalizacao(solution, lambda x: self.funcao_objetivo(x)[0], restricoes, population)

        return fitness  # Retorna apenas o valor do fitness (um escalar)

    def run_optimization(self, lower_bounds, upper_bounds, pop_size=50):
        """
        Executa a otimização múltiplas vezes e retorna as métricas para o volume V.
        
        Parameters:
        - lower_bounds: limites inferiores para as variáveis de decisão.
        - upper_bounds: limites superiores para as variáveis de decisão.
        - pop_size: tamanho da população.
        
        Returns:
        - métricas de Melhor, Mediana, Média, Desvio Padrão e Pior para o volume V.
        """
        # Calcular o número de epochs com base no número total de avaliações e no tamanho da população
        epochs = self.num_evaluations // pop_size

        results = []
        for _ in range(self.num_executions):
            # Inicializar a população
            population = np.random.uniform(lower_bounds, upper_bounds, (pop_size, len(lower_bounds)))

            # Otimização usando a model_class passada com a função objetiva penalizada
            problem = {
                "obj_func": lambda solution: self.penalized_objective_function(solution, population),
                "bounds": FloatVar(lb=lower_bounds, ub=upper_bounds),
                "minmax": "min",
                "log_to": None,
            }

            model = self.model_class(epoch=epochs, pop_size=pop_size)
            model.solve(problem)

            best_solution = model.g_best.solution
            best_fitness = model.g_best.target.fitness

            # Armazenar os valores de V para a melhor solução
            V_best, _ = self.funcao_objetivo(best_solution)
            results.append(V_best)

        # Calcular as métricas
        melhor = np.min(results)
        mediana = np.median(results)
        media = np.mean(results)
        dp = np.std(results)
        pior = np.max(results)

        return melhor, mediana, media, dp, pior


def main():

    st.set_page_config(page_title="Otimização de GA e PSO com Restrições (Kalyanmoy Deb)", page_icon="📊", layout="wide")

    # Parâmetros do problema
    number_of_constraints = 3 # Número de variáveis de decisão, no caso do problema da mola são x1, x2 e x3

    model_classes = {"GA": GA.BaseGA, "PSO": PSO.OriginalPSO}

    # Interface gráfica
    st.title("Otimização de GA e PSO com Restrições")
    st.sidebar.title("Configurações")

    with st.sidebar:
        with st.form(key="config_form"):
            num_executions = st.number_input("Número de execuções", min_value=1, max_value=100, value=35, step=1, key="num_executions")
            num_evaluations = st.number_input("Número total de avaliações", min_value=1000, max_value=100000, value=36000, step=1000, key="num_evaluations")
            pop_size = st.number_input("Tamanho da população", min_value=1, max_value=200, value=50, step=1, key="pop_size")
            submit_button = st.form_submit_button("Executar")

    if not submit_button:
        return
    
    # Limites das variáveis de decisão
    lower_bounds = [2.0, 0.25, 0.05]
    upper_bounds = [15.0, 1.3, 2.0]

    resultados = []
    col1, col2 = st.columns(2)

    # Percorrer cada algoritmo de otimização
    with st.spinner("Executando otimizações..."):
        start_time = time.time()
        for key in model_classes.keys():
            optimizer = CustomPenaltyOptimization(
                number_of_constraints=number_of_constraints,
                model_class=model_classes[key],
                num_executions=num_executions,
                num_evaluations=num_evaluations
            )
            # Executar a otimização e obter as métricas
            melhor, mediana, media, dp, pior = optimizer.run_optimization(lower_bounds, upper_bounds, pop_size=pop_size)
            resultados.append((key, melhor, mediana, media, dp, pior))
        end_time = time.time()
        # Calcular horas, minutos e segundos que foram necessários para a execução
        tempo_execucao = end_time - start_time
        horas = int(tempo_execucao // 3600)
        minutos = int((tempo_execucao % 3600) // 60)
        segundos = int(tempo_execucao % 60)

    st.success(f"Execução finalizada em {horas} horas, {minutos} minutos e {segundos} segundos.")  

    # Criar dataframe com os resultados
    df = pd.DataFrame(resultados, columns=["Algoritmo", "Melhor", "Mediana", "Média", "Desvio Padrão", "Pior"])

    col1.write("Resultados")
    col1.write(df)
    # Gráfico de barras unificado
    fig = px.bar(df, x="Algoritmo", y="Melhor", color="Algoritmo", title="Melhor valor de V para cada algoritmo")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()