import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import numpy as np
import sys
import os
import time
import pandas as pd
evaluation_counter = 0

def read_instance(directory, types):
    """
    Per ogni file (istanza) nella directory, esegue num_runs iterazioni dell'HGA e raccoglie:
      - La curva di convergenza per ogni iterazione.
      - Il costo finale ottenuto in ogni iterazione.
    Tra tutte le istanze viene selezionata quella con il costo finale minimo (e std minore in caso di parità).
    Prepara quindi i dati per la visualizzazione dei grafici e la tabella dei risultati.
    """
    global evaluation_counter

    if types.lower() == "grid":
        if len(sys.argv) < 4:
            print(f"Usage: python {sys.argv[0]} <directory> Grid <first_dimension_graph>")
            sys.exit(1)
        path_part = f"{types}_{sys.argv[3]}_{sys.argv[3]}_"
    elif types.lower() in ["rand", "random"]:
        if len(sys.argv) < 5:
            print(f"Usage: python {sys.argv[0]} <directory> Rand <num_nodes> <num_edges>")
            sys.exit(1)
        path_part = f"{types}_{sys.argv[3]}_{sys.argv[4]}_"
    else:
        path_part = ""

    file_names = sorted(os.listdir(directory))
    filtered_files = [file for file in file_names if path_part in file]

    n_file = 1
    instances_results = []
    num_runs = 10
    results = []

    for file in filtered_files:
        print(f"Elaborazione file {n_file}: '{file}'")
        path = os.path.join(directory, file)
        G, nodes, edges, weights, weights_aux, adj_dict, seed = parse_fvs_file(path)
        maxCost = sum(weights)
        print(f"MaxCost: {maxCost}")
        result_data = [G, nodes, edges, weights, weights_aux, adj_dict]

        costs = []           # Costo finale per iterazione
        evaluations = []     # Numero di valutazioni per iterazione
        all_iter_best_costs = []  # Curve di convergenza per ogni run
        best_solution_runs = []   # Migliore soluzione per ogni run
        start_time = time.time()

        for j in range(num_runs):
            print(f"  Iterazione {j+1}/{num_runs} (seed {seed})")
            evaluation_counter = 0
            best_solution, best_cost, best_costs = hybrid_genetic_algorithm(result_data, maxCost)
            print(f"    Costo: {best_cost}, Valutazioni: {evaluation_counter}")
            costs.append(best_cost)
            evaluations.append(evaluation_counter)
            all_iter_best_costs.append(best_costs)
            best_solution_runs.append(best_solution)

        end_time = time.time()
        best_run_idx = costs.index(min(costs))
        instance_result = {
            "G": G,
            "best_solution": best_solution_runs[best_run_idx],
            "all_iter_best_costs": all_iter_best_costs,
            "nodes": nodes,
            "edges": edges,
            "best_instance_cost": min(costs),
            "std": np.std(costs)
        }
        instances_results.append(instance_result)
        n_file += 1

        duration = end_time - start_time
        formatted_tempo_medio = "{:.2f}".format(duration / num_runs)
        results_row = {
            "seed": seed, 
            "min_costo": min(costs), 
            "media_costi": "{:.2f}".format(sum(costs) / len(costs)), 
            "dev_std": "{:.2f}".format(np.std(costs)), 
            "evaluations": sum(evaluations) / len(evaluations), 
            "tempo": f"{formatted_tempo_medio} s"
        }
        results.append(results_row)
        print(f"  Risultati file '{file}': Tempo: {duration:.2f} s, Costo minimo: {min(costs)}\n")
    
    print("Risultati aggregati:")
    best_instance_costs = [inst["best_instance_cost"] for inst in instances_results]
    print(f"  Media migliori soluzioni: {np.mean(best_instance_costs)}")
    print(f"  Deviazione standard: {np.std(best_instance_costs)}")
    
    best_instance = min(instances_results, key=lambda inst: (inst["best_instance_cost"], inst["std"]))
    print(f"Istanza migliore: Costo {best_instance['best_instance_cost']} con std {best_instance['std']}")
    
    fileDirRes = f"risultati/{types.lower()}_{nodes}_{edges}/"
    if not os.path.exists(fileDirRes):
        os.makedirs(fileDirRes, exist_ok=True)
    filePathRes = f"{fileDirRes}data.jpeg"
    title = f"Risultati istanza '{types}_{nodes}_{edges}'"
    params = get_params(nodes, edges)
    description = (
        f"Risultati dell'algoritmo HGA eseguito sull'istanza '{types}_{nodes}_{edges}'.\n"
        f"Media costo: {np.mean(best_instance_costs)}\n"
        f"Istanza migliore: costo {best_instance['best_instance_cost']} con std {best_instance['std']}\n"
        f"Parametri usati:\n"
        f"pop_size = {params[0]}\n"
        f"generations = {params[1]}\n"
        f"k = {params[2]}\n"
        f"mutation_rate = {params[3]}\n"
        f"crossover_rate = {params[4]}\n"
        f"local_search_sample_size = {params[5]}"
    )
    save_table_as_png(results, filePathRes, title, description)

    concatenated_convergence = []
    for run_curve in best_instance["all_iter_best_costs"]:
        concatenated_convergence.extend(run_curve)
    final_costs_per_iteration = [run_curve[-1] for run_curve in best_instance["all_iter_best_costs"]]

    visualizza_grafici(best_instance["G"], best_instance["best_solution"],
                       concatenated_convergence, final_costs_per_iteration,
                       best_instance["nodes"], best_instance["edges"])

def save_table_as_png(results, filepath, title, description):
    """
    Converte i risultati in una tabella e la salva come immagine.
    """
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                     colColours=["#f2f2f2"] * len(df.columns), bbox=[0, 0.64, 1, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    table.scale(1, 1)
    plt.title(title, fontsize=7, fontweight="bold", pad=0)
    plt.figtext(0.5, 0.35, description, horizontalalignment='center', fontsize=5)
    plt.savefig(filepath, bbox_inches='tight', dpi=200, transparent=True, format="jpeg")
    plt.close()

def parse_fvs_file(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file if line.strip()]
    typeF = lines[1].split(":")[1].strip()
    nodes = int(lines[3].split(":")[1].strip())
    edges = int(lines[4].split(":")[1].strip())
    seed = int(lines[6].split(":")[1].strip())
    print(f"Tipo: {typeF}, Nodi: {nodes}, Archi: {edges}")
    try:
        nw_index = lines.index("NODE_WEIGHT_SECTION")
    except ValueError:
        print("NODE_WEIGHT_SECTION non trovato.")
        sys.exit(1)
    weights = []
    for i in range(nw_index + 1, nw_index + 1 + nodes):
        parts = lines[i].split()
        weights.append(int(parts[1]) if len(parts) >= 2 else 0)
    print("Fine lettura pesi")
    matrix_header_index = nw_index + 1 + nodes
    matrix_start = matrix_header_index + 1
    first_matrix_line = lines[matrix_start].split()
    if len(first_matrix_line) < nodes:
        adj_dict = {i: [] for i in range(nodes)}
        for i in range(nodes):
            row_line = lines[matrix_start + i].split()
            for j in range(len(row_line)):
                if j < i and int(row_line[j]) == 1:
                    adj_dict[i].append(j)
                    adj_dict[j].append(i)
    else:
        adjacency_matrix = [list(map(int, lines[matrix_start + i].split())) for i in range(nodes)]
        adj_dict = {i: [j for j in range(nodes) if i != j and adjacency_matrix[i][j] == 1] for i in range(nodes)}
    G = nx.Graph()
    weights_aux = {i: weights[i] for i in range(nodes)}
    for i, w in weights_aux.items():
        G.add_node(i, weight=w)
    for i in range(nodes):
        for j in adj_dict[i]:
            if i < j:
                G.add_edge(i, j)
    return G, nodes, edges, weights, weights_aux, adj_dict, seed

def repair_solution(graph, sol, weights, edges_list):
    """
    Ripara la soluzione affinché il grafo risultante sia aciclico.
    """
    new_sol = sol.copy()
    while not is_fvs(graph, new_sol, edges_list):
        remaining = [i for i, bit in enumerate(new_sol) if bit == 0]
        subG = graph.subgraph(remaining)
        try:
            cycle = nx.find_cycle(subG, orientation='ignore')
        except nx.NetworkXNoCycle:
            break
        vertices_in_cycle = {u for u, v, _ in cycle} | {v for u, v, _ in cycle}
        candidate = min(vertices_in_cycle, key=lambda v: weights[v])
        new_sol[candidate] = 1
    return new_sol

def smooth_curve(values, window_size=5):
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

def visualizza_grafici(G, best_solution, concatenated_convergence, final_costs_per_iteration, n, m):
    """
    Visualizza grafici: grafo iniziale, grafo finale, convergenza e costi finali per iterazione.
    """
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 2)
    
    ax_left_top = plt.subplot(gs[0, 0])
    pos = nx.spring_layout(G)
    node_labels = {node: f"{node}\n{G.nodes[node].get('weight','')}" for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue',
            node_size=500, edge_color='gray', ax=ax_left_top)
    ax_left_top.set_title(f"Grafo Iniziale ({n} nodi, {m} archi)")
    
    ax_left_bottom = plt.subplot(gs[1, 0])
    fvs = {i for i, bit in enumerate(best_solution) if bit == 1}
    G_final = G.subgraph(set(G.nodes()) - fvs)
    pos_final = nx.spring_layout(G_final)
    nx.draw(G_final, pos_final, with_labels=True, node_color='lightgreen',
            node_size=500, edge_color='gray', ax=ax_left_bottom)
    ax_left_bottom.set_title("Grafo Finale (Aciclico)")
    
    ax_right_top = plt.subplot(gs[0, 1])
    ax_right_top.plot(range(len(concatenated_convergence)), concatenated_convergence,
                      marker='', linestyle='-', color='r', label="Convergenza")
    ax_right_top.set_xlabel("Generazione (iterazioni totali)")
    ax_right_top.set_ylabel("Costo migliore")
    ax_right_top.set_title("Andamento di Convergenza")
    ax_right_top.legend()
    ax_right_top.grid(True)
    
    ax_right_bottom = plt.subplot(gs[1, 1])
    ax_right_bottom.plot(range(len(final_costs_per_iteration)), final_costs_per_iteration,
                         marker='o', linestyle='-', color='g', label="Costo Finale Iterazione")
    ax_right_bottom.set_xlabel("Iterazione")
    ax_right_bottom.set_ylabel("Costo migliore")
    ax_right_bottom.set_title("Migliori Costi per Iterazione")
    ax_right_bottom.legend()
    ax_right_bottom.grid(True)
    
    plt.tight_layout()
    plt.show()

def has_cycle(graph, remaining):
    """
    Verifica se il sottografo dei nodi in 'remaining' contiene cicli (DFS).
    """
    visited = set()
    stack = set()
    def dfs(node, parent):
        visited.add(node)
        stack.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in remaining:
                continue
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor in stack and neighbor != parent:
                return True
        stack.remove(node)
        return False
    for node in remaining:
        if node not in visited:
            if dfs(node, None):
                return True
    return False

def is_fvs(graph, sol, edges_list):
    """
    Verifica se, rimuovendo i nodi indicati in 'sol', il grafo diventa aciclico.
    """
    remaining = {node for node, bit in enumerate(sol) if bit == 0}
    return not has_cycle(graph, remaining)

def fitness(graph, sol, weights, maxCost, edges_list):
    global evaluation_counter
    evaluation_counter += 1    
    if is_fvs(graph, sol, edges_list):
        return sum(weights[i] for i, bit in enumerate(sol) if bit == 1)
    else:
        return maxCost

def tournament_selection(population, fitness_values, k=3):
    indices = random.sample(range(len(population)), k)
    best_index = min(indices, key=lambda i: fitness_values[i])
    return population[best_index]

def crossover(parent1, parent2):
    p1, p2 = list(parent1), list(parent2)
    if len(p1) < 2 or len(p2) < 2:
        return parent1, parent2
    point = random.randint(1, len(p1) - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    return [1 - bit if random.random() < mutation_rate else bit for bit in individual]

def local_search(graph, individual, weights, maxCost, edges_list, sample_size=None):
    current = repair_solution(graph, individual, weights, edges_list)
    best = current.copy()
    best_fit = fitness(graph, best, weights, maxCost, edges_list)
    improved = True
    while improved:
        improved = False
        indices = list(range(len(best)))
        if sample_size is not None:
            indices = random.sample(indices, min(sample_size, len(best)))
        for i in indices:
            neighbor = best.copy()
            neighbor[i] = 1 - neighbor[i]
            new_fit = fitness(graph, neighbor, weights, maxCost, edges_list)
            if new_fit < best_fit:
                best, best_fit = neighbor, new_fit
                improved = True
                break
    return best

def get_params(n, m):
    pop_size = 10
    generations = 50
    k = 4
    mutation_rate = 0.065
    crossover_rate = 0.8
    local_search_sample_size = min(5, n)
    print(f"Parametri: pop_size={pop_size}, generazioni={generations}, k={k}, mut_rate={mutation_rate}, crossover={crossover_rate}")
    return pop_size, generations, k, mutation_rate, crossover_rate, local_search_sample_size

def initPopulation(pop_size, num_nodes):
    return [[random.randint(0, 1) for _ in range(num_nodes)] for _ in range(pop_size)]

def hybrid_genetic_algorithm(result, maxCost):
    print(f"result: Nodi-{result[1]}; Archi-{result[2]}")
    graph, num_nodes, _, weights, _, _ = result[:6]
    pop_size, generations, k, mutation_rate, crossover_rate, ls_sample = get_params(num_nodes, result[2])
    population = initPopulation(pop_size, num_nodes)
    edges_list = list(graph.edges())
    best_fit = maxCost
    best_solution = population[0]
    best_costs = []
    for gen in range(generations):
        fitness_values = [fitness(graph, sol, weights, maxCost, edges_list) for sol in population]
        current_best = min(fitness_values)
        if current_best < best_fit:
            best_fit = current_best
            best_solution = population[fitness_values.index(current_best)]
        best_costs.append(best_fit)
        new_population = []
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness_values, k)
            p2 = tournament_selection(population, fitness_values, k)
            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            c1 = local_search(graph, c1, weights, maxCost, edges_list, sample_size=ls_sample)
            c2 = local_search(graph, c2, weights, maxCost, edges_list, sample_size=ls_sample)
            new_population.extend([c1, c2])
        population = new_population[:pop_size]
        if gen % 10 == 0:
            print(f"Generazione {gen}: Best fitness = {best_fit}/{maxCost}")
    best_solution_cost = fitness(graph, best_solution, weights, maxCost, edges_list)
    print(f"Migliore soluzione trovata: {best_solution} con costo {best_solution_cost} / {sum(best_solution)}")
    return best_solution, best_solution_cost, best_costs

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <directory> <type> [specific_args_types]")
        sys.exit(1)
    directoryPath = sys.argv[1]
    types = sys.argv[2]
    read_instance(directoryPath, types)

if __name__ == "__main__":
    main()
