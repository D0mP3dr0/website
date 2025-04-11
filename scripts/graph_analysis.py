import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import kneighbors_graph

def construct_rbs_network(rbs_data, threshold_distance=1000):
    """
    Constrói um grafo representando a rede de RBSs onde as conexões
    são estabelecidas baseadas na distância e sobreposição de cobertura
    
    Args:
        rbs_data: DataFrame com dados das estações base
        threshold_distance: Distância máxima para criar conexão (metros)
    """
    # Extrair coordenadas
    coords = rbs_data[['latitude', 'longitude']].values
    
    # Criar matriz de adjacência baseada na distância
    A = kneighbors_graph(coords, n_neighbors=10, mode='distance')
    
    # Converter para NetworkX graph
    G = nx.from_scipy_sparse_matrix(A)
    
    # Adicionar atributos aos nós
    for i, row in enumerate(rbs_data.iterrows()):
        G.nodes[i]['lat'] = row[1]['latitude']
        G.nodes[i]['lon'] = row[1]['longitude']
        G.nodes[i]['power'] = row[1]['transmitter_power']
        G.nodes[i]['height'] = row[1]['antenna_height']
        G.nodes[i]['operator'] = row[1]['operator']
    
    return G

def analyze_network_properties(G):
    """
    Analisa as propriedades do grafo de rede RBS
    
    Args:
        G: NetworkX graph representando a rede RBS
    
    Returns:
        Dict com propriedades da rede
    """
    properties = {}
    
    # Calcular métricas básicas
    properties['num_nodes'] = G.number_of_nodes()
    properties['num_edges'] = G.number_of_edges()
    properties['density'] = nx.density(G)
    
    # Calcular distribuição de grau
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    properties['avg_degree'] = np.mean(degree_sequence)
    properties['max_degree'] = max(degree_sequence)
    
    # Calcular centralidade
    centrality = nx.betweenness_centrality(G)
    properties['max_centrality'] = max(centrality.values())
    properties['avg_centrality'] = np.mean(list(centrality.values()))
    
    # Calcular coeficiente de clustering
    clustering = nx.clustering(G)
    properties['avg_clustering'] = np.mean(list(clustering.values()))
    
    # Calcular componentes conectados
    components = list(nx.connected_components(G))
    properties['num_components'] = len(components)
    
    return properties

def detect_communities(G, algorithm='louvain'):
    """
    Detecta comunidades na rede RBS
    
    Args:
        G: NetworkX graph representando a rede RBS
        algorithm: Algoritmo a ser usado (louvain, label_propagation)
    
    Returns:
        Dict mapeando nós para comunidades
    """
    try:
        import community as community_louvain
        if algorithm == 'louvain':
            return community_louvain.best_partition(G)
        else:
            return {node: i for i, comp in enumerate(nx.algorithms.community.label_propagation_communities(G)) for node in comp}
    except ImportError:
        print('Package python-louvain not found, using label propagation instead')
        return {node: i for i, comp in enumerate(nx.algorithms.community.label_propagation_communities(G)) for node in comp}

def identify_critical_nodes(G, method='betweenness'):
    """
    Identifica nós críticos na rede cuja remoção 
    mais impactaria na conectividade
    
    Args:
        G: NetworkX graph representando a rede RBS
        method: Método de identificação (betweenness, degree)
    
    Returns:
        Lista de nós críticos (top 5%)
    """
    if method == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    else:
        centrality = {node: d for node, d in G.degree()}
    
    # Ordenar nós por centralidade
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    # Retornar top 5%
    num_critical = max(1, int(len(sorted_nodes) * 0.05))
    return [node for node, cent in sorted_nodes[:num_critical]]

if __name__ == '__main__':
    # Exemplo de uso
    print('Graph Analysis Module for RBS Networks') 