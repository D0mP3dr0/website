#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'railway')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
RAILWAY_FILE = os.path.join(INPUT_DIR, 'railway_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

def load_data():
    """Carrega os dados processados da rede ferroviária."""
    print("Carregando dados da rede ferroviária...")
    
    data = {}
    
    try:
        data['railway'] = gpd.read_file(RAILWAY_FILE)
        print(f"Ferrovias: {len(data['railway'])} registros")
    except Exception as e:
        print(f"Erro ao carregar ferrovias: {str(e)}")
        data['railway'] = None
    
    try:
        data['sorocaba'] = gpd.read_file(SOROCABA_FILE)
        print(f"Área de estudo: {len(data['sorocaba'])} registros")
    except Exception as e:
        print(f"Erro ao carregar área de estudo: {str(e)}")
        data['sorocaba'] = None
    
    # Verificar CRS e garantir que todos estejam no mesmo sistema
    for key, gdf in data.items():
        if gdf is not None:
            print(f"CRS de {key}: {gdf.crs}")
    
    # Padronizar CRS para SIRGAS 2000 (EPSG:4674)
    target_crs = "EPSG:4674"
    for key, gdf in data.items():
        if gdf is not None and gdf.crs != target_crs:
            data[key] = gdf.to_crs(target_crs)
            print(f"Reprojetado {key} para {target_crs}")
    
    return data

def create_railway_network_map(data, output_path):
    """Cria um mapa interativo da rede ferroviária usando Folium."""
    print("Criando mapa interativo da rede ferroviária...")
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    elif data['railway'] is not None:
        # Usar o centro da rede ferroviária
        railway = data['railway'].to_crs(epsg=4326)
        center_lat = railway.geometry.centroid.y.mean()
        center_lon = railway.geometry.centroid.x.mean()
    else:
        print("Dados insuficientes para criar o mapa")
        return
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=12,
                  tiles='CartoDB positron')
    
    # Adicionar mini mapa
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    
    # Adicionar escala
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar estações ferroviárias como marcadores (se existissem estações nos dados)
    # Este é um exemplo, que pode ser implementado se tivermos dados de estações
    """
    if data['railway_stations'] is not None:
        stations = data['railway_stations'].to_crs(epsg=4326)
        for idx, row in stations.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=row['name'] if 'name' in row and pd.notna(row['name']) else "Estação ferroviária",
                icon=folium.Icon(color='red', icon='train', prefix='fa'),
                tooltip=row['name'] if 'name' in row and pd.notna(row['name']) else "Estação ferroviária"
            ).add_to(m)
    """
    
    # Adicionar área de estudo (Sorocaba)
    if data['sorocaba'] is not None:
        # Converter para GeoJSON para o Folium
        sorocaba_json = sorocaba.to_json()
        folium.GeoJson(
            data=sorocaba_json,
            name='Área de Estudo',
            style_function=lambda x: {
                'fillColor': '#ffff00',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.1
            }
        ).add_to(m)
    
    # Adicionar rede ferroviária
    if data['railway'] is not None:
        railway = data['railway'].to_crs(epsg=4326)
        railway_json = railway.to_json()
        
        # Verificar se existem colunas relevantes para estilização
        # Poderia ser usado para diferentes tipos de ferrovia, eletrificada/não-eletrificada etc.
        style_column = None
        if 'railway' in railway.columns:
            style_column = 'railway'
        elif 'type' in railway.columns:
            style_column = 'type'
        
        if style_column:
            # Criar função de estilo com base no tipo de ferrovia
            def get_railway_style(feature):
                railway_type = feature['properties'].get(style_column, '')
                
                # Definir cores diferentes para diferentes tipos de ferrovia
                if railway_type == 'rail':
                    color = '#1f78b4'  # Azul escuro para ferrovia principal
                    weight = 3
                elif railway_type == 'tram':
                    color = '#33a02c'  # Verde para bondes
                    weight = 2
                elif railway_type == 'subway':
                    color = '#e31a1c'  # Vermelho para metrô
                    weight = 3
                elif railway_type == 'light_rail':
                    color = '#ff7f00'  # Laranja para VLT
                    weight = 2
                else:
                    color = '#6a3d9a'  # Roxo para outros tipos
                    weight = 2
                
                # Adicionar estilo de linha tracejada para ferrovias não eletrificadas
                if 'electrified' in feature['properties']:
                    if feature['properties']['electrified'] == 'no':
                        dash_array = '5, 5'  # Linha tracejada
                    else:
                        dash_array = None  # Linha contínua
                else:
                    dash_array = None
                
                return {
                    'color': color,
                    'weight': weight,
                    'opacity': 0.8,
                    'dashArray': dash_array
                }
            
            # Criar GeoJSON com estilo baseado no tipo
            folium.GeoJson(
                data=railway_json,
                name='Rede Ferroviária',
                style_function=get_railway_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'railway', 'operator', 'gauge_mm', 'electrified', 'length_km'],
                    aliases=['Nome:', 'Tipo:', 'Operador:', 'Bitola (mm):', 'Eletrificada:', 'Comprimento (km):'],
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
        else:
            # Estilo simples se não houver coluna para diferenciar tipos
            folium.GeoJson(
                data=railway_json,
                name='Rede Ferroviária',
                style_function=lambda x: {
                    'color': '#1f78b4',
                    'weight': 3,
                    'opacity': 0.8
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'length_km', 'sinuosity'] if all(col in railway.columns for col in ['name', 'length_km', 'sinuosity']) else None,
                    aliases=['Nome:', 'Comprimento (km):', 'Sinuosidade:'] if all(col in railway.columns for col in ['name', 'length_km', 'sinuosity']) else None,
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
    
    # Adicionar legenda
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 100px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;
                ">
    <b>Legenda</b><br>
    <i style="background: #ffff00; opacity: 0.3; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo<br>
    <i style="background: none; border: 3px solid #1f78b4; display: inline-block; width: 18px; height: 5px;"></i> Ferrovia<br>
    <i style="background: none; border: 3px dashed #1f78b4; display: inline-block; width: 18px; height: 5px;"></i> Ferrovia Não Eletrificada<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar controle de camadas por último para incluir todas as camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def create_railway_map_with_basemap(data, output_path):
    """Cria um mapa estático de alta qualidade da rede ferroviária com camada base usando Contextily."""
    print("Criando mapa estático da rede ferroviária com camada base...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    if data['sorocaba'] is not None:
        area_bounds = data['sorocaba'].to_crs(epsg=3857).total_bounds
        area = data['sorocaba'].to_crs(epsg=3857)
        area.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8)
    else:
        area_bounds = data['railway'].to_crs(epsg=3857).total_bounds
    
    # Adicionar rede ferroviária com estilo baseado em atributos (se disponíveis)
    railway = data['railway'].to_crs(epsg=3857)
    
    # Verificar se existe coluna de tipo
    if 'railway' in railway.columns:
        # Mapear tipos para cores
        type_colors = {
            'rail': '#1f78b4',      # Azul escuro
            'tram': '#33a02c',      # Verde
            'subway': '#e31a1c',    # Vermelho
            'light_rail': '#ff7f00' # Laranja
        }
        
        # Plotar cada tipo com cores diferentes
        for railway_type, color in type_colors.items():
            subset = railway[railway['railway'] == railway_type]
            if not subset.empty:
                subset.plot(ax=ax, color=color, linewidth=2, alpha=0.8, label=railway_type)
    else:
        # Plotar toda a rede com uma única cor
        railway.plot(ax=ax, color='#1f78b4', linewidth=2, alpha=0.8, label='Ferrovia')
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    
    # Configurar limites do mapa baseados na área de estudo
    ax.set_xlim([area_bounds[0], area_bounds[2]])
    ax.set_ylim([area_bounds[1], area_bounds[3]])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título e legenda
    plt.title('Rede Ferroviária - Sorocaba', fontsize=16, pad=20)
    plt.legend(loc='lower right')
    
    # Adicionar escala (aproximada)
    # Cálculo do comprimento da barra de escala (aproximadamente 5km)
    scale_length = 5000  # metros
    scale_x_start = area_bounds[0] + (area_bounds[2] - area_bounds[0]) * 0.05
    scale_y = area_bounds[1] + (area_bounds[3] - area_bounds[1]) * 0.05
    ax.plot([scale_x_start, scale_x_start + scale_length], [scale_y, scale_y], 'k-', linewidth=2)
    ax.text(scale_x_start + scale_length/2, scale_y + (area_bounds[3] - area_bounds[1]) * 0.01, 
            '5 km', ha='center', va='bottom', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa estático salvo em: {output_path}")

def plot_railway_types_distribution(data, output_path):
    """Plota a distribuição dos tipos de ferrovia."""
    print("Criando gráfico de distribuição dos tipos de ferrovia...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Verificar se a coluna do tipo ferroviário existe
    if 'railway' in data['railway'].columns:
        # Contar frequência de cada tipo
        railway_counts = data['railway']['railway'].value_counts().sort_index()
        
        # Definir cores para o gráfico
        colors = plt.cm.tab10(np.linspace(0, 1, len(railway_counts)))
        
        # Criar barras
        ax = railway_counts.plot(kind='bar', color=colors)
        
        # Adicionar valores acima das barras
        for i, v in enumerate(railway_counts):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # Configurar gráfico
        plt.title('Distribuição dos Tipos de Ferrovia', fontsize=14)
        plt.xlabel('Tipo de Ferrovia', fontsize=12)
        plt.ylabel('Número de Segmentos', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de distribuição dos tipos de ferrovia salvo em: {output_path}")
    else:
        print("Coluna 'railway' não encontrada no dataset")

def plot_railway_length_distribution(data, output_path):
    """Plota a distribuição de comprimentos da rede ferroviária."""
    print("Criando gráfico de distribuição de comprimentos da rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    if 'length_km' not in data['railway'].columns:
        print("Dados de comprimento não disponíveis")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Filtrar valores válidos
    lengths = data['railway']['length_km'].dropna()
    
    # Criar histograma
    sns.histplot(lengths, bins=20, kde=True, color='steelblue')
    
    # Adicionar linha vertical na média
    mean_length = lengths.mean()
    plt.axvline(x=mean_length, color='r', linestyle='--', alpha=0.7)
    plt.text(mean_length + 0.1, plt.ylim()[1] * 0.9, f'Média: {mean_length:.2f} km', 
             color='r', fontweight='bold')
    
    # Configurar gráfico
    plt.title('Distribuição do Comprimento dos Segmentos Ferroviários', fontsize=14)
    plt.xlabel('Comprimento (km)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de distribuição de comprimentos salvo em: {output_path}")

def create_network_analysis(data, output_path):
    """Realiza análise de rede na rede ferroviária usando NetworkX."""
    print("Realizando análise de rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    # Criar grafo a partir dos segmentos ferroviários
    G = nx.Graph()
    
    # Adicionar nós e arestas
    for idx, row in data['railway'].iterrows():
        if isinstance(row.geometry, LineString):
            # Extrair pontos de início e fim como identificadores de nós
            start_point = tuple(row.geometry.coords[0])
            end_point = tuple(row.geometry.coords[-1])
            
            # Adicionar nós com coordenadas
            G.add_node(start_point, pos=start_point)
            G.add_node(end_point, pos=end_point)
            
            # Adicionar aresta com atributos
            attributes = {
                'weight': row['length_km'] if 'length_km' in row and pd.notna(row['length_km']) else 1.0,
                'type': row['railway'] if 'railway' in row and pd.notna(row['railway']) else 'unknown',
                'name': row['name'] if 'name' in row and pd.notna(row['name']) else ''
            }
            
            G.add_edge(start_point, end_point, **attributes)
    
    print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
    
    # Calcular métricas de centralidade
    print("Calculando métricas de centralidade...")
    
    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
    except Exception as e:
        print(f"Erro ao calcular betweenness centrality: {str(e)}")
        betweenness = {node: 0 for node in G.nodes()}
    
    # Closeness centrality
    try:
        closeness = nx.closeness_centrality(G, distance='weight')
    except Exception as e:
        print(f"Erro ao calcular closeness centrality: {str(e)}")
        closeness = {node: 0 for node in G.nodes()}
    
    # Degree centrality
    degree = nx.degree_centrality(G)
    
    # Adicionar métricas aos nós
    nx.set_node_attributes(G, betweenness, 'betweenness')
    nx.set_node_attributes(G, closeness, 'closeness')
    nx.set_node_attributes(G, degree, 'degree')
    
    # Criar visualização da rede
    plt.figure(figsize=(12, 10))
    
    # Obter posições dos nós
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G)
    
    # Desenhar arestas com espessura baseada no tipo
    edge_width = []
    edge_color = []
    for u, v, attrs in G.edges(data=True):
        if attrs.get('type') == 'rail':
            width = 2.0
            color = '#1f78b4'  # Azul para linha principal
        elif attrs.get('type') == 'tram':
            width = 1.5
            color = '#33a02c'  # Verde para bonde
        else:
            width = 1.0
            color = '#6a3d9a'  # Roxo para outros
        
        edge_width.append(width)
        edge_color.append(color)
    
    # Desenhar nós com tamanho baseado em betweenness
    node_size = [5000 * G.nodes[node].get('betweenness', 0) + 20 for node in G.nodes()]
    
    # Colorir nós com base em closeness
    node_color = [G.nodes[node].get('closeness', 0) for node in G.nodes()]
    
    # Desenhar grafo
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color=edge_color, alpha=0.6)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                                  node_color=node_color, cmap=plt.cm.viridis, alpha=0.7)
    
    # Adicionar colorbar
    plt.colorbar(nodes, label='Closeness Centrality')
    
    # Adicionar título
    plt.title('Análise de Rede - Ferrovia', fontsize=16)
    
    # Criar legendas para tipos de aresta
    edge_legend_elements = [
        Line2D([0], [0], color='#1f78b4', lw=2, label='Ferrovia Principal'),
        Line2D([0], [0], color='#33a02c', lw=1.5, label='Bonde/VLT'),
        Line2D([0], [0], color='#6a3d9a', lw=1, label='Outros')
    ]
    
    # Adicionar legenda
    plt.legend(handles=edge_legend_elements, loc='upper right', title='Tipo de Ferrovia')
    
    plt.tight_layout()
    plt.axis('off')
    
    # Salvar figura
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de rede salva em: {output_path}")
    
    return G

def analyze_railway_sinuosity(data, output_path):
    """Analisa e visualiza a sinuosidade da rede ferroviária."""
    print("Analisando sinuosidade da rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    if 'sinuosity' not in data['railway'].columns:
        print("Dados de sinuosidade não disponíveis")
        return
    
    # Filtrar dados válidos
    sinuosity_data = data['railway'].dropna(subset=['sinuosity'])
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Histograma de sinuosidade
    sns.histplot(sinuosity_data['sinuosity'], ax=ax1, bins=20, kde=True, color='darkorange')
    ax1.set_title('Distribuição de Sinuosidade da Rede Ferroviária', fontsize=14)
    ax1.set_xlabel('Índice de Sinuosidade', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar linhas de referência e anotações
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(x=1.2, color='green', linestyle='--', alpha=0.7)
    ax1.text(1.0, ax1.get_ylim()[1]*0.9, 'Linha Reta', rotation=90, color='red', alpha=0.8)
    ax1.text(1.2, ax1.get_ylim()[1]*0.9, 'Sinuoso', rotation=90, color='green', alpha=0.8)
    
    # 2. Mapa de sinuosidade
    railway_proj = data['railway'].to_crs(epsg=3857).copy()
    
    # Criar camada base
    if data['sorocaba'] is not None:
        data['sorocaba'].to_crs(epsg=3857).boundary.plot(ax=ax2, color='black', linewidth=1)
    
    # Plotar ferrovias coloridas por sinuosidade
    railway_proj.plot(column='sinuosity', cmap='plasma', linewidth=2, 
                      legend=True, ax=ax2, 
                      legend_kwds={'label': 'Índice de Sinuosidade', 'orientation': 'vertical'})
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax2, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    
    # Configurar segundo gráfico
    ax2.set_title('Mapa de Sinuosidade da Rede Ferroviária', fontsize=14)
    ax2.set_axis_off()
    
    # Adicionar estatísticas de sinuosidade
    stats = sinuosity_data['sinuosity'].describe()
    stats_text = f"""
    Estatísticas de Sinuosidade:
    - Média: {stats['mean']:.3f}
    - Mediana: {stats['50%']:.3f}
    - Mínimo: {stats['min']:.3f}
    - Máximo: {stats['max']:.3f}
    - Desvio Padrão: {stats['std']:.3f}
    """
    
    plt.figtext(0.5, 0.01, stats_text, ha='center', va='bottom', 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
               fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de sinuosidade salva em: {output_path}")

def create_railway_heatmap(data, output_path):
    """Cria um mapa de calor da rede ferroviária baseado na densidade de linhas."""
    print("Criando mapa de calor da rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    else:
        railway = data['railway'].to_crs(epsg=4326)
        center_lat = railway.geometry.centroid.y.mean()
        center_lon = railway.geometry.centroid.x.mean()
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=13,
                  tiles='CartoDB dark_matter')
    
    # Adicionar mini mapa
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    
    # Gerar pontos para o mapa de calor a partir das linhas ferroviárias
    heat_data = []
    railway = data['railway'].to_crs(epsg=4326)
    
    # Extrair pontos das geometrias para criar o mapa de calor
    for _, row in railway.iterrows():
        if isinstance(row.geometry, LineString):
            # Extrair pontos ao longo da linha
            for i in range(len(row.geometry.coords)):
                # Usar pesos maiores para segmentos principais se tivermos a informação do tipo
                if 'railway' in row and isinstance(row['railway'], str):
                    weight = 3.0 if row['railway'] == 'rail' else 1.0
                else:
                    weight = 1.0
                
                # Adicionar cada ponto com seu peso
                heat_data.append([row.geometry.coords[i][1], 
                                 row.geometry.coords[i][0], 
                                 weight])
    
    # Adicionar mapa de calor
    HeatMap(
        data=heat_data,
        radius=15,
        max_zoom=13,
        blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
    ).add_to(m)
    
    # Adicionar contorno da área de estudo
    if data['sorocaba'] is not None:
        folium.GeoJson(
            data=sorocaba.to_json(),
            name='Área de Estudo',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'white',
                'weight': 2,
                'fillOpacity': 0
            }
        ).add_to(m)
    
    # Adicionar escala
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar título
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Mapa de Calor - Rede Ferroviária</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa de calor ferroviário salvo em: {output_path}")
    return output_path

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para dados ferroviários ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado ferroviário pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo da rede ferroviária
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_ferrovias.html')
    create_railway_network_map(data, interactive_map_path)
    
    # 2. Mapa estático com camada base
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_ferrovias.png')
    create_railway_map_with_basemap(data, static_map_path)
    
    # 3. Distribuição de tipos ferroviários (se existir a coluna)
    types_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_tipos_ferrovias.png')
    plot_railway_types_distribution(data, types_dist_path)
    
    # 4. Distribuição de comprimentos
    length_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_comprimentos_ferrovias.png')
    plot_railway_length_distribution(data, length_dist_path)
    
    # 5. Análise de rede
    network_analysis_path = os.path.join(OUTPUT_DIR, 'analise_rede_ferrovias.png')
    create_network_analysis(data, network_analysis_path)
    
    # 6. Análise de sinuosidade
    sinuosity_path = os.path.join(OUTPUT_DIR, 'analise_sinuosidade_ferrovias.png')
    analyze_railway_sinuosity(data, sinuosity_path)
    
    # 7. Mapa de calor - comentado temporariamente devido a problemas
    # heatmap_path = os.path.join(OUTPUT_DIR, 'mapa_calor_ferrovias.html')
    # create_railway_heatmap(data, heatmap_path)
    print("Nota: O mapa de calor não foi gerado devido a problemas de compatibilidade com a biblioteca.")

if __name__ == "__main__":
    main() 