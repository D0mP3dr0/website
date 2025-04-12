#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para criar mapas interativos para o projeto de geoprocessamento de Sorocaba.
Gera visualizações usando a biblioteca Folium para exibição no site.
"""

import folium
import random

# Coordenadas centrais de Sorocaba
SOROCABA_LAT = -23.5015
SOROCABA_LON = -47.4526

def create_road_network_map():
    """Cria um mapa interativo da rede viária de Sorocaba."""
    # Criar mapa base
    m = folium.Map(location=[SOROCABA_LAT, SOROCABA_LON], zoom_start=12, 
                  tiles='OpenStreetMap')
    
    # Adicionar título ao mapa
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Rede Viária de Sorocaba</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Adicionar um marcador para o centro da cidade
    folium.Marker(
        [SOROCABA_LAT, SOROCABA_LON],
        popup="Centro de Sorocaba",
        tooltip="Centro da Cidade",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    
    # Salvar o mapa como HTML
    m.save("maps/road_network_interactive.html")
    print("Mapa de rede viária criado com sucesso!")

def create_hydrography_map():
    """Cria um mapa interativo da hidrografia de Sorocaba."""
    # Criar mapa base
    m = folium.Map(location=[SOROCABA_LAT, SOROCABA_LON], zoom_start=12, 
                  tiles='CartoDB positron')
    
    # Adicionar título ao mapa
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Hidrografia de Sorocaba</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Adicionar algumas características hídricas simuladas
    # Rio Sorocaba
    folium.PolyLine(
        [[-23.4815, -47.4726], [-23.4915, -47.4626], [-23.5015, -47.4526], 
         [-23.5115, -47.4426], [-23.5215, -47.4326]],
        color="blue",
        weight=5,
        opacity=0.7,
        tooltip="Rio Sorocaba"
    ).add_to(m)
    
    # Adicionar áreas de risco simuladas
    for i in range(5):
        # Gerar coordenadas aleatórias próximas ao centro
        lat = SOROCABA_LAT + random.uniform(-0.02, 0.02)
        lon = SOROCABA_LON + random.uniform(-0.02, 0.02)
        
        # Adicionar um círculo para área de risco
        folium.Circle(
            location=[lat, lon],
            radius=300,  # metros
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.4,
            tooltip=f"Área de Risco {i+1}"
        ).add_to(m)
    
    # Salvar o mapa como HTML
    m.save("maps/hydrography_interactive.html")
    print("Mapa de hidrografia criado com sucesso!")

def create_landuse_map():
    """Cria um mapa interativo do uso do solo de Sorocaba."""
    # Criar mapa base
    m = folium.Map(location=[SOROCABA_LAT, SOROCABA_LON], zoom_start=12, 
                  tiles='CartoDB positron')
    
    # Adicionar título ao mapa
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>Uso do Solo em Sorocaba</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Simular diferentes tipos de uso do solo com polígonos
    land_use_types = [
        {"name": "Urbano", "color": "red"},
        {"name": "Agricultura", "color": "green"},
        {"name": "Industrial", "color": "purple"},
        {"name": "Vegetação Natural", "color": "darkgreen"},
        {"name": "Água", "color": "blue"}
    ]
    
    # Adicionar polígonos de uso do solo (simplificados para exemplo)
    for i, land_use in enumerate(land_use_types):
        # Gerar coordenadas aleatórias para o polígono
        center_lat = SOROCABA_LAT + random.uniform(-0.03, 0.03)
        center_lon = SOROCABA_LON + random.uniform(-0.03, 0.03)
        
        # Criar um polígono simples
        points = []
        for j in range(6):  # hexágono
            angle = j * 60  # 360 / 6
            lat = center_lat + 0.01 * math.sin(math.radians(angle))
            lon = center_lon + 0.01 * math.cos(math.radians(angle))
            points.append([lat, lon])
        
        # Adicionar o polígono ao mapa
        folium.Polygon(
            locations=points,
            color=land_use["color"],
            fill=True,
            fill_color=land_use["color"],
            fill_opacity=0.4,
            tooltip=land_use["name"]
        ).add_to(m)
    
    # Salvar o mapa como HTML
    m.save("maps/landuse_interactive.html")
    print("Mapa de uso do solo criado com sucesso!")

def create_heatmap():
    """Cria um mapa de calor da densidade urbana de Sorocaba."""
    # Criar mapa base
    m = folium.Map(location=[SOROCABA_LAT, SOROCABA_LON], zoom_start=12, 
                  tiles='CartoDB dark_matter')
    
    # Adicionar título ao mapa
    title_html = '''
    <h3 align="center" style="font-size:16px; color: white;"><b>Mapa de Densidade Urbana de Sorocaba</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Gerar pontos aleatórios para o mapa de calor
    heat_data = []
    for _ in range(500):
        # Gerar coordenadas aleatórias com maior concentração no centro
        lat = SOROCABA_LAT + random.normalvariate(0, 0.02)
        lon = SOROCABA_LON + random.normalvariate(0, 0.02)
        heat_data.append([lat, lon])
    
    # Adicionar o mapa de calor
    folium.plugins.HeatMap(heat_data, radius=15).add_to(m)
    
    # Salvar o mapa como HTML
    m.save("maps/density_heatmap.html")
    print("Mapa de calor de densidade criado com sucesso!")

if __name__ == "__main__":
    try:
        import math
        import folium.plugins
        
        print("Criando mapas interativos...")
        create_road_network_map()
        create_hydrography_map()
        create_landuse_map()
        create_heatmap()
        print("Todos os mapas foram criados com sucesso!")
    except ImportError as e:
        print(f"Erro: {e}")
        print("Por favor, instale as bibliotecas necessárias com:")
        print("pip install folium") 