import os
import requests
import time

# Cria pastas para organizar as imagens
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Pastas para categorias de imagens
base_dir = "images/geoprocessing"
ensure_dir(base_dir)
ensure_dir(f"{base_dir}/hidrografia")
ensure_dir(f"{base_dir}/buildings")
ensure_dir(f"{base_dir}/landuse")
ensure_dir(f"{base_dir}/inmet")
ensure_dir(f"{base_dir}/roads")
ensure_dir(f"{base_dir}/natural_areas")

# Lista de URLs para baixar
image_urls = [
    # Hidrografia
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/hidrografia/analise_rede_hidrografia.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/hidrografia/analise_sinuosidade.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/hidrografia/distribuicao_strahler.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/hidrografia/comprimento_por_strahler.png",
    
    # Buildings
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/buildings/analise_morfologia.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/buildings/analise_altura_edificios.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/buildings/distribuicao_classes_edificios.png",
    
    # Land use
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/landuse/analise_fragmentacao_landuse.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/landuse/mapa_calor_landuse.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/landuse/distribuicao_area_landuse.png",
    
    # INMET
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/inmet/rosa_dos_ventos_static.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/inmet/temperatura_mensal_heatmap.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/inmet/precipitacao_por_estacao.png",
    
    # Roads
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/roads/distribuicao_classes_vias.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/roads/mapa_estatico_vias.png",
    
    # Natural areas
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/natural_areas/histograma_area_ha.png",
    "https://raw.githubusercontent.com/D0mP3dr0/geoprocessing_gnn/main/outputs/visualizations/natural_areas/mapa_coropletico_area_ha.png"
]

# Função para determinar o diretório correto para salvar cada imagem
def get_save_dir(url):
    if "hidrografia" in url:
        return f"{base_dir}/hidrografia"
    elif "buildings" in url:
        return f"{base_dir}/buildings"
    elif "landuse" in url:
        return f"{base_dir}/landuse"
    elif "inmet" in url:
        return f"{base_dir}/inmet"
    elif "roads" in url:
        return f"{base_dir}/roads"
    elif "natural_areas" in url:
        return f"{base_dir}/natural_areas"
    else:
        return base_dir

# Baixa as imagens
for url in image_urls:
    try:
        filename = url.split("/")[-1]
        save_dir = get_save_dir(url)
        save_path = os.path.join(save_dir, filename)
        
        print(f"Baixando {filename}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Imagem salva em {save_path}")
        else:
            print(f"Erro ao baixar {url}: Status code {response.status_code}")
        
        # Pausa breve para não sobrecarregar o servidor
        time.sleep(0.5)
            
    except Exception as e:
        print(f"Erro ao processar {url}: {e}")

print("Download concluído!") 