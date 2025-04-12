import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# Configuração da figura
plt.figure(figsize=(12, 6))
plt.axis('off')
ax = plt.gca()

# Cores
colors = {
    'acquisition': '#3498db',
    'preprocessing': '#2ecc71',
    'analysis': '#e74c3c',
    'visualization': '#9b59b6',
    'modeling': '#f39c12',
    'report': '#1abc9c'
}

# Posições dos blocos
positions = {
    'acquisition': [0.1, 0.5, 0.1, 0.3],  # [x, y, width, height]
    'preprocessing': [0.25, 0.5, 0.1, 0.3],
    'exploration': [0.4, 0.5, 0.1, 0.3],
    'spatial_analysis': [0.55, 0.5, 0.1, 0.3],
    'modeling': [0.7, 0.5, 0.1, 0.3],
    'visualization': [0.85, 0.5, 0.1, 0.3],
}

# Textos
titles = {
    'acquisition': 'Aquisição\nde Dados',
    'preprocessing': 'Pré-\nprocessamento',
    'exploration': 'Análise\nExploratória',
    'spatial_analysis': 'Análise\nEspacial',
    'modeling': 'Modelagem\nde Redes',
    'visualization': 'Visualização\ne Relatórios',
}

# Desenhar blocos
for key, pos in positions.items():
    color = colors.get(key.split('_')[0], '#34495e')  # Obtém cor ou usa padrão
    rect = Rectangle((pos[0], pos[1]), pos[2], pos[3], 
                    facecolor=color, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    plt.text(pos[0] + pos[2]/2, pos[1] + pos[3]/2, titles[key], 
             ha='center', va='center', color='white', fontweight='bold')

# Adicionar setas
for i in range(len(positions) - 1):
    keys = list(positions.keys())
    start = positions[keys[i]]
    end = positions[keys[i+1]]
    
    arrow = FancyArrowPatch(
        (start[0] + start[2], start[1] + start[3]/2),
        (end[0], end[1] + end[3]/2),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2,
        color='gray'
    )
    ax.add_patch(arrow)

# Adicionar título
plt.title('Fluxo de Trabalho - Geoprocessamento com GNN', fontsize=16, fontweight='bold', pad=20)

# Ajustar limites
plt.xlim(0, 1)
plt.ylim(0, 1)

# Adicionar textos informativos na parte inferior
notes = [
    'GeoPandas', 'Shapely', 'NetworkX', 'Rasterio',
    'GeoPandas', 'NetworkX', 'Scikit-learn', 'GNN', 'Folium'
]
for i, note in enumerate(notes):
    pos_x = 0.1 + (i * 0.1)
    if pos_x < 0.95:  # Limitar para não sair da figura
        plt.text(pos_x, 0.3, note, ha='center', va='center', 
                fontsize=8, color='gray', rotation=20)

# Salvar figura
plt.tight_layout()
plt.savefig('images/geoprocessing/workflow_diagram.png', dpi=300, bbox_inches='tight')
print("Diagrama de fluxo de trabalho criado em images/geoprocessing/workflow_diagram.png") 