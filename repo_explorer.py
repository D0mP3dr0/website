import requests
import base64
import json

# URLs base
repo_owner = "D0mP3dr0"
repo_name = "geoprocessing_gnn"
base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"

def get_readme():
    """Obtém o conteúdo do README.md"""
    url = f"{base_url}/readme"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        content = base64.b64decode(data['content']).decode('utf-8')
        return content
    else:
        return f"Erro ao obter README: {response.status_code}"

def get_directory_contents(path=""):
    """Obtém o conteúdo de um diretório específico"""
    url = f"{base_url}/contents/{path}" if path else f"{base_url}/contents"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Erro ao obter conteúdo do diretório {path}: {response.status_code}"

def explore_directory(path="", level=0):
    """Explora um diretório e seus subdiretórios recursivamente"""
    contents = get_directory_contents(path)
    
    if isinstance(contents, str):  # Se for uma mensagem de erro
        return contents
    
    result = []
    indent = "  " * level
    
    for item in contents:
        if item['type'] == 'dir':
            result.append(f"{indent}📁 {item['name']}/")
            subdir_path = f"{path}/{item['name']}" if path else item['name']
            subdir_contents = explore_directory(subdir_path, level + 1)
            if isinstance(subdir_contents, list):
                result.extend(subdir_contents)
            else:
                result.append(f"{indent}  ⚠️ {subdir_contents}")
        else:
            result.append(f"{indent}📄 {item['name']} ({item['size']} bytes)")
    
    return result

def get_file_content(path):
    """Obtém o conteúdo de um arquivo específico"""
    url = f"{base_url}/contents/{path}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['size'] > 1000000:  # Arquivos muito grandes
            return "Arquivo muito grande para exibir"
        if data['encoding'] == 'base64':
            return base64.b64decode(data['content']).decode('utf-8')
        else:
            return f"Codificação não suportada: {data['encoding']}"
    else:
        return f"Erro ao obter arquivo {path}: {response.status_code}"

def get_requirements():
    """Obtém o conteúdo do arquivo requirements.txt"""
    return get_file_content("requirements.txt")

def main():
    # Obter README
    print("=== README.md ===")
    readme = get_readme()
    print(readme[:500] + "..." if len(readme) > 500 else readme)
    print("\n")
    
    # Estrutura do repositório
    print("=== Estrutura do Repositório ===")
    structure = explore_directory()
    print("\n".join(structure))
    print("\n")
    
    # Requirements
    print("=== Requirements ===")
    reqs = get_requirements()
    print(reqs)
    print("\n")
    
    # Verificar algum arquivo Python importante
    important_files = ["run_workflow.py", "data_analysis.py"]
    for file in important_files:
        print(f"=== Primeiras linhas de {file} ===")
        content = get_file_content(file)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("\n")

if __name__ == "__main__":
    main() 