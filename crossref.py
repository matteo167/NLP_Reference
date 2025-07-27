import requests

def get_journal_info(issn):
    url = f"https://api.crossref.org/journals/{issn}"
    try:
        response = requests.get(url, timeout=10)  # Timeout de 10 segundos
        response.raise_for_status()  # Levanta exceção para erros HTTP
        data = response.json()
        
        # Extrai os campos com tratamento para chaves ausentes
        message = data.get("message", {})
        journal_info = {
            "title": message.get("title", "Não disponível"),
            "publisher": message.get("publisher", "Não disponível"),
            "ISSN": message.get("ISSN", []),
            "counts": {
                "total_dois": message.get("counts", {}).get("total-dois", 0),  # Número total de artigos
                "current_dois": message.get("counts", {}).get("current-dois", 0)  # Artigos ativos
            }
        }
        return journal_info
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Erro na requisição: {str(e)}", "status_code": 500}

# Exemplo de uso
issn = "2236-6695"  # Nature (pode ser com ou sem hífen)
data = get_journal_info(issn)

if "error" not in data:
    print("\n=== METADADOS DA REVISTA ===")
    print(f"Título: {data['title']}")
    print(f"Editora: {data['publisher']}")
    print(f"Artigos registrados: {data['counts']['total_dois']} (ativos: {data['counts']['current_dois']})")
else:
    print(f"Erro: {data['error']}")