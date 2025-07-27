# Sistema de Pesquisa de Revistas Científicas

## Visão Geral
Este sistema foi desenvolvido para auxiliar na pesquisa de revistas científicas de múltiplas áreas, utilizando dados da base de periódicos da CAPES Sucupira. O sistema combina técnicas de RAG (Retrieval-Augmented Generation) com agentes inteligentes para fornecer informações detalhadas sobre revistas acadêmicas, incluindo seu estrato Qualis, que indica a qualidade da publicação.

## Principais Funcionalidades

- 🔍 Busca semântica em múltiplas áreas científicas  
- 🏷️ Informações sobre o estrato **Qualis** das revistas  
- 🔗 Integração com a **API Crossref** para dados adicionais  
- 🤖 Sistema de agentes para processamento inteligente das consultas  

## Estrutura de Arquivos
.
├── criar_embbendings_chroma.py   # Script para gerar embeddings do dataset
├── main.py                       # Script principal do sistema de agentes
├── rag.py                        # Script de teste do sistema RAG
├── requirements.txt              # Dependências do projeto
├── sucupira_chroma_db/           # Banco de dados Chroma com os embeddings
│   ├── [arquivos do Chroma]
│   └── chroma.sqlite3
├── sucupira.csv                  # Dataset original da CAPES Sucupira
└── testes/                       # Pasta com scripts de teste
    ├── criar_embbendings_csv.py  # Testes de geração de embeddings
    ├── crossref.py               # Testes da API Crossref
    ├── teste[1-6].py             # Diversos scripts de teste


## Pré-requisitos
-Python 3.10
-Conda (recomendado para gerenciamento de ambientes)
-Ollama (para execução local dos modelos LLM)

## Instalação e Configuração:
Crie e ative um ambiente Conda:
```bash
conda create -n sucupira_env python=3.9
conda activate sucupira_env
```
Instale as dependências:
```bash
pip install -r requirements.txt
```
Certifique-se que o Ollama está rodando localmente na porta 11434

## Como Usar

1. **Gerar os embeddings** (necessário na primeira execução):

    ```bash
    python3 criar_embbendings_chroma.py
    ```

2. **Testar o sistema RAG** (opcional):

    ```bash
    python3 rag.py
    ```

3. **Executar o sistema completo de agentes**:

    ```bash
    python3 main.py
    ```

## Personalização:
Para alterar a área de pesquisa ou o modelo LLM utilizado, edite as seguintes variáveis no arquivo main.py:
```python
llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")
area = "Computação e Medicina"
```

## Sobre o Dataset
O sistema utiliza a base da CAPES Sucupira como fonte principal porque:
-Contém o estrato Qualis de cada revista
-Oferece uma avaliação padronizada da qualidade das publicações
-Abrange múltiplas áreas do conhecimento

## Tecnologias Utilizadas
-RAG (Retrieval-Augmented Generation): Para busca semântica usando o modelo paraphrase-MiniLM-L6-v2
-ChromaDB: Para armazenamento e consulta dos embeddings
-Ollama: Para execução local de modelos LLM
-Crossref API: Para obtenção de informações adicionais sobre as revistas

## Testes
A pasta testes/ contém diversos scripts utilizados durante o desenvolvimento para validar diferentes componentes do sistema, incluindo:
-Geração de embeddings
-Integração com a API Crossref
-Testes de funcionalidades específicas

Observações
-O sistema foi otimizado para trabalhar com o modelo Llama3 (3B) via Ollama, mas pode ser adaptado para outros modelos LLM
-A primeira execução pode demorar enquanto os embeddings são gerados e indexados
-Para grandes volumes de pesquisa, recomenda-se verificar os recursos disponíveis na máquina