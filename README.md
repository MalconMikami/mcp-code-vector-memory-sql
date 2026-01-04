# 🧠 Code Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-orange.svg)](https://modelcontextprotocol.io)
[![SQLite](https://img.shields.io/badge/SQLite-Vector-blue.svg)](https://sqlite.org)
[![CPU-Only](https://img.shields.io/badge/Embeddings-CPU%20Only-red.svg)](https://github.com/qdrant/fastembed)

**Servidor MCP de memória híbrida para OpenCode/VS Code com busca vetorial, textual e grafo de conhecimento - tudo local e privado.**

---

## 🎯 Por que Code Memory?

Nascemos da necessidade de ter **memória persistente e inteligente** para assistentes de código, mas com privacidade e controle total. Diferente de outras soluções que dependem de APIs externas ou armazenamento global, o Code Memory oferece:

- **🔒 Privacidade por sessão**: Cada sessão tem seu escopo isolado (session_id obrigatório)
- **🚀 Busca híbrida inteligente**: Vector + FTS5 + re-rank por recência e prioridade
- **🧠 Grafo de conhecimento**: Entidades e relações extraídas automaticamente
- **💭 Resumos locais**: LLM em CPU (GGUF) para resumos sem enviar dados para nuvem
- **⚡ Performance otimizada**: SQLite com pragmas ajustados e cache inteligente
- **🛡️ Segurança built-in**: Filtro automático de conteúdo sensível e deduplicação

---

## 🏆 Arquitetura Única

![Arquitetura Híbrida](https://img.shields.io/badge/Architecture-Hybrid%20Search%20%2B%20Graph%20%2B%20Local%20LLM-brightgreen)

O Code Memory combina **3 camadas de busca** em uma solução unificada:

### 🔍 Busca Vetorial (sqlite-vec)
- Embeddings em CPU com fastembed
- Índice vetorial otimizado com sqlite-vec
- Oversample inteligente para melhor recall

### 📝 Busca Textual (FTS5)
- Índice full-text search com FTS5
- Re-rank híbrido combinando scores
- Suporte a busca exata e fuzzy

### 🕸️ Grafo de Conhecimento
- Entidades extraídas com tree-sitter
- Relações semânticas entre conceitos
- Busca contextual no grafo

### 🤖 Resumos Locais (Opcional)
- LLM local com GGUF (llama-cpp-python)
- Resumos automáticos sem enviar dados para nuvem
- Configurável por variáveis de ambiente

---

## ⚡ Features Principais

### 🔒 Privacidade e Isolamento
- **Session isolation**: `session_id` obrigatório em todas as operações
- **Filtro de sensível**: Detecção automática de API keys, secrets, passwords
- **Deduplicação inteligente**: Hash-based com janela temporal
- **Armazenamento local**: SQLite, zero dependência de serviços externos

### 🧠 Busca Híbrida Avançada
- **Vector search**: Embeddings com fastembed (CPU-only)
- **FTS5 re-rank**: Busca textual com re-rank semântico
- **Recência e prioridade**: Algoritmo de ranking customizável
- **Oversample**: Recuperação inteligente com oversample fator

### 🕸️ Grafo de Conhecimento
- **Extração automática**: Tree-sitter para funções, classes, variáveis
- **Entidades e relações**: Grafo semântico com observações
- **Busca no grafo**: Semantic search sobre entidades
- **Relações customizáveis**: Tipos de relação flexíveis

### 🤖 Inteligência Local
- **Resumos automáticos**: GGUF via llama-cpp-python
- **Tags inteligentes**: Extração heurística de keywords
- **Prioridade dinâmica**: Sistema de prioridades 1-5
- **Metadata flexível**: JSON metadata para contexto extra

### ⚙️ Operação e Observabilidade
- **Health checks**: Endpoints de saúde e diagnósticos
- **Logs estruturados**: Configuráveis por arquivo/diretório
- **Maintenance tools**: Vacuum, prune, purge manuais
- **Metrics internas**: Contadores e estatísticas

---

## 🚀 Quick Start

### Instalação

```bash
# Básico
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate     # Windows
pip install -e .

# Com grafo e resumos locais
pip install -e ".[graph,summary]"

# Desenvolvimento
pip install -e ".[dev]"
```

### Configuração MCP

Exemplo `opencode.json`:

```json
{
  "mcpServers": {
    "code-memory": {
      "command": "python",
      "args": ["-m", "code_memory"],
      "env": {
        "CODE_MEMORY_DB_DIR": "C:/path/to/your/workspace",
        "CODE_MEMORY_LOG_DIR": "C:/Users/you/.cache/code-memory/logs",
        "CODE_MEMORY_ENABLE_GRAPH": "1",
        "CODE_MEMORY_ENABLE_FTS": "1",
        "CODE_MEMORY_ENABLE_VEC": "1"
      }
    }
  }
}
```

### Primeiros Passos

```python
# Lembrar contexto
remember(
    content="Implementei autenticação JWT no UserService com middleware de validação",
    session_id="project-123",
    kind="feature",
    summary="Autenticação JWT adicionada",
    tags="auth,jwt,security",
    priority=1,  # Alta prioridade
    metadata_json={"component": "UserService", "files": ["user.py"]}
)

# Buscar contexto relevante
search_memory(
    query="autenticação JWT",
    session_id="project-123",
    limit=5,
    top_p=0.6
)

# Obter grafo de conhecimento
get_context_graph(
    query="UserService",
    limit=10
)
```

---

## 🎛️ Configuração Avançada

### Modelos e Embeddings

| Variável | Descrição | Padrão |
|---|---|---|
| `CODE_MEMORY_EMBED_MODEL` | Modelo de embedding | `BAAI/bge-small-en-v1.5` |
| `CODE_MEMORY_EMBED_DIM` | Dimensão do embedding | `384` |
| `CODE_MEMORY_MODEL_DIR` | Cache de modelos | `~/.cache/code-memory` |

### Busca e Ranking

| Variável | Descrição | Padrão |
|---|---|---|
| `CODE_MEMORY_TOP_K` | Limite base por busca | `12` |
| `CODE_MEMORY_TOP_P` | Filtro por recência | `0.6` |
| `CODE_MEMORY_OVERSAMPLE_K` | Fator de oversample | `4` |
| `CODE_MEMORY_RECENCY_WEIGHT` | Peso da recência | `0.2` |
| `CODE_MEMORY_PRIORITY_WEIGHT` | Peso da prioridade | `0.15` |
| `CODE_MEMORY_FTS_BONUS` | Bônus FTS | `0.1` |

### Features Toggle

| Variável | Descrição | Padrão |
|---|---|---|
| `CODE_MEMORY_ENABLE_VEC` | Busca vetorial | `1` |
| `CODE_MEMORY_ENABLE_FTS` | Busca textual | `1` |
| `CODE_MEMORY_ENABLE_GRAPH` | Grafo de conhecimento | `0` |

### Resumos Locais (GGUF)

| Variável | Descrição | Padrão |
|---|---|---|
| `CODE_MEMORY_SUMMARY_MODEL` | Caminho do modelo GGUF | `""` |
| `CODE_MEMORY_SUMMARY_CTX` | Context window | `2048` |
| `CODE_MEMORY_SUMMARY_THREADS` | Threads | `4` |
| `CODE_MEMORY_SUMMARY_MAX_TOKENS` | Max tokens | `200` |
| `CODE_MEMORY_SUMMARY_TEMPERATURE` | Temperatura | `0.2` |
| `CODE_MEMORY_AUTO_INSTALL` | Auto-install llama-cpp | `1` |

---

## 🛠️ Ferramentas MCP

### Core Tools
- **`remember(content, session_id, kind, summary, tags, priority, metadata_json)`**
  - Armazena memória com vector + FTS + entidades
  - `session_id` obrigatório
  - Prioridade: 1 (alta) a 5 (baixa)

- **`search_memory(query, session_id, limit, top_p)`**
  - Busca semântica + vector com re-rank FTS
  - `session_id` obrigatório para escopo
  - Ranking por recência e prioridade

### Graph Tools
- **`upsert_entity(name, entity_type, observations_json, memory_id)`**
- **`add_relation(source, target, relation_type, memory_id)`**
- **`get_entity(name)`**
- **`get_context_graph(query, limit)`**

### Management Tools
- **`list_recent(limit)`** - Memórias mais recentes
- **`list_entities(memory_id)`** - Entidades de uma memória
- **`maintenance(action, confirm, session_id, older_than_days)`** - Manutenção manual
- **`health()`** - Health check completo
- **`diagnostics()`** - Diagnósticos detalhados

---

## 📊 Modelo de Dados

### Schema Principal
```sql
-- Tabela principal de memórias
memories (
    id, session_id, kind, content, summary, 
    tags, priority, metadata, hash, created_at
)

-- Vetores para busca semântica
vec_memories (
    rowid, embedding[float384]  -- sqlite-vec
)

-- Índice textual para FTS5
memories_fts (
    content, summary, tags, metadata  -- FTS5
)

-- Entidades extraídas
entities (
    memory_id, entity_type, name, source, path
)
```

### Grafo de Conhecimento (Opcional)
```sql
-- Entidades do grafo
graph_entities (
    id, name, entity_type, created_at
)

-- Observações das entidades
graph_observations (
    entity_id, content, memory_id, created_at
)

-- Relações entre entidades
graph_relations (
    source_id, target_id, relation_type, memory_id
)

-- Vetores das entidades (opcional)
vec_graph_entities (
    rowid, embedding[float384]
)
```

---

## 🏆 Comparação com Alternativas

| Feature | Code Memory | mcp-memory-libsql | @modelcontextprotocol/server-memory |
|---|---|---|---|
| **Armazenamento** | SQLite (local) | libSQL (local/remoto) | JSONL (local) |
| **Busca Vetorial** | ✅ sqlite-vec | ✅ libSQL vector | ❌ |
| **Busca Textual** | ✅ FTS5 + re-rank | ❌ | ❌ |
| **Session Isolation** | ✅ Obrigatório | ❌ Global | ❌ Global |
| **Grafo de Conhecimento** | ✅ Opcional | ✅ Básico | ✅ Básico |
| **Extração de Entidades** | ✅ tree-sitter | ❌ | ❌ |
| **Resumos Locais** | ✅ GGUF (CPU) | ❌ | ❌ |
| **Filtro de Sensível** | ✅ Automático | ❌ | ❌ |
| **Deduplicação** | ✅ Hash-based | ❌ | ❌ |
| **Re-rank Híbrido** | ✅ Vector + FTS | ❌ | ❌ |
| **Configuração** | ✅ 50+ env vars | ✅ Básica | ❌ Mínima |
| **Logs Estruturados** | ✅ Configurável | ❌ | ❌ |
| **Health Checks** | ✅ Completos | ❌ | ❌ |
| **Performance** | ✅ Otimizada | ✅ Boa | ❌ Básica |

> **Nota**: Comparação baseada na análise dos repositórios e documentação pública.

---

## 🎯 Casos de Uso

### Para Desenvolvedores
- **Contexto contínuo**: Lembrar decisões de arquitetura entre sessões
- **Documentação viva**: Auto-documentação de código e decisões
- **Busca inteligente**: Encontrar código relevante por semântica
- **Grafo de conhecimento**: Visualizar relações entre componentes
- **Small Language Models**: Uso de SLMs para resumos e classificação locais
- **Busca Híbrida**: Combinação de Vector + FTS + Graph para máxima precisão

### Para Equipes
- **Conhecimento compartilhado**: Base de conhecimento do projeto
- **Onboarding acelerado**: Novos membros entendem o contexto rapidamente
- **Decisões rastreáveis**: Histórico de decisões e evolução
- **Padrões identificados**: Detectar padrões e boas práticas
- **Memória Corporativa**: Conhecimento acumulado do time

### Para Arquitetura
- **Visão holística**: Entender interdependências do sistema
- **Evolução do código**: Acompanhar mudanças e refatorações
- **Análise de impacto**: Avaliar impacto de mudanças
- **Documentação automática**: Manter docs atualizadas
- **Grafos de Dependências**: Mapeamento automático de relações entre componentes

---

## 🔧 Arquitetura Técnica

### Fluxo do `remember`
1. **Session Resolution**: `session_id` (input → context → env)
2. **Content Filtering**: Detecta e remove conteúdo sensível
3. **Deduplication**: Verifica hash em janela temporal (5min)
4. **Local Summary**: Gera resumo com GGUF (se habilitado)
5. **Auto Tags**: Extrai keywords heurísticas
6. **Embedding**: Gera vetor com fastembed
7. **Storage**: Grava em memories + vec_memories + FTS
8. **Entity Extraction**: Tree-sitter para funções/classes
9. **Graph Update**: Atualiza grafo de conhecimento

### Fluxo do `search_memory`
1. **Query Embedding**: Gera vetor do query
2. **Vector Search**: Recupera candidatos (oversample)
3. **FTS Re-rank**: Aplica bônus de matches textuais
4. **Hybrid Ranking**: Combina distância + recência + prioridade
5. **Top-P Filtering**: Reduz resultados por fator de recência
6. **Session Scoping**: Filtra por session_id

### Fluxo do `get_context_graph`
- **Sem query**: Retorna grafo completo (limitado)
- **Com query**: Busca semântica nas entidades do grafo
- **Relações**: Inclui relações diretas e indiretas

---

## 📈 Performance e Otimizações

### SQLite Optimizations
```sql
PRAGMA journal_mode=WAL;          -- Concurrent reads/writes
PRAGMA synchronous=NORMAL;        -- Balance safety/speed
PRAGMA temp_store=MEMORY;         -- Temp tables in RAM
PRAGMA cache_size=-20000;         -- 20MB cache
PRAGMA mmap_size=268435456;       -- 256MB memory map
PRAGMA page_size=8192;           -- Larger pages
PRAGMA busy_timeout=10000;        -- 10s timeout
```

### Embedding Performance
- **CPU-only**: fastembed otimizado para CPU
- **Model cache**: Cache persistente de modelos
- **Batch processing**: Processamento em lote
- **Lazy loading**: Carregamento sob demanda

### Search Optimizations
- **Oversample**: Recupera 4x candidatos para melhor recall
- **Hybrid ranking**: Algoritmo de ranking customizável
- **Index strategy**: Índices compostos otimizados
- **Query planning**: Planejamento inteligente de queries

---

## 🧪 Testes e Qualidade

### Testes Disponíveis
```bash
# Smoke test (rápido)
python tests/test_memory.py

# Testes completos
pytest

# Testes de integração
pytest tests/integration/

# Coverage
pytest --cov=code_memory
```

### Qualidade do Código
- **Type hints**: Anotações de tipo completas
- **Error handling**: Tratamento robusto de exceções
- **Logging**: Logs estruturados em todos os níveis
- **Validation**: Validação de inputs e sanitização
- **Resource management**: Gerenciamento adequado de conexões

---

## 📚 Documentação Adicional

- **[Arquitetura](docs/ARQUITETURA.md)** - Detalhes técnicos da arquitetura
- **[Configuração](docs/CONFIGURACAO.md)** - Guia completo de configuração
- **[Operação](docs/OPERACAO.md)** - Guia de operação e manutenção
- **[Técnicas de Memória](docs/MEMORY_TECHNIQUES.md)** - Deep dive em técnicas avançadas de memória
- **[Small/Nano Language Models](docs/SMALL_NANO_MODELS.md)** - Guia completo de SLMs e Nano Models
- **[Hybrid Search](docs/HYBRID_SEARCH.md)** - Arquitetura detalhada de busca híbrida
- **[API Reference](docs/API.md)** - Referência completa da API

---

## 🚀 Roadmap

### v0.2 - Próximo Release
- [ ] Multi-language embeddings
- [ ] Advanced graph algorithms
- [ ] Web dashboard
- [ ] Backup/restore tools

### v0.3 - Future
- [ ] Distributed mode
- [ ] Advanced analytics
- [ ] Custom entity extractors
- [ ] Plugin system

---

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma feature branch
3. Faça commit das mudanças
4. Abra um Pull Request

### Development Setup
```bash
git clone https://github.com/MalconMikami/mcp-code-vector-memory-sql
cd mcp-code-vector-memory-sql
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

---

## 📄 Licença

MIT License - ver [LICENSE](LICENSE) para detalhes.

---

## 🙏 Agradecimentos

- **FastMCP** - Framework MCP para Python
- **fastembed** - Embeddings em CPU otimizados
- **sqlite-vec** - Extensão vetorial para SQLite
- **tree-sitter** - Parser para extração de entidades
- **llama-cpp-python** - Runtime para modelos GGUF

---

## 📞 Contato

- **Issues**: [GitHub Issues](https://github.com/MalconMikami/mcp-code-vector-memory-sql/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MalconMikami/mcp-code-vector-memory-sql/discussions)
- **Email**: [malcon.mikami@example.com]

---

<div align="center">

**🧠 Code Memory - Memória Inteligente para Desenvolvedores**

[⭐ Star](https://github.com/MalconMikami/mcp-code-vector-memory-sql) • [🍴 Fork](https://github.com/MalconMikami/mcp-code-vector-memory-sql/fork) • [📖 Docs](docs/)

</div>