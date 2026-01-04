# 🧠 Hybrid Search: Vector + FTS + Graph

## 📋 Sumário
- [Introdução](#introdução)
- [Fundamentos do Hybrid Search](#fundamentos-do-hybrid-search)
- [Vector Search](#vector-search)
- [Full-Text Search (FTS)](#full-text-search-fts)
- [Graph Search](#graph-search)
- [Algoritmos de Fusão](#algoritmos-de-fusão)
- [Implementação no Code Memory](#implementação-no-code-memory)
- [Otimizações e Performance](#otimizações-e-performance)
- [Casos de Uso](#casos-de-uso)
- [Métricas e Avaliação](#métricas-e-avaliação)

---

## 🎯 Introdução

Hybrid Search é a técnica de combinar múltiplos métodos de busca para obter resultados mais precisos e relevantes. O Code Memory implementa uma arquitetura híbrida sofisticada que combina **Vector Search**, **Full-Text Search (FTS)** e **Graph Search** para fornecer a melhor experiência de recuperação de contexto para desenvolvedores.

---

## 🔍 Fundamentos do Hybrid Search

### Por que Hybrid Search?

Cada técnica de busca tem suas forças e fraquezas:

| Técnica | Forças | Fraquezas | Ideal para |
|---|---|---|---|
| **Vector Search** | Semântica, contexto, similaridade | Exatidão de termos, performance | Significado, conceitos |
| **FTS** | Exatidão de termos, performance | Semântica, contexto | Termos específicos, código |
| **Graph Search** | Relações, contexto estrutural | Escalabilidade, complexidade | Entidades, dependências |

A combinação inteligente dessas técnicas permite superar as limitações individuais.

### Arquitetura Geral

```
Query → [Vector Search] → Candidates₁
     → [FTS Search]   → Candidates₂  
     → [Graph Search]  → Candidates₃
                              ↓
                        [Fusion Algorithm]
                              ↓
                        [Re-ranking]
                              ↓
                        [Final Results]
```

---

## 🔷 Vector Search

### Fundamentos

Vector Search utiliza **embeddings** para representar o significado semântico do texto:

```python
# Exemplo de embedding
text = "Implement JWT authentication"
embedding = [0.1, -0.3, 0.8, ..., 0.2]  # 384 dimensões
```

### Implementação no Code Memory

#### 1. Geração de Embeddings
```python
class VectorEmbedding:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.dimension = 384
    
    def embed(self, text: str) -> List[float]:
        """Gera embedding do texto"""
        embeddings = list(self.model.embed([text]))
        return embeddings[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings em lote"""
        return list(self.model.embed(texts))
```

#### 2. Índice Vetorial com SQLite-VEC
```python
class VectorIndex:
    def __init__(self, db_path: Path, dimension: int = 384):
        self.db_path = db_path
        self.dimension = dimension
        self._setup_vector_table()
    
    def _setup_vector_table(self):
        """Configura tabela vetorial"""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories 
            USING vec0(embedding float[{self.dimension}])
        """)
        
        conn.commit()
        conn.close()
    
    def insert_vector(self, row_id: int, embedding: List[float]):
        """Insere vetor no índice"""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        serialized = struct.pack(f"{len(embedding)}f", *embedding)
        conn.execute(
            "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            (row_id, serialized)
        )
        
        conn.commit()
        conn.close()
    
    def search_vectors(self, query_embedding: List[float], k: int = 10):
        """Busca vetores similares"""
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        serialized_query = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        
        rows = conn.execute(f"""
            SELECT 
                v.rowid,
                v.distance
            FROM vec_memories v
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance
            LIMIT ?
        """, (serialized_query, k, k)).fetchall()
        
        conn.close()
        return rows
```

#### 3. Otimizações de Performance
```python
class OptimizedVectorSearch:
    def __init__(self, vector_index: VectorIndex):
        self.vector_index = vector_index
        self.embedding_cache = LRUCache(maxsize=1000)
        self.oversample_factor = 4
    
    def search_with_oversample(self, query: str, limit: int = 10):
        """Busca com oversample para melhor recall"""
        # Cache de embeddings
        if query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            query_embedding = self.vector_index.embed(query)
            self.embedding_cache[query] = query_embedding
        
        # Oversample: busca mais candidatos que o necessário
        oversample_k = limit * self.oversample_factor
        candidates = self.vector_index.search_vectors(
            query_embedding, 
            k=oversample_k
        )
        
        return candidates
```

---

## 📝 Full-Text Search (FTS)

### Fundamentos

FTS utiliza índices invertidos para busca exata de termos:

```sql
-- Índice FTS5
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    summary, 
    tags,
    metadata
);
```

### Implementação no Code Memory

#### 1. Configuração FTS5
```python
class FTSIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._setup_fts_table()
    
    def _setup_fts_table(self):
        """Configura tabela FTS5"""
        conn = sqlite3.connect(self.db_path)
        
        # Cria tabela FTS
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                summary,
                tags,
                metadata
            )
        """)
        
        # Configura triggers para sincronização automática
        self._setup_fts_triggers(conn)
        
        conn.commit()
        conn.close()
    
    def _setup_fts_triggers(self, conn):
        """Configura triggers para manter FTS sincronizado"""
        # Trigger INSERT
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, summary, tags, metadata)
                VALUES (new.id, new.content, new.summary, new.tags, new.metadata);
            END;
        """)
        
        # Trigger UPDATE
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.id;
                INSERT INTO memories_fts(rowid, content, summary, tags, metadata)
                VALUES (new.id, new.content, new.summary, new.tags, new.metadata);
            END;
        """)
        
        # Trigger DELETE
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.id;
            END;
        """)
```

#### 2. Busca FTS Avançada
```python
class FTSearch:
    def __init__(self, fts_index: FTSIndex):
        self.fts_index = fts_index
    
    def search_fts(self, query: str, session_id: Optional[str] = None, limit: int = 10):
        """Busca usando FTS5"""
        conn = sqlite3.connect(self.fts_index.db_path)
        
        # Query FTS com suporte a session_id
        if session_id:
            sql = """
                SELECT m.id, m.session_id, m.content, m.summary, m.tags, m.priority, m.created_at
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ? 
                AND m.session_id = ?
                ORDER BY rank
                LIMIT ?
            """
            params = (query, session_id, limit)
        else:
            sql = """
                SELECT m.id, m.session_id, m.content, m.summary, m.tags, m.priority, m.created_at
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            params = (query, limit)
        
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def search_with_boosting(self, query: str, boost_fields: List[str] = None):
        """Busca FTS com boosting de campos"""
        if not boost_fields:
            boost_fields = ["content", "summary", "tags"]
        
        # Constrói query com boosting
        boosted_query = []
        for field in boost_fields:
            boosted_query.append(f"{field}:{query}")
        
        final_query = " OR ".join(boosted_query)
        return self.search_fts(final_query)
    
    def _row_to_dict(self, row):
        """Converte row do SQLite para dicionário"""
        return {
            "id": row[0],
            "session_id": row[1],
            "content": row[2],
            "summary": row[3],
            "tags": row[4],
            "priority": row[5],
            "created_at": row[6],
            "score": 0.0,  # Será calculado depois
            "source": "fts"
        }
```

#### 3. Otimizações FTS
```python
class OptimizedFTS:
    def __init__(self, ftsearch: FTSearch):
        self.ftsearch = ftsearch
        self.query_cache = LRUCache(maxsize=500)
    
    def search_with_stemming(self, query: str, **kwargs):
        """Busca com stemming (redução de palavras à raiz)"""
        # Aplica stemming simples
        stemmed_terms = []
        for term in query.split():
            if term.endswith('ing'):
                stemmed_terms.append(term[:-3])
            elif term.endswith('ed'):
                stemmed_terms.append(term[:-2])
            elif term.endswith('s'):
                stemmed_terms.append(term[:-1])
            else:
                stemmed_terms.append(term)
        
        stemmed_query = " OR ".join(stemmed_terms)
        return self.ftsearch.search_fts(stemmed_query, **kwargs)
    
    def search_with_fuzzy(self, query: str, fuzzy_distance: int = 1, **kwargs):
        """Busca com correspondência aproximada"""
        # Implementa fuzzy search usando NEAR
        fuzzy_query = f'NEAR("{query}", {fuzzy_distance})'
        return self.ftsearch.search_fts(fuzzy_query, **kwargs)
```

---

## 🕸️ Graph Search

### Fundamentos

Graph Search explora relações entre entidades no código:

```sql
-- Entidades e relações
CREATE TABLE graph_entities (id, name, entity_type);
CREATE TABLE graph_relations (source_id, target_id, relation_type);
```

### Implementação no Code Memory

#### 1. Estrutura do Grafo
```python
class CodeGraph:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._setup_graph_tables()
    
    def _setup_graph_tables(self):
        """Configura tabelas do grafo"""
        conn = sqlite3.connect(self.db_path)
        
        # Tabela de entidades
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de observações
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                memory_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(entity_id) REFERENCES graph_entities(id) ON DELETE CASCADE
            )
        """)
        
        # Tabela de relações
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                memory_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source_id) REFERENCES graph_entities(id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES graph_entities(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
```

#### 2. Busca no Grafo
```python
class GraphSearch:
    def __init__(self, code_graph: CodeGraph):
        self.code_graph = code_graph
    
    def search_entities(self, query: str, limit: int = 10):
        """Busca entidades por nome"""
        conn = sqlite3.connect(self.code_graph.db_path)
        
        rows = conn.execute("""
            SELECT id, name, entity_type, created_at
            FROM graph_entities
            WHERE name LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", limit)).fetchall()
        
        conn.close()
        
        return [self._entity_row_to_dict(row) for row in rows]
    
    def search_relations(self, entity_name: str, limit: int = 10):
        """Busca relações de uma entidade"""
        conn = sqlite3.connect(self.code_graph.db_path)
        
        rows = conn.execute("""
            SELECT r.relation_type, s.name, t.name
            FROM graph_relations r
            JOIN graph_entities s ON r.source_id = s.id
            JOIN graph_entities t ON r.target_id = t.id
            WHERE s.name = ? OR t.name = ?
            ORDER BY r.created_at DESC
            LIMIT ?
        """, (entity_name, entity_name, limit)).fetchall()
        
        conn.close()
        
        return [self._relation_row_to_dict(row) for row in rows]
    
    def search_contextual(self, query: str, limit: int = 10):
        """Busca contextual no grafo"""
        conn = sqlite3.connect(self.code_graph.db_path)
        
        # Busca entidades relacionadas
        rows = conn.execute("""
            SELECT DISTINCT e.id, e.name, e.entity_type, e.created_at,
                   COUNT(o.id) as observation_count
            FROM graph_entities e
            LEFT JOIN graph_observations o ON e.id = o.entity_id
            WHERE e.name LIKE ? OR e.entity_type LIKE ?
            GROUP BY e.id
            ORDER BY observation_count DESC, e.created_at DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit)).fetchall()
        
        conn.close()
        
        return [self._entity_row_to_dict(row) for row in rows]
    
    def _entity_row_to_dict(self, row):
        """Converte row de entidade para dicionário"""
        return {
            "id": row[0],
            "name": row[1],
            "entity_type": row[2],
            "created_at": row[3],
            "observation_count": row[4] if len(row) > 4 else 0,
            "source": "graph"
        }
    
    def _relation_row_to_dict(self, row):
        """Converte row de relação para dicionário"""
        return {
            "relation_type": row[0],
            "source": row[1],
            "target": row[2],
            "source": "graph"
        }
```

#### 3. Graph Traversal
```python
class GraphTraversal:
    def __init__(self, graph_search: GraphSearch):
        self.graph_search = graph_search
    
    def find_related_entities(self, entity_name: str, max_depth: int = 2, limit: int = 20):
        """Encontra entidades relacionadas (traversal)"""
        visited = set()
        queue = [(entity_name, 0)]
        results = []
        
        while queue and len(results) < limit:
            current_entity, depth = queue.pop(0)
            
            if current_entity in visited or depth > max_depth:
                continue
            
            visited.add(current_entity)
            
            # Busca relações diretas
            relations = self.graph_search.search_relations(current_entity, limit=50)
            
            for relation in relations:
                related_entity = relation["target"] if relation["source"] == current_entity else relation["source"]
                
                if related_entity not in visited:
                    results.append({
                        "entity": related_entity,
                        "relation": relation["relation_type"],
                        "depth": depth + 1,
                        "path": f"{entity_name} -> {relation['relation_type']} -> {related_entity}"
                    })
                    
                    if depth < max_depth:
                        queue.append((related_entity, depth + 1))
        
        return results[:limit]
    
    def find_shortest_path(self, source: str, target: str, max_depth: int = 5):
        """Encontra o caminho mais curto entre duas entidades"""
        from collections import deque
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            if len(path) > max_depth:
                continue
            
            # Busca vizinhos
            relations = self.graph_search.search_relations(current, limit=50)
            
            for relation in relations:
                neighbor = relation["target"] if relation["source"] == current else relation["source"]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return None  # Caminho não encontrado
```

---

## 🔄 Algoritmos de Fusão

### 1. Score Fusion

#### Weighted Score Fusion
```python
class WeightedScoreFusion:
    def __init__(self, weights: Dict[str, float] = None):
        self.default_weights = {
            "vector": 0.5,
            "fts": 0.3,
            "graph": 0.2
        }
        self.weights = weights or self.default_weights
    
    def fuse_results(self, vector_results: List[Dict], 
                    fts_results: List[Dict], 
                    graph_results: List[Dict]) -> List[Dict]:
        """Funde resultados usando pesos"""
        
        # Normaliza scores
        vector_normalized = self._normalize_scores(vector_results, "vector")
        fts_normalized = self._normalize_scores(fts_results, "fts")
        graph_normalized = self._normalize_scores(graph_results, "graph")
        
        # Combina resultados
        combined_results = {}
        
        # Adiciona resultados vector
        for result in vector_normalized:
            result_id = result["id"]
            combined_results[result_id] = result.copy()
            combined_results[result_id]["vector_score"] = result["score"]
            combined_results[result_id]["final_score"] = result["score"] * self.weights["vector"]
        
        # Adiciona resultados FTS
        for result in fts_normalized:
            result_id = result["id"]
            if result_id in combined_results:
                combined_results[result_id]["fts_score"] = result["score"]
                combined_results[result_id]["final_score"] += result["score"] * self.weights["fts"]
                combined_results[result_id]["sources"].append("fts")
            else:
                result_copy = result.copy()
                result_copy["fts_score"] = result["score"]
                result_copy["final_score"] = result["score"] * self.weights["fts"]
                result_copy["sources"] = ["fts"]
                combined_results[result_id] = result_copy
        
        # Adiciona resultados graph
        for result in graph_normalized:
            result_id = result["id"]
            if result_id in combined_results:
                combined_results[result_id]["graph_score"] = result["score"]
                combined_results[result_id]["final_score"] += result["score"] * self.weights["graph"]
                combined_results[result_id]["sources"].append("graph")
            else:
                result_copy = result.copy()
                result_copy["graph_score"] = result["score"]
                result_copy["final_score"] = result["score"] * self.weights["graph"]
                result_copy["sources"] = ["graph"]
                combined_results[result_id] = result_copy
        
        # Ordena por score final
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        return final_results
    
    def _normalize_scores(self, results: List[Dict], source: str):
        """Normaliza scores para [0, 1]"""
        if not results:
            return []
        
        scores = [r.get("score", 0) for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # Todos os scores são iguais
            for result in results:
                result["score"] = 0.5
        else:
            # Normalização linear
            for result in results:
                original_score = result.get("score", 0)
                normalized = (original_score - min_score) / (max_score - min_score)
                result["score"] = normalized
        
        # Adiciona informação da fonte
        for result in results:
            result["sources"] = [source]
        
        return results
```

### 2. Reciprocal Rank Fusion (RRF)

```python
class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k  # Constante para RRF
    
    def fuse_results(self, vector_results: List[Dict], 
                    fts_results: List[Dict], 
                    graph_results: List[Dict]) -> List[Dict]:
        """Funde resultados usando Reciprocal Rank Fusion"""
        
        # Calcula RRF scores
        rrf_scores = {}
        
        # Vector results
        for rank, result in enumerate(vector_results):
            result_id = result["id"]
            rrf_score = 1.0 / (self.k + rank + 1)
            rrf_scores[result_id] = {
                "rrf_score": rrf_score,
                "result": result,
                "sources": ["vector"]
            }
        
        # FTS results
        for rank, result in enumerate(fts_results):
            result_id = result["id"]
            rrf_score = 1.0 / (self.k + rank + 1)
            
            if result_id in rrf_scores:
                rrf_scores[result_id]["rrf_score"] += rrf_score
                rrf_scores[result_id]["sources"].append("fts")
            else:
                rrf_scores[result_id] = {
                    "rrf_score": rrf_score,
                    "result": result,
                    "sources": ["fts"]
                }
        
        # Graph results
        for rank, result in enumerate(graph_results):
            result_id = result["id"]
            rrf_score = 1.0 / (self.k + rank + 1)
            
            if result_id in rrf_scores:
                rrf_scores[result_id]["rrf_score"] += rrf_score
                rrf_scores[result_id]["sources"].append("graph")
            else:
                rrf_scores[result_id] = {
                    "rrf_score": rrf_score,
                    "result": result,
                    "sources": ["graph"]
                }
        
        # Ordena por RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Prepara resultado final
        final_results = []
        for item in sorted_results:
            result = item["result"].copy()
            result["rrf_score"] = item["rrf_score"]
            result["sources"] = item["sources"]
            final_results.append(result)
        
        return final_results
```

### 3. Adaptive Fusion

```python
class AdaptiveFusion:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.weight_calculator = DynamicWeightCalculator()
    
    def fuse_results(self, query: str, vector_results: List[Dict], 
                    fts_results: List[Dict], graph_results: List[Dict]) -> List[Dict]:
        """Funde resultados com pesos adaptativos baseados na query"""
        
        # Analisa características da query
        query_features = self.query_analyzer.analyze(query)
        
        # Calcula pesos dinâmicos
        dynamic_weights = self.weight_calculator.calculate_weights(query_features)
        
        # Usa weighted fusion com pesos dinâmicos
        fusion = WeightedScoreFusion(dynamic_weights)
        return fusion.fuse_results(vector_results, fts_results, graph_results)


class QueryAnalyzer:
    def analyze(self, query: str) -> Dict[str, float]:
        """Analisa características da query"""
        features = {}
        
        # Comprimento da query
        features["length"] = len(query.split())
        
        # Presença de termos técnicos
        technical_terms = ["function", "class", "method", "variable", "api", "database"]
        features["technical_density"] = sum(1 for term in technical_terms if term in query.lower()) / len(technical_terms)
        
        # Presença de aspas (busca exata)
        features["has_quotes"] = 1.0 if '"' in query else 0.0
        
        # Presença de operadores booleanos
        boolean_operators = ["AND", "OR", "NOT"]
        features["boolean_density"] = sum(1 for op in boolean_operators if op.upper() in query.upper()) / len(boolean_operators)
        
        return features


class DynamicWeightCalculator:
    def calculate_weights(self, query_features: Dict[str, float]) -> Dict[str, float]:
        """Calcula pesos dinâmicos baseados nas features da query"""
        
        # Pesos base
        base_weights = {
            "vector": 0.5,
            "fts": 0.3,
            "graph": 0.2
        }
        
        # Ajustes baseados nas features
        weights = base_weights.copy()
        
        # Query curta → mais peso para vector (semântica)
        if query_features["length"] < 3:
            weights["vector"] += 0.2
            weights["fts"] -= 0.1
            weights["graph"] -= 0.1
        
        # Termos técnicos → mais peso para graph
        if query_features["technical_density"] > 0.3:
            weights["graph"] += 0.2
            weights["vector"] -= 0.1
            weights["fts"] -= 0.1
        
        # Aspas → mais peso para FTS (busca exata)
        if query_features["has_quotes"] > 0:
            weights["fts"] += 0.3
            weights["vector"] -= 0.2
            weights["graph"] -= 0.1
        
        # Operadores booleanos → mais peso para FTS
        if query_features["boolean_density"] > 0:
            weights["fts"] += 0.2
            weights["vector"] -= 0.1
            weights["graph"] -= 0.1
        
        # Normaliza pesos
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        return weights
```

---

## 🚀 Implementação no Code Memory

### HybridSearchManager

```python
class HybridSearchManager:
    def __init__(self, memory_store: 'MemoryStore'):
        self.memory_store = memory_store
        
        # Componentes de busca
        self.vector_search = OptimizedVectorSearch(memory_store.vector_index)
        self.ftsearch = OptimizedFTS(FTSearch(memory_store.db_path))
        self.graph_search = GraphSearch(CodeGraph(memory_store.db_path))
        
        # Algoritmos de fusão
        self.weighted_fusion = WeightedScoreFusion()
        self.rrf_fusion = ReciprocalRankFusion()
        self.adaptive_fusion = AdaptiveFusion()
        
        # Configurações
        self.fusion_method = os.getenv("CODE_MEMORY_FUSION_METHOD", "weighted")
        self.enable_graph_search = ENABLE_GRAPH
    
    def search(self, query: str, session_id: Optional[str] = None, 
              limit: int = 10, top_p: float = 0.6, **kwargs) -> List[Dict]:
        """Busca híbrida principal"""
        
        # 1. Busca em cada fonte
        vector_results = self._search_vector(query, session_id, limit)
        fts_results = self._search_fts(query, session_id, limit)
        graph_results = self._search_graph(query, session_id, limit) if self.enable_graph_search else []
        
        # 2. Funde resultados
        if self.fusion_method == "weighted":
            fused_results = self.weighted_fusion.fuse_results(vector_results, fts_results, graph_results)
        elif self.fusion_method == "rrf":
            fused_results = self.rrf_fusion.fuse_results(vector_results, fts_results, graph_results)
        elif self.fusion_method == "adaptive":
            fused_results = self.adaptive_fusion.fuse_results(query, vector_results, fts_results, graph_results)
        else:
            # Fallback para weighted
            fused_results = self.weighted_fusion.fuse_results(vector_results, fts_results, graph_results)
        
        # 3. Aplica filtros e ranking adicional
        filtered_results = self._apply_filters(fused_results, session_id, top_p)
        
        # 4. Limita resultados
        final_results = filtered_results[:limit]
        
        return final_results
    
    def _search_vector(self, query: str, session_id: Optional[str], limit: int) -> List[Dict]:
        """Busca vetorial"""
        try:
            # Busca vetorial com oversample
            vector_candidates = self.vector_search.search_with_oversample(query, limit)
            
            # Recupera dados completos dos candidatos
            results = []
            for row_id, distance in vector_candidates:
                memory_data = self.memory_store._get_memory_by_id(row_id)
                if memory_data and (session_id is None or memory_data["session_id"] == session_id):
                    memory_data["vector_distance"] = distance
                    memory_data["score"] = 1.0 - distance  # Converte distância para score
                    results.append(memory_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca vetorial: {e}")
            return []
    
    def _search_fts(self, query: str, session_id: Optional[str], limit: int) -> List[Dict]:
        """Busca textual"""
        try:
            fts_results = self.ftsearch.search_fts(query, session_id, limit)
            
            # Adiciona scores FTS
            for result in fts_results:
                result["fts_score"] = 1.0  # FTS já retorna ordenado por relevância
                result["score"] = 1.0
            
            return fts_results
            
        except Exception as e:
            logger.error(f"Erro na busca FTS: {e}")
            return []
    
    def _search_graph(self, query: str, session_id: Optional[str], limit: int) -> List[Dict]:
        """Busca no grafo"""
        try:
            # Busca entidades
            graph_results = self.graph_search.search_contextual(query, limit)
            
            # Converte para formato consistente
            results = []
            for entity in graph_results:
                # Cria resultado artificial baseado na entidade
                result = {
                    "id": f"graph_{entity['id']}",
                    "session_id": session_id,
                    "content": f"Entity: {entity['name']} ({entity['entity_type']})",
                    "summary": f"Found in knowledge graph",
                    "tags": entity['entity_type'],
                    "priority": 3,
                    "created_at": entity['created_at'],
                    "graph_score": 1.0,
                    "score": 1.0,
                    "sources": ["graph"]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca no grafo: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict], session_id: Optional[str], top_p: float) -> List[Dict]:
        """Aplica filtros e ranking adicional"""
        
        # Filtra por session_id se especificado
        if session_id:
            results = [r for r in results if r.get("session_id") == session_id]
        
        # Aplica ranking por recência e prioridade
        results = self._apply_recency_priority_ranking(results, top_p)
        
        return results
    
    def _apply_recency_priority_ranking(self, results: List[Dict], top_p: float) -> List[Dict]:
        """Aplica ranking por recência e prioridade"""
        if not results:
            return results
        
        # Constantes de ranking
        recency_weight = float(os.getenv("CODE_MEMORY_RECENCY_WEIGHT", "0.2"))
        priority_weight = float(os.getenv("CODE_MEMORY_PRIORITY_WEIGHT", "0.15"))
        
        # Calcula timestamps
        now = time.time()
        for result in results:
            created_at = result.get("created_at", "")
            if created_at:
                try:
                    timestamp = time.mktime(time.strptime(created_at, "%Y-%m-%d %H:%M:%S"))
                    age_hours = (now - timestamp) / 3600
                except:
                    age_hours = 24 * 30  # Default para 30 dias
            else:
                age_hours = 24 * 30
            
            # Calcula score de recência (mais recente = maior)
            recency_score = max(0, 1.0 - age_hours / (24 * 30))  # Decai em 30 dias
            
            # Calcula score de prioridade
            priority = result.get("priority", 3)
            priority_score = (6 - priority) / 5.0  # Inverte: prioridade 1 = score 1.0
            
            # Combina scores
            base_score = result.get("score", 0.5)
            final_score = (
                base_score * 0.7 +
                recency_score * recency_weight +
                priority_score * priority_weight
            )
            
            result["final_score"] = final_score
            result["recency_score"] = recency_score
            result["priority_score"] = priority_score
        
        # Aplica filtro top_p (recência)
        if top_p < 1.0:
            results.sort(key=lambda x: x.get("recency_score", 0), reverse=True)
            keep_count = max(1, int(len(results) * top_p))
            top_results = results[:keep_count]
            
            # Reordena por score final
            top_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return top_results
        else:
            results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return results
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de busca"""
        return {
            "fusion_method": self.fusion_method,
            "enable_graph_search": self.enable_graph_search,
            "vector_cache_size": len(self.vector_search.embedding_cache),
            "fts_cache_size": len(self.ftsearch.query_cache),
            "components": {
                "vector_search": "available",
                "ftsearch": "available", 
                "graph_search": "available" if self.enable_graph_search else "disabled"
            }
        }
```

---

## ⚡ Otimizações e Performance

### 1. Parallel Search

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ParallelHybridSearch:
    def __init__(self, hybrid_search: HybridSearchManager):
        self.hybrid_search = hybrid_search
        self.max_workers = 3  # Um por componente de busca
    
    def search_parallel(self, query: str, **kwargs) -> List[Dict]:
        """Executa buscas em paralelo"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submete buscas em paralelo
            futures = {
                executor.submit(self.hybrid_search._search_vector, query, **kwargs): "vector",
                executor.submit(self.hybrid_search._search_fts, query, **kwargs): "fts",
                executor.submit(self.hybrid_search._search_graph, query, **kwargs): "graph"
            }
            
            # Coleta resultados
            results = {}
            for future in as_completed(futures):
                search_type = futures[future]
                try:
                    results[search_type] = future.result(timeout=10)
                except Exception as e:
                    logger.error(f"Erro na busca {search_type}: {e}")
                    results[search_type] = []
        
        # Funde resultados
        return self.hybrid_search.weighted_fusion.fuse_results(
            results.get("vector", []),
            results.get("fts", []),
            results.get("graph", [])
        )
```

### 2. Caching Inteligente

```python
class IntelligentCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtém item do cache"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                
                # Verifica TTL
                if time.time() - item["timestamp"] < self.ttl:
                    self.access_times[key] = time.time()
                    return item["data"]
                else:
                    # Remove item expirado
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            return None
    
    def set(self, key: str, data: Any):
        """Armazena item no cache"""
        with self.lock:
            # Remove item mais antigo se necessário
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            # Armazena novo item
            self.cache[key] = {
                "data": data,
                "timestamp": time.time()
            }
            self.access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hit_rate": getattr(self, 'hit_count', 0) / max(getattr(self, 'total_count', 1), 1)
            }


class CachedHybridSearch:
    def __init__(self, hybrid_search: HybridSearchManager):
        self.hybrid_search = hybrid_search
        self.query_cache = IntelligentCache(max_size=500, ttl=1800)  # 30 minutos
        self.result_cache = IntelligentCache(max_size=200, ttl=3600)  # 1 hora
    
    def search(self, query: str, **kwargs) -> List[Dict]:
        """Busca com caching"""
        
        # Gera chave de cache
        cache_key = self._generate_cache_key(query, **kwargs)
        
        # Verifica cache de resultados
        cached_result = self.result_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Executa busca
        results = self.hybrid_search.search(query, **kwargs)
        
        # Cacheia resultado
        self.result_cache.set(cache_key, results)
        
        return results
    
    def _generate_cache_key(self, query: str, **kwargs) -> str:
        """Gera chave de cache"""
        import hashlib
        
        # Normaliza parâmetros
        normalized_kwargs = {k: v for k, v in sorted(kwargs.items()) 
                           if k in ["session_id", "limit", "top_p"]}
        
        # Gera hash
        key_data = f"{query}:{str(normalized_kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

### 3. Query Optimization

```python
class QueryOptimizer:
    def __init__(self):
        self.query_history = {}
        self.performance_stats = {}
    
    def optimize_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Otimiza query para melhor performance"""
        
        # 1. Expansão de query (se necessário)
        expanded_query = self._expand_query(query, context)
        
        # 2. Limpeza de query
        cleaned_query = self._clean_query(expanded_query)
        
        # 3. Balanceamento de termos
        balanced_query = self._balance_terms(cleaned_query)
        
        return balanced_query
    
    def _expand_query(self, query: str, context: Dict[str, Any]) -> str:
        """Expande query com sinônimos e termos relacionados"""
        # Implementa expansão baseada em contexto do projeto
        expansions = {
            "auth": ["authentication", "authorization", "login", "security"],
            "db": ["database", "sql", "query", "table"],
            "api": ["endpoint", "service", "rest", "http"],
            "ui": ["interface", "frontend", "view", "component"],
            "bug": ["error", "issue", "problem", "exception", "fix"]
        }
        
        expanded_terms = []
        for term in query.split():
            expanded_terms.append(term)
            
            # Adiciona expansões se disponíveis
            if term.lower() in expansions:
                expanded_terms.extend(expansions[term.lower()])
        
        return " ".join(expanded_terms)
    
    def _clean_query(self, query: str) -> str:
        """Limpa e normaliza query"""
        import re
        
        # Remove caracteres especiais (exceto aspas para busca exata)
        cleaned = re.sub(r'[^\w\s"]', ' ', query)
        
        # Remove espaços extras
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _balance_terms(self, query: str) -> str:
        """Balancea termos para evitar bias"""
        terms = query.split()
        
        # Se query é muito longa, mantém apenas termos mais importantes
        if len(terms) > 10:
            # Implementa heurística para selecionar termos importantes
            important_terms = []
            for term in terms:
                # Mantém termos técnicos e em caixa alta
                if term.isupper() or self._is_technical_term(term):
                    important_terms.append(term)
            
            # Se ainda não tem termos suficientes, pega os primeiros
            if len(important_terms) < 5:
                important_terms.extend(terms[:5])
            
            return " ".join(important_terms[:10])
        
        return query
    
    def _is_technical_term(self, term: str) -> bool:
        """Verifica se termo é técnico"""
        technical_indicators = [
            "function", "class", "method", "variable", "constant",
            "api", "http", "sql", "json", "xml", "html", "css",
            "react", "vue", "angular", "node", "python", "java"
        ]
        
        return term.lower() in technical_indicators
```

---

## 🎯 Casos de Uso

### 1. Busca por Contexto de Código

```python
# Exemplo: Buscar contexto sobre autenticação
query = "JWT authentication implementation"
results = hybrid_search.search(
    query=query,
    session_id="project-123",
    limit=5
)

# Resultados combinariam:
# - Vector: Memórias semanticamente relacionadas (auth, security, tokens)
# - FTS: Memórias com termos exatos ("JWT", "authentication")
# - Graph: Entidades como UserService, AuthController, TokenMiddleware
```

### 2. Busca por Padrões de Arquitetura

```python
# Exemplo: Buscar padrões de arquitetura
query = "repository pattern data access"
results = hybrid_search.search(
    query=query,
    session_id="project-123",
    limit=8
)

# Resultados combinariam:
# - Vector: Conceitos relacionados (data layer, persistence, ORM)
# - FTS: Termos exatos ("repository", "pattern", "data", "access")
# - Graph: Classes Repository, interfaces IDataAccess, relações "implements"
```

### 3. Busca por Problemas Específicos

```python
# Exemplo: Buscar bugs relacionados
query = "memory leak connection pool"
results = hybrid_search.search(
    query=query,
    session_id="project-123",
    limit=10,
    top_p=0.8  # Priorizar resultados mais recentes
)

# Resultados combinariam:
# - Vector: Problemas relacionados (leaks, resources, connections)
# - FTS: Termos exatos ("memory", "leak", "connection", "pool")
# - Graph: Classes ConnectionPool, ResourceManager, relações "uses"
```

---

## 📊 Métricas e Avaliação

### 1. Métricas de Qualidade

```python
class SearchMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_search_quality(self, query: str, results: List[Dict], 
                               relevant_ids: List[int]) -> Dict[str, float]:
        """Avalia qualidade da busca"""
        
        if not results or not relevant_ids:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        # Calcula métricas
        retrieved_ids = [r.get("id") for r in results]
        relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
        
        precision = len(relevant_retrieved) / len(retrieved_ids)
        recall = len(relevant_retrieved) / len(relevant_ids)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Métricas adicionais
        mrr = self._calculate_mrr(results, relevant_ids)
        ndcg = self._calculate_ndcg(results, relevant_ids)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg,
            "total_results": len(results),
            "relevant_results": len(relevant_retrieved)
        }
    
    def _calculate_mrr(self, results: List[Dict], relevant_ids: List[int]) -> float:
        """Calcula Mean Reciprocal Rank"""
        for i, result in enumerate(results):
            if result.get("id") in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, results: List[Dict], relevant_ids: List[int], k: int = 10) -> float:
        """Calcula Normalized Discounted Cumulative Gain"""
        dcg = 0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
        
        if idcg == 0:
            return 0.0
        
        for i, result in enumerate(results[:k]):
            if result.get("id") in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)
        
        return dcg / idcg
```

### 2. A/B Testing Framework

```python
class SearchABTest:
    def __init__(self):
        self.test_results = {}
    
    def run_comparison_test(self, queries: List[str], 
                           method_a: str, method_b: str,
                           relevant_judgments: Dict[str, List[int]]) -> Dict[str, Any]:
        """Compara dois métodos de busca"""
        
        results_a = []
        results_b = []
        
        metrics = SearchMetrics()
        
        for query in queries:
            # Executa busca com método A
            results_a_method = self._search_with_method(query, method_a)
            quality_a = metrics.evaluate_search_quality(
                query, results_a_method, relevant_judgments.get(query, [])
            )
            results_a.append(quality_a)
            
            # Executa busca com método B
            results_b_method = self._search_with_method(query, method_b)
            quality_b = metrics.evaluate_search_quality(
                query, results_b_method, relevant_judgments.get(query, [])
            )
            results_b.append(quality_b)
        
        # Calcula estatísticas
        stats_a = self._calculate_statistics(results_a)
        stats_b = self._calculate_statistics(results_b)
        
        # Teste de significância
        significance = self._calculate_significance(results_a, results_b)
        
        return {
            "method_a": {
                "name": method_a,
                "stats": stats_a
            },
            "method_b": {
                "name": method_b,
                "stats": stats_b
            },
            "significance": significance,
            "winner": method_a if stats_a["mean_f1"] > stats_b["mean_f1"] else method_b
        }
    
    def _calculate_statistics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calcula estatísticas dos resultados"""
        import numpy as np
        
        metrics = ["precision", "recall", "f1", "mrr", "ndcg"]
        stats = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                stats[f"mean_{metric}"] = np.mean(values)
                stats[f"std_{metric}"] = np.std(values)
                stats[f"min_{metric}"] = np.min(values)
                stats[f"max_{metric}"] = np.max(values)
        
        return stats
```

---

## 🎯 Conclusão

O Hybrid Search do Code Memory representa o estado da arte em recuperação de informação para desenvolvedores, combinando:

1. **Vector Search** - Para similaridade semântica e contexto
2. **FTS** - Para precisão de termos e busca exata
3. **Graph Search** - Para relações estruturais e dependências
4. **Fusion Algorithms** - Para combinar inteligentemente os resultados
5. **Performance Optimizations** - Para busca em tempo real

Essa abordagem híbrida permite que o sistema entenda tanto o **significado** quanto a **forma** do código, fornecendo resultados mais relevantes e contextuais do que qualquer técnica individualmente.

---

## 📚 Referências

### Papers Acadêmicos
- "Learning to Combine Signals for Diverse and Relevant Search Results"
- "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Fusion"
- "A Survey on Graph-based Neural Networks for Text Classification"

### Ferramentas e Frameworks
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [SQLite-VEC](https://github.com/asg017/sqlite-vec)
- [FAISS](https://github.com/facebookresearch/faiss)

### Comunidades
- [Information Retrieval Community](https://ir.community/)
- [Search Engines Community](https://www.searchengines.community/)

---

*Documentação em evolução contínua. Contribuições são bem-vindas!*