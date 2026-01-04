# 🧠 Técnicas Avançadas de Memória e Small Language Models

## 📋 Sumário
- [Introdução](#introdução)
- [Técnicas de Memória em Sistemas de IA](#técnicas-de-memória-em-sistemas-de-ia)
- [Small Language Models (SLMs)](#small-language-models-slms)
- [Nano Language Models](#nano-language-models)
- [Integração com Code Memory](#integração-com-code-memory)
- [Implementações Práticas](#implementações-práticas)
- [Comparativo de Abordagens](#comparativo-de-abordagens)
- [Futuro e Tendências](#futuro-e-tendências)

---

## 🎯 Introdução

O Code Memory implementa uma arquitetura de memória híbrida que combina as melhores técnicas de memória de sistemas de IA com a eficiência de Small Language Models (SLMs) e Nano Language Models. Este documento explora em detalhes as técnicas fundamentais e como elas se aplicam ao contexto de desenvolvimento de software.

---

## 🧠 Técnicas de Memória em Sistemas de IA

### 1. Arquitetura de Memória Hierárquica

#### Memória de Curto Prazo (Working Memory)
```python
# Context window atual do LLM
working_memory = {
    "current_context": [],  # ~4K-32K tokens
    "recent_interactions": [],  # Últimas 10-20 interações
    "active_variables": [],  # Variáveis em escopo
}
```

#### Memória de Médio Prazo (Episodic Memory)
```python
# Implementado pelo Code Memory
episodic_memory = {
    "session_memories": [],  # Por session_id
    "recent_context": [],  # Últimas 100-500 interações
    "semantic_index": {},  # Índice vetorial
}
```

#### Memória de Longo Prazo (Semantic Memory)
```python
# Grafo de conhecimento do Code Memory
semantic_memory = {
    "knowledge_graph": {},  # Entidades e relações
    "code_patterns": {},  # Padrões de código
    "project_wisdom": {},  # Decisões arquitetônicas
}
```

### 2. Técnicas de Retrieval-Augmented Generation (RAG)

#### RAG Clássico
```
Query → Vector Search → Context → LLM → Response
```

#### RAG Híbrido (Implementado pelo Code Memory)
```
Query → [Vector Search + FTS + Graph] → Context Ranking → LLM → Response
```

#### RAG Multi-Hop
```python
def multi_hop_retrieval(query, max_hops=3):
    context = []
    current_query = query
    
    for hop in range(max_hops):
        results = search_memory(current_query)
        context.extend(results)
        
        # Gera próxima query baseada nos resultados
        if hop < max_hops - 1:
            current_query = generate_followup_query(results)
    
    return context
```

### 3. Técnicas de Memória Associativa

#### Memória Espacial (Spatial Memory)
```python
# Mapeamento de arquivos e diretórios
spatial_memory = {
    "file_structure": {},
    "code_locations": {},
    "dependency_graph": {},
}
```

#### Memória Temporal (Temporal Memory)
```python
# Implementado pelo ranking do Code Memory
temporal_memory = {
    "recency_weight": 0.2,
    "decay_function": "exponential",
    "time_windows": ["1h", "1d", "1w", "1m"],
}
```

#### Memória Episódica (Episodic Memory)
```python
# Session isolation do Code Memory
episodic_memory = {
    "session_boundaries": {},
    "context_continuity": {},
    "narrative_coherence": {},
}
```

### 4. Técnicas de Consolidation

#### Memory Consolidation
```python
def consolidate_memories(session_id):
    # Identifica padrões
    patterns = extract_patterns(session_id)
    
    # Gera resumos com SLM
    summaries = generate_summaries(patterns)
    
    # Atualiza grafo de conhecimento
    update_knowledge_graph(summaries)
    
    # Remove redundâncias
    prune_redundant_memories(session_id)
```

#### Forgetting Curve Management
```python
def apply_forgetting_curve(memory, age_days):
    # Curva de esquecimento de Ebbinghaus
    retention_rate = math.exp(-age_days / 10.0)
    
    # Ajusta prioridade baseado na retenção
    adjusted_priority = memory.priority * retention_rate
    
    return max(1, adjusted_priority)
```

---

## 🤖 Small Language Models (SLMs)

### Definição e Características

Small Language Models são modelos com **1B-10B parâmetros** otimizados para:
- **Eficiência computacional** (CPU-friendly)
- **Baixo consumo de memória** (<8GB RAM)
- **Inferência rápida** (<100ms por token)
- **Especialização de domínio**

### Arquiteturas Comuns de SLMs

#### 1. Transformer Eficiente
```python
class EfficientTransformer:
    def __init__(self):
        self.attention_heads = 16  # vs 32+ em LLMs
        self.hidden_size = 768     # vs 4096+ em LLMs
        self.num_layers = 12       # vs 24+ em LLMs
        self.intermediate_size = 3072  # Reduzido
```

#### 2. Mixture of Experts (MoE) Leve
```python
class LightweightMoE:
    def __init__(self):
        self.num_experts = 4       # vs 8+ em MoE grandes
        self.top_k = 1             # vs 2+ em MoE grandes
        self.expert_capacity = 128  # Reduzido
```

#### 3. State Space Models (SSMs)
```python
class StateSpaceModel:
    def __init__(self):
        self.state_dim = 64        # Compacto
        self.sequence_length = 4096  # Eficiente para sequências longas
        self.complexity = O(n)      # Linear vs O(n²) em Transformers
```

### SLMs Populares e Suas Aplicações

#### Phi-3 (Microsoft)
- **Parâmetros**: 3.8B
- **Context Window**: 4K-128K
- **Especialidade**: Raciocínio e código
- **Uso no Code Memory**: Resumos e tags

#### Llama 3.1 8B (Meta)
- **Parâmetros**: 8B
- **Context Window**: 128K
- **Especialidade**: Conversação e instruções
- **Uso no Code Memory**: Geração de contexto

#### Gemma 2B (Google)
- **Parâmetros**: 2B
- **Context Window**: 8K
- **Especialidade**: Eficiência e portabilidade
- **Uso no Code Memory**: Processamento leve

#### Qwen 2.5 7B (Alibaba)
- **Parâmetros**: 7B
- **Context Window**: 32K-128K
- **Especialidade**: Multilingue e código
- **Uso no Code Memory**: Suporte a múltiplas linguagens

### Técnicas de Otimização para SLMs

#### 1. Quantização
```python
def quantize_model(model, bits=4):
    # Quantização para 4 bits
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint4
    )
    return quantized
```

#### 2. Pruning
```python
def prune_model(model, sparsity=0.3):
    # Remove 30% dos pesos menos importantes
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=sparsity
    )
```

#### 3. Knowledge Distillation
```python
def distill_knowledge(teacher_model, student_model, data):
    # Treina SLM com conhecimento de LLM
    for batch in data:
        with torch.no_grad():
            teacher_outputs = teacher_model(batch)
        
        student_outputs = student_model(batch)
        
        # KL divergence loss
        loss = F.kl_div(
            F.log_softmax(student_outputs, dim=1),
            F.softmax(teacher_outputs, dim=1),
            reduction='batchmean'
        )
        
        loss.backward()
```

---

## 🦠 Nano Language Models

### Definição e Características

Nano Language Models são modelos **<1B parâmetros** extremamente otimizados para:
- **Dispositivos edge** (mobile, IoT)
- **Inferência em tempo real** (<10ms)
- **Consumo mínimo de energia** (<1W)
- **Tarefas específicas** e altamente especializadas

### Arquiteturas de Nano Models

#### 1. TinyBERT
```python
class TinyBERT:
    def __init__(self):
        self.hidden_size = 128     # Muito pequeno
        self.num_layers = 2        # Mínimo
        self.attention_heads = 2   # Reduzido
        self.vocab_size = 30522    # Vocabulário padrão
```

#### 2. MobileBERT
```python
class MobileBERT:
    def __init__(self):
        self.hidden_size = 512
        self.num_layers = 24
        self.attention_heads = 8
        self.bottleneck_size = 256  # Otimizado para mobile
```

#### 3. DistilBERT
```python
class DistilBERT:
    def __init__(self):
        self.hidden_size = 768
        self.num_layers = 6        # 50% do BERT original
        self.attention_heads = 12
        self.distillation_loss = True
```

### Aplicações de Nano Models no Code Memory

#### 1. Classificação Rápida de Tags
```python
class NanoTagger:
    def __init__(self):
        self.model = load_nano_model("distilbert-base-uncased")
        self.categories = ["bug", "feature", "refactor", "docs", "test"]
    
    def classify(self, text):
        outputs = self.model(text)
        predicted_class = torch.argmax(outputs.logits, dim=1)
        return self.categories[predicted_class]
```

#### 2. Detecção de Entidades
```python
class NanoEntityExtractor:
    def __init__(self):
        self.model = load_nano_model("dbmdz/bert-large-cased-finetuned-conll03-english")
    
    def extract_entities(self, code):
        entities = []
        tokens = self.model.tokenizer(code)
        outputs = self.model(tokens)
        
        for token, label in zip(tokens, outputs.labels):
            if label != "O":
                entities.append({"token": token, "type": label})
        
        return entities
```

#### 3. Geração de Resumos Ultra-Rápidos
```python
class NanoSummarizer:
    def __init__(self):
        self.model = load_nano_model("t5-small")
        self.max_length = 50
    
    def summarize(self, text):
        input_ids = self.model.tokenizer(
            f"summarize: {text}", 
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=2,
            early_stopping=True
        )
        
        return self.model.tokenizer.decode(outputs[0])
```

---

## 🔗 Integração com Code Memory

### Arquitetura Híbrida de Modelos

```python
class CodeMemoryModelOrchestrator:
    def __init__(self):
        # Nano models para tarefas rápidas
        self.nano_tagger = NanoTagger()
        self.nano_entity_extractor = NanoEntityExtractor()
        
        # SLM para tarefas médias
        self.slm_summarizer = SLMModel("phi-3-mini-4k-instruct")
        self.slm_context_generator = SLMModel("llama-3.1-8b-instruct")
        
        # LLM remoto (opcional) para tarefas complexas
        self.llm_client = None  # OpenAI/Anthropic/etc
    
    def process_memory(self, content, session_id):
        # 1. Extração rápida com nano models
        tags = self.nano_tagger.classify(content)
        entities = self.nano_entity_extractor.extract_entities(content)
        
        # 2. Resumo com SLM
        summary = self.slm_summarizer.summarize(content)
        
        # 3. Geração de contexto (se necessário)
        if self.needs_context_expansion(content):
            context = self.slm_context_generator.generate_context(content)
        else:
            context = ""
        
        return {
            "tags": tags,
            "entities": entities,
            "summary": summary,
            "context": context
        }
```

### Pipeline de Processamento Otimizado

```python
class OptimizedMemoryPipeline:
    def __init__(self):
        self.models = {
            "nano": self.load_nano_models(),
            "slm": self.load_slm_models(),
            "llm": self.load_llm_models()  # Opcional
        }
        
        self.cache = ModelCache()
        self.scheduler = TaskScheduler()
    
    def process_batch(self, memories):
        # Processamento em lote otimizado
        tasks = []
        
        for memory in memories:
            # Agenda tarefas baseado na complexidade
            if self.is_simple(memory):
                tasks.append(("nano", memory))
            elif self.is_medium(memory):
                tasks.append(("slm", memory))
            else:
                tasks.append(("llm", memory))
        
        # Executa em paralelo quando possível
        results = self.scheduler.execute_parallel(tasks)
        
        return results
```

### Estratégias de Cache e Reuso

```python
class ModelCache:
    def __init__(self):
        self.embedding_cache = LRUCache(maxsize=1000)
        self.summary_cache = LRUCache(maxsize=500)
        self.tag_cache = LRUCache(maxsize=2000)
    
    def get_cached_embedding(self, text):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Gera e cacheia
        embedding = self.generate_embedding(text)
        self.embedding_cache[cache_key] = embedding
        
        return embedding
```

---

## 🛠️ Implementações Práticas

### 1. Configuração de SLMs para Resumos

```python
# Configuração no pyproject.toml
[project.optional-dependencies]
slm = [
    "transformers>=4.35.0",
    "torch>=2.0.0",
    "accelerate>=0.20.0",
    "bitsandbytes>=0.41.0"  # Para quantização
]

# Variáveis de ambiente
CODE_MEMORY_SLM_MODEL="microsoft/Phi-3-mini-4k-instruct"
CODE_MEMORY_SLM_QUANTIZE="4bit"
CODE_MEMORY_SLM_DEVICE="cpu"
CODE_MEMORY_SLM_MAX_TOKENS=200
CODE_MEMORY_SLM_TEMPERATURE=0.2
```

### 2. Implementação de SLM Manager

```python
class SLMManager:
    def __init__(self):
        self.model_name = os.getenv("CODE_MEMORY_SLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        self.device = os.getenv("CODE_MEMORY_SLM_DEVICE", "cpu")
        self.quantize = os.getenv("CODE_MEMORY_SLM_QUANTIZE", "4bit")
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import BitsAndBytesConfig
            
            # Configuração de quantização
            if self.quantize == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            
            # Carrega tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Carrega modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Configura padding
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"SLM carregado: {self.model_name} ({self.quantize})")
            
        except Exception as e:
            logger.error(f"Falha ao carregar SLM: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_summary(self, content, max_tokens=200):
        if not self.model or not self.tokenizer:
            return ""
        
        try:
            prompt = f"""Summarize this code-related content in 1-2 sentences:
            
            {content}
            
            Summary:"""
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Extrai apenas a parte gerada
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Falha ao gerar resumo com SLM: {e}")
            return ""
    
    def is_available(self):
        return self.model is not None and self.tokenizer is not None
```

### 3. Nano Model para Classificação de Tags

```python
class NanoTagClassifier:
    def __init__(self):
        self.categories = {
            "bug": ["error", "fix", "issue", "problem", "crash"],
            "feature": ["add", "implement", "create", "new", "feature"],
            "refactor": ["refactor", "cleanup", "optimize", "improve", "restructure"],
            "docs": ["document", "readme", "comment", "doc", "guide"],
            "test": ["test", "spec", "assert", "mock", "coverage"],
            "config": ["config", "setting", "env", "parameter", "option"],
            "deploy": ["deploy", "release", "build", "package", "publish"]
        }
    
    def classify(self, content):
        content_lower = content.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            scores[category] = score
        
        # Retorna a categoria com maior score, ou "general"
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "general"
    
    def classify_multiple(self, content, max_tags=3):
        content_lower = content.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            scores[category] = score
        
        # Retorna top N categorias
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, score in sorted_categories if score > 0][:max_tags]
```

### 4. Integração com MemoryStore

```python
class EnhancedMemoryStore(MemoryStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Inicializa modelos
        self.slm_manager = SLMManager()
        self.nano_classifier = NanoTagClassifier()
        
        # Cache de embeddings
        self.embedding_cache = {}
    
    def add(self, content, session_id=None, **kwargs):
        # 1. Classificação de tags com nano model
        if not kwargs.get("tags"):
            tags = self.nano_classifier.classify_multiple(content)
            kwargs["tags"] = ",".join(tags)
        
        # 2. Geração de resumo com SLM (se disponível)
        if not kwargs.get("summary") and self.slm_manager.is_available():
            summary = self.slm_manager.generate_summary(content)
            kwargs["summary"] = summary
        
        # 3. Processamento normal
        return super().add(content, session_id, **kwargs)
    
    def search_with_context_expansion(self, query, session_id=None, limit=10):
        # Busca inicial
        results = self.search(query, session_id, limit)
        
        # Expansão de contexto com SLM (se necessário)
        if len(results) < limit and self.slm_manager.is_available():
            expanded_query = self.slm_manager.expand_query(query)
            additional_results = self.search(expanded_query, session_id, limit - len(results))
            results.extend(additional_results)
        
        return results[:limit]
```

---

## 📊 Comparativo de Abordagens

### Tabela Comparativa de Modelos

| Característica | Nano Models (<1B) | Small Models (1-10B) | Large Models (10B+) |
|---|---|---|---|
| **Memória RAM** | <2GB | 4-16GB | 32GB+ |
| **Inferência** | <10ms | 50-200ms | 500ms+ |
| **Precisão** | 70-85% | 85-95% | 95-99% |
| **Especialização** | Alta | Média | Baixa |
| **Custo** | Mínimo | Baixo | Alto |
| **Privacidade** | Total | Total | Parcial |
| **Uso Ideal** | Classificação, extração | Resumos, contexto | Geração complexa |

### Tabela Comparativa de Técnicas de Memória

| Técnica | Vantagens | Desvantagens | Uso no Code Memory |
|---|---|---|---|
| **RAG Simples** | Fácil implementação | Contexto limitado | Base da busca vetorial |
| **RAG Híbrido** | Alta precisão | Complexidade média | ✅ Implementado |
| **Memória Episódica** | Contexto rico | Alto overhead | ✅ Session isolation |
| **Grafo de Conhecimento** | Relações semânticas | Manutenção complexa | ✅ Graph entities |
| **Memória Associativa** | Alta relevância | Implementação complexa | 🔄 Parcialmente |
| **Consolidation** | Eficiência a longo prazo | Processamento pesado | 🔄 Planejado |

---

## 🔮 Futuro e Tendências

### 1. Modelos Especializados em Código

#### Code-Specific SLMs
```python
# Modelos futuros especializados em código
class CodeSLM:
    def __init__(self):
        self.model = "codellama-7b-code"  # Especializado em código
        self.vocab_size = 32000  # Vocabulário de código expandido
        self.max_context = 16384  # Contexto longo para arquivos grandes
```

#### Multi-Modal Code Models
```python
class MultiModalCodeModel:
    def __init__(self):
        self.text_encoder = "codebert-base"
        self.visual_encoder = "vit-base"  # Para diagramas
        self.audio_encoder = "wav2vec2"  # Para discussões em áudio
```

### 2. Técnicas Avançadas de Memória

#### Hierarchical Memory Networks
```python
class HierarchicalMemory:
    def __init__(self):
        self.level_0 = "working_memory"    # Contexto atual
        self.level_1 = "episodic_memory"   # Sessão atual
        self.level_2 = "semantic_memory"   # Projeto inteiro
        self.level_3 = "procedural_memory" # Padrões e best practices
```

#### Continual Learning
```python
class ContinualMemory:
    def __init__(self):
        self.memory_replay = []
        self.elastic_weight_consolidation = True
        self.knowledge_distillation = True
```

### 3. Otimizações de Performance

#### Dynamic Model Loading
```python
class DynamicModelLoader:
    def __init__(self):
        self.model_pool = {}
        self.usage_stats = {}
    
    def load_model_on_demand(self, model_name, task_complexity):
        if task_complexity == "low":
            return self.load_nano_model(model_name)
        elif task_complexity == "medium":
            return self.load_slm_model(model_name)
        else:
            return self.load_llm_model(model_name)
```

#### Adaptive Quantization
```python
class AdaptiveQuantization:
    def __init__(self):
        self.precision_levels = ["fp32", "fp16", "int8", "int4"]
    
    def adjust_precision(self, model, task_requirements):
        if task_requirements.speed_critical:
            return self.quantize_to_int4(model)
        elif task_requirements.quality_critical:
            return self.keep_fp32(model)
        else:
            return self.quantize_to_int8(model)
```

### 4. Integração com Ferramentas de Desenvolvimento

#### IDE Integration
```python
class IDEMemoryIntegration:
    def __init__(self):
        self.vscode_extension = VSCodeExtension()
        self.intellij_plugin = IntelliJPlugin()
        self.neovim_plugin = NeovimPlugin()
    
    def provide_contextual_suggestions(self, file_path, cursor_position):
        context = self.get_relevant_context(file_path, cursor_position)
        suggestions = self.generate_suggestions(context)
        return suggestions
```

#### Version Control Integration
```python
class VCMemoryIntegration:
    def __init__(self):
        self.git_analyzer = GitAnalyzer()
        self.commit_memory = CommitMemory()
    
    def remember_commit_context(self, commit_hash):
        files_changed = self.git_analyzer.get_changed_files(commit_hash)
        commit_message = self.git_analyzer.get_commit_message(commit_hash)
        
        for file_path in files_changed:
            content = self.git_analyzer.get_file_content(commit_hash, file_path)
            self.remember(
                content=content,
                context=f"Commit {commit_hash}: {commit_message}",
                file_path=file_path
            )
```

---

## 🎯 Conclusão

O Code Memory representa o estado da arte em memória para assistentes de código, combinando:

1. **Técnicas avançadas de memória** - RAG híbrido, grafos de conhecimento, consolidação
2. **Small Language Models** - Eficiência e privacidade sem sacrificar qualidade
3. **Nano Language Models** - Processamento ultra-rápido para tarefas específicas
4. **Arquitetura otimizada** - Performance, escalabilidade e flexibilidade

A abordagem híbrida permite que o sistema se adapte a diferentes necessidades, desde classificações rápidas com nano models até geração de contexto complexa com SLMs, mantendo tudo local e privado.

Esta arquitetura posiciona o Code Memory como uma solução única no mercado, oferecendo o melhor dos dois mundos: a inteligência de modelos grandes com a eficiência e privacidade de modelos pequenos.

---

## 📚 Referências e Leitura Adicional

### Papers Acadêmicos
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "TinyBERT: Distilling BERT for Natural Language Understanding" (Jiao et al., 2020)

### Modelos e Frameworks
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastEmbed](https://github.com/qdrant/fastembed)
- [SQLite-VEC](https://github.com/asg017/sqlite-vec)

### Comunidades e Discussões
- [RAG Community](https://rag.community/)
- [Small Language Models](https://github.com/awesome-small-language-models)
- [Local AI](https://localai.io/)

---

*Este documento está em evolução contínua. Contribuições e correções são bem-vindas!*