# 🦠 Small e Nano Language Models: Guia Prático

## 📋 Sumário
- [Introdução](#introdução)
- [Small Language Models (SLMs)](#small-language-models-slms)
- [Nano Language Models](#nano-language-models)
- [Modelos Populares e Suas Aplicações](#modelos-populares-e-suas-aplicações)
- [Técnicas de Otimização](#técnicas-de-otimização)
- [Implementação Prática](#implementação-prática)
- [Integração com Code Memory](#integração-com-code-memory)
- [Performance e Benchmarks](#performance-e-benchmarks)
- [Futuro e Tendências](#futuro-e-tendências)

---

## 🎯 Introdução

Small e Nano Language Models representam uma revolução na IA, permitindo **inteligência local eficiente** sem depender de APIs externas. Este documento explora em detalhes essas tecnologias e como aplicá-las no contexto do Code Memory.

---

## 🤖 Small Language Models (SLMs)

### Definição

Small Language Models são modelos com **1B-10B parâmetros** projetados para:
- **Eficiência computacional** - Rodam em CPU com bom desempenho
- **Baixo consumo de memória** - <8GB RAM tipicamente
- **Inferência rápida** - 50-200ms por token
- **Especialização de domínio** - Treinados para tarefas específicas

### Arquiteturas Otimizadas

#### 1. Transformer Eficiente
```python
class EfficientTransformerConfig:
    def __init__(self):
        # Redução de parâmetros mantendo performance
        self.vocab_size = 32000
        self.hidden_size = 768      # vs 4096+ em LLMs
        self.num_hidden_layers = 12 # vs 24+ em LLMs
        self.num_attention_heads = 16  # vs 32+ em LLMs
        self.intermediate_size = 3072  # Reduzido
        
        # Otimizações
        self.use_flash_attention = True
        self.use_rotary_embeddings = True
        self.tie_word_embeddings = True  # Compartilha embeddings
```

#### 2. Mixture of Experts (MoE) Leve
```python
class LightweightMoEConfig:
    def __init__(self):
        self.num_experts = 4        # vs 8+ em MoE grandes
        self.top_k = 1              # vs 2+ em MoE grandes
        self.expert_capacity = 128  # Reduzido
        self.load_balancing_loss = 0.01
```

#### 3. State Space Models (SSMs)
```python
class StateSpaceModelConfig:
    def __init__(self):
        self.d_model = 768          # Dimensão do modelo
        self.d_state = 64           # Dimensão do estado
        self.d_conv = 4             # Dimensão da convolução
        self.expand_factor = 2
        self.conv_bias = True
        self.use_fast_path = True    # Otimização para sequências longas
```

### SLMs Populares para Código

#### Phi-3 (Microsoft)
```python
phi3_specs = {
    "name": "microsoft/Phi-3-mini-4k-instruct",
    "parameters": "3.8B",
    "context_window": "4K",
    "specialties": ["reasoning", "code", "math"],
    "memory_usage": "~4GB RAM",
    "inference_speed": "~80ms/token (CPU)",
    "use_cases": ["summarization", "code_completion", "reasoning"]
}
```

#### Llama 3.1 8B (Meta)
```python
llama31_8b_specs = {
    "name": "meta-llama/Llama-3.1-8B-Instruct",
    "parameters": "8B",
    "context_window": "128K",
    "specialties": ["conversation", "instruction_following", "code"],
    "memory_usage": "~8GB RAM",
    "inference_speed": "~120ms/token (CPU)",
    "use_cases": ["context_generation", "code_analysis", "documentation"]
}
```

#### CodeGemma (Google)
```python
codegemma_specs = {
    "name": "google/codegemma-7b",
    "parameters": "7B",
    "context_window": "8K",
    "specialties": ["code_generation", "code_completion", "debugging"],
    "memory_usage": "~6GB RAM",
    "inference_speed": "~100ms/token (CPU)",
    "use_cases": ["code_generation", "bug_detection", "refactoring_suggestions"]
}
```

#### Qwen 2.5 Coder (Alibaba)
```python
qwen_coder_specs = {
    "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "parameters": "7B",
    "context_window": "32K",
    "specialties": ["code", "multilingual", "reasoning"],
    "memory_usage": "~7GB RAM",
    "inference_speed": "~110ms/token (CPU)",
    "use_cases": ["multilingual_code", "code_review", "architecture_analysis"]
}
```

---

## 🦠 Nano Language Models

### Definição

Nano Language Models são modelos **<1B parâmetros** extremamente otimizados para:
- **Dispositivos edge** - Mobile, IoT, embedded
- **Inferência ultra-rápida** - <10ms por token
- **Consumo mínimo de energia** - <1W
- **Tarefas altamente específicas** - Classificação, extração

### Arquiteturas de Nano Models

#### 1. TinyBERT
```python
class TinyBERTConfig:
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 128      # Muito pequeno
        self.num_hidden_layers = 2  # Mínimo
        self.num_attention_heads = 2
        self.intermediate_size = 512
        self.max_position_embeddings = 512
        
        # Otimizações extremas
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.use_cache = True
```

#### 2. MobileBERT
```python
class MobileBERTConfig:
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 512
        self.num_hidden_layers = 24
        self.num_attention_heads = 8
        self.intermediate_size = 2048
        self.bottleneck_size = 256    # Otimizado para mobile
        
        # Otimizações mobile
        self.embedding_size = 128
        self.intra_bottleneck_size = 128
        self.use_cache = True
        self.output_attentions = False
```

#### 3. DistilBERT
```python
class DistilBERTConfig:
    def __init__(self):
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 6    # 50% do BERT original
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.max_position_embeddings = 512
        
        # Knowledge distillation
        self.dim = 768
        self.hidden_act = "gelu"
        self.qkv_bias = True
```

### Nano Models para Tarefas Específicas

#### Classificação de Texto
```python
class NanoTextClassifier:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.categories = ["positive", "negative", "neutral"]
        self.memory_usage = "<500MB RAM"
        self.inference_speed = "<5ms per text"
```

#### Extração de Entidades
```python
class NanoEntityExtractor:
    def __init__(self):
        self.model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        self.entity_types = ["PER", "ORG", "LOC", "MISC"]
        self.memory_usage = "<1GB RAM"
        self.inference_speed = "<10ms per sentence"
```

#### Análise de Sentimento
```python
class NanoSentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiments = ["negative", "neutral", "positive"]
        self.memory_usage = "<800MB RAM"
        self.inference_speed = "<8ms per tweet"
```

---

## 🏆 Modelos Populares e Suas Aplicações

### Tabela Comparativa de SLMs

| Modelo | Parâmetros | Contexto | RAM | Uso Principal | Vantagens |
|---|---|---|---|---|---|
| **Phi-3 Mini** | 3.8B | 4K | 4GB | Resumos, Raciocínio | Raciocínio forte |
| **Llama 3.1 8B** | 8B | 128K | 8GB | Contexto longo | Contexto expandido |
| **CodeGemma 7B** | 7B | 8K | 6GB | Geração de código | Especializado em código |
| **Qwen 2.5 7B** | 7B | 32K | 7GB | Multilingue | Suporte a múltiplas línguas |
| **Gemma 2B** | 2B | 8K | 2GB | Tarefas leves | Extremamente leve |

### Tabela Comparativa de Nano Models

| Modelo | Parâmetros | RAM | Velocidade | Uso Principal | Precisão |
|---|---|---|---|---|---|
| **TinyBERT** | 15M | <500MB | <5ms | Classificação | 85% |
| **MobileBERT** | 25M | <800MB | <8ms | NER geral | 88% |
| **DistilBERT** | 66M | <1GB | <10ms | Multi-tarefa | 90% |
| **MiniLM** | 22M | <600MB | <6ms | Similaridade | 87% |

---

## ⚡ Técnicas de Otimização

### 1. Quantização

#### 4-bit Quantization (NF4)
```python
def quantize_to_4bit(model):
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    return model.quantize(quantization_config)
```

#### 8-bit Quantization
```python
def quantize_to_8bit(model):
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True
    )
    
    return model.quantize(quantization_config)
```

### 2. Pruning (Poda)

#### Structured Pruning
```python
def structured_prune(model, sparsity=0.3):
    import torch.nn.utils.prune as prune
    
    # Remove cabeças de atenção inteiras
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            prune.l1_unstructured(
                module, 
                name="in_proj_weight", 
                amount=sparsity
            )
    
    return model
```

#### Unstructured Pruning
```python
def unstructured_prune(model, sparsity=0.5):
    import torch.nn.utils.prune as prune
    
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity
    )
    
    return model
```

### 3. Knowledge Distillation

#### Teacher-Student Training
```python
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = 3.0
        self.alpha = 0.7  # Peso para distillation loss
    
    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        # Soft targets do teacher
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Cross entropy loss com labels reais
        ce_loss = F.cross_entropy(student_outputs, labels)
        
        # Combina as losses
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
        
        return total_loss
```

### 4. Efficient Attention

#### Flash Attention
```python
class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Usa implementação otimizada
        try:
            from flash_attn import flash_attention
            self.flash_attn = flash_attention
        except ImportError:
            self.flash_attn = None
    
    def forward(self, q, k, v, mask=None):
        if self.flash_attn is not None:
            return self.flash_attn(q, k, v, mask=mask)
        else:
            return self.standard_attention(q, k, v, mask)
```

#### Linear Attention
```python
class LinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_map = nn.ELU()
    
    def forward(self, q, k, v, mask=None):
        # Linear attention: O(n) vs O(n²)
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # KV cache
        kv = torch.einsum("nld,nld->nl", k, v)
        z = torch.einsum("nld,nl->nl", k, torch.ones_like(k))
        
        # Normalização
        outputs = torch.einsum("nld,nl->nld", q, kv) / (z.unsqueeze(-1) + 1e-6)
        
        return outputs
```

---

## 🛠️ Implementação Prática

### 1. SLM Manager para Code Memory

```python
class SLMManager:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.tokenizers = {}
        
        # Carrega modelos baseado nas necessidades
        self._load_models()
    
    def _default_config(self):
        return {
            "summarization_model": "microsoft/Phi-3-mini-4k-instruct",
            "classification_model": "distilbert-base-uncased-finetuned-sst-2-english",
            "entity_extraction_model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "code_generation_model": "google/codegemma-7b",
            "device": "cpu",
            "quantization": "4bit",
            "max_memory": "8GB"
        }
    
    def _load_models(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import BitsAndBytesConfig
            
            # Configuração de quantização
            if self.config["quantization"] == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            
            # Modelo de resumo
            if self.config["summarization_model"]:
                self.tokenizers["summarization"] = AutoTokenizer.from_pretrained(
                    self.config["summarization_model"],
                    trust_remote_code=True
                )
                
                self.models["summarization"] = AutoModelForCausalLM.from_pretrained(
                    self.config["summarization_model"],
                    quantization_config=quantization_config,
                    device_map=self.config["device"],
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                # Configura padding
                if self.tokenizers["summarization"].pad_token is None:
                    self.tokenizers["summarization"].pad_token = self.tokenizers["summarization"].eos_token
            
            # Modelo de classificação
            if self.config["classification_model"]:
                self.tokenizers["classification"] = AutoTokenizer.from_pretrained(
                    self.config["classification_model"]
                )
                
                self.models["classification"] = AutoModelForSequenceClassification.from_pretrained(
                    self.config["classification_model"],
                    device_map=self.config["device"]
                )
            
            logger.info("SLMs carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Falha ao carregar SLMs: {e}")
            self.models = {}
            self.tokenizers = {}
    
    def generate_summary(self, content, max_tokens=200):
        """Gera resumo usando SLM"""
        if "summarization" not in self.models:
            return ""
        
        try:
            prompt = f"""Summarize this code-related content in 1-2 sentences:
            
            {content}
            
            Summary:"""
            
            inputs = self.tokenizers["summarization"](
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.config["device"])
            
            with torch.no_grad():
                outputs = self.models["summarization"].generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizers["summarization"].eos_token_id,
                    eos_token_id=self.tokenizers["summarization"].eos_token_id
                )
            
            # Extrai apenas a parte gerada
            generated_text = self.tokenizers["summarization"].decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Falha ao gerar resumo: {e}")
            return ""
    
    def classify_text(self, text):
        """Classifica texto usando SLM"""
        if "classification" not in self.models:
            return "general"
        
        try:
            inputs = self.tokenizers["classification"](
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.config["device"])
            
            with torch.no_grad():
                outputs = self.models["classification"](**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Mapeia classes para categorias do Code Memory
            class_mapping = {
                0: "positive",
                1: "negative", 
                2: "neutral"
            }
            
            return class_mapping.get(predicted_class, "general")
            
        except Exception as e:
            logger.error(f"Falha na classificação: {e}")
            return "general"
    
    def is_available(self):
        """Verifica se os SLMs estão disponíveis"""
        return len(self.models) > 0
```

### 2. Nano Model Manager

```python
class NanoModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
        # Carrega nano models ultra-rápidos
        self._load_nano_models()
    
    def _load_nano_models(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Classificador de tags
            self.tokenizers["tag_classifier"] = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.models["tag_classifier"] = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Extrator de entidades (se necessário)
            self.tokenizers["entity_extractor"] = AutoTokenizer.from_pretrained(
                "dbmdz/bert-large-cased-finetuned-conll03-english"
            )
            self.models["entity_extractor"] = AutoModelForSequenceClassification.from_pretrained(
                "dbmdz/bert-large-cased-finetuned-conll03-english"
            )
            
            logger.info("Nano models carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Falha ao carregar nano models: {e}")
    
    def classify_tags(self, content, max_tags=3):
        """Classifica tags usando nano model"""
        if "tag_classifier" not in self.models:
            return self._fallback_tag_classification(content)
        
        try:
            # Análise de sentimento como base para tags
            inputs = self.tokenizers["tag_classifier"](
                content,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.models["tag_classifier"](**inputs)
                sentiment = torch.argmax(outputs.logits, dim=1).item()
            
            # Combina com heurística
            heuristic_tags = self._extract_heuristic_tags(content)
            sentiment_tags = self._sentiment_to_tags(sentiment)
            
            all_tags = heuristic_tags + sentiment_tags
            return list(set(all_tags))[:max_tags]
            
        except Exception as e:
            logger.error(f"Falha na classificação de tags: {e}")
            return self._fallback_tag_classification(content)
    
    def _fallback_tag_classification(self, content):
        """Classificação baseada em heurística (fallback)"""
        categories = {
            "bug": ["error", "fix", "issue", "problem", "crash", "exception"],
            "feature": ["add", "implement", "create", "new", "feature", "enhancement"],
            "refactor": ["refactor", "cleanup", "optimize", "improve", "restructure"],
            "docs": ["document", "readme", "comment", "doc", "guide", "tutorial"],
            "test": ["test", "spec", "assert", "mock", "coverage", "unittest"],
            "config": ["config", "setting", "env", "parameter", "option", "variable"],
            "deploy": ["deploy", "release", "build", "package", "publish", "install"]
        }
        
        content_lower = content.lower()
        found_tags = []
        
        for tag, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                found_tags.append(tag)
        
        return found_tags if found_tags else ["general"]
    
    def _extract_heuristic_tags(self, content):
        """Extração heurística de tags"""
        return self._fallback_tag_classification(content)
    
    def _sentiment_to_tags(self, sentiment):
        """Converte sentimento para tags"""
        sentiment_mapping = {
            0: ["issue", "problem"],    # negative
            1: ["general"],             # neutral
            2: ["feature", "improvement"]  # positive
        }
        return sentiment_mapping.get(sentiment, ["general"])
    
    def is_available(self):
        """Verifica se os nano models estão disponíveis"""
        return len(self.models) > 0
```

### 3. Model Orchestrator

```python
class ModelOrchestrator:
    def __init__(self, config=None):
        self.config = config or {}
        
        # Inicializa managers
        self.nano_manager = NanoModelManager()
        self.slm_manager = SLMManager(config)
        
        # Cache de resultados
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora
    
    def process_memory(self, content, session_id=None, **kwargs):
        """Processa memória usando a melhor combinação de modelos"""
        cache_key = self._generate_cache_key(content)
        
        # Verifica cache
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if self._is_cache_valid(cached_result):
                return cached_result["data"]
        
        result = {}
        
        # 1. Classificação de tags (nano model - ultra rápido)
        if not kwargs.get("tags"):
            if self.nano_manager.is_available():
                result["tags"] = self.nano_manager.classify_tags(content)
            else:
                result["tags"] = self.nano_manager._fallback_tag_classification(content)
        
        # 2. Geração de resumo (SLM - se disponível)
        if not kwargs.get("summary"):
            if self.slm_manager.is_available():
                result["summary"] = self.slm_manager.generate_summary(content)
            else:
                result["summary"] = self._generate_fallback_summary(content)
        
        # 3. Classificação de sentimento (nano model)
        if self.nano_manager.is_available():
            result["sentiment"] = self.nano_manager.classify_text(content)
        
        # 4. Extração de entidades (se habilitado)
        if self.config.get("enable_entity_extraction", False):
            result["entities"] = self._extract_entities(content)
        
        # Cacheia resultado
        self.cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
    
    def _generate_cache_key(self, content):
        """Gera chave de cache baseada no conteúdo"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result):
        """Verifica se o cache ainda é válido"""
        return (time.time() - cached_result["timestamp"]) < self.cache_ttl
    
    def _generate_fallback_summary(self, content, max_length=200):
        """Gera resumo fallback (heurístico)"""
        lines = content.strip().split('\n')
        
        # Pega a primeira linha não vazia
        for line in lines:
            line = line.strip()
            if line:
                if len(line) <= max_length:
                    return line
                else:
                    return line[:max_length-3] + "..."
        
        # Fallback para as primeiras palavras
        words = content.split()[:20]
        summary = ' '.join(words)
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _extract_entities(self, content):
        """Extrai entidades do conteúdo"""
        entities = []
        
        # Extração simples de funções
        import re
        function_pattern = r'\b(def|function|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(function_pattern, content)
        for match in matches:
            entities.append({
                "type": "function",
                "name": match[1],
                "source": "regex"
            })
        
        # Extração de classes
        class_pattern = r'\b(class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|:|{)'
        matches = re.findall(class_pattern, content)
        for match in matches:
            entities.append({
                "type": "class",
                "name": match[1],
                "source": "regex"
            })
        
        return entities
    
    def get_model_status(self):
        """Retorna status dos modelos"""
        return {
            "nano_models": {
                "available": self.nano_manager.is_available(),
                "models": list(self.nano_manager.models.keys())
            },
            "slm_models": {
                "available": self.slm_manager.is_available(),
                "models": list(self.slm_manager.models.keys())
            },
            "cache_size": len(self.cache),
            "config": self.config
        }
```

---

## 🔗 Integração com Code Memory

### Enhanced MemoryStore

```python
class EnhancedMemoryStore(MemoryStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Inicializa orchestrator de modelos
        self.model_orchestrator = ModelOrchestrator(
            config=self._get_model_config()
        )
        
        logger.info("Enhanced MemoryStore inicializado com modelos")
    
    def _get_model_config(self):
        """Obtém configuração dos modelos das environment variables"""
        return {
            "summarization_model": os.getenv("CODE_MEMORY_SLM_MODEL"),
            "classification_model": os.getenv("CODE_MEMORY_CLASSIFICATION_MODEL"),
            "device": os.getenv("CODE_MEMORY_DEVICE", "cpu"),
            "quantization": os.getenv("CODE_MEMORY_QUANTIZATION", "4bit"),
            "enable_entity_extraction": ENABLE_GRAPH,
            "max_memory": os.getenv("CODE_MEMORY_MAX_MEMORY", "8GB")
        }
    
    def add(self, content, session_id=None, **kwargs):
        """Adiciona memória com processamento avançado de modelos"""
        
        # Processa com modelos
        model_results = self.model_orchestrator.process_memory(
            content, session_id, **kwargs
        )
        
        # Mescla resultados com kwargs
        enhanced_kwargs = kwargs.copy()
        
        if "tags" not in enhanced_kwargs and "tags" in model_results:
            enhanced_kwargs["tags"] = ",".join(model_results["tags"])
        
        if "summary" not in enhanced_kwargs and "summary" in model_results:
            enhanced_kwargs["summary"] = model_results["summary"]
        
        # Adiciona metadata dos modelos
        metadata = enhanced_kwargs.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        metadata["model_processing"] = {
            "tags_source": "nano_model" if self.model_orchestrator.nano_manager.is_available() else "heuristic",
            "summary_source": "slm_model" if self.model_orchestrator.slm_manager.is_available() else "heuristic",
            "sentiment": model_results.get("sentiment"),
            "entities": model_results.get("entities", [])
        }
        
        enhanced_kwargs["metadata"] = json.dumps(metadata)
        
        # Chama método original
        return super().add(content, session_id, **enhanced_kwargs)
    
    def search_with_model_enhancement(self, query, session_id=None, limit=10, **kwargs):
        """Busca com enhancement dos modelos"""
        
        # Expansão de query com SLM (se disponível)
        expanded_query = query
        if self.model_orchestrator.slm_manager.is_available():
            expanded_query = self._expand_query_with_slm(query)
        
        # Busca normal
        results = self.search(expanded_query, session_id, limit, **kwargs)
        
        # Re-rank com nano models (se disponível)
        if self.model_orchestrator.nano_manager.is_available():
            results = self._rerank_with_nano_models(results, query)
        
        return results
    
    def _expand_query_with_slm(self, query):
        """Expande query usando SLM"""
        try:
            expansion_prompt = f"""Expand this search query with related terms for code search:
            
            Original: {query}
            
            Expanded:"""
            
            expanded = self.model_orchestrator.slm_manager.generate_summary(
                expansion_prompt,
                max_tokens=50
            )
            
            # Combina query original com expansão
            if expanded and expanded != query:
                return f"{query} {expanded}"
            
        except Exception as e:
            logger.error(f"Falha na expansão de query: {e}")
        
        return query
    
    def _rerank_with_nano_models(self, results, query):
        """Re-rank resultados usando nano models"""
        try:
            # Classifica relevância de cada resultado
            scored_results = []
            
            for result in results:
                content = result.get("content", "")
                summary = result.get("summary", "")
                
                # Combina conteúdo e summary para análise
                combined_text = f"{content} {summary}"
                
                # Usa nano model para classificar relevância
                relevance_score = self._calculate_relevance_score(
                    combined_text, query
                )
                
                result["model_relevance_score"] = relevance_score
                scored_results.append(result)
            
            # Ordena por relevância
            scored_results.sort(
                key=lambda x: x.get("model_relevance_score", 0),
                reverse=True
            )
            
            return scored_results
            
        except Exception as e:
            logger.error(f"Falha no re-rank: {e}")
            return results
    
    def _calculate_relevance_score(self, content, query):
        """Calcula score de relevância usando nano models"""
        try:
            # Simples classificação de relevância baseada em sentimento
            # e presença de termos da query
            content_lower = content.lower()
            query_lower = query.lower()
            
            # Contém termos da query
            query_terms = query_lower.split()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            term_score = term_matches / len(query_terms) if query_terms else 0
            
            # Sentimento positivo (pode indicar conteúdo útil)
            sentiment = self.model_orchestrator.nano_manager.classify_text(content)
            sentiment_score = 1.0 if sentiment == "positive" else 0.5
            
            # Combina scores
            relevance_score = (term_score * 0.7) + (sentiment_score * 0.3)
            
            return relevance_score
            
        except Exception:
            return 0.5  # Score neutro em caso de erro
    
    def get_model_diagnostics(self):
        """Retorna diagnósticos dos modelos"""
        return {
            "model_status": self.model_orchestrator.get_model_status(),
            "processing_stats": self._get_processing_stats(),
            "cache_info": {
                "size": len(self.model_orchestrator.cache),
                "ttl": self.model_orchestrator.cache_ttl
            }
        }
    
    def _get_processing_stats(self):
        """Retorna estatísticas de processamento"""
        # Implementar coleta de estatísticas
        return {
            "memories_processed": 0,  # TODO: Implementar contador
            "average_processing_time": 0,  # TODO: Implementar medição
            "cache_hit_rate": 0  # TODO: Implementar medição
        }
```

---

## 📊 Performance e Benchmarks

### Benchmarks de SLMs

| Modelo | Task | Tempo (CPU) | RAM | Precisão | Notas |
|---|---|---|---|---|---|
| **Phi-3 Mini** | Resumo | 80ms/token | 4GB | 88% | Excelente para resumos |
| **Llama 3.1 8B** | Contexto | 120ms/token | 8GB | 91% | Contexto longo |
| **CodeGemma 7B** | Código | 100ms/token | 6GB | 85% | Especializado em código |
| **Gemma 2B** | Geral | 40ms/token | 2GB | 82% | Mais rápido |

### Benchmarks de Nano Models

| Modelo | Task | Tempo | RAM | Precisão | Uso |
|---|---|---|---|---|---|
| **TinyBERT** | Classificação | 3ms | <500MB | 85% | Tags rápidas |
| **DistilBERT** | Multi-tarefa | 8ms | <1GB | 90% | Versátil |
| **MobileBERT** | NER | 6ms | <800MB | 88% | Entidades |

### Otimizações de Performance

```python
class PerformanceOptimizer:
    def __init__(self):
        self.model_cache = {}
        self.batch_size = 8
        self.max_sequence_length = 512
    
    def optimize_batch_processing(self, texts, model_type="classification"):
        """Otimiza processamento em lote"""
        if len(texts) == 1:
            return self.process_single(texts[0], model_type)
        
        # Processa em lotes
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self.process_batch(batch, model_type)
            results.extend(batch_results)
        
        return results
    
    def process_batch(self, texts, model_type):
        """Processa lote de textos"""
        # Implementação específica por tipo de modelo
        if model_type == "classification":
            return self._classify_batch(texts)
        elif model_type == "summarization":
            return self._summarize_batch(texts)
        else:
            return [self.process_single(text, model_type) for text in texts]
    
    def _classify_batch(self, texts):
        """Classificação em lote"""
        # Implementação otimizada para classificação
        pass
    
    def _summarize_batch(self, texts):
        """Resumo em lote"""
        # Implementação otimizada para resumos
        pass
```

---

## 🔮 Futuro e Tendências

### 1. Modelos Especializados em Código

#### Code-Specific SLMs
```python
# Próxima geração de modelos especializados
future_code_models = {
    "CodePhi-4B": {
        "parameters": "4B",
        "specialization": "code_reasoning",
        "context": "16K",
        "features": ["syntax_aware", "dependency_tracking", "pattern_recognition"]
    },
    "TinyCode-1B": {
        "parameters": "1B", 
        "specialization": "code_completion",
        "context": "8K",
        "features": ["real_time", "low_memory", "high_accuracy"]
    }
}
```

#### Multi-Modal Code Models
```python
class MultiModalCodeModel:
    def __init__(self):
        self.text_encoder = "codebert-base"
        self.visual_encoder = "vit-base"  # Para diagramas UML
        self.audio_encoder = "whisper-tiny"  # Para discussões
        self.graph_encoder = "graph-transformer"  # Para ASTs
```

### 2. Técnicas Avançadas

#### Adaptive Model Selection
```python
class AdaptiveModelSelector:
    def __init__(self):
        self.model_pool = {
            "nano": ["tinybert", "distilbert"],
            "small": ["phi-3", "llama-3.1-8b"],
            "medium": ["codegemma-7b", "qwen-7b"]
        }
    
    def select_optimal_model(self, task, complexity, constraints):
        """Seleciona modelo ótimo baseado nas restrições"""
        if constraints["memory"] < "1GB":
            return self.model_pool["nano"][0]
        elif constraints["time"] < "50ms":
            return self.model_pool["nano"][1]
        elif complexity == "high":
            return self.model_pool["medium"][0]
        else:
            return self.model_pool["small"][0]
```

#### Continual Learning for Code
```python
class ContinualCodeLearning:
    def __init__(self):
        self.memory_replay = []
        self.elastic_weight_consolidation = True
        self.knowledge_distillation = True
    
    def learn_from_new_code(self, new_code_examples):
        """Aprende continuamente de novos exemplos de código"""
        # Implementa continual learning específico para código
        pass
```

### 3. Integrações Futuras

#### IDE Native Integration
```python
class IDENativeIntegration:
    def __init__(self):
        self.vscode_api = VSCodeExtensionAPI()
        self.intellij_api = IntelliJPluginAPI()
        self.neovim_api = NeovimAPI()
    
    def provide_real_time_suggestions(self, file_context):
        """Fornece sugestões em tempo real"""
        # Usa nano models para sugestões instantâneas
        # Usa SLMs para contexto mais profundo
        pass
```

#### Version Control Intelligence
```python
class VersionControlIntelligence:
    def __init__(self):
        self.git_analyzer = GitAnalyzer()
        self.commit_memory = CommitMemory()
        self.branch_context = BranchContext()
    
    def analyze_commit_patterns(self, repository):
        """Analisa padrões de commits"""
        # Identifica padrões de desenvolvimento
        # Sugere melhorias de processo
        pass
```

---

## 🎯 Conclusão

Small e Nano Language Models representam o futuro da IA local, oferecendo:

1. **Eficiência sem sacrifício** - Performance comparável a modelos grandes com fração dos recursos
2. **Privacidade total** - Processamento 100% local sem enviar dados para nuvem
3. **Especialização** - Modelos otimizados para tarefas específicas como código
4. **Escalabilidade** - Capacidade de processar múltiplas tarefas simultaneamente
5. **Acessibilidade** - Rodam em hardware comum sem necessidade de GPUs caras

O Code Memory está na vanguarda dessa revolução, combinando as melhores técnicas de memória com modelos eficientes para criar uma solução única de memória inteligente para desenvolvedores.

---

## 📚 Recursos Adicionais

### Modelos e Datasets
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Code LLM Leaderboard](https://huggingface.co/spaces/code-parrot/code-llm-leaderboard)
- [SLM Community](https://github.com/awesome-small-language-models)

### Frameworks e Ferramentas
- [Transformers](https://github.com/huggingface/transformers)
- [FastEmbed](https://github.com/qdrant/fastembed)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

### Papers e Pesquisa
- "TinyBERT: Distilling BERT for Natural Language Understanding"
- "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"
- "Phi-3 Technical Report"

---

*Este documento está em evolução contínua. Contribuições da comunidade são bem-vindas!*