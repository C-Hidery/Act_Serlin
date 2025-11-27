# Serlin-Transformer By Ryan Crepa - 增强版（修复词汇表同步）
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import sqlite3
import pickle
import hashlib
from collections import defaultdict, deque, OrderedDict
import datetime
import math
class SerlinConfig:
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        think_steps=3,
        max_length=50
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.think_steps = think_steps
        self.max_length = max_length
serlin_config = SerlinConfig()
class LongTermMemory:
    """长期记忆系统"""
    
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """初始化记忆数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                personality_profile TEXT,
                preferences TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 对话记忆表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                input_text TEXT,
                response_text TEXT,
                sentiment REAL,
                topics TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # 知识记忆表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_text TEXT UNIQUE,
                value_text TEXT,
                confidence REAL,
                source TEXT,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 自我反思表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                reflection_text TEXT,
                improvement_suggestions TEXT,
                quality_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, user_id, input_text, response_text, sentiment, topics):
        """存储对话记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        # 确保用户存在
        cursor.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    
        # 处理情感值
        if isinstance(sentiment, torch.Tensor):
            sentiment_value = sentiment.mean().item()
        else:
            sentiment_value = float(sentiment)
    
        cursor.execute('''
            INSERT INTO conversations (user_id, input_text, response_text, sentiment, topics)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, input_text, response_text, sentiment_value, json.dumps(topics)))
    
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id, limit=10):
        """获取用户对话历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT input_text, response_text, sentiment, topics, timestamp
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'input': row[0],
            'response': row[1],
            'sentiment': row[2],
            'topics': json.loads(row[3]),
            'timestamp': row[4]
        } for row in results]
    
    def store_knowledge(self, key, value, confidence=1.0, source="conversation"):
        """存储知识"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge (key_text, value_text, confidence, source, last_accessed)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (key, value, confidence, source))
        
        conn.commit()
        conn.close()
    
    def retrieve_knowledge(self, key):
        """检索知识"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT value_text, confidence FROM knowledge 
            WHERE key_text = ? AND confidence > 0.5
            ORDER BY confidence DESC
        ''', (key,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.update_knowledge_access(key)
            return result[0], result[1]
        return None, 0.0
    
    def update_knowledge_access(self, key):
        """更新知识访问时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE knowledge SET last_accessed = CURRENT_TIMESTAMP 
            WHERE key_text = ?
        ''', (key,))
        
        conn.commit()
        conn.close()
    
    def store_reflection(self, conversation_id, reflection, suggestions, quality_score):
        """存储自我反思"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reflections (conversation_id, reflection_text, improvement_suggestions, quality_score)
            VALUES (?, ?, ?, ?)
        ''', (conversation_id, reflection, suggestions, quality_score))
        
        conn.commit()
        conn.close()

class KnowledgeBase:
    """知识库系统"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.domain_knowledge = self.load_domain_knowledge()
    
    def load_domain_knowledge(self):
        """加载领域知识"""
        return {
            "technology": {
                "python": "Python是一种高级编程语言，以简洁易读著称",
                "ai": "人工智能是计算机科学的一个分支，致力于创造智能机器",
                "machine learning": "机器学习是AI的子领域，让计算机从数据中学习"
            },
            "entertainment": {
                "movies": "电影是一种重要的娱乐形式",
                "music": "音乐可以表达情感和创造氛围",
                "games": "游戏可以提供娱乐和挑战"
            }
        }
    
    def query_knowledge(self, query, domain=None):
        """查询知识"""
        # 先从长期记忆查询
        knowledge, confidence = self.memory.retrieve_knowledge(query)
        if knowledge:
            return knowledge, confidence
        
        # 从领域知识查询
        if domain and domain in self.domain_knowledge:
            if query in self.domain_knowledge[domain]:
                return self.domain_knowledge[domain][query], 0.8
        
        # 从通用领域知识查询
        for domain_knowledge in self.domain_knowledge.values():
            if query in domain_knowledge:
                return domain_knowledge[query], 0.7
        
        return None, 0.0
    
    def learn_from_conversation(self, user_input, response):
        """从对话中学习新知识"""
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:
                if word in response.lower():
                    self.memory.store_knowledge(word, response, 0.6, "conversation_learning")

class MultiTurnContext:
    """多轮对话上下文管理"""
    
    def __init__(self, max_context_length=10):
        self.max_context_length = max_context_length
        self.conversation_context = deque(maxlen=max_context_length)
        self.current_topics = set()
    
    def add_turn(self, user_input, ai_response, sentiment, extracted_topics):
        """添加一轮对话到上下文"""
        turn = {
            'user_input': user_input,
            'ai_response': ai_response,
            'sentiment': sentiment,
            'topics': extracted_topics,
            'timestamp': datetime.datetime.now()
        }
        self.conversation_context.append(turn)
        
        # 更新当前话题
        self.current_topics.update(extracted_topics)
        # 限制话题数量
        if len(self.current_topics) > serlin_config.nhead:
            self.current_topics = set(list(self.current_topics)[-serlin_config.nhead:])
    
    def get_context_text(self):
        """获取上下文文本"""
        if not self.conversation_context:
            return ""
        
        context_texts = []
        for turn in list(self.conversation_context)[-serlin_config.think_steps:]:
            context_texts.extend([turn['user_input'], turn['ai_response']])
        
        return " ".join(context_texts)
    
    def get_recent_topics(self):
        """获取最近的话题"""
        return list(self.current_topics)[-5:]

class PersonalityAdaptation:
    """个性化适应系统"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.user_profiles = {}
    
    def get_user_profile(self, user_id):
        """获取用户个性画像"""
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT personality_profile, preferences FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return json.loads(result[0]), json.loads(result[1])
        else:
            # 默认个性画像
            default_profile = {
                "formality": 0.5,
                "humor_level": 0.3,
                "detail_level": 0.6,
                "empathy_level": 0.7,
                "curiosity_level": 0.5
            }
            default_prefs = {
                "preferred_topics": [],
                "avoided_topics": [],
                "communication_style": "balanced"
            }
            return default_profile, default_prefs
    
    def update_user_profile(self, user_id, user_input, response, sentiment):
        """基于对话更新用户画像"""
        profile, prefs = self.get_user_profile(user_id)
        
        # 分析用户输入特征来更新画像
        input_lower = user_input.lower()
        
        # 更新正式程度
        if any(word in input_lower for word in ['您好', '请问', '麻烦']):
            profile["formality"] = min(1.0, profile["formality"] + 0.1)
        elif any(word in input_lower for word in ['嘿', '嗨', '哈哈']):
            profile["formality"] = max(0.0, profile["formality"] - 0.1)
        
        # 更新幽默感
        if any(word in input_lower for word in ['笑话', '搞笑', '幽默']):
            profile["humor_level"] = min(1.0, profile["humor_level"] + 0.15)
        
        # 更新情感水平
        if sentiment > 0.6:
            profile["empathy_level"] = min(1.0, profile["empathy_level"] + 0.05)
        
        # 保存更新后的画像
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, personality_profile, preferences)
            VALUES (?, ?, ?)
        ''', (user_id, json.dumps(profile), json.dumps(prefs)))
        
        conn.commit()
        conn.close()
        
        return profile

class SelfReflection:
    """自我反思系统"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
    
    def analyze_response_quality(self, user_input, ai_response, sentiment):
        """分析回应质量"""
        quality_score = 0.5  # 基础分
        
        # 基于长度评估
        if len(ai_response.split()) >= 5 and len(ai_response.split()) <= serlin_config.max_length:
            quality_score += 0.2
        
        # 处理情感值
        if isinstance(sentiment, torch.Tensor):
            sentiment_value = sentiment.mean().item()
        else:
            sentiment_value = float(sentiment)
        
        # 基于情感一致性
        if sentiment_value > 0.3 and sentiment_value < 0.8:
            quality_score += 0.1
        
        # 基于问题相关性
        user_words = set(user_input.lower().split())
        response_words = set(ai_response.lower().split())
        common_words = user_words.intersection(response_words)
        if len(common_words) > 0:
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def generate_improvement_suggestions(self, user_input, ai_response, quality_score):
        """生成改进建议"""
        suggestions = []
        
        if quality_score < 0.6:
            if len(ai_response.split()) < 3:
                suggestions.append("回应过于简短，可以提供更多细节")
            elif len(ai_response.split()) > 60:
                suggestions.append("回应可能过长，考虑更简洁表达")
            
            if "?" in user_input and "?" not in ai_response:
                suggestions.append("用户的问题可能需要更直接的答案")
        
        return suggestions
    
    def reflect_on_conversation(self, conversation_id, user_input, ai_response, sentiment):
        """对对话进行反思"""
        quality_score = self.analyze_response_quality(user_input, ai_response, sentiment)
        suggestions = self.generate_improvement_suggestions(user_input, ai_response, quality_score)
        
        reflection_text = f"回应质量评分: {quality_score:.2f}. "
        if suggestions:
            reflection_text += "改进建议: " + "; ".join(suggestions)
        else:
            reflection_text += "这次回应质量不错。"
        
        # 存储反思结果
        self.memory.store_reflection(conversation_id, reflection_text, 
                                   json.dumps(suggestions), quality_score)
        
        return reflection_text, suggestions, quality_score

class PositionalEncoding(nn.Module):
    """位置编码 - 适配batch_first"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer，不参与训练
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerThinkingLayer(nn.Module):
    """Transformer思考层 - 适配batch_first"""
    
    def __init__(self, d_model, nhead, num_layers, think_steps=serlin_config.think_steps):
        super(TransformerThinkingLayer, self).__init__()
        self.d_model = d_model
        self.think_steps = think_steps
        
        # 思考Transformer层 - 设置batch_first=True
        self.thinking_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                batch_first=True  # 添加这个参数
            ),
            num_layers=num_layers
        )
        
        # 思考融合层
        self.thought_fusion = nn.Linear(d_model * think_steps, d_model)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, context, knowledge_vector=None, memory_vector=None):
        # context形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = context.size()
        
        # 多步思考过程
        thoughts = []
        current_thought = context
        
        for step in range(self.think_steps):
            # 应用Transformer思考
            thought_output = self.thinking_transformer(current_thought)
            
            # 集成知识和记忆（如果提供）
            if knowledge_vector is not None:
                # knowledge_vector: [batch_size, d_model]
                knowledge_expanded = knowledge_vector.unsqueeze(1).expand(-1, seq_len, -1)
                thought_output = thought_output + knowledge_expanded
            
            if memory_vector is not None:
                # memory_vector: [batch_size, d_model]
                memory_expanded = memory_vector.unsqueeze(1).expand(-1, seq_len, -1)
                thought_output = thought_output + memory_expanded
            
            thoughts.append(thought_output)
            current_thought = thought_output
        
        # 融合所有思考步骤
        if len(thoughts) > 1:
            # 沿着特征维度拼接
            thought_cat = torch.cat(thoughts, dim=-1)  # [batch_size, seq_len, d_model * think_steps]
            final_thought = self.thought_fusion(thought_cat)
        else:
            final_thought = thoughts[0]
        
        # 层归一化
        final_thought = self.layer_norm(final_thought)
        
        return final_thought  # [batch_size, seq_len, d_model]

class TransformerDialogueAI(nn.Module):
    """基于Transformer的对话AI - 修复batch_first警告"""
    
    def __init__(self, vocab_size, idx2word, d_model=512, nhead=serlin_config.nhead, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 think_steps=3, max_length=100, dropout=0.1):
        super(TransformerDialogueAI, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.idx2word = idx2word
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)
        
        # Transformer编码器 - 设置batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer思考层
        self.thinking_layer = TransformerThinkingLayer(
            d_model=d_model,
            nhead=nhead,
            num_layers=2,
            think_steps=think_steps
        )
        
        # Transformer解码器 - 设置batch_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 个性适配器
        self.personality_adapter = nn.Linear(d_model + 5, d_model)
        
        # 情感和主题分析
        self.sentiment_analysis = nn.Linear(d_model, 3)
        self.topic_analysis = nn.Linear(d_model, 10)
        
        # 初始化参数
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def encode(self, src, src_mask=None):
        """编码输入序列"""
        # src形状: [batch_size, seq_len]
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)
        
        # 编码器期望输入: [batch_size, seq_len, d_model]
        memory = self.encoder(src_embedded, src_mask)
        return memory  # [batch_size, seq_len, d_model]
    
    def thinking_process(self, memory, knowledge_vector=None, memory_vector=None, personality_vector=None):
        """思考过程"""
        # memory形状: [batch_size, seq_len, d_model]
        
        # 应用思考层
        thought_memory = self.thinking_layer(memory, knowledge_vector, memory_vector)
        
        # 个性适配
        if personality_vector is not None:
            batch_size, seq_len, d_model = thought_memory.size()
            personality_expanded = personality_vector.unsqueeze(1).expand(-1, seq_len, -1)
            thought_with_personality = torch.cat([thought_memory, personality_expanded], dim=-1)
            thought_memory = torch.tanh(self.personality_adapter(thought_with_personality))
        
        return thought_memory  # [batch_size, seq_len, d_model]
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """解码生成回应"""
        # tgt形状: [batch_size, seq_len]
        # memory形状: [batch_size, seq_len, d_model]
        
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # 解码器期望输入: [batch_size, seq_len, d_model]
        output = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.output_projection(output)  # [batch_size, seq_len, vocab_size]
    
    def forward(self, src, tgt=None, knowledge_vector=None, 
                memory_vector=None, personality_vector=None, teacher_forcing_ratio=0.5):
        
        batch_size = src.size(0)
        device = src.device
        
        # 检查输入索引是否超出范围
        if src.max() >= self.vocab_size:
            print(f"错误: 输入包含超出词汇表的索引! max_index={src.max()}, vocab_size={self.vocab_size}")
            # 将超出范围的索引替换为UNK
            src = torch.clamp(src, 0, self.vocab_size - 1)
        
        # 编码输入
        memory = self.encode(src)
        
        # 思考过程
        thought_memory = self.thinking_process(memory, knowledge_vector, memory_vector, personality_vector)
        
        # 解码
        if tgt is not None:
            tgt_len = tgt.size(1)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 检查目标索引是否超出范围
            if tgt_input.max() >= self.vocab_size:
                tgt_input = torch.clamp(tgt_input, 0, self.vocab_size - 1)
            
            # 创建目标序列掩码
            tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # 解码
            decoder_output = self.decode(tgt_input, thought_memory, 
                                       tgt_mask=tgt_mask, memory_mask=None)
            
            # 转置输出以匹配目标形状
            decoder_output = decoder_output.transpose(0, 1)
        else:
            # 推理模式 - 生成回应
            decoder_output, generated_tokens = self.generate_autoregressive(thought_memory, batch_size, device)
            tgt_output = None
        
        # 分析情感和主题
        context_representation = thought_memory[:, 0, :]
        sentiment = self.sentiment_analysis(context_representation)
        topics = self.topic_analysis(context_representation)
        
        return {
            'output': decoder_output,
            'sentiment': sentiment,
            'topics': topics,
            'memory': thought_memory,
            'generated_tokens': generated_tokens if tgt is None else None
        }
    
    def generate_autoregressive(self, memory, batch_size, device):
        """改进的自回归生成 - 防止PAD泛滥"""
        # 初始化为SOS标记
        tgt = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        
        outputs = []
        generated_tokens = []
        
        # 生成参数
        temperature = 0.9
        top_k = 30
        repetition_penalty = 1.2
        
        for i in range(min(20, self.max_length)):
            # 创建目标序列掩码
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # 解码
            output = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=None)
            
            # 获取最后一个时间步的预测
            next_word_logits = output[:, -1, :]
            
            # 应用重复惩罚
            for token in set(generated_tokens):
                next_word_logits[0, token] /= repetition_penalty
            
            # 禁止生成PAD和SOS作为内容
            next_word_logits[0, 0] = -float('Inf')  # PAD
            next_word_logits[0, 1] = -float('Inf')  # SOS
            next_word_logits[0, 3] = -float('Inf')  # UNK
            
            # 温度采样 + top-k
            if temperature > 0:
                next_word_logits = next_word_logits / temperature
                
                # top-k过滤
                if top_k > 0:
                    indices_to_remove = next_word_logits < torch.topk(next_word_logits, top_k)[0][..., -1, None]
                    next_word_logits[indices_to_remove] = -float('Inf')
                
                probabilities = torch.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
            else:
                next_word = next_word_logits.argmax(dim=-1, keepdim=True)
            
            word = self.idx2word.get(next_word.item(), '<UNK>')
            
            # 如果生成EOS，提前停止
            if next_word.item() == 2:  # EOS
                break
            
            # 如果连续生成太多无效标记，提前停止
            if word in ['<PAD>', '<SOS>', '<UNK>']:
                if generated_tokens.count(next_word.item()) > 3:
                    break
            
            # 添加到序列
            tgt = torch.cat([tgt, next_word], dim=1)
            outputs.append(output[:, -1:, :])
            generated_tokens.append(next_word.item())
        
        if outputs:
            final_output = torch.cat(outputs, dim=1)
            return final_output, generated_tokens
        else:
            return torch.zeros(batch_size, 1, self.vocab_size).to(device), []
    
    def generate_square_subsequent_mask(self, sz):
        """生成序列掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class DialogueDataProcessor:
    """对话数据处理器"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
        
    def build_vocab(self, dialogues, min_freq=1):
        """构建词汇表"""
        word_freq = defaultdict(int)
    
        for dialogue in dialogues:
            for text in [dialogue['input'], dialogue['output']]:
                words = self.tokenize(text)
                for word in words:
                    word_freq[word] += 1
    
        print(f"发现 {len(word_freq)} 个不同的词")
    
        # 添加特殊标记
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
    
        # 添加所有词
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
    
        self.vocab_size = len(self.word2idx)
        print(f"词汇表构建完成，大小: {self.vocab_size}")
        print(f"示例词汇: {list(self.word2idx.keys())[:20]}")
    
        return self.vocab_size
    
    def auto_expand_vocab(self, text, min_freq=1):
        """自动扩展词汇表 - 修复版"""
        new_words = []
        words = self.tokenize(text)

        for word in words:
            # 跳过空词和特殊标记
            if not word.strip() or word in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                continue
        
            if word not in self.word2idx:
                # 检查词汇表是否达到上限（可选）
                #if self.vocab_size >= 15000:
                #   print(f"警告: 词汇表已达到上限 {self.vocab_size}，跳过添加新词")
                #   continue
            
                # 添加新词到词汇表
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                new_words.append(word)

        if new_words:
            print(f"词汇表自动扩展: 添加了 {len(new_words)} 个新词")
            self.save_vocab()
    
        return new_words

    def save_vocab(self, file_path="vocab.json"):
        """保存词汇表到文件"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"词汇表已保存到: {file_path}")

    def load_vocab(self, file_path="vocab.json"):
        """从文件加载词汇表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
            self.vocab_size = vocab_data['vocab_size']
            print(f"词汇表已从 {file_path} 加载，大小: {self.vocab_size}")
            return True
        except FileNotFoundError:
            print(f"词汇表文件 {file_path} 不存在")
            return False
        except Exception as e:
            print(f"加载词汇表失败: {e}")
            return False

    def add_basic_vocabulary(self):
        """添加基础词汇"""
        basic_words = [
            '我', '你', '他', '她', '它', '我们', '你们', '他们', 
            '这', '那', '哪', '谁', '什么', '怎么', '为什么',
            '是', '有', '在', '的', '了', '着', '过',
            '不', '没', '很', '非常', '真', '太',
            '说', '问', '回答', '告诉', '知道', '想', '觉得', 
            '可以', '能够', '会', '要', '需要', '应该',
            '做', '学习', '帮助', '理解', '解释',
            '问题', '答案', '事情', '东西', '时间', '地方',
            '人', '朋友', '老师', '学生', '电脑', '手机',
            '今天', '明天', '昨天', '现在', '以后',
            '好', '坏', '对', '错', '高兴', '开心', '难过',
            '喜欢', '爱', '讨厌', '希望', '期待',
            '吗', '呢', '吧', '啊', '呀', '哦',
            '因为', '所以', '但是', '然后', '如果',
            '一', '二', '两', '三', '四', '五', '十', '百', '千', '万',
            '个', '只', '条', '件', '次', '些',
            '上', '下', '左', '右', '前', '后', '里', '外',
            '年', '月', '日', '小时', '分钟', '秒'
        ]
    
        added_count = 0
        for word in basic_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                added_count += 1
    
        if added_count > 0:
            print(f"添加了 {added_count} 个基础词汇")
            self.save_vocab()
    
        return added_count
    
    def tokenize(self, text):
        """分词函数"""
        if text in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
            return [text]
    
        tokens = []
        i = 0
        n = len(text)
    
        while i < n:
            if text[i] == '<' and '>' in text[i:]:
                end = text.find('>', i) + 1
                if end > i:
                    potential_special = text[i:end]
                    if potential_special in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                        tokens.append(potential_special)
                        i = end
                        continue
        
            char = text[i]
        
            if '\u4e00' <= char <= '\u9fff':
                tokens.append(char)
                i += 1
            elif char.isalpha():
                start = i
                while i < n and text[i].isalpha():
                    i += 1
                tokens.append(text[start:i].lower())
            elif char.isdigit():
                start = i
                while i < n and text[i].isdigit():
                    i += 1
                tokens.append(text[start:i])
            else:
                if char.strip():
                    tokens.append(char)
                i += 1
    
        return tokens
    
    def encode(self, text, auto_expand=True):
        """将文本编码为索引序列"""
        words = self.tokenize(text)
        indices = []
    
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                if auto_expand:
                    self.auto_expand_vocab(word)
                    indices.append(self.word2idx[word])
                else:
                    indices.append(self.word2idx['<UNK>'])
    
        return [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def get_vocab_size(self):
        """获取词汇表大小"""
        return self.vocab_size
    
    def sync_vocab_to_array(self):
        """同步词汇表到数组 - 确保所有词都在词汇表中"""
        print(f"同步词汇表到数组，当前词汇表大小: {self.vocab_size}")
        
        # 确保所有特殊标记都存在
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            if token not in self.word2idx:
                self.word2idx[token] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = token
                print(f"添加特殊标记: {token}")
        
        # 更新词汇表大小
        self.vocab_size = len(self.word2idx)
        print(f"同步后词汇表大小: {self.vocab_size}")
        
        return self.vocab_size

class TransformerTrainer:
    """改进的Transformer训练器"""
    
    def __init__(self, model, processor, learning_rate=0.0001):
        self.model = model
        self.processor = processor
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 改进的损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            reduction='mean'
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        
    def train_epoch(self, dataloader):
        """改进的训练epoch"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            input_seq = batch['input']
            target_seq = batch['output']
            
            # 检查输入是否超出词汇表范围
            if input_seq.max() >= self.model.vocab_size or target_seq.max() >= self.model.vocab_size:
                print(f"警告: 输入或目标包含超出词汇表的索引!")
                continue
            
            # 前向传播
            outputs = self.model(input_seq, target_seq)
            predictions = outputs['output']
            
            # 计算损失 - 只计算非PAD位置
            pred_flat = predictions.reshape(-1, predictions.size(-1))
            target_flat = target_seq[:, 1:].reshape(-1)  # 移除SOS用于目标
            
            # 过滤掉PAD位置
            non_pad_mask = target_flat != 0
            if non_pad_mask.sum() == 0:
                continue  # 跳过全PAD的批次
                
            pred_non_pad = pred_flat[non_pad_mask]
            target_non_pad = target_flat[non_pad_mask]
            
            loss = self.criterion(pred_non_pad, target_non_pad)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        self.scheduler.step()
        return total_loss / total_batches if total_batches > 0 else 0

# 示例训练数据
training_dialogues = [
    {"input": "你好", "output": "你好！我是Serlin，很高兴认识你。"},
    {"input": "你好吗", "output": "我很好，谢谢关心！你呢？"},
    {"input": "你叫什么名字", "output": "我叫Serlin，是一个AI助手。"},
    {"input": "你能做什么", "output": "我可以回答问题、聊天、学习新知识。"},
    {"input": "今天天气怎么样", "output": "我无法获取实时天气，建议查看天气预报。"},
    {"input": "给我讲个笑话", "output": "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 == Dec 25！"},
    {"input": "什么是人工智能", "output": "人工智能是让机器模拟人类智能的技术。"},
    {"input": "你会学习吗", "output": "是的，我会从对话中学习并改进自己。"},
    {"input": "再见", "output": "再见！期待下次聊天。"},
    {"input": "谢谢", "output": "不客气！随时为你服务。"},
]

class SerlinTransformer:
    """基于Transformer的Serlin系统 - 增强版"""
    
    def __init__(self, d_model,nhead,num_encoder_layers,num_decoder_layers,think_steps,max_length,model_save_path="serlin_transformer.pth"):
        self.model_save_path = model_save_path
        self.training_data = []
        self.conversation_history = []
    
        # 初始化记忆系统
        self.memory = LongTermMemory()
        self.knowledge_base = KnowledgeBase(self.memory)
        self.context_manager = MultiTurnContext()
        self.personality_adaptation = PersonalityAdaptation(self.memory)
        self.self_reflection = SelfReflection(self.memory)
    
        # 初始化数据处理器
        self.processor = DialogueDataProcessor()
    
        # 尝试加载现有词汇表，否则构建新的
        if not self.processor.load_vocab():
            print("构建初始词汇表...")
            vocab_size = self.processor.build_vocab(training_dialogues, min_freq=1)
            self.processor.add_basic_vocabulary()
            print(f"最终词汇表大小: {self.processor.vocab_size}")
        
        # 同步词汇表到数组
        self.processor.sync_vocab_to_array()
    
        # 使用Transformer模型
        self.model = TransformerDialogueAI(
            vocab_size=self.processor.vocab_size,
            d_model=d_model,
            idx2word=self.processor.idx2word,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            think_steps=think_steps,
            max_length=max_length
        )
    
        # 初始化训练器
        self.trainer = TransformerTrainer(self.model, self.processor)
    
        # 尝试加载已有模型
        self.load_model()
    
    def _create_knowledge_vector(self, knowledge, confidence):
        """创建知识向量"""
        if knowledge:
            knowledge_hash = hashlib.md5(knowledge.encode()).hexdigest()
            knowledge_int = int(knowledge_hash[:serlin_config.nhead], 16)
            vector = np.random.RandomState(knowledge_int).randn(serlin_config.d_model)
            return torch.tensor(vector * confidence, dtype=torch.float32).unsqueeze(0)
        return torch.zeros(1, serlin_config.d_model)
    
    def _create_memory_vector(self, user_id, user_input):
        """创建记忆向量"""
        user_history = self.memory.get_user_history(user_id, limit=5)
        
        if not user_history:
            return torch.zeros(1, serlin_config.d_model)
        
        memory_text = " ".join([conv['input'] + " " + conv['response'] for conv in user_history])
        memory_hash = hashlib.md5(memory_text.encode()).hexdigest()
        memory_int = int(memory_hash[:serlin_config.nhead], 16)
        vector = np.random.RandomState(memory_int).randn(serlin_config.d_model)
        
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    
    def process_user_input(self, user_id, user_input):
        """处理用户输入的全流程"""
        self.processor.auto_expand_vocab(user_input)
        self.sync_model_vocab()
        self.processor.sync_vocab_to_array()
        user_history = self.memory.get_user_history(user_id)
        user_profile, user_prefs = self.personality_adaptation.get_user_profile(user_id)
        
        # 知识检索
        knowledge, confidence = self.knowledge_base.query_knowledge(user_input)
        knowledge_vector = self._create_knowledge_vector(knowledge, confidence)
        
        # 记忆检索
        memory_vector = self._create_memory_vector(user_id, user_input)
        
        # 生成回应
        response_data = self._generate_response(user_input, knowledge_vector, memory_vector, user_profile)
        
        topics = self._extract_topics(user_input, response_data['response'])
        
        # 处理情感值
        sentiment_tensor = response_data['sentiment']
        if isinstance(sentiment_tensor, torch.Tensor):
            sentiment_value = sentiment_tensor.mean().item()
        else:
            sentiment_value = float(sentiment_tensor)
        
        # 存储对话
        self.memory.store_conversation(user_id, user_input, response_data['response'], 
                                     sentiment_value, topics)
        
        # 自我反思
        reflection, suggestions, quality = self.self_reflection.reflect_on_conversation(
            len(user_history) + 1, user_input, response_data['response'], sentiment_value
        )
        
        # 学习
        self.knowledge_base.learn_from_conversation(user_input, response_data['response'])
        
        # 更新用户画像
        self.personality_adaptation.update_user_profile(
            user_id, user_input, response_data['response'], sentiment_value
        )
        
        # 更新上下文
        self.context_manager.add_turn(user_input, response_data['response'], 
                                    sentiment_value, topics)
        
        result = {
            'response': response_data['response'],
            'sentiment': sentiment_tensor,
            'sentiment_value': sentiment_value,
            'topics': topics,
            'reflection': reflection,
            'quality_score': quality,
            'user_profile': user_profile
        }
        
        if knowledge:
            result['knowledge_used'] = f"使用了知识: {knowledge[:serlin_config.max_length]}..."
        if len(user_history) > 0:
            result['memory_accessed'] = f"访问了{len(user_history)}条历史记录"
            
        return result

    def _generate_response(self, user_input, knowledge_vector, memory_vector, user_profile):
        """生成回应"""
        try:
            # 编码输入前同步词汇表
            self.sync_model_vocab()
        
            input_seq = self.processor.encode(user_input, auto_expand=False)
            
            # 检查输入序列是否有效
            if not input_seq or len(input_seq) == 0:
                return {
                    'response': "抱歉，我没有理解您的输入。",
                    'sentiment': torch.tensor([0.5, 0.3, 0.2])
                }
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long)
        
            # 检查输入索引是否超出范围
            if input_tensor.max() >= self.model.vocab_size:
                print(f"警告: 输入包含超出词汇表的索引! max_index={input_tensor.max()}, vocab_size={self.model.vocab_size}")
                # 将超出范围的索引替换为UNK
                input_tensor = torch.clamp(input_tensor, 0, self.model.vocab_size - 1)
        
            personality_tensor = torch.tensor([[
                user_profile['formality'],
                user_profile['humor_level'], 
                user_profile['detail_level'],
                user_profile['empathy_level'],
                user_profile['curiosity_level']
            ]], dtype=torch.float32)
        
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor, 
                                   knowledge_vector=knowledge_vector, 
                                   memory_vector=memory_vector, 
                                   personality_vector=personality_tensor)
            
                predictions = outputs['output']
            
                # 检查输出形状
                if 'generated_tokens' in outputs and outputs['generated_tokens']:
                    response_indices = outputs['generated_tokens']
                else:
                    response_indices = predictions.argmax(dim=-1)[0].cpu().numpy()
                
                response = self.processor.decode(response_indices)
                response = self.postprocess_response(response)
            
                return {
                    'response': response,
                    'sentiment': outputs.get('sentiment', torch.tensor([0.5, 0.3, 0.2]))
                }
    
        except Exception as e:
            print(f"生成回应时出错: {e}")
            return {
                'response': "抱歉，我遇到了一些技术问题，请稍后再试。",
                'sentiment': torch.tensor([0.5, 0.3, 0.2])
            }
                
    def postprocess_response(self, response):
        """改进的后处理函数"""
        if not response or response.strip() == "":
            return "我还在学习如何回答这个问题。"
    
        words = response.split()
        valid_words = []
    
        for word in words:
            # 放宽过滤条件
            if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] or not word.strip():
                continue
            # 允许单个中文字符
            if len(word) == 1 and ('\u4e00' <= word <= '\u9fff'):
                valid_words.append(word)
            # 允许较短的词
            elif len(word) > 0:
                valid_words.append(word)
    
        # 放宽有效词数量要求
        if len(valid_words) == 0:
            return "我还在学习如何回答这个问题。"
        elif len(valid_words) == 1:
            # 单个词时，尝试构建更有意义的回应
            word = valid_words[0]
            if word in ['你好', '嗨', '嘿']:
                return f"{word}！我是Serlin。"
            elif word in ['谢谢', '感谢']:
                return f"{word}！不客气。"
            elif word in ['再见', '拜拜']:
                return f"{word}！下次见。"
            else:
                return f"{word}。"
    
        final_response = ' '.join(valid_words)
        final_response = final_response.strip('.,!?;，。！？；')
    
        # 确保回应以合适的标点结束
        if not any(final_response.endswith(p) for p in ['。', '！', '？', '.', '!', '?']):
            final_response += '。'
    
        return final_response
    
    def _extract_topics(self, user_input, response):
        """提取话题"""
        common_words = {'的', '了', '是', '在', '我', '你', '他', '她', '它', '这', '那', '吗', '呢', '啊', '吧', '哦'}
        words = set(user_input.lower().split() + response.lower().split())
        topics = [word for word in words if len(word) > 1 and word not in common_words and word != '<unk>']
        return topics[:serlin_config.think_steps]
    
    def sync_model_vocab(self):
        """修复的词汇表同步方法 - 确保嵌入层正确扩展"""
        current_vocab_size = self.processor.vocab_size
        model_vocab_size = self.model.vocab_size

        if current_vocab_size != model_vocab_size:
            print(f"词汇表大小不匹配: 模型={model_vocab_size}, 处理器={current_vocab_size}")
            print("重新初始化Transformer模型...")
    
            # 保存重要参数
            old_state_dict = None
            try:
                old_state_dict = self.model.state_dict().copy()
            except:
                pass
    
            # 重新初始化模型
            self.model = TransformerDialogueAI(
                vocab_size=current_vocab_size,
                idx2word=self.processor.idx2word,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                think_steps=3,
                max_length=100
            )
    
            # 尝试恢复参数
            if old_state_dict is not None:
                try:
                    new_state_dict = self.model.state_dict()
                
                    # 恢复嵌入层参数
                    if 'embedding.weight' in old_state_dict and 'embedding.weight' in new_state_dict:
                        old_embedding = old_state_dict['embedding.weight']
                        new_embedding = new_state_dict['embedding.weight']
                        min_size = min(old_embedding.size(0), new_embedding.size(0))
                        new_embedding[:min_size] = old_embedding[:min_size]
                        new_state_dict['embedding.weight'] = new_embedding
                        print(f"恢复了嵌入层前{min_size}个参数")
                
                    # 恢复其他匹配的参数
                    for name, param in old_state_dict.items():
                        if name in new_state_dict and new_state_dict[name].shape == param.shape:
                            new_state_dict[name] = param
                
                    self.model.load_state_dict(new_state_dict)
                    print("模型参数已成功恢复")
                except Exception as e:
                    print(f"恢复参数失败: {e}，将使用新模型")
    
            # 更新训练器
            self.trainer = TransformerTrainer(self.model, self.processor)
            print("Transformer模型已重新初始化")
    
        return current_vocab_size

    def chat(self, user_id, user_input, show_thinking=True):
        """对话方法"""
        result = self.process_user_input(user_id, user_input)
        
        if show_thinking:
            self._display_enhanced_thinking(result, user_input)
        
        self.conversation_history.append({
            'user': user_input,
            'ai': result['response'],
            'result': result
        })
        
        return result['response']
    
    def _display_enhanced_thinking(self, result, user_input):
        """显示增强的思考过程"""
        print(f"用户输入: '{user_input}'")
        print("Serlin思考过程:")
        
        print("分析结果:")
        print(f"  情感值: {result['sentiment_value']:.3f}")
        print(f"  识别话题: {', '.join(result['topics'])}")
        print(f"  回应质量: {result['quality_score']:.3f}")
        
        profile = result['user_profile']
        print("个性适配:")
        print(f"  正式程度: {profile['formality']:.3f}")
        print(f"  幽默水平: {profile['humor_level']:.3f}")
        print(f"  详细程度: {profile['detail_level']:.3f}")
        print(f"  同理心水平: {profile['empathy_level']:.3f}")
        print(f"  好奇心水平: {profile['curiosity_level']:.3f}")
        
        print(f"自我反思: {result['reflection']}")
        
        # 显示知识和记忆使用情况
        self._display_knowledge_memory_usage(result)
    
    def _display_knowledge_memory_usage(self, result):
        """显示知识和记忆使用情况"""
        if 'knowledge_used' in result:
            print(f"知识使用: {result['knowledge_used']}")
        if 'memory_accessed' in result:
            print(f"记忆访问: {result['memory_accessed']}")

    def get_system_status(self, user_id):
        """获取系统状态"""
        user_history = self.memory.get_user_history(user_id)
        profile, prefs = self.personality_adaptation.get_user_profile(user_id)
        
        print("\n=== 系统状态 ===")
        print(f"用户ID: {user_id}")
        print(f"对话历史记录: {len(user_history)} 条")
        print(f"当前话题: {', '.join(self.context_manager.get_recent_topics())}")
        print(f"个性画像:")
        print(f"  正式程度: {profile['formality']:.2f}")
        print(f"  幽默水平: {profile['humor_level']:.2f}")
        print(f"  详细程度: {profile['detail_level']:.2f}")
        print(f"  同理心水平: {profile['empathy_level']:.2f}")
        print(f"  好奇心水平: {profile['curiosity_level']:.2f}")
        print(f"训练数据数量: {len(self.training_data)}")
        print(f"词汇表大小: {self.processor.vocab_size}")
        print(f"模型词汇表大小: {self.model.vocab_size}")
        print("================")

    def add_training_data(self, questions, answers):
        """添加训练数据"""
        for q, a in zip(questions, answers):
            if len(q.strip()) == 0 or len(a.strip()) == 0:
                continue
            
            self.training_data.append({"input": q, "output": a})
            
            # 自动扩展词汇表
            self.processor.auto_expand_vocab(q)
            self.processor.auto_expand_vocab(a)
            
            # 同步词汇表到数组
            self.processor.sync_vocab_to_array()
            
            # 同步模型
            self.sync_model_vocab()

    def create_dataloader(self, batch_size=2):
        """创建数据加载器"""
        if not self.training_data:
            return []
        
        inputs = []
        outputs = []
    
        for dialogue in self.training_data:
            try:
                # 编码时不要自动扩展词汇表，避免干扰
                input_seq = self.processor.encode(dialogue['input'], auto_expand=False)
                output_seq = self.processor.encode(dialogue['output'], auto_expand=False)
            
                # 检查序列质量
                if len(input_seq) < 2 or len(output_seq) < 2:
                    continue
                
                # 确保序列以EOS结束
                if output_seq[-1] != self.processor.word2idx['<EOS>']:
                    output_seq = output_seq[:-1] + [self.processor.word2idx['<EOS>']]
            
                # 填充到合适长度
                max_len = 15
                input_seq = input_seq[:max_len] + [0] * (max_len - len(input_seq))
                output_seq = output_seq[:max_len] + [0] * (max_len - len(output_seq))
            
                inputs.append(input_seq)
                outputs.append(output_seq)
            
            except Exception as e:
                print(f"处理训练数据时出错: {e}")
                continue
    
        if not inputs:
            print("没有有效的训练数据!")
            return []
    
        print(f"创建数据加载器: {len(inputs)} 条有效数据")
        return [{
            'input': torch.tensor(inputs, dtype=torch.long),
            'output': torch.tensor(outputs, dtype=torch.long)
        }]

    def improved_train(self, epochs=100, batch_size=4, save_after_training=True, early_stopping=True):
        """改进的训练方法"""
        if not self.training_data:
            print("没有训练数据，请先添加训练数据！")
            return None
    
        if len(self.training_data) < 5:
            print("训练数据太少，无法有效训练")
            return None
    
        self.sync_model_vocab()
        print(f"开始改进版Transformer训练，使用 {len(self.training_data)} 条训练数据...")
        print(f"词汇表大小: {self.processor.vocab_size}")
        print(f"模型词汇表大小: {self.model.vocab_size}")
        print(f"批次大小: {batch_size}, 训练轮数: {epochs}")
    
        dataloader = self.create_dataloader(batch_size)
        if not dataloader:
            print("数据加载器创建失败")
            return None
    
        best_loss = float('inf')
        patience = serlin_config.nhead
        patience_counter = 0
    
        train_losses = []
        for epoch in range(epochs):
            try:
                loss = self.trainer.train_epoch(dataloader)
                train_losses.append(loss)
            
                if epoch % 5 == 0:
                    print(f'Epoch {epoch:3d}, Loss: {loss:.4f}, LR: {self.trainer.scheduler.get_last_lr()[0]:.6f}')
            
                # 早停检查
                if early_stopping:
                    if epoch % 5 == 0:
                        val_loss = self._compute_validation_loss(dataloader)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            patience_counter = 0
                            if save_after_training:
                                self.save_model(f"best_{self.model_save_path}")
                            print(f"✅ 发现更好模型，验证损失: {val_loss:.4f}")
                        else:
                            patience_counter += 1
                            print(f"⏳ 早停计数: {patience_counter}/{patience}")
                
                    if patience_counter >= patience:
                        print(f"🛑 早停触发，在第 {epoch} 轮停止训练")
                        break
            
                # 每15个epoch测试一次模型
                if epoch % 15 == 0 and epoch > 0:
                    self._test_model_improved()
                
            except Exception as e:
                print(f"第 {epoch} 轮训练出错: {e}")
                continue
    
        if save_after_training:
            final_path = self.save_model()
            print(f"最终模型已保存到: {final_path}")
    
        print("训练完成！")
        final_loss = train_losses[-1] if train_losses else 0
        return final_loss

    def _compute_validation_loss(self, dataloader):
        """计算验证损失"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
    
        with torch.no_grad():
            for batch in dataloader:
                input_seq = batch['input']
                target_seq = batch['output']
            
                outputs = self.model(input_seq, target_seq)
                predictions = outputs['output']
            
                # 计算损失
                pred_flat = predictions.reshape(-1, predictions.size(-1))
                target_flat = target_seq[:, 1:].reshape(-1)
            
                loss = self.trainer.criterion(pred_flat, target_flat)
                total_loss += loss.item()
                total_batches += 1
    
        self.model.train()
        return total_loss / total_batches if total_batches > 0 else float('inf')

    def _test_model_improved(self):
        """改进的模型测试"""
        self.model.eval()
        with torch.no_grad():
            test_inputs = ["你好", "你叫什么名字", "你会做什么", "再见", "谢谢"]
            print("\n=== 模型测试 ===")
        
            for test_input in test_inputs:
                try:
                    input_seq = self.processor.encode(test_input, auto_expand=False)
                    input_tensor = torch.tensor([input_seq], dtype=torch.long)
                
                    # 检查输入索引是否超出范围
                    if input_tensor.max() >= self.model.vocab_size:
                        print(f"警告: 测试输入 '{test_input}' 包含超出词汇表的索引!")
                        input_tensor = torch.clamp(input_tensor, 0, self.model.vocab_size - 1)
                
                    outputs = self.model(input_tensor)
                    predictions = outputs['output']
                
                    if predictions.numel() == 0:
                        response = "[无输出]"
                        raw_response = "[无输出]"
                    else:
                        response_indices = predictions.argmax(dim=-1)[0].cpu().numpy()
                        raw_response = self.processor.decode(response_indices)
                        response = self.postprocess_response(raw_response)
                
                    print(f"  输入: '{test_input}'")
                    print(f"  原始输出: '{raw_response}'")
                    print(f"  处理后: '{response}'")
                    print()
                
                except Exception as e:
                    print(f"  测试 '{test_input}' 时出错: {e}")
        
            print("================")
    
        self.model.train()

    def validate_training_data(self):
        """验证训练数据"""
        if not self.training_data:
            print("没有训练数据")
            return False

        print(f"\n=== 训练数据验证 ===")
        print(f"训练数据数量: {len(self.training_data)}")

        valid_count = 0
        for i, data in enumerate(self.training_data):
            try:
                # 先自动扩展词汇表
                self.processor.auto_expand_vocab(data['input'])
                self.processor.auto_expand_vocab(data['output'])
            
                # 同步词汇表到数组
                self.processor.sync_vocab_to_array()
            
                # 重新同步模型词汇表
                self.sync_model_vocab()
            
                # 再次编码
                input_encoded = self.processor.encode(data['input'], auto_expand=False)
                output_encoded = self.processor.encode(data['output'], auto_expand=False)
            
                # 检查是否有未知词
                has_unk = any(idx == self.processor.word2idx['<UNK>'] for idx in input_encoded + output_encoded)
            
                status = "✅ 有效" if not has_unk else "❌ 包含未知词"
                input_preview = data['input'][:20] + "..." if len(data['input']) > 20 else data['input']
                output_preview = data['output'][:20] + "..." if len(data['output']) > 20 else data['output']
                print(f"{i+1}. 输入: '{input_preview}'")
                print(f"    输出: '{output_preview}' - {status}")
            
                if not has_unk:
                    valid_count += 1
                
            except Exception as e:
                print(f"{i+1}. 数据验证出错: {e}")
                continue

        print(f"验证结果: {valid_count}/{len(self.training_data)} 条数据有效")
    
        # 放宽验证标准，只要有数据就允许训练
        is_valid = valid_count > 0
        if is_valid:
            print("✅ 训练数据验证通过")
        else:
            print("❌ 训练数据验证失败")
    
        print("=== 验证结束 ===\n")

        return is_valid

    def interactive_training_with_options(self):
        """带选项的交互式训练模式"""
        print("\n=== Serlin Transformer 交互式训练模式 ===")
        print("输入训练数据，格式：")
        print("  问题：你的问题")
        print("  期望回复：期望的回答") 
        print("输入 '完成' 结束数据输入")
        print("输入 '训练' 开始训练")
        print("输入 '退出' 返回主菜单")
        print("=" * 50)
    
        questions = []
        answers = []
    
        while True:
            try:
                user_input = input("\n> ").strip()
            
                if user_input.lower() in ['退出', 'exit']:
                    return False
                elif user_input.lower() in ['完成', 'done']:
                    break
                elif user_input.lower() in ['训练', 'train']:
                    if questions and answers:
                        print(f"准备训练，共 {len(questions)} 对问答数据")
                        self.add_training_data(questions, answers)
                        
                        # 询问训练参数
                        print("\n=== 训练参数设置 ===")
                        try:
                            epochs = int(input("训练轮数 (默认50): ") or "50")
                            batch_size = int(input("批次大小 (默认2): ") or "2")
                            learning_rate = float(input("学习率 (默认0.0001): ") or "0.0001")
                        except ValueError:
                            print("输入无效，使用默认参数")
                            epochs = 50
                            batch_size = 2
                            learning_rate = 0.0001
                        
                        # 更新学习率
                        if hasattr(self.trainer.optimizer, 'param_groups'):
                            for param_group in self.trainer.optimizer.param_groups:
                                param_group['lr'] = learning_rate
                        
                        print(f"开始训练: {epochs}轮, 批次大小: {batch_size}, 学习率: {learning_rate}")
                        
                        # 使用带调试的训练
                        self.improved_train(epochs=epochs, batch_size=batch_size)
                    
                        questions = []
                        answers = []
                    else:
                        print("没有训练数据，请先添加数据！")
                    continue
            
                # 解析训练数据
                if user_input.startswith("问题："):
                    question = user_input[serlin_config.think_steps:].strip()
                    questions.append(question)
                    print(f"已记录问题: {question}")
                elif user_input.startswith("期望回复："):
                    answer = user_input[5:].strip()
                    answers.append(answer)
                    print(f"已记录期望回复: {answer}")
                else:
                    print("格式错误！请使用：")
                    print("  问题：你的问题")
                    print("  期望回复：期望的回答")
        
            except Exception as e:
                print(f"处理输入时出错: {e}")
                continue
    
        return True

    def batch_train_from_json(self, file_path):
        """从JSON文件批量训练"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            # 支持多种JSON格式
            if isinstance(training_data, list):
                # 格式1: [{"input": "问题", "output": "回答"}, ...]
                questions = [item['input'] for item in training_data if 'input' in item and 'output' in item]
                answers = [item['output'] for item in training_data if 'input' in item and 'output' in item]
            elif isinstance(training_data, dict) and 'training_data' in training_data:
                # 格式2: {"training_data": [{"input": "问题", "output": "回答"}, ...]}
                questions = [item['input'] for item in training_data['training_data'] if 'input' in item and 'output' in item]
                answers = [item['output'] for item in training_data['training_data'] if 'input' in item and 'output' in item]
            else:
                print("不支持的JSON格式")
                return False
        
            if questions and answers:
                print(f"从JSON文件加载了 {len(questions)} 对训练数据")
                
                # 询问训练参数
                print("\n=== 训练参数设置 ===")
                try:
                    epochs = int(input("训练轮数 (默认100): ") or "100")
                    batch_size = int(input("批次大小 (默认4): ") or "4")
                    learning_rate = float(input("学习率 (默认0.0001): ") or "0.0001")
                    early_stopping = input("启用早停机制? (y/N): ").strip().lower() == 'y'
                except ValueError:
                    print("输入无效，使用默认参数")
                    epochs = 100
                    batch_size = 4
                    learning_rate = 0.0001
                    early_stopping = True
                
                # 更新学习率
                if hasattr(self.trainer.optimizer, 'param_groups'):
                    for param_group in self.trainer.optimizer.param_groups:
                        param_group['lr'] = learning_rate
                
                self.add_training_data(questions, answers)
                
                print(f"开始自动训练...")
                print(f"参数: {epochs}轮, 批次大小: {batch_size}, 学习率: {learning_rate}, 早停: {early_stopping}")
                
                self.improved_train(
                    epochs=epochs, 
                    batch_size=batch_size, 
                    save_after_training=True,
                    early_stopping=early_stopping
                )
                return True
            else:
                print("JSON文件中没有找到有效的训练数据")
                return False
            
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在")
            return False
        except json.JSONDecodeError:
            print(f"JSON文件格式错误: {file_path}")
            return False
        except Exception as e:
            print(f"读取JSON文件时出错: {e}")
            return False

    def create_training_template(self, file_path="training_template.json"):
        """创建训练数据模板"""
        template = {
            "description": "Serlin训练数据模板",
            "version": "1.0",
            "created_date": datetime.datetime.now().isoformat(),
            "training_data": [
                {
                    "input": "你好",
                    "output": "你好！我是Serlin，很高兴认识你。"
                },
                {
                    "input": "你叫什么名字",
                    "output": "我叫Serlin，是一个基于Transformer的AI助手。"
                },
                {
                    "input": "你能做什么",
                    "output": "我可以回答问题、聊天、学习新知识，还能进行多轮对话。"
                }
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        print(f"训练数据模板已创建: {file_path}")
        return file_path

    def show_training_options(self):
        """显示训练选项"""
        print("\n=== 训练选项 ===")
        print("1. 交互式训练 - 手动输入训练数据")
        print("2. JSON文件批量训练 - 从JSON文件加载")
        print("3. 创建训练模板 - 生成JSON模板文件")
        print("4. 返回主菜单")
        
        choice = input("请选择训练方式 (1-4): ").strip()
        
        if choice == '1':
            self.interactive_training_with_options()
        elif choice == '2':
            file_path = input("请输入JSON训练文件路径: ").strip()
            if file_path:
                self.batch_train_from_json(file_path)
        elif choice == '3':
            file_name = input("请输入模板文件名 (默认training_template.json): ").strip() or "training_template.json"
            self.create_training_template(file_name)
        elif choice == '4':
            return
        else:
            print("无效选择")

    def show_training_status(self):
        """显示训练状态"""
        print(f"\n训练数据数量: {len(self.training_data)}")
        if self.training_data:
            print("最近5条训练数据:")
            for i, data in enumerate(self.training_data[-5:], 1):
                print(f"  {i}. 问题: {data['input']}")
                print(f"     期望: {data['output']}")

    def get_conversation_summary(self):
        """获取对话摘要"""
        if not self.conversation_history:
            return "暂无对话历史"
        
        summary = f"与用户的对话摘要:\n"
        summary += f"总对话轮数: {len(self.conversation_history)}\n"
        
        # 计算平均回应质量
        avg_quality = np.mean([conv['result']['quality_score'] for conv in self.conversation_history])
        summary += f"平均回应质量: {avg_quality:.3f}\n"
        
        # 提取常见话题
        all_topics = []
        for conv in self.conversation_history:
            all_topics.extend(conv['result']['topics'])
        
        if all_topics:
            from collections import Counter
            topic_counts = Counter(all_topics)
            common_topics = topic_counts.most_common(serlin_config.think_steps)
            summary += f"常见话题: {', '.join([f'{topic}({count})' for topic, count in common_topics])}\n"
        
        return summary

    def _convert_tensors_to_serializable(self, obj):
        """将Tensor对象转换为可JSON序列化的格式"""
        if isinstance(obj, torch.Tensor):
            # 如果是单个值的Tensor，转换为float
            if obj.numel() == 1:
                return obj.item()
            # 如果是向量，转换为列表
            else:
                return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # 对于其他无法序列化的类型，转换为字符串
            return str(obj)

    def export_conversation(self, filename=None):
        """修复的导出对话历史方法"""
        if filename is None:
            filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
        # 转换对话历史中的Tensor对象
        serializable_history = self._convert_tensors_to_serializable(self.conversation_history)
    
        export_data = {
            'export_time': datetime.datetime.now().isoformat(),
            'conversations': serializable_history
        }
    
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
            print(f"对话已导出到: {filename}")
            return filename
        except Exception as e:
            print(f"导出失败: {e}")
            return None

    def reset_model(self):
        """重置模型到初始状态"""
        print("重置模型...")
    
        # 重新初始化模型
        self.model = TransformerDialogueAI(
            vocab_size=self.processor.vocab_size,
            idx2word=self.processor.idx2word,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            think_steps=3,
            max_length=100
        )
    
        # 重新初始化训练器
        self.trainer = TransformerTrainer(self.model, self.processor)
    
        # 清除训练数据（可选）
        self.training_data = []
    
        print("模型已重置")

    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = self.model_save_path
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'word2idx': self.processor.word2idx,
            'idx2word': self.processor.idx2word,
            'vocab_size': self.processor.vocab_size,
            'training_data': self.training_data
        }
        
        torch.save(checkpoint, path)
        print(f"Transformer模型已保存到: {path}")
        return path
    
    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            path = self.model_save_path
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            saved_vocab_size = checkpoint.get('vocab_size', 0)
            current_vocab_size = self.processor.vocab_size
            
            if saved_vocab_size != current_vocab_size:
                print(f"词汇表大小不匹配: 保存的模型({saved_vocab_size}) vs 当前模型({current_vocab_size})")
                return self._rebuild_model_from_checkpoint(checkpoint, path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'training_data' in checkpoint:
                self.training_data = checkpoint['training_data']
            
            print(f"Transformer模型已从 {path} 加载")
            print(f"词汇表大小: {self.processor.vocab_size}")
            return True
            
        except FileNotFoundError:
            print(f"模型文件 {path} 不存在，将使用初始模型")
            return False
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return self._rebuild_model_from_checkpoint(checkpoint, path)
    
    def _rebuild_model_from_checkpoint(self, checkpoint, path):
        """从检查点重建模型"""
        try:
            saved_word2idx = checkpoint.get('word2idx')
            saved_idx2word = checkpoint.get('idx2word')
            saved_vocab_size = checkpoint.get('vocab_size', 0)
            
            if not saved_word2idx or not saved_idx2word:
                print("检查点中没有找到有效的词汇表信息")
                return False
            
            self.processor.word2idx = saved_word2idx
            self.processor.idx2word = saved_idx2word
            self.processor.vocab_size = saved_vocab_size
            
            self.model = TransformerDialogueAI(
                vocab_size=saved_vocab_size,
                idx2word=self.processor.idx2word,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                think_steps=3,
                max_length=100
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer = TransformerTrainer(self.model, self.processor)
            
            if 'training_data' in checkpoint:
                self.training_data = checkpoint['training_data']
            
            print(f"Transformer模型已从 {path} 重建并加载")
            print(f"词汇表大小: {self.processor.vocab_size}")
            return True
            
        except Exception as e:
            print(f"重建模型失败: {e}")
            return False

def print_help():
    print("可用命令:")
    print("  '退出'/'exit'/'quit' - 结束对话")
    print("  '状态' - 查看系统状态")
    print("  '摘要' - 显示对话摘要")
    print("  '导出' - 导出对话历史到文件")
    print("  '训练' - 进入训练模式")
    print("  '训练选项' - 显示训练选项菜单")
    print("  '训练状态' - 显示训练状态")
    print("  '加载模型' - 从文件加载模型")
    print("  '帮助' - 显示此帮助信息")
    print("  '静默' - 切换思考过程显示（开/关）")
    print("  '词汇表' - 查看词汇表")
    print("  '扩展词汇' - 添加词汇")
    print("  '重载词汇表' - 重新载入词汇表")
    print("  '修复模型' - 手动同步词汇表")
    print("  '验证数据' - 验证训练数据")
    print("  '重置模型' - 重置模型")
    print("  '创建模板' - 创建训练数据模板")
    print("  '设置参数' - 设置AI参数")
    print("-" * 50)

def main():
    # 参数配置文件路径
    config_file = "serlin_config.json"
    
    # 尝试从JSON文件加载参数，如果不存在则创建默认文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            print(f"从配置文件 {config_file} 加载参数...")
            
            # 更新SerlinConfig实例的参数
            serlin_config.d_model = config_data.get('d_model', serlin_config.d_model)
            serlin_config.nhead = config_data.get('nhead', serlin_config.nhead)
            serlin_config.num_encoder_layers = config_data.get('num_encoder_layers', serlin_config.num_encoder_layers)
            serlin_config.num_decoder_layers = config_data.get('num_decoder_layers', serlin_config.num_decoder_layers)
            serlin_config.think_steps = config_data.get('think_steps', serlin_config.think_steps)
            serlin_config.max_length = config_data.get('max_length', serlin_config.max_length)
            
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在，创建默认配置文件...")
        # 创建默认配置
        default_config = {
            "d_model": serlin_config.d_model,
            "nhead": serlin_config.nhead,
            "num_encoder_layers": serlin_config.num_encoder_layers,
            "num_decoder_layers": serlin_config.num_decoder_layers,
            "think_steps": serlin_config.think_steps,
            "max_length": serlin_config.max_length,
            "description": "Serlin Transformer 配置文件",
            "created_date": datetime.datetime.now().isoformat(),
            "notes": "修改这些参数后需要重启系统才能生效"
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"默认配置文件已创建: {config_file}")
    
    except Exception as e:
        print(f"加载配置文件时出错: {e}，使用默认参数")
    
    print("初始化Transformer版Serlin系统...")
    print("当前参数：")
    print(f"  d_model: {serlin_config.d_model}")
    print(f"  nhead: {serlin_config.nhead}")
    print(f"  编码器层数: {serlin_config.num_encoder_layers}")
    print(f"  解码器层数: {serlin_config.num_decoder_layers}")
    print(f"  思考步骤: {serlin_config.think_steps}")
    print(f"  最大序列长度: {serlin_config.max_length}")
    
    # 初始化
    trainer = SerlinTransformer(
        d_model=serlin_config.d_model,
        nhead=serlin_config.nhead,
        num_encoder_layers=serlin_config.num_encoder_layers,
        num_decoder_layers=serlin_config.num_decoder_layers,
        think_steps=serlin_config.think_steps,
        max_length=serlin_config.max_length
    )
    
    print("系统初始化完成！")
    print("增强版Transformer思考式对话Serlin已就绪")
    
    user_id = input("请输入用户ID：").strip() or "default_user"
    
    print(f"\n欢迎，用户 {user_id}！")
    
    print_help()
    
    show_thinking = True  # 默认显示思考过程
    
    while True:
        try:
            user_input = input(f"{user_id}: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['退出', 'exit', 'quit']:
                # 退出前显示摘要
                summary = trainer.get_conversation_summary()
                print(f"\n{summary}")
                export_choice = input("是否导出对话历史？(y/N): ").strip().lower()
                if export_choice == 'y':
                    trainer.export_conversation()
                print("感谢使用Serlin，再见！")
                break
                
            elif user_input.lower() in ['状态', 'status']:
                trainer.get_system_status(user_id)
                continue
                
            elif user_input.lower() in ['摘要', 'summary']:
                summary = trainer.get_conversation_summary()
                print(f"\n{summary}")
                continue
                
            elif user_input.lower() in ['设置参数', 'set']:
                # 先显示当前参数
                print("\n当前参数：")
                print(f"1. d_model: {serlin_config.d_model}")
                print(f"2. nhead: {serlin_config.nhead}")
                print(f"3. 编码器层数: {serlin_config.num_encoder_layers}")
                print(f"4. 解码器层数: {serlin_config.num_decoder_layers}")
                print(f"5. 思考步骤: {serlin_config.think_steps}")
                print(f"6. 最大序列长度: {serlin_config.max_length}")
                
                try:
                    # 获取新参数
                    new_d_model = int(input(f"请输入d_model (当前: {serlin_config.d_model}): ") or serlin_config.d_model)
                    new_nhead = int(input(f"请输入nhead (当前: {serlin_config.nhead}): ") or serlin_config.nhead)
                    new_num_encoder_layers = int(input(f"请输入编码器层数 (当前: {serlin_config.num_encoder_layers}): ") or serlin_config.num_encoder_layers)
                    new_num_decoder_layers = int(input(f"请输入解码器层数 (当前: {serlin_config.num_decoder_layers}): ") or serlin_config.num_decoder_layers)
                    new_think_steps = int(input(f"请输入思考步骤 (当前: {serlin_config.think_steps}): ") or serlin_config.think_steps)
                    new_max_length = int(input(f"请输入最大序列长度 (当前: {serlin_config.max_length}): ") or serlin_config.max_length)
                    
                    # 更新SerlinConfig实例
                    serlin_config.d_model = new_d_model
                    serlin_config.nhead = new_nhead
                    serlin_config.num_encoder_layers = new_num_encoder_layers
                    serlin_config.num_decoder_layers = new_num_decoder_layers
                    serlin_config.think_steps = new_think_steps
                    serlin_config.max_length = new_max_length
                    
                    # 保存到配置文件
                    config_data = {
                        "d_model": serlin_config.d_model,
                        "nhead": serlin_config.nhead,
                        "num_encoder_layers": serlin_config.num_encoder_layers,
                        "num_decoder_layers": serlin_config.num_decoder_layers,
                        "think_steps": serlin_config.think_steps,
                        "max_length": serlin_config.max_length,
                        "description": "Serlin Transformer 配置文件",
                        "last_modified": datetime.datetime.now().isoformat(),
                        "notes": "修改这些参数后需要重启系统才能生效"
                    }
                    
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, ensure_ascii=False, indent=2)
                    
                    print("参数已更新并保存到配置文件！")
                    print("注意：需要重启系统才能使新的参数生效")
                    
                except ValueError:
                    print("输入无效，参数未更改")
                continue
                
            elif user_input.lower() in ['词汇表', 'vocab']:
                print(f"\n词汇表大小: {trainer.processor.vocab_size}")
                print(f"模型词汇表大小: {trainer.model.vocab_size}")
                print(f"词汇表前50个词: {list(trainer.processor.word2idx.keys())[:50]}")
                continue

            elif user_input.lower() in ['扩展词汇', 'expand vocab']:
                words_to_add = input("请输入要添加的词汇（用空格分隔）: ").strip()
                if words_to_add:
                    added = trainer.processor.auto_expand_vocab(words_to_add)
                    if added:
                        print(f"成功添加 {len(added)} 个词汇")
                        # 同步词汇表到数组
                        trainer.processor.sync_vocab_to_array()
                        # 同步模型
                        trainer.sync_model_vocab()
                    else:
                        print("没有新词汇需要添加")
                continue
                
            elif user_input.lower() in ['验证数据', 'validate data']:
                trainer.validate_training_data()
                continue
                
            elif user_input.lower() in ['修复模型', 'fix model']:
                print("执行模型修复...")
                trainer.sync_model_vocab()
                print("修复完成")
                continue
                
            elif user_input.lower() in ['重载词汇表', 'reload vocab']:
                if trainer.processor.load_vocab():
                    print("词汇表重载成功")
                    # 同步词汇表到数组
                    trainer.processor.sync_vocab_to_array()
                    # 需要重新初始化模型以适应新的词汇表大小
                    trainer.model = TransformerDialogueAI(
                        vocab_size=trainer.processor.vocab_size,
                        idx2word=trainer.processor.idx2word,
                        d_model=serlin_config.d_model,
                        nhead=serlin_config.nhead,
                        num_encoder_layers=serlin_config.num_encoder_layers,
                        num_decoder_layers=serlin_config.num_decoder_layers,
                        think_steps=serlin_config.think_steps,
                        max_length=serlin_config.max_length
                    )
                    trainer.trainer = TransformerTrainer(trainer.model, trainer.processor)
                    print("模型已重新初始化以适应新词汇表")
                continue
                
            elif user_input.lower() in ['重置模型', 'reset model']:
                trainer.reset_model()
                continue
                
            elif user_input.lower() in ['导出', 'export']:
                filename = trainer.export_conversation()
                print(f"对话已导出到: {filename}")
                continue
                
            elif user_input.lower() in ['训练', 'train']:
                print("\n进入训练模式...")
                trainer.interactive_training_with_options()
                print("返回主对话模式")
                continue
                
            elif user_input.lower() in ['训练选项', 'training options']:
                trainer.show_training_options()
                continue
                
            elif user_input.lower() in ['训练状态', 'training status']:
                trainer.show_training_status()
                continue
                
            elif user_input.lower() in ['加载模型', 'load model']:
                model_path = input("请输入模型文件路径（留空使用默认）: ").strip()
                if model_path:
                    trainer.load_model(model_path)
                else:
                    trainer.load_model()
                continue

            elif user_input.lower() in ['创建模板', 'create template']:
                file_name = input("请输入模板文件名 (默认training_template.json): ").strip() or "training_template.json"
                trainer.create_training_template(file_name)
                continue

            elif user_input.lower() in ['帮助', 'help']:
                print_help()
                continue
                
            elif user_input.lower() in ['静默', 'silent', 'quiet']:
                show_thinking = not show_thinking
                print(f"思考过程显示: {'开启' if show_thinking else '关闭'}")
                continue
            
            # 处理普通对话
            response = trainer.chat(user_id, user_input, show_thinking=show_thinking)
            print(f"Serlin: {response}")
            
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在退出...")
            summary = trainer.get_conversation_summary()
            print(f"\n{summary}")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重新输入或输入'退出'结束对话")
if __name__ == "__main__":
    main()