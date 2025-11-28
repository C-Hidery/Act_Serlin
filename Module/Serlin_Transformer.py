# Serlin transformer server main file
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import sqlite3
import pickle
import hashlib
import socket
import threading
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
    """Long-term memory system"""
    
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize memory database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User information table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                personality_profile TEXT,
                preferences TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Conversation memory table
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
        
        # Knowledge memory table
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
        
        # Self-reflection table
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
        """Store conversation record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        # Ensure user exists
        cursor.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    
        # Process sentiment value
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
        """Get user conversation history"""
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
        """Store knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge (key_text, value_text, confidence, source, last_accessed)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (key, value, confidence, source))
        
        conn.commit()
        conn.close()
    
    def retrieve_knowledge(self, key):
        """Retrieve knowledge"""
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
        """Update knowledge access time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE knowledge SET last_accessed = CURRENT_TIMESTAMP 
            WHERE key_text = ?
        ''', (key,))
        
        conn.commit()
        conn.close()
    
    def store_reflection(self, conversation_id, reflection, suggestions, quality_score):
        """Store self-reflection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reflections (conversation_id, reflection_text, improvement_suggestions, quality_score)
            VALUES (?, ?, ?, ?)
        ''', (conversation_id, reflection, suggestions, quality_score))
        
        conn.commit()
        conn.close()

class KnowledgeBase:
    """Knowledge base system"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.domain_knowledge = self.load_domain_knowledge()
    
    def load_domain_knowledge(self):
        """Load domain knowledge"""
        return {
            "technology": {
                "python": "Python is a high-level programming language known for its readability and simplicity",
                "ai": "Artificial Intelligence is a branch of computer science focused on creating intelligent machines",
                "machine learning": "Machine learning is a subset of AI that enables computers to learn from data"
            },
            "entertainment": {
                "movies": "Movies are an important form of entertainment",
                "music": "Music can express emotions and create atmosphere",
                "games": "Games can provide entertainment and challenges"
            }
        }
    
    def query_knowledge(self, query, domain=None):
        """Query knowledge"""
        # First query from long-term memory
        knowledge, confidence = self.memory.retrieve_knowledge(query)
        if knowledge:
            return knowledge, confidence
        
        # Query from domain knowledge
        if domain and domain in self.domain_knowledge:
            if query in self.domain_knowledge[domain]:
                return self.domain_knowledge[domain][query], 0.8
        
        # Query from general domain knowledge
        for domain_knowledge in self.domain_knowledge.values():
            if query in domain_knowledge:
                return domain_knowledge[query], 0.7
        
        return None, 0.0
    
    def learn_from_conversation(self, user_input, response):
        """Learn new knowledge from conversation"""
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:
                if word in response.lower():
                    self.memory.store_knowledge(word, response, 0.6, "conversation_learning")

class MultiTurnContext:
    """Multi-turn conversation context management"""
    
    def __init__(self, max_context_length=10):
        self.max_context_length = max_context_length
        self.conversation_context = deque(maxlen=max_context_length)
        self.current_topics = set()
    
    def add_turn(self, user_input, ai_response, sentiment, extracted_topics):
        """Add a conversation turn to context"""
        turn = {
            'user_input': user_input,
            'ai_response': ai_response,
            'sentiment': sentiment,
            'topics': extracted_topics,
            'timestamp': datetime.datetime.now()
        }
        self.conversation_context.append(turn)
        
        # Update current topics
        self.current_topics.update(extracted_topics)
        # Limit number of topics
        if len(self.current_topics) > serlin_config.nhead:
            self.current_topics = set(list(self.current_topics)[-serlin_config.nhead:])
    
    def get_context_text(self):
        """Get context text"""
        if not self.conversation_context:
            return ""
        
        context_texts = []
        for turn in list(self.conversation_context)[-serlin_config.think_steps:]:
            context_texts.extend([turn['user_input'], turn['ai_response']])
        
        return " ".join(context_texts)
    
    def get_recent_topics(self):
        """Get recent topics"""
        return list(self.current_topics)[-5:]

class PersonalityAdaptation:
    """Personalized adaptation system"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.user_profiles = {}
    
    def get_user_profile(self, user_id):
        """Get user personality profile"""
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT personality_profile, preferences FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return json.loads(result[0]), json.loads(result[1])
        else:
            # Default personality profile
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
        """Update user profile based on conversation"""
        profile, prefs = self.get_user_profile(user_id)
        
        # Analyze user input characteristics to update profile
        input_lower = user_input.lower()
        
        # Update formality level
        if any(word in input_lower for word in ['hello', 'please', 'thank you']):
            profile["formality"] = min(1.0, profile["formality"] + 0.1)
        elif any(word in input_lower for word in ['hey', 'hi', 'haha']):
            profile["formality"] = max(0.0, profile["formality"] - 0.1)
        
        # Update humor level
        if any(word in input_lower for word in ['joke', 'funny', 'humor']):
            profile["humor_level"] = min(1.0, profile["humor_level"] + 0.15)
        
        # Update empathy level
        if sentiment > 0.6:
            profile["empathy_level"] = min(1.0, profile["empathy_level"] + 0.05)
        
        # Save updated profile
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
    """Self-reflection system"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
    
    def analyze_response_quality(self, user_input, ai_response, sentiment):
        """Analyze response quality"""
        quality_score = 0.5  # Base score
        
        # Evaluate based on length
        if len(ai_response.split()) >= 5 and len(ai_response.split()) <= serlin_config.max_length:
            quality_score += 0.2
        
        # Process sentiment value
        if isinstance(sentiment, torch.Tensor):
            sentiment_value = sentiment.mean().item()
        else:
            sentiment_value = float(sentiment)
        
        # Based on sentiment consistency
        if sentiment_value > 0.3 and sentiment_value < 0.8:
            quality_score += 0.1
        
        # Based on question relevance
        user_words = set(user_input.lower().split())
        response_words = set(ai_response.lower().split())
        common_words = user_words.intersection(response_words)
        if len(common_words) > 0:
            quality_score += 0.2
        
        return min(1.0, quality_score)
    
    def generate_improvement_suggestions(self, user_input, ai_response, quality_score):
        """Generate improvement suggestions"""
        suggestions = []
        
        if quality_score < 0.6:
            if len(ai_response.split()) < 3:
                suggestions.append("Response is too brief, provide more details")
            elif len(ai_response.split()) > 60:
                suggestions.append("Response may be too long, consider more concise expression")
            
            if "?" in user_input and "?" not in ai_response:
                suggestions.append("User's question may need a more direct answer")
        
        return suggestions
    
    def reflect_on_conversation(self, conversation_id, user_input, ai_response, sentiment):
        """Reflect on conversation"""
        quality_score = self.analyze_response_quality(user_input, ai_response, sentiment)
        suggestions = self.generate_improvement_suggestions(user_input, ai_response, quality_score)
        
        reflection_text = f"Response quality score: {quality_score:.2f}. "
        if suggestions:
            reflection_text += "Improvement suggestions: " + "; ".join(suggestions)
        else:
            reflection_text += "This response quality is good."
        
        # Store reflection result
        self.memory.store_reflection(conversation_id, reflection_text, 
                                   json.dumps(suggestions), quality_score)
        
        return reflection_text, suggestions, quality_score

class PositionalEncoding(nn.Module):
    """Positional Encoding - adapted for batch_first"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer, not trainable
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerThinkingLayer(nn.Module):
    """Enhanced Transformer Thinking Layer with sophisticated reasoning mechanism"""
    
    def __init__(self, d_model, nhead, num_layers, think_steps=serlin_config.think_steps):
        super(TransformerThinkingLayer, self).__init__()
        self.d_model = d_model
        self.think_steps = think_steps
        
        # Multi-head reasoning transformers for different thinking aspects
        self.reasoning_transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=nhead,
                    batch_first=True
                ),
                num_layers=num_layers
            ) for _ in range(think_steps)
        ])
        
        # Attention mechanisms for different information sources
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=max(1, nhead//2), 
            batch_first=True
        )
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=max(1, nhead//2), 
            batch_first=True
        )
        
        # Reasoning state controllers
        self.reasoning_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.Sigmoid()
            ) for _ in range(think_steps)
        ])
        
        # Intermediate thought processors
        self.thought_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(think_steps)
        ])
        
        # Thought fusion layer with residual connections
        self.thought_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Confidence scoring for each reasoning step
        self.confidence_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for _ in range(think_steps)
        ])
        
        # Reasoning trajectory tracking
        self.trajectory_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Thinking process monitoring
        self.thinking_monitor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, think_steps),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, context, knowledge_vector=None, memory_vector=None):
        # context shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = context.size()
        device = context.device
        
        # Initialize thinking process
        thoughts = []
        confidence_scores = []
        current_thought = context
        reasoning_trajectory = []
        
        # Multi-step thinking process with enhanced reasoning
        for step in range(self.think_steps):
            # Apply specialized reasoning transformer for this step
            thought_output = self.reasoning_transformers[step](current_thought)
            
            # Enhanced knowledge integration with attention
            if knowledge_vector is not None:
                knowledge_expanded = knowledge_vector.unsqueeze(1).expand(-1, seq_len, -1)
                # Use attention to selectively incorporate knowledge
                attended_knowledge, _ = self.knowledge_attention(
                    thought_output, knowledge_expanded, knowledge_expanded
                )
            else:
                attended_knowledge = torch.zeros_like(thought_output)
            
            # Enhanced memory integration with attention
            if memory_vector is not None:
                memory_expanded = memory_vector.unsqueeze(1).expand(-1, seq_len, -1)
                # Use attention to selectively incorporate memory
                attended_memory, _ = self.memory_attention(
                    thought_output, memory_expanded, memory_expanded
                )
            else:
                attended_memory = torch.zeros_like(thought_output)
            
            # Dynamic gating mechanism for information fusion
            context_mean = thought_output.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
            
            # Ensure all tensors have the same shape before concatenation
            gate_input = torch.cat([
                context_mean,  # [batch_size, seq_len, d_model]
                attended_knowledge,  # [batch_size, seq_len, d_model]  
                attended_memory  # [batch_size, seq_len, d_model]
            ], dim=-1)  # Results in [batch_size, seq_len, d_model * 3]
            
            reasoning_gate = self.reasoning_gates[step](gate_input)
            
            # Fuse information with gating
            fused_thought = thought_output + reasoning_gate * attended_knowledge + (1 - reasoning_gate) * attended_memory
            
            # Refine thought with intermediate processing
            refined_thought = self.thought_refiners[step](fused_thought)
            
            # Calculate confidence score for this reasoning step
            step_confidence = self.confidence_scorers[step](refined_thought.mean(dim=1))
            confidence_scores.append(step_confidence)
            
            # Store thought and update trajectory
            thoughts.append(refined_thought)
            reasoning_trajectory.append(refined_thought.mean(dim=1, keepdim=True))
            
            # Prepare for next reasoning step
            current_thought = refined_thought
        
        # Encode reasoning trajectory
        if reasoning_trajectory:
            trajectory_input = torch.cat(reasoning_trajectory, dim=1)
            trajectory_encoded, _ = self.trajectory_encoder(trajectory_input)
            trajectory_final = trajectory_encoded[:, -1, :].unsqueeze(1).expand(-1, seq_len, -1)
        else:
            trajectory_final = torch.zeros_like(context)
        
        # Hierarchical fusion of all thinking steps - FIXED dimension issues
        if len(thoughts) > 0:
            # Fix: Properly handle confidence scores for weighted average
            if confidence_scores:
                # Stack confidence scores and apply softmax
                confidence_stack = torch.stack(confidence_scores, dim=1)  # [batch_size, think_steps, 1]
                thought_weights = torch.softmax(confidence_stack.squeeze(-1), dim=-1)  # [batch_size, think_steps]
                
                # Apply weights to thoughts - FIXED dimension issue
                weighted_thoughts = []
                for i, thought in enumerate(thoughts):
                    # thought shape: [batch_size, seq_len, d_model]
                    # thought_weights shape: [batch_size, think_steps]
                    weight = thought_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
                    weighted_thought = thought * weight.expand(-1, seq_len, d_model)
                    weighted_thoughts.append(weighted_thought)
                
                # Sum weighted thoughts
                final_thought_from_thinking = sum(weighted_thoughts)
            else:
                # Simple average if no confidence scores
                final_thought_from_thinking = sum(thoughts) / len(thoughts)
            
            # Fuse with original context and trajectory - FIXED dimension issue
            # Use mean pooling to get global representation
            thinking_global = final_thought_from_thinking.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
            trajectory_global = trajectory_final.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
            
            fusion_input = torch.cat([thinking_global, trajectory_global], dim=-1)  # [batch_size, 1, d_model * 2]
            
            # Final fusion with residual connection
            final_thought_global = self.thought_fusion(fusion_input)  # [batch_size, 1, d_model]
            final_thought = final_thought_global.expand(-1, seq_len, -1) + context  # Residual connection
        else:
            final_thought = context
        
        # Layer normalization
        final_thought = self.layer_norm(final_thought)
        
        # Thinking process monitoring output
        thinking_weights = self.thinking_monitor(final_thought.mean(dim=1))
        
        # Store thinking diagnostics
        self.thinking_diagnostics = {
            'confidence_scores': torch.stack(confidence_scores, dim=1) if confidence_scores else None,
            'thinking_weights': thinking_weights,
            'reasoning_trajectory': reasoning_trajectory
        }
        
        return final_thought  # [batch_size, seq_len, d_model]
    #'reasoning_trajectory': reasoning_trajectory
class TransformerDialogueAI(nn.Module):
    """Transformer-based Dialogue AI - Fixed batch_first warnings"""
    
    def __init__(self, vocab_size, idx2word, d_model=512, nhead=serlin_config.nhead, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 think_steps=3, max_length=100, dropout=0.1):
        super(TransformerDialogueAI, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.idx2word = idx2word
        self.think_steps = think_steps
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Enhanced Transformer thinking layer
        self.thinking_layer = TransformerThinkingLayer(
            d_model=d_model,
            nhead=nhead,
            num_layers=2,
            think_steps=think_steps
        )
        
        # Context analyzer for thinking guidance
        self.context_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, think_steps * 3),
            nn.Tanh()
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Enhanced personality adapter - FIXED input dimension
        self.personality_adapter = nn.Sequential(
            nn.Linear(d_model + 5 + think_steps, d_model * 2),  # Added think_steps dimension
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Sentiment and topic analysis
        self.sentiment_analysis = nn.Linear(d_model, 3)
        self.topic_analysis = nn.Linear(d_model, 10)
        
        # Thinking quality assessment
        self.thinking_quality = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def encode(self, src, src_mask=None):
        """Encode input sequence"""
        # src shape: [batch_size, seq_len]
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)
        
        # Encoder expects input: [batch_size, seq_len, d_model]
        memory = self.encoder(src_embedded, src_mask)
        return memory  # [batch_size, seq_len, d_model]
    
    def thinking_process(self, memory, knowledge_vector=None, memory_vector=None, personality_vector=None):
        """Enhanced thinking process with context guidance"""
        # Analyze context to guide thinking
        context_analysis = self.context_analyzer(memory.mean(dim=1))
        context_analysis = context_analysis.view(-1, self.think_steps, 3)
        
        # Apply enhanced thinking layer
        thought_memory = self.thinking_layer(memory, knowledge_vector, memory_vector)
        
        # Assess thinking quality
        thinking_quality_score = self.thinking_quality(thought_memory.mean(dim=1))
        
        # Enhanced personality adaptation with thinking context
        if personality_vector is not None:
            batch_size, seq_len, d_model = thought_memory.size()
            personality_expanded = personality_vector.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Include thinking diagnostics in personality adaptation
            thinking_context = self.thinking_layer.thinking_diagnostics['thinking_weights']
            
            # Fix: Ensure thinking_context has correct shape [batch_size, think_steps]
            if thinking_context.dim() == 1:
                thinking_context = thinking_context.unsqueeze(0)
            
            thinking_context_expanded = thinking_context.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Ensure all dimensions match before concatenation
            thought_with_personality = torch.cat([
                thought_memory, 
                personality_expanded, 
                thinking_context_expanded
            ], dim=-1)
            
            thought_memory = self.personality_adapter(thought_with_personality)
        
        # Store thinking diagnostics
        self.thinking_diagnostics = {
            **self.thinking_layer.thinking_diagnostics,
            'context_analysis': context_analysis,
            'thinking_quality': thinking_quality_score
        }
        
        return thought_memory
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decode to generate response"""
        # tgt shape: [batch_size, seq_len]
        # memory shape: [batch_size, seq_len, d_model]
        
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # Decoder expects input: [batch_size, seq_len, d_model]
        output = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.output_projection(output)  # [batch_size, seq_len, vocab_size]
    
    def forward(self, src, tgt=None, knowledge_vector=None, 
                memory_vector=None, personality_vector=None, teacher_forcing_ratio=0.5):
        
        batch_size = src.size(0)
        device = src.device
        
        # Check if input indices are out of range
        if src.max() >= self.vocab_size:
            print(f"Warning: Input contains indices beyond vocabulary! max_index={src.max()}, vocab_size={self.vocab_size}")
            src = torch.clamp(src, 0, self.vocab_size - 1)
        
        # Encode input
        memory = self.encode(src)
        
        # Enhanced thinking process with diagnostics
        thought_memory = self.thinking_process(memory, knowledge_vector, memory_vector, personality_vector)
        
        # Decode
        if tgt is not None:
            tgt_len = tgt.size(1)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            if tgt_input.max() >= self.vocab_size:
                tgt_input = torch.clamp(tgt_input, 0, self.vocab_size - 1)
            
            tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            decoder_output = self.decode(tgt_input, thought_memory, 
                                       tgt_mask=tgt_mask, memory_mask=None)
            
            decoder_output = decoder_output.transpose(0, 1)
        else:
            decoder_output, generated_tokens = self.generate_autoregressive(thought_memory, batch_size, device)
            tgt_output = None
        
        # Enhanced sentiment and topics analysis with thinking context
        context_representation = thought_memory[:, 0, :]
        sentiment = self.sentiment_analysis(context_representation)
        topics = self.topic_analysis(context_representation)
        
        return {
            'output': decoder_output,
            'sentiment': sentiment,
            'topics': topics,
            'memory': thought_memory,
            'generated_tokens': generated_tokens if tgt is None else None,
            'thinking_diagnostics': self.thinking_diagnostics
        }
    
    def generate_autoregressive(self, memory, batch_size, device):
        """Improved autoregressive generation - prevent PAD overflow"""
        # Initialize with SOS token
        tgt = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        
        outputs = []
        generated_tokens = []
        
        # Generation parameters
        temperature = 0.9
        top_k = 30
        repetition_penalty = 1.2
        
        for i in range(min(20, self.max_length)):
            # Create target sequence mask
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Decode
            output = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=None)
            
            # Get last timestep prediction
            next_word_logits = output[:, -1, :]
            
            # Apply repetition penalty
            for token in set(generated_tokens):
                next_word_logits[0, token] /= repetition_penalty
            
            # Prevent generating PAD and SOS as content
            next_word_logits[0, 0] = -float('Inf')  # PAD
            next_word_logits[0, 1] = -float('Inf')  # SOS
            next_word_logits[0, 3] = -float('Inf')  # UNK
            
            # Temperature sampling + top-k
            if temperature > 0:
                next_word_logits = next_word_logits / temperature
                
                # top-k filtering
                if top_k > 0:
                    indices_to_remove = next_word_logits < torch.topk(next_word_logits, top_k)[0][..., -1, None]
                    next_word_logits[indices_to_remove] = -float('Inf')
                
                probabilities = torch.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
            else:
                next_word = next_word_logits.argmax(dim=-1, keepdim=True)
            
            word = self.idx2word.get(next_word.item(), '<UNK>')
            
            # If EOS generated, stop early
            if next_word.item() == 2:  # EOS
                break
            
            # If too many invalid tokens generated consecutively, stop early
            if word in ['<PAD>', '<SOS>', '<UNK>']:
                if generated_tokens.count(next_word.item()) > 3:
                    break
            
            # Add to sequence
            tgt = torch.cat([tgt, next_word], dim=1)
            outputs.append(output[:, -1:, :])
            generated_tokens.append(next_word.item())
        
        if outputs:
            final_output = torch.cat(outputs, dim=1)
            return final_output, generated_tokens
        else:
            return torch.zeros(batch_size, 1, self.vocab_size).to(device), []
    
    def generate_square_subsequent_mask(self, sz):
        """Generate sequence mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class DialogueDataProcessor:
    """Dialogue data processor"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4
        
    def build_vocab(self, dialogues, min_freq=1):
        """Build vocabulary"""
        word_freq = defaultdict(int)
    
        for dialogue in dialogues:
            for text in [dialogue['input'], dialogue['output']]:
                words = self.tokenize(text)
                for word in words:
                    word_freq[word] += 1
    
        print(f"Found {len(word_freq)} distinct words")
    
        # Add special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
    
        # Add all words
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
    
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built, size: {self.vocab_size}")
        print(f"Sample vocabulary: {list(self.word2idx.keys())[:20]}")
    
        return self.vocab_size
    
    def auto_expand_vocab(self, text, min_freq=1):
        """Auto-expand vocabulary - fixed version"""
        new_words = []
        words = self.tokenize(text)

        for word in words:
            # Skip empty words and special tokens
            if not word.strip() or word in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                continue
        
            if word not in self.word2idx:
                # Check if vocabulary has reached limit (optional)
                #if self.vocab_size >= 15000:
                #   print(f"Warning: Vocabulary has reached limit {self.vocab_size}, skipping new word addition")
                #   continue
            
                # Add new word to vocabulary
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                new_words.append(word)

        if new_words:
            print(f"Vocabulary auto-expanded: Added {len(new_words)} new words")
            self.save_vocab()
    
        return new_words

    def save_vocab(self, file_path="vocab.json"):
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to: {file_path}")

    def load_vocab(self, file_path="vocab.json"):
        """Load vocabulary from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        
            self.word2idx = vocab_data['word2idx']
            self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
            self.vocab_size = vocab_data['vocab_size']
            print(f"Vocabulary loaded from {file_path}, size: {self.vocab_size}")
            return True
        except FileNotFoundError:
            print(f"Vocabulary file {file_path} does not exist")
            return False
        except Exception as e:
            print(f"Failed to load vocabulary: {e}")
            return False

    def add_basic_vocabulary(self):
        """Add basic vocabulary"""
        basic_words = [
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 
            'this', 'that', 'which', 'who', 'what', 'how', 'why',
            'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 
            'can', 'could', 'will', 'would', 'should', 'may', 'might',
            'say', 'ask', 'answer', 'tell', 'know', 'think', 'feel', 
            'good', 'bad', 'right', 'wrong', 'happy', 'sad', 
            'like', 'love', 'hate', 'want', 'need', 
            'hello', 'hi', 'hey', 'goodbye', 'bye', 'thanks', 'thank you',
            'yes', 'no', 'okay', 'ok', 'please', 'sorry',
            'time', 'day', 'today', 'tomorrow', 'yesterday', 'now', 'later',
            'person', 'people', 'friend', 'teacher', 'student', 'computer', 'phone',
            'work', 'learn', 'help', 'understand', 'explain',
            'problem', 'question', 'answer', 'thing', 'something', 'nothing',
            'one', 'two', 'three', 'four', 'five', 'ten', 'hundred', 'thousand', 'million',
            'first', 'second', 'third', 'last', 'next', 'previous',
            'up', 'down', 'left', 'right', 'front', 'back', 'inside', 'outside',
            'year', 'month', 'week', 'day', 'hour', 'minute', 'second'
        ]
    
        added_count = 0
        for word in basic_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                added_count += 1
    
        if added_count > 0:
            print(f"Added {added_count} basic vocabulary words")
            self.save_vocab()
    
        return added_count
    
    def tokenize(self, text):
        """Tokenization function"""
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
        
            if char.isalpha():
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
        """Encode text to index sequence"""
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
        """Decode index sequence to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.vocab_size
    
    def sync_vocab_to_array(self):
        """Sync vocabulary to array - ensure all words are in vocabulary"""
        print(f"Syncing vocabulary to array, current vocabulary size: {self.vocab_size}")
        
        # Ensure all special tokens exist
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            if token not in self.word2idx:
                self.word2idx[token] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = token
                print(f"Added special token: {token}")
        
        # Update vocabulary size
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size after sync: {self.vocab_size}")
        
        return self.vocab_size

class TransformerTrainer:
    """Improved Transformer trainer"""
    
    def __init__(self, model, processor, learning_rate=0.0001):
        self.model = model
        self.processor = processor
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Improved loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            reduction='mean'
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        
    def train_epoch(self, dataloader):
        """Improved training epoch"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            input_seq = batch['input']
            target_seq = batch['output']
            
            # Check if input is beyond vocabulary range
            if input_seq.max() >= self.model.vocab_size or target_seq.max() >= self.model.vocab_size:
                print(f"Warning: Input or target contains indices beyond vocabulary!")
                continue
            
            # Forward propagation
            outputs = self.model(input_seq, target_seq)
            predictions = outputs['output']
            
            # Calculate loss - only non-PAD positions
            pred_flat = predictions.reshape(-1, predictions.size(-1))
            target_flat = target_seq[:, 1:].reshape(-1)  # Remove SOS for target
            
            # Filter out PAD positions
            non_pad_mask = target_flat != 0
            if non_pad_mask.sum() == 0:
                continue  # Skip all-PAD batches
                
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

# Example training data
training_dialogues = [
    {"input": "hello", "output": "Hello! I am Serlin, nice to meet you."},
    {"input": "how are you", "output": "I'm doing well, thank you for asking! How about you?"},
    {"input": "what is your name", "output": "My name is Serlin, I am an AI assistant."},
    {"input": "what can you do", "output": "I can answer questions, chat, and learn new knowledge."},
    {"input": "what is the weather today", "output": "I cannot access real-time weather, please check a weather forecast."},
    {"input": "tell me a joke", "output": "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!"},
    {"input": "what is artificial intelligence", "output": "Artificial Intelligence is technology that enables machines to simulate human intelligence."},
    {"input": "can you learn", "output": "Yes, I can learn from conversations and improve myself."},
    {"input": "goodbye", "output": "Goodbye! Looking forward to our next chat."},
    {"input": "thank you", "output": "You're welcome! I'm here to help anytime."},
]

class SerlinTransformer:
    """Transformer-based Serlin System - Enhanced Version"""
    
    def __init__(self, d_model,nhead,num_encoder_layers,num_decoder_layers,think_steps,max_length,model_save_path="serlin_transformer.pth"):
        self.model_save_path = model_save_path
        self.training_data = []
        self.conversation_history = []
    
        # Initialize memory system
        self.memory = LongTermMemory()
        self.knowledge_base = KnowledgeBase(self.memory)
        self.context_manager = MultiTurnContext()
        self.personality_adaptation = PersonalityAdaptation(self.memory)
        self.self_reflection = SelfReflection(self.memory)
    
        # Initialize data processor
        self.processor = DialogueDataProcessor()
    
        # Try to load existing vocabulary, otherwise build new one
        if not self.processor.load_vocab():
            print("Building initial vocabulary...")
            vocab_size = self.processor.build_vocab(training_dialogues, min_freq=1)
            self.processor.add_basic_vocabulary()
            print(f"Final vocabulary size: {self.processor.vocab_size}")
        
        # Sync vocabulary to array
        self.processor.sync_vocab_to_array()
    
        # Use Transformer model
        self.model = TransformerDialogueAI(
            vocab_size=self.processor.vocab_size,
            d_model=d_model,
            idx2word=self.processor.idx2word,
            nhead=serlin_config.nhead,
            num_encoder_layers=serlin_config.num_encoder_layers,
            num_decoder_layers=serlin_config.num_decoder_layers,
            think_steps=serlin_config.think_steps,
            max_length=serlin_config.max_length
        )
    
        # Initialize trainer
        self.trainer = TransformerTrainer(self.model, self.processor)
    
        # Try to load existing model
        self.load_model()
    
    def _create_knowledge_vector(self, knowledge, confidence):
        """Create knowledge vector"""
        if knowledge:
            knowledge_hash = hashlib.md5(knowledge.encode()).hexdigest()
            knowledge_int = int(knowledge_hash[:serlin_config.nhead], 16)
            vector = np.random.RandomState(knowledge_int).randn(serlin_config.d_model)
            return torch.tensor(vector * confidence, dtype=torch.float32).unsqueeze(0)
        return torch.zeros(1, serlin_config.d_model)
    
    def _create_memory_vector(self, user_id, user_input):
        """Create memory vector"""
        user_history = self.memory.get_user_history(user_id, limit=5)
        
        if not user_history:
            return torch.zeros(1, serlin_config.d_model)
        
        memory_text = " ".join([conv['input'] + " " + conv['response'] for conv in user_history])
        memory_hash = hashlib.md5(memory_text.encode()).hexdigest()
        memory_int = int(memory_hash[:serlin_config.nhead], 16)
        vector = np.random.RandomState(memory_int).randn(serlin_config.d_model)
        
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    
    def process_user_input(self, user_id, user_input):
        """Full process for user input"""
        self.processor.auto_expand_vocab(user_input)
        self.sync_model_vocab()
        self.processor.sync_vocab_to_array()
        user_history = self.memory.get_user_history(user_id)
        user_profile, user_prefs = self.personality_adaptation.get_user_profile(user_id)
        
        # Knowledge retrieval
        knowledge, confidence = self.knowledge_base.query_knowledge(user_input)
        knowledge_vector = self._create_knowledge_vector(knowledge, confidence)
        
        # Memory retrieval
        memory_vector = self._create_memory_vector(user_id, user_input)
        
        # Generate response
        response_data = self._generate_response(user_input, knowledge_vector, memory_vector, user_profile)
        
        topics = self._extract_topics(user_input, response_data['response'])
        
        # Process sentiment value
        sentiment_tensor = response_data['sentiment']
        if isinstance(sentiment_tensor, torch.Tensor):
            sentiment_value = sentiment_tensor.mean().item()
        else:
            sentiment_value = float(sentiment_tensor)
        
        # Store conversation
        self.memory.store_conversation(user_id, user_input, response_data['response'], 
                                     sentiment_value, topics)
        
        # Self-reflection
        reflection, suggestions, quality = self.self_reflection.reflect_on_conversation(
            len(user_history) + 1, user_input, response_data['response'], sentiment_value
        )
        
        # Learning
        self.knowledge_base.learn_from_conversation(user_input, response_data['response'])
        
        # Update user profile
        self.personality_adaptation.update_user_profile(
            user_id, user_input, response_data['response'], sentiment_value
        )
        
        # Update context
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
            result['knowledge_used'] = f"Used knowledge: {knowledge[:serlin_config.max_length]}..."
        if len(user_history) > 0:
            result['memory_accessed'] = f"Accessed {len(user_history)} historical records"
            
        return result

    def _generate_response(self, user_input, knowledge_vector, memory_vector, user_profile):
        """Generate response"""
        try:
            # Sync vocabulary before encoding input
            self.sync_model_vocab()
        
            input_seq = self.processor.encode(user_input, auto_expand=False)
            
            # Check if input sequence is valid
            if not input_seq or len(input_seq) == 0:
                return {
                    'response': "Sorry, I didn't understand your input.",
                    'sentiment': torch.tensor([0.5, 0.3, 0.2])
                }
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long)
        
            # Check if input indices are out of range
            if input_tensor.max() >= self.model.vocab_size:
                print(f"Warning: Input contains indices beyond vocabulary! max_index={input_tensor.max()}, vocab_size={self.model.vocab_size}")
                # Replace out-of-range indices with UNK
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
            
                # Check output shape
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
            print(f"Error generating response: {e}")
            return {
                'response': "Sorry, I encountered some technical issues, please try again later.",
                'sentiment': torch.tensor([0.5, 0.3, 0.2])
            }
                
    def postprocess_response(self, response):
        """Improved post-processing function"""
        if not response or response.strip() == "":
            return "I'm still learning how to answer this question."
    
        words = response.split()
        valid_words = []
    
        for word in words:
            # Relax filtering conditions
            if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] or not word.strip():
                continue
            # Allow single characters
            if len(word) == 1 and word.isalpha():
                valid_words.append(word)
            # Allow shorter words
            elif len(word) > 0:
                valid_words.append(word)
    
        # Relax valid word count requirement
        if len(valid_words) == 0:
            return "I'm still learning how to answer this question."
        elif len(valid_words) == 1:
            # For single word, try to build more meaningful response
            word = valid_words[0]
            if word in ['hello', 'hi', 'hey']:
                return f"{word}! I am Serlin."
            elif word in ['thanks', 'thank']:
                return f"{word}! You're welcome."
            elif word in ['goodbye', 'bye']:
                return f"{word}! See you next time."
            else:
                return f"{word}."
    
        final_response = ' '.join(valid_words)
        final_response = final_response.strip('.,!?;')
    
        # Ensure response ends with appropriate punctuation
        if not any(final_response.endswith(p) for p in ['.', '!', '?']):
            final_response += '.'
    
        return final_response
    
    def _extract_topics(self, user_input, response):
        """Extract topics"""
        common_words = {'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
        words = set(user_input.lower().split() + response.lower().split())
        topics = [word for word in words if len(word) > 1 and word not in common_words and word != '<unk>']
        return topics[:serlin_config.think_steps]
    
    def sync_model_vocab(self):
        """Fixed vocabulary synchronization method - ensure embedding layer expands correctly"""
        current_vocab_size = self.processor.vocab_size
        model_vocab_size = self.model.vocab_size

        if current_vocab_size != model_vocab_size:
            print(f"Vocabulary size mismatch: model={model_vocab_size}, processor={current_vocab_size}")
            print("Reinitializing Transformer model...")
    
            # Save important parameters
            old_state_dict = None
            try:
                old_state_dict = self.model.state_dict().copy()
            except:
                pass
    
            # Reinitialize model
            self.model = TransformerDialogueAI(
                vocab_size=current_vocab_size,
                idx2word=self.processor.idx2word,
                d_model=serlin_config.d_model,
                nhead=serlin_config.nhead,
                num_encoder_layers=serlin_config.num_encoder_layers,
                num_decoder_layers=serlin_config.num_decoder_layers,
                think_steps=serlin_config.think_steps,
                max_length=serlin_config.max_length
            )
    
            # Try to restore parameters
            if old_state_dict is not None:
                try:
                    new_state_dict = self.model.state_dict()
                
                    # Restore embedding layer parameters
                    if 'embedding.weight' in old_state_dict and 'embedding.weight' in new_state_dict:
                        old_embedding = old_state_dict['embedding.weight']
                        new_embedding = new_state_dict['embedding.weight']
                        min_size = min(old_embedding.size(0), new_embedding.size(0))
                        new_embedding[:min_size] = old_embedding[:min_size]
                        new_state_dict['embedding.weight'] = new_embedding
                        print(f"Restored first {min_size} parameters of embedding layer")
                
                    # Restore other matching parameters
                    for name, param in old_state_dict.items():
                        if name in new_state_dict and new_state_dict[name].shape == param.shape:
                            new_state_dict[name] = param
                
                    self.model.load_state_dict(new_state_dict)
                    print("Model parameters successfully restored")
                except Exception as e:
                    print(f"Failed to restore parameters: {e}, will use new model")
    
            # Update trainer
            self.trainer = TransformerTrainer(self.model, self.processor)
            print("Transformer model reinitialized")
    
        return current_vocab_size

    def chat(self, user_id, user_input, show_thinking=True):
        """Chat method"""
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
        """Display enhanced thinking process with detailed reasoning"""
        print(f"User input: '{user_input}'")
        print("Serlin enhanced thinking process:")
        
        # Display thinking diagnostics if available
        if 'thinking_diagnostics' in result:
            diagnostics = result['thinking_diagnostics']
            
            print("Reasoning Diagnostics:")
            if 'confidence_scores' in diagnostics and diagnostics['confidence_scores'] is not None:
                conf_scores = diagnostics['confidence_scores'].squeeze().tolist()
                print(f"  Step confidence scores: {[f'{score:.3f}' for score in conf_scores]}")
            
            if 'thinking_weights' in diagnostics:
                thinking_weights = diagnostics['thinking_weights'].squeeze().tolist()
                print(f"  Thinking step weights: {[f'{weight:.3f}' for weight in thinking_weights]}")
            
            if 'thinking_quality' in diagnostics:
                quality_score = diagnostics['thinking_quality'].item()
                print(f"  Overall thinking quality: {quality_score:.3f}")
            
            if 'context_analysis' in diagnostics:
                context_analysis = diagnostics['context_analysis'].squeeze().tolist()
                print(f"  Context analysis: {context_analysis}")
        
        print("Analysis results:")
        print(f"  Sentiment value: {result['sentiment_value']:.3f}")
        print(f"  Identified topics: {', '.join(result['topics'])}")
        print(f"  Response quality: {result['quality_score']:.3f}")
        
        profile = result['user_profile']
        print("Personality adaptation:")
        print(f"  Formality level: {profile['formality']:.3f}")
        print(f"  Humor level: {profile['humor_level']:.3f}")
        print(f"  Detail level: {profile['detail_level']:.3f}")
        print(f"  Empathy level: {profile['empathy_level']:.3f}")
        print(f"  Curiosity level: {profile['curiosity_level']:.3f}")
        
        print(f"Self-reflection: {result['reflection']}")
        
        # Display knowledge and memory usage
        self._display_knowledge_memory_usage(result)
        
        # Display thinking process summary
        print("Thinking process completed")
    
    def _display_knowledge_memory_usage(self, result):
        """Display knowledge and memory usage"""
        if 'knowledge_used' in result:
            print(f"Knowledge used: {result['knowledge_used']}")
        if 'memory_accessed' in result:
            print(f"Memory accessed: {result['memory_accessed']}")

    def get_system_status(self, user_id):
        """Get system status"""
        user_history = self.memory.get_user_history(user_id)
        profile, prefs = self.personality_adaptation.get_user_profile(user_id)
        
        status = {
            "user_id": user_id,
            "conversation_history_count": len(user_history),
            "current_topics": self.context_manager.get_recent_topics(),
            "personality_profile": profile,
            "training_data_count": len(self.training_data),
            "vocabulary_size": self.processor.vocab_size,
            "model_vocabulary_size": self.model.vocab_size
        }
        
        return status

    def add_training_data(self, questions, answers):
        """Add training data"""
        for q, a in zip(questions, answers):
            if len(q.strip()) == 0 or len(a.strip()) == 0:
                continue
            
            self.training_data.append({"input": q, "output": a})
            
            # Auto-expand vocabulary
            self.processor.auto_expand_vocab(q)
            self.processor.auto_expand_vocab(a)
            
            # Sync vocabulary to array
            self.processor.sync_vocab_to_array()
            
            # Sync model
            self.sync_model_vocab()

    def create_dataloader(self, batch_size=2):
        """Create data loader"""
        if not self.training_data:
            return []
        
        inputs = []
        outputs = []
    
        for dialogue in self.training_data:
            try:
                # Encode without auto-expanding vocabulary to avoid interference
                input_seq = self.processor.encode(dialogue['input'], auto_expand=False)
                output_seq = self.processor.encode(dialogue['output'], auto_expand=False)
            
                # Check sequence quality
                if len(input_seq) < 2 or len(output_seq) < 2:
                    continue
                
                # Ensure sequence ends with EOS
                if output_seq[-1] != self.processor.word2idx['<EOS>']:
                    output_seq = output_seq[:-1] + [self.processor.word2idx['<EOS>']]
            
                # Pad to appropriate length
                max_len = 15
                input_seq = input_seq[:max_len] + [0] * (max_len - len(input_seq))
                output_seq = output_seq[:max_len] + [0] * (max_len - len(output_seq))
            
                inputs.append(input_seq)
                outputs.append(output_seq)
            
            except Exception as e:
                print(f"Error processing training data: {e}")
                continue
    
        if not inputs:
            print("No valid training data!")
            return []
    
        print(f"Created data loader: {len(inputs)} valid data")
        return [{
            'input': torch.tensor(inputs, dtype=torch.long),
            'output': torch.tensor(outputs, dtype=torch.long)
        }]

    def improved_train(self, epochs=100, batch_size=4, save_after_training=True, early_stopping=True):
        """Improved training method"""
        if not self.training_data:
            return {"status": "error", "message": "No training data"}
    
        if len(self.training_data) < 5:
            return {"status": "error", "message": "Too little training data for effective training"}
    
        self.sync_model_vocab()
        print(f"Starting improved Transformer training, using {len(self.training_data)} training data...")
        print(f"Vocabulary size: {self.processor.vocab_size}")
        print(f"Model vocabulary size: {self.model.vocab_size}")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
    
        dataloader = self.create_dataloader(batch_size)
        if not dataloader:
            return {"status": "error", "message": "Data loader creation failed"}
    
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
            
                # Early stopping check
                if early_stopping:
                    if epoch % 5 == 0:
                        val_loss = self._compute_validation_loss(dataloader)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            patience_counter = 0
                            if save_after_training:
                                self.save_model(f"best_{self.model_save_path}")
                            print(f"Found better model, validation loss: {val_loss:.4f}")
                        else:
                            patience_counter += 1
                            print(f"Early stopping count: {patience_counter}/{patience}")
                
                    if patience_counter >= patience:
                        print(f"Early stopping triggered, stopped at epoch {epoch}")
                        break
            
                # Test model every 15 epochs
                if epoch % 15 == 0 and epoch > 0:
                    self._test_model_improved()
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                continue
    
        if save_after_training:
            final_path = self.save_model()
            print(f"Final model saved to: {final_path}")
    
        print("Training completed!")
        final_loss = train_losses[-1] if train_losses else 0
        
        return {
            "status": "success", 
            "message": "Training completed",
            "final_loss": final_loss,
            "epochs_completed": len(train_losses)
        }

    def _compute_validation_loss(self, dataloader):
        """Compute validation loss"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
    
        with torch.no_grad():
            for batch in dataloader:
                input_seq = batch['input']
                target_seq = batch['output']
            
                outputs = self.model(input_seq, target_seq)
                predictions = outputs['output']
            
                # Calculate loss
                pred_flat = predictions.reshape(-1, predictions.size(-1))
                target_flat = target_seq[:, 1:].reshape(-1)
            
                loss = self.trainer.criterion(pred_flat, target_flat)
                total_loss += loss.item()
                total_batches += 1
    
        self.model.train()
        return total_loss / total_batches if total_batches > 0 else float('inf')

    def _test_model_improved(self):
        """Improved model testing"""
        self.model.eval()
        with torch.no_grad():
            test_inputs = ["hello", "what is your name", "what can you do", "goodbye", "thank you"]
            print("\n=== Model Testing ===")
        
            for test_input in test_inputs:
                try:
                    input_seq = self.processor.encode(test_input, auto_expand=False)
                    input_tensor = torch.tensor([input_seq], dtype=torch.long)
                
                    # Check if input indices are out of range
                    if input_tensor.max() >= self.model.vocab_size:
                        print(f"Warning: Test input '{test_input}' contains indices beyond vocabulary!")
                        input_tensor = torch.clamp(input_tensor, 0, self.model.vocab_size - 1)
                
                    outputs = self.model(input_tensor)
                    predictions = outputs['output']
                
                    if predictions.numel() == 0:
                        response = "[no output]"
                        raw_response = "[no output]"
                    else:
                        response_indices = predictions.argmax(dim=-1)[0].cpu().numpy()
                        raw_response = self.processor.decode(response_indices)
                        response = self.postprocess_response(raw_response)
                
                    print(f"  Input: '{test_input}'")
                    print(f"  Raw output: '{raw_response}'")
                    print(f"  Processed: '{response}'")
                    print()
                
                except Exception as e:
                    print(f"  Error testing '{test_input}': {e}")
        
            print("================")
    
        self.model.train()

    def validate_training_data(self):
        """Validate training data"""
        if not self.training_data:
            return {"status": "error", "message": "No training data"}

        valid_count = 0
        for i, data in enumerate(self.training_data):
            try:
                # First auto-expand vocabulary
                self.processor.auto_expand_vocab(data['input'])
                self.processor.auto_expand_vocab(data['output'])
            
                # Sync vocabulary to array
                self.processor.sync_vocab_to_array()
            
                # Re-sync model vocabulary
                self.sync_model_vocab()
            
                # Encode again
                input_encoded = self.processor.encode(data['input'], auto_expand=False)
                output_encoded = self.processor.encode(data['output'], auto_expand=False)
            
                # Check for unknown words
                has_unk = any(idx == self.processor.word2idx['<UNK>'] for idx in input_encoded + output_encoded)
            
                if not has_unk:
                    valid_count += 1
                
            except Exception as e:
                continue

        # Relax validation standards, allow training as long as there is data
        is_valid = valid_count > 0
        
        return {
            "status": "success" if is_valid else "error",
            "valid_count": valid_count,
            "total_count": len(self.training_data),
            "is_valid": is_valid
        }

    def get_conversation_summary(self):
        """Get conversation summary"""
        if not self.conversation_history:
            return "No conversation history yet"
        
        summary = f"Conversation summary with user:\n"
        summary += f"Total conversation turns: {len(self.conversation_history)}\n"
        
        # Calculate average response quality
        avg_quality = np.mean([conv['result']['quality_score'] for conv in self.conversation_history])
        summary += f"Average response quality: {avg_quality:.3f}\n"
        
        # Extract common topics
        all_topics = []
        for conv in self.conversation_history:
            all_topics.extend(conv['result']['topics'])
        
        if all_topics:
            from collections import Counter
            topic_counts = Counter(all_topics)
            common_topics = topic_counts.most_common(serlin_config.think_steps)
            summary += f"Common topics: {', '.join([f'{topic}({count})' for topic, count in common_topics])}\n"
        
        return summary

    def _convert_tensors_to_serializable(self, obj):
        """Convert Tensor objects to JSON serializable format"""
        if isinstance(obj, torch.Tensor):
            # If single value Tensor, convert to float
            if obj.numel() == 1:
                return obj.item()
            # If vector, convert to list
            else:
                return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For other non-serializable types, convert to string
            return str(obj)

    def export_conversation(self, filename=None):
        """Fixed export conversation history method"""
        if filename is None:
            filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
        # Convert Tensor objects in conversation history
        serializable_history = self._convert_tensors_to_serializable(self.conversation_history)
    
        export_data = {
            'export_time': datetime.datetime.now().isoformat(),
            'conversations': serializable_history
        }
    
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
            print(f"Conversation exported to: {filename}")
            return filename
        except Exception as e:
            print(f"Export failed: {e}")
            return None

    def reset_model(self):
        """Reset model to initial state"""
        print("Resetting model...")
    
        # Reinitialize model
        self.model = TransformerDialogueAI(
            vocab_size=self.processor.vocab_size,
            idx2word=self.processor.idx2word,
            d_model=serlin_config.d_model,
            nhead=serlin_config.nhead,
            num_encoder_layers=serlin_config.num_encoder_layers,
            num_decoder_layers=serlin_config.num_decoder_layers,
            think_steps=serlin_config.think_steps,
            max_length=serlin_config.max_length
        )
    
        # Reinitialize trainer
        self.trainer = TransformerTrainer(self.model, self.processor)
    
        # Clear training data (optional)
        self.training_data = []
    
        print("Model reset")
        return {"status": "success", "message": "Model reset successfully"}

    def save_model(self, path=None):
        """Save model"""
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
        print(f"Transformer model saved to: {path}")
        return path
    
    def load_model(self, path=None):
        """Load model"""
        if path is None:
            path = self.model_save_path
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            saved_vocab_size = checkpoint.get('vocab_size', 0)
            current_vocab_size = self.processor.vocab_size
            
            if saved_vocab_size != current_vocab_size:
                print(f"Vocabulary size mismatch: saved model({saved_vocab_size}) vs current model({current_vocab_size})")
                return self._rebuild_model_from_checkpoint(checkpoint, path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'training_data' in checkpoint:
                self.training_data = checkpoint['training_data']
            
            print(f"Transformer model loaded from {path}")
            print(f"Vocabulary size: {self.processor.vocab_size}")
            return {"status": "success", "message": "Model loaded successfully"}
            
        except FileNotFoundError:
            print(f"Model file {path} does not exist, will use initial model")
            return {"status": "error", "message": "Model file not found"}
        except Exception as e:
            print(f"Error loading model: {e}")
            return {"status": "error", "message": f"Error loading model: {e}"}
    
    def _rebuild_model_from_checkpoint(self, checkpoint, path):
        """Rebuild model from checkpoint"""
        try:
            saved_word2idx = checkpoint.get('word2idx')
            saved_idx2word = checkpoint.get('idx2word')
            saved_vocab_size = checkpoint.get('vocab_size', 0)
            
            if not saved_word2idx or not saved_idx2word:
                print("No valid vocabulary information found in checkpoint")
                return {"status": "error", "message": "No valid vocabulary in checkpoint"}
            
            self.processor.word2idx = saved_word2idx
            self.processor.idx2word = saved_idx2word
            self.processor.vocab_size = saved_vocab_size
            
            self.model = TransformerDialogueAI(
                vocab_size=saved_vocab_size,
                idx2word=self.processor.idx2word,
                d_model=serlin_config.d_model,
                nhead=serlin_config.nhead,
                num_encoder_layers=serlin_config.num_encoder_layers,
                num_decoder_layers=serlin_config.num_decoder_layers,
                think_steps=serlin_config.think_steps,
                max_length=serlin_config.max_length
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer = TransformerTrainer(self.model, self.processor)
            
            if 'training_data' in checkpoint:
                self.training_data = checkpoint['training_data']
            
            print(f"Transformer model rebuilt and loaded from {path}")
            print(f"Vocabulary size: {self.processor.vocab_size}")
            return {"status": "success", "message": "Model rebuilt and loaded successfully"}
            
        except Exception as e:
            print(f"Failed to rebuild model: {e}")
            return {"status": "error", "message": f"Failed to rebuild model: {e}"}

    def get_vocabulary_info(self):
        """Get vocabulary information"""
        
        return {
            "status": "success",
            "vocab_size": self.processor.vocab_size,
            "model_vocab_size": self.model.vocab_size,
            "sample_words": list(self.processor.word2idx.keys())[:50]
        }

    def expand_vocabulary(self, words_text):
        """Expand vocabulary with new words"""
        added_words = self.processor.auto_expand_vocab(words_text)
        self.processor.sync_vocab_to_array()
        self.sync_model_vocab()
        
        return {
            "status": "success",
            "added_words": added_words,
            "new_vocab_size": self.processor.vocab_size
        }

    def reload_vocabulary(self):
        """Reload vocabulary from file"""
        if self.processor.load_vocab():
            self.processor.sync_vocab_to_array()
            # Reinitialize model to adapt to new vocabulary size
            self.model = TransformerDialogueAI(
                vocab_size=self.processor.vocab_size,
                idx2word=self.processor.idx2word,
                d_model=serlin_config.d_model,
                nhead=serlin_config.nhead,
                num_encoder_layers=serlin_config.num_encoder_layers,
                num_decoder_layers=serlin_config.num_decoder_layers,
                think_steps=serlin_config.think_steps,
                max_length=serlin_config.max_length
            )
            self.trainer = TransformerTrainer(self.model, self.processor)
            return {"status": "success", "message": "Vocabulary reloaded and model reinitialized"}
        else:
            return {"status": "error", "message": "Failed to reload vocabulary"}

    def fix_model_vocab(self):
        """Manually sync vocabulary"""
        self.sync_model_vocab()
        return {"status": "success", "message": "Model vocabulary synchronized"}

    def create_training_template(self):
        """Create training data template"""
        template = {
            "description": "Serlin training data template",
            "version": "1.0",
            "created_date": datetime.datetime.now().isoformat(),
            "training_data": [
                {
                    "input": "hello",
                    "output": "Hello! I am Serlin, nice to meet you."
                },
                {
                    "input": "what is your name",
                    "output": "My name is Serlin, I am a Transformer-based AI assistant."
                },
                {
                    "input": "what can you do",
                    "output": "I can answer questions, chat, learn new knowledge, and conduct multi-turn conversations."
                }
            ]
        }
        
        #with open(file_path, 'w', encoding='utf-8') as f:
        #   json.dump(template, f, ensure_ascii=False, indent=2)
        
        #print(f"Training data template created: {file_path}")
        return template

    def batch_train_from_json(self, file_path):
        """Batch train from JSON file"""
        try:
            
            training_data = file_path
            
            # Support multiple JSON formats
            if isinstance(training_data, list):
                # Format 1: [{"input": "question", "output": "answer"}, ...]
                questions = [item['input'] for item in training_data if 'input' in item and 'output' in item]
                answers = [item['output'] for item in training_data if 'input' in item and 'output' in item]
            elif isinstance(training_data, dict) and 'training_data' in training_data:
                # Format 2: {"training_data": [{"input": "question", "output": "answer"}, ...]}
                questions = [item['input'] for item in training_data['training_data'] if 'input' in item and 'output' in item]
                answers = [item['output'] for item in training_data['training_data'] if 'input' in item and 'output' in item]
            else:
                return {"status": "error", "message": "Unsupported JSON format"}
        
            if questions and answers:
                self.add_training_data(questions, answers)
                return {
                    "status": "success", 
                    "message": f"Loaded {len(questions)} training pairs from JSON file",
                    "training_data_count": len(self.training_data)
                }
            else:
                return {"status": "error", "message": "No valid training data found in JSON file"}
            
        except json.JSONDecodeError:
            return {"status": "error", "message": f"JSON file format error: {file_path}"}
        except Exception as e:
            return {"status": "error", "message": f"Error reading JSON file: {e}"}