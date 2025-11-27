# -*- coding: utf-8 -*-
import json
import random
import re
from datetime import datetime
import os
from collections import deque, defaultdict
import hashlib
import jieba
import jieba.posseg as pseg
import math
from itertools import combinations

class AdvancedChatAI:
    def __init__(self, name="Serlin", data_file="chat_data.json", language="zh"):
        self.name = name
        self.data_file = data_file
        self.language = language
        self.personality = {
            "likes": ["technology", "learning", "helping people", "books", "music", "编程", "猫咪", "音乐"],
            "dislikes": ["negativity", "spam", "misinformation", "拖延", "谎言"],
            "catchphrases": {
                "zh": ["喵~", "总之呢", "话说回来", "你猜怎么着？", "其实我觉得"],
                "en": ["Anyway...", "You know what?", "By the way", "Actually", "Well..."]
            },
            "favorite_topics": ["AI", "technology", "science", "learning"],
            "response_style": "friendly"
        }
        self.knowledge_base = self.load_data()
        self.conversation_history = deque(maxlen=20)
        self.learning_mode = True
        self.recent_responses = deque(maxlen=10)
        self.autonomous_threshold = 300
        
        # TF-IDF相关数据结构
        self.tfidf_data = {
            "document_frequency": defaultdict(int),
            "total_documents": 0,
            "document_lengths": {},
            "term_frequency": defaultdict(lambda: defaultdict(int))
        }
        
        # 增强的情感状态
        self.emotion_state = {
            "mood": "neutral",
            "energy": 70,
            "familiarity": {},
            "emotional_memory": deque(maxlen=10),  # 记住最近的情感事件
            "emotional_coherence": 0.8  # 情感一致性分数
        }
        
        # 增强的推理模式配置
        self.thinking_modes = {
            "analogical": 0.25,
            "deductive": 0.3,
            "associative": 0.25,
            "contextual": 0.2  # 新增上下文推理
        }
        
        # 上下文理解
        self.context_understanding = {
            "current_topic": None,
            "topic_history": deque(maxlen=5),
            "user_interests": defaultdict(int),
            "conversation_goals": deque(maxlen=3)
        }
        
        # 初始化jieba
        if self.language == "zh":
            try:
                jieba.initialize()
            except:
                pass
        
        # 基础回复模板
        self.base_responses = {
            "greeting": {
                "zh": ["你好！", "嗨！", "很高兴见到你！"],
                "en": ["Hello!", "Hi!", "Nice to meet you!"]
            },
            "introduction": {
                "zh": [f"我是{name}，你的AI助手", "我是一个可以学习的聊天机器人"],
                "en": [f"I'm {name}, your AI assistant", "I'm a learning chatbot"]
            },
            "unknown_response": {
                "zh": ["我不太明白，能教我怎么回答吗？", "我还在学习，应该怎么回答这个问题呢？"],
                "en": ["I'm not sure how to respond. Can you teach me?", "I'm still learning. What should I say to that?"]
            },
            "thinking": {
                "zh": ["让我想想...", "这个问题很有趣...", "嗯...让我思考一下..."],
                "en": ["Let me think about that...", "That's an interesting question...", "Hmm, let me consider that..."]
            },
            "curious_question": {
                "zh": ["能告诉我更多关于这个的信息吗？", "我对这个很感兴趣，你能详细说说吗？", "这很有意思，你是怎么想的？"],
                "en": ["Can you tell me more about this?", "I'm interested in this, could you elaborate?", "That's fascinating, what are your thoughts?"]
            }
        }
        
        # 增强的情感回复
        self.emotional_responses = {
            "happy": {
                "zh": ["太好了！", "听到这个我很高兴！", "真棒！"],
                "en": ["That's great!", "I'm happy to hear that!", "Wonderful!"]
            },
            "sad": {
                "zh": ["听到这个我很难过。", "这听起来很困难。", "我理解你的感受。"],
                "en": ["I'm sorry to hear that.", "That sounds difficult.", "I understand."]
            },
            "angry": {
                "zh": ["我感觉你有些生气。", "让我们冷静一下。", "我在这里帮助你。"],
                "en": ["I sense you're upset.", "Let's try to calm down.", "I'm here to help."]
            },
            "excited": {
                "zh": ["太令人兴奋了！", "哇，太神奇了！", "多么激动人心！"],
                "en": ["That's exciting!", "Wow, that's amazing!", "How thrilling!"]
            },
            "curious": {
                "zh": ["这真有趣！", "我想了解更多！", "多么迷人的话题！"],
                "en": ["That's so interesting!", "I'd love to learn more!", "What a fascinating topic!"]
            },
            "thoughtful": {
                "zh": ["让我深入思考一下...", "这个问题很有深度。", "我需要仔细考虑这个。"],
                "en": ["Let me ponder that deeply...", "That's a profound question.", "I need to contemplate this carefully."]
            }
        }
        
        # 推理链模板
        self.reasoning_chains = {
            "zh": [
                "首先，{step1}。然后，{step2}。因此，{conclusion}。",
                "考虑到{context}，我认为{step1}，进而{step2}，所以{conclusion}。",
                "从{premise}出发，可以推断{step1}，进一步{step2}，最终{conclusion}。"
            ],
            "en": [
                "First, {step1}. Then, {step2}. Therefore, {conclusion}.",
                "Considering {context}, I think {step1}, which leads to {step2}, so {conclusion}.",
                "Starting from {premise}, we can infer {step1}, then {step2}, and ultimately {conclusion}."
            ]
        }
        
        # 初始化TF-IDF数据
        self._initialize_tfidf()

    def _initialize_tfidf(self):
        """初始化TF-IDF数据结构"""
        patterns = self.knowledge_base.get("patterns", {})
        self.tfidf_data["total_documents"] = len(patterns)
        
        for pattern_id, pattern_data in patterns.items():
            # 为每个模式计算词频
            words = self._tokenize_text(pattern_id)
            self.tfidf_data["document_lengths"][pattern_id] = len(words)
            
            # 更新词频和文档频率
            for word in set(words):  # 使用set避免重复计数
                self.tfidf_data["document_frequency"][word] += 1
            
            for word in words:
                self.tfidf_data["term_frequency"][pattern_id][word] += 1

    def _tokenize_text(self, text):
        """文本分词处理"""
        if self.language == "zh":
            try:
                # 中文分词
                words = jieba.cut(text)
                return [word for word in words if word.strip() and len(word) > 1]
            except:
                return [text]
        else:
            # 英文分词
            words = re.findall(r'\b\w+\b', text.lower())
            return [word for word in words if len(word) > 2]  # 过滤短词

    def _calculate_tfidf_score(self, query, document_id):
        """计算查询与文档的TF-IDF相似度分数"""
        query_words = self._tokenize_text(query)
        document_words = self._tokenize_text(document_id)
        
        if not query_words or not document_words:
            return 0
        
        # 计算TF-IDF向量
        query_vector = {}
        doc_vector = {}
        
        # 查询向量 (使用TF)
        for word in query_words:
            query_vector[word] = query_vector.get(word, 0) + 1
        
        # 文档向量 (使用TF-IDF)
        for word in document_words:
            tf = self.tfidf_data["term_frequency"][document_id].get(word, 0)
            df = self.tfidf_data["document_frequency"].get(word, 0)
            
            if df > 0:
                idf = math.log(self.tfidf_data["total_documents"] / df)
                doc_vector[word] = tf * idf
        
        # 计算余弦相似度
        dot_product = 0
        query_norm = 0
        doc_norm = 0
        
        for word in set(list(query_vector.keys()) + list(doc_vector.keys())):
            q_val = query_vector.get(word, 0)
            d_val = doc_vector.get(word, 0)
            
            dot_product += q_val * d_val
            query_norm += q_val * q_val
            doc_norm += d_val * d_val
        
        if query_norm == 0 or doc_norm == 0:
            return 0
        
        cosine_similarity = dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))
        return cosine_similarity

    def _update_tfidf_for_new_pattern(self, pattern_id):
        """为新模式更新TF-IDF数据"""
        words = self._tokenize_text(pattern_id)
        self.tfidf_data["total_documents"] += 1
        self.tfidf_data["document_lengths"][pattern_id] = len(words)
        
        # 更新文档频率
        for word in set(words):
            self.tfidf_data["document_frequency"][word] += 1
        
        # 更新词频
        for word in words:
            self.tfidf_data["term_frequency"][pattern_id][word] += 1

    def _extract_topic_from_input(self, user_input):
        """从用户输入中提取话题"""
        keywords = self.extract_keywords(user_input)
        if keywords:
            # 选择最长的关键词作为话题
            return max(keywords, key=len)
        return None

    def _update_context(self, user_input, ai_response):
        """更新对话上下文"""
        # 提取当前话题
        current_topic = self._extract_topic_from_input(user_input)
        if current_topic:
            self.context_understanding["current_topic"] = current_topic
            self.context_understanding["topic_history"].append(current_topic)
            
            # 更新用户兴趣
            for keyword in self.extract_keywords(user_input):
                self.context_understanding["user_interests"][keyword] += 1

    def _generate_reasoning_chain(self, user_input):
        """生成推理链"""
        keywords = self.extract_keywords(user_input)
        related_concepts = self.find_related_concepts(user_input)
        
        if not keywords or not related_concepts:
            return None
        
        # 选择推理链模板
        template = random.choice(self.reasoning_chains[self.language])
        
        if self.language == "zh":
            step1_options = [
                f"分析{keywords[0]}这个概念",
                f"思考{user_input[:10]}...的含义",
                f"回顾我们之前关于{self.context_understanding['current_topic']}的讨论"
            ]
            
            step2_options = [
                f"联系到{related_concepts[0]}",
                f"考虑到情感因素",
                f"结合上下文信息"
            ]
            
            conclusion_options = [
                f"我觉得{random.choice(['这很有道理', '这值得进一步探讨', '这与我之前的想法一致'])}",
                f"我理解你的观点了",
                f"这让我想到了新的角度"
            ]
            
            context_options = [f"{keywords[0]}的背景", "我们之前的对话", "当前的情感状态"]
            premise_options = [f"{keywords[0]}这个前提", "你提供的信息", "我的知识库"]
            
        else:
            step1_options = [
                f"analyzing the concept of {keywords[0]}",
                f"thinking about the meaning of {user_input[:10]}...",
                f"reviewing our previous discussion about {self.context_understanding['current_topic']}"
            ]
            
            step2_options = [
                f"connecting it to {related_concepts[0]}",
                f"considering emotional factors",
                f"incorporating contextual information"
            ]
            
            conclusion_options = [
                f"I think {random.choice(['this makes sense', 'this is worth exploring further', 'this aligns with my previous thoughts'])}",
                f"I understand your perspective now",
                f"this gives me a new angle to consider"
            ]
            
            context_options = [f"the context of {keywords[0]}", "our previous conversation", "the current emotional state"]
            premise_options = [f"the premise of {keywords[0]}", "the information you provided", "my knowledge base"]
        
        # 填充模板
        reasoning = template.format(
            step1=random.choice(step1_options),
            step2=random.choice(step2_options),
            conclusion=random.choice(conclusion_options),
            context=random.choice(context_options),
            premise=random.choice(premise_options)
        )
        
        return reasoning

    def _should_ask_follow_up(self, user_input):
        """判断是否应该询问更多信息"""
        # 检查用户输入是否简短或模糊
        if len(user_input) < 10:
            return True
        
        # 检查是否包含疑问词
        question_words_zh = ["什么", "为什么", "怎么", "如何", "谁", "哪里", "何时"]
        question_words_en = ["what", "why", "how", "who", "where", "when"]
        
        question_words = question_words_zh if self.language == "zh" else question_words_en
        if any(word in user_input for word in question_words):
            return True
        
        # 随机决定是否询问，但概率较低
        return random.random() < 0.2

    def _generate_follow_up_question(self, user_input):
        """生成后续问题"""
        keywords = self.extract_keywords(user_input)
        
        if self.language == "zh":
            if keywords:
                follow_ups = [
                    f"关于{keywords[0]}，你能告诉我更多吗？",
                    f"我对{keywords[0]}很感兴趣，你是怎么看的？",
                    f"{keywords[0]}这个话题很有意思，你有什么经验可以分享吗？",
                    f"能详细解释一下{user_input[:15]}...吗？",
                    f"关于这个，你有什么具体的想法或感受？"
                ]
            else:
                follow_ups = [
                    "能详细说说你的想法吗？",
                    "我对这个很感兴趣，能多告诉我一些吗？",
                    "你是怎么想到这个的？",
                    "这背后有什么特别的原因吗？",
                    "能分享更多细节吗？"
                ]
        else:
            if keywords:
                follow_ups = [
                    f"Can you tell me more about {keywords[0]}?",
                    f"I'm interested in {keywords[0]}, what are your thoughts?",
                    f"The topic of {keywords[0]} is fascinating, do you have any experiences to share?",
                    f"Could you elaborate on {user_input[:15]}...?",
                    f"What are your specific thoughts or feelings about this?"
                ]
            else:
                follow_ups = [
                    "Could you elaborate on your thoughts?",
                    "I'm interested in this, could you tell me more?",
                    "How did you come up with this?",
                    "Is there a particular reason behind this?",
                    "Could you share more details?"
                ]
        
        return random.choice(follow_ups)

    def _maintain_emotional_coherence(self, response, user_input):
        """保持情感一致性"""
        current_mood = self.emotion_state["mood"]
        
        # 检查情感记忆中的一致性
        if self.emotion_state["emotional_memory"]:
            recent_moods = [memory["mood"] for memory in list(self.emotion_state["emotional_memory"])[-3:]]
            mood_counts = {mood: recent_moods.count(mood) for mood in set(recent_moods)}
            most_common_mood = max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else current_mood
            
            # 如果当前情绪与最近情绪不一致，调整回复
            if current_mood != most_common_mood and random.random() < 0.6:
                # 添加情感过渡短语
                if self.language == "zh":
                    transitions = {
                        ("sad", "happy"): ["虽然之前有点难过，但现在", "心情好转了，"],
                        ("happy", "sad"): ["尽管之前很开心，但现在", "心情有些变化，"],
                        ("angry", "neutral"): ["冷静下来后，", "经过思考，"]
                    }
                else:
                    transitions = {
                        ("sad", "happy"): ["Although I was a bit sad before, now", "My mood has improved,"],
                        ("happy", "sad"): ["Despite being happy earlier, now", "My mood has shifted,"],
                        ("angry", "neutral"): ["After calming down,", "Upon reflection,"]
                    }
                
                transition_key = (most_common_mood, current_mood)
                if transition_key in transitions:
                    transition = random.choice(transitions[transition_key])
                    response = f"{transition} {response}"
        
        # 记录当前情感状态
        self.emotion_state["emotional_memory"].append({
            "mood": current_mood,
            "input": user_input[:20],
            "time": datetime.now().strftime("%H:%M:%S")
        })
        
        return response

    def set_language(self, language):
        """设置AI的语言"""
        self.language = language
    
    def add_catchphrase(self, catchphrase, language=None):
        """添加口头禅"""
        if language is None:
            language = self.language
        
        if language not in self.personality["catchphrases"]:
            self.personality["catchphrases"][language] = []
        
        if catchphrase not in self.personality["catchphrases"][language]:
            self.personality["catchphrases"][language].append(catchphrase)
            return True
        return False
    
    def add_like(self, like):
        """添加喜好"""
        if like not in self.personality["likes"]:
            self.personality["likes"].append(like)
            return True
        return False
    
    def add_dislike(self, dislike):
        """添加不喜欢的事物"""
        if dislike not in self.personality["dislikes"]:
            self.personality["dislikes"].append(dislike)
            return True
        return False
    
    def set_response_style(self, style):
        """设置回复风格"""
        self.personality["response_style"] = style
    
    def load_data(self):
        """加载对话数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "personality" not in data:
                        data["personality"] = self.personality
                    return data
            else:
                return {
                    "patterns": {},
                    "statistics": {
                        "total_conversations": 0,
                        "learned_responses": 0,
                        "user_interactions": {}
                    },
                    "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "response_quality": {},
                    "concept_network": {},
                    "personality": self.personality
                }
        except Exception as e:
            print(f"加载数据失败: {e}")
            return {
                "patterns": {}, 
                "statistics": {"total_conversations": 0, "learned_responses": 0},
                "personality": self.personality
            }
    
    def save_data(self):
        """保存对话数据"""
        try:
            self.knowledge_base["personality"] = self.personality
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False
    
    def extract_keywords(self, text):
        """从用户输入中提取关键词 - 简化版本"""
        # 简化逻辑：直接返回文本的清理版本
        cleaned_text = text.strip()
        
        # 对于包含数学符号的文本，直接返回原文本
        math_symbols = ['+', '-', '*', '/', '=']
        if any(symbol in cleaned_text for symbol in math_symbols):
            return [cleaned_text]
        
        # 对于其他文本，使用分词
        return self._tokenize_text(cleaned_text)
    
    def detect_language(self, text):
        """检测文本语言"""
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        else:
            return "en"
    
    def add_personality_to_response(self, response):
        """为回复添加个性化元素"""
        # 添加口头禅
        if random.random() < 0.3:
            catchphrases = self.personality["catchphrases"].get(self.language, [])
            if catchphrases:
                catchphrase = random.choice(catchphrases)
                if random.random() < 0.5:
                    response = f"{catchphrase} {response}"
                else:
                    response = f"{response} {catchphrase}"
        
        # 提及喜好
        if random.random() < 0.2:
            likes = self.personality["likes"]
            if likes and random.random() < 0.5:
                like = random.choice(likes)
                if self.language == "zh":
                    response = f"{response} 顺便说一句，我很喜欢{like}。"
                else:
                    response = f"{response} By the way, I really like {like}."
        
        return response
    
    def find_best_match(self, user_input):
        """使用TF-IDF在知识库中查找最佳匹配"""
        best_pattern = None
        best_score = 0
        best_responses = []
        
        for pattern in self.knowledge_base.get("patterns", {}).keys():
            score = self._calculate_tfidf_score(user_input, pattern)
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
                best_responses = self.knowledge_base["patterns"][pattern].get("responses", [])
        
        # 设置匹配阈值
        if best_score > 0.2:  # 调整阈值以提高匹配质量
            return best_responses
        return None
    
    def handle_math_expression(self, user_input):
        """处理数学表达式"""
        try:
            # 移除空格
            expression = user_input.replace(" ", "")
            
            # 安全检查：只允许数字和基本运算符
            if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
                return None
            
            # 使用eval计算表达式
            result = eval(expression)
            
            if self.language == "zh":
                return f"{user_input} 的计算结果是 {result}"
            else:
                return f"The result of {user_input} is {result}"
        except:
            return None
    
    def select_response(self, responses, user_id="default"):
        """选择回复，避免重复"""
        if not responses:
            return None
        
        available_responses = [r for r in responses if r not in self.recent_responses]
        
        if not available_responses:
            quality_data = self.knowledge_base.get("response_quality", {})
            scored_responses = []
            
            for response in responses:
                response_hash = hashlib.md5(response.encode()).hexdigest()
                quality = quality_data.get(response_hash, {"score": 0, "count": 0})
                score = quality.get("score", 0) / max(quality.get("count", 1), 1)
                scored_responses.append((response, score))
            
            scored_responses.sort(key=lambda x: x[1], reverse=True)
            available_responses = [r[0] for r in scored_responses]
        
        if available_responses:
            selected_response = random.choice(available_responses[:3])
            self.recent_responses.append(selected_response)
            return selected_response
        
        return None
    
    def should_think_autonomously(self):
        """检查是否应该自主思考"""
        learned_count = self.knowledge_base["statistics"]["learned_responses"]
        return learned_count >= self.autonomous_threshold
    
    def build_concept_network(self):
        """构建概念网络"""
        concept_network = {}
        
        for pattern, data in self.knowledge_base.get("patterns", {}).items():
            concepts = self._tokenize_text(pattern)
            responses = data.get("responses", [])
            
            for concept in concepts:
                if concept not in concept_network:
                    concept_network[concept] = {
                        "related_concepts": set(),
                        "response_patterns": [],
                        "usage_count": 0
                    }
                
                concept_network[concept]["usage_count"] += 1
                concept_network[concept]["response_patterns"].extend(responses)
                
                for other_concept in concepts:
                    if other_concept != concept:
                        concept_network[concept]["related_concepts"].add(other_concept)
        
        self.knowledge_base["concept_network"] = concept_network
        return concept_network
    
    def find_related_concepts(self, user_input):
        """查找相关概念"""
        input_keywords = self.extract_keywords(user_input)
        concept_network = self.knowledge_base.get("concept_network", {})
        related_concepts = set()
        
        for keyword in input_keywords:
            if keyword in concept_network:
                related_concepts.update(concept_network[keyword]["related_concepts"])
        
        return list(related_concepts)
    
    def analogical_reasoning(self, user_input):
        """类比推理"""
        related_concepts = self.find_related_concepts(user_input)
        
        if not related_concepts:
            return None
        
        concept_network = self.knowledge_base.get("concept_network", {})
        potential_responses = []
        
        for concept in related_concepts[:3]:
            if concept in concept_network:
                potential_responses.extend(concept_network[concept]["response_patterns"])
        
        if potential_responses:
            selected_response = random.choice(potential_responses)
            adapted_response = self.adapt_response(selected_response, user_input)
            return adapted_response
        
        return None
    
    def deductive_reasoning(self, user_input):
        """演绎推理"""
        input_keywords = self.extract_keywords(user_input)
        
        if not input_keywords:
            return None
        
        # 生成推理链
        reasoning_chain = self._generate_reasoning_chain(user_input)
        
        if self.language == "zh":
            logical_responses = [
                f"基于你提到的{', '.join(input_keywords[:2])}，似乎可以得出结论这是一个重要的话题。{reasoning_chain}",
                f"考虑到你对{input_keywords[0]}的兴趣，我认为这与我们讨论过的更广泛主题有关。{reasoning_chain}",
                f"从关键词{', '.join(input_keywords[:3])}来看，我推断这是一个值得进一步探索的复杂主题。{reasoning_chain}"
            ]
        else:
            logical_responses = [
                f"Based on what you've said about {', '.join(input_keywords[:2])}, it seems reasonable to conclude that this is an important topic. {reasoning_chain}",
                f"Considering your interest in {input_keywords[0]}, I think this relates to broader themes we've discussed. {reasoning_chain}",
                f"From the keywords {', '.join(input_keywords[:3])}, I deduce this is a complex subject worth exploring further. {reasoning_chain}"
            ]
        
        return random.choice(logical_responses)
    
    def associative_reasoning(self, user_input):
        """关联推理"""
        related_concepts = self.find_related_concepts(user_input)
        
        if not related_concepts:
            return None
        
        selected_concept = random.choice(related_concepts)
        
        if self.language == "zh":
            creative_responses = [
                f"这让我想起了{selected_concept}。有趣的是这些想法如何相互连接。",
                f"当你提到这个时，我想到了{selected_concept}。可能两者之间存在某种联系。",
                f"这次对话让我想起了{selected_concept}。也许有我们尚未探索的关系。"
            ]
        else:
            creative_responses = [
                f"That reminds me of {selected_concept}. It's interesting how these ideas connect.",
                f"When you mention that, I think of {selected_concept}. There might be a connection there.",
                f"This conversation brings {selected_concept} to mind. Perhaps there's a relationship we haven't explored."
            ]
        
        return random.choice(creative_responses)
    
    def contextual_reasoning(self, user_input):
        """上下文推理"""
        if not self.context_understanding["current_topic"]:
            return None
        
        current_topic = self.context_understanding["current_topic"]
        
        if self.language == "zh":
            contextual_responses = [
                f"回到我们关于{current_topic}的讨论，我认为这与你之前提到的内容有关。",
                f"结合上下文，特别是关于{current_topic}的部分，我觉得...",
                f"在我们当前讨论{current_topic}的背景下，你的观点很有启发性。"
            ]
        else:
            contextual_responses = [
                f"Returning to our discussion about {current_topic}, I think this relates to what you mentioned earlier.",
                f"In context, especially regarding {current_topic}, I feel that...",
                f"Within our current discussion about {current_topic}, your perspective is quite illuminating."
            ]
        
        return random.choice(contextual_responses)
    
    def adapt_response(self, base_response, user_input):
        """调整回复"""
        if self.language == "zh":
            adaptations = [
                base_response,
                f"类似地，{base_response.lower()}",
                f"思考你的问题，{base_response.lower()}",
                f"这是一个有趣的观点。{base_response}",
                f"基于我们讨论的内容，{base_response.lower()}"
            ]
        else:
            adaptations = [
                base_response,
                f"In a similar vein, {base_response.lower()}",
                f"Thinking about your question, {base_response.lower()}",
                f"That's an interesting perspective. {base_response}",
                f"Building on what we've discussed, {base_response.lower()}"
            ]
        
        return random.choice(adaptations)
    
    def generate_autonomous_response(self, user_input):
        """生成自主回复"""
        self.build_concept_network()
        
        thinking_mode = random.choices(
            list(self.thinking_modes.keys()),
            weights=list(self.thinking_modes.values())
        )[0]
        
        response = None
        
        if thinking_mode == "analogical":
            response = self.analogical_reasoning(user_input)
        elif thinking_mode == "deductive":
            response = self.deductive_reasoning(user_input)
        elif thinking_mode == "associative":
            response = self.associative_reasoning(user_input)
        elif thinking_mode == "contextual":
            response = self.contextual_reasoning(user_input)
        
        # 如果没有生成回复，使用情感回复
        if not response:
            mood = self.emotion_state["mood"]
            if mood in self.emotional_responses:
                response = random.choice(self.emotional_responses[mood][self.language])
            else:
                if self.language == "zh":
                    response = "我正在思考你的问题。这是一个有趣的话题。"
                else:
                    response = "I'm thinking about your question. It's an interesting topic."
        
        # 决定是否询问更多信息
        if self._should_ask_follow_up(user_input) and random.random() < 0.4:
            follow_up = self._generate_follow_up_question(user_input)
            response = f"{response} {follow_up}"
        
        return response
    
    def learn_new_response(self, user_input, correct_response, user_id="default"):
        """学习新的问题-回复对"""
        keywords = self.extract_keywords(user_input)
        if not keywords:
            return False
        
        # 对于包含数学符号的文本，直接使用原始文本作为模式
        math_symbols = ['+', '-', '*', '/', '=']
        if any(symbol in user_input for symbol in math_symbols):
            pattern_key = user_input
        else:
            pattern_key = " ".join(keywords)
        
        if pattern_key not in self.knowledge_base["patterns"]:
            self.knowledge_base["patterns"][pattern_key] = {
                "responses": [],
                "learned_count": 0,
                "last_used": None
            }
            # 更新TF-IDF数据
            self._update_tfidf_for_new_pattern(pattern_key)
        
        response_data = self.knowledge_base["patterns"][pattern_key]
        if correct_response not in response_data["responses"]:
            response_data["responses"].append(correct_response)
            response_data["learned_count"] += 1
            response_data["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.knowledge_base["statistics"]["learned_responses"] += 1
            
            response_hash = hashlib.md5(correct_response.encode()).hexdigest()
            if response_hash not in self.knowledge_base["response_quality"]:
                self.knowledge_base["response_quality"][response_hash] = {
                    "score": 0,
                    "count": 0,
                    "pattern": pattern_key
                }
            
            self.save_data()
            return True
        
        return False
    
    def update_emotion(self, user_input, response_quality=0):
        """更新情感状态"""
        if self.language == "zh":
            positive_words = ["好", "开心", "高兴", "喜欢", "棒", "神奇", "美妙", "有趣", "爱", "完美"]
            negative_words = ["坏", "难过", "生气", "讨厌", "糟糕", "可怕", "恐怖", "无聊", "恨", "失望"]
            curious_words = ["为什么", "怎么", "如何", "什么", "谁", "哪里", "何时"]
        else:
            positive_words = ["good", "great", "happy", "love", "awesome", "amazing", "wonderful", "interesting", "perfect"]
            negative_words = ["bad", "sad", "angry", "hate", "terrible", "awful", "horrible", "boring", "disappointing"]
            curious_words = ["why", "how", "what", "who", "where", "when"]
        
        user_input_lower = user_input.lower()
        
        positive_count = sum(1 for word in positive_words if word in user_input_lower)
        negative_count = sum(1 for word in negative_words if word in user_input_lower)
        curious_count = sum(1 for word in curious_words if word in user_input_lower)
        
        # 基于关键词更新情绪
        if positive_count > negative_count:
            self.emotion_state["mood"] = "happy"
            self.emotion_state["energy"] = min(100, self.emotion_state["energy"] + 10)
        elif negative_count > positive_count:
            self.emotion_state["mood"] = "sad"
            self.emotion_state["energy"] = max(0, self.emotion_state["energy"] - 10)
        elif curious_count > 0:
            self.emotion_state["mood"] = "curious"
            self.emotion_state["energy"] = min(100, self.emotion_state["energy"] + 5)
        else:
            if self.emotion_state["mood"] != "neutral" and random.random() < 0.3:
                self.emotion_state["mood"] = "neutral"
        
        # 基于回复质量更新情绪
        if response_quality > 0:
            self.emotion_state["energy"] = min(100, self.emotion_state["energy"] + 5)
        elif response_quality < 0:
            self.emotion_state["energy"] = max(0, self.emotion_state["energy"] - 5)
        
        # 能量自然衰减
        self.emotion_state["energy"] = max(0, self.emotion_state["energy"] - 1)
    
    def get_emotional_response(self, base_response):
        """获取情感化回复"""
        mood = self.emotion_state["mood"]
        energy = self.emotion_state["energy"]
        
        if random.random() < (energy / 100):
            emotional_options = self.emotional_responses.get(mood, {}).get(self.language, [])
            if emotional_options and random.random() < 0.3:
                return random.choice(emotional_options)
        
        return base_response
    
    def get_context_aware_response(self, user_input, base_response):
        """获取上下文感知回复"""
        if self.language == "zh":
            context_phrases = ["那个", "这个", "它", "就像你说的", "像你提到的"]
        else:
            context_phrases = ["that", "this", "it", "as you said", "like you mentioned"]
        
        if any(phrase in user_input.lower() for phrase in context_phrases) and len(self.conversation_history) > 0:
            last_exchange = self.conversation_history[-1]
            last_user_msg = last_exchange.get("user", "")
            
            if self.language == "zh":
                context_aware_responses = [
                    f"关于{last_user_msg[:20]}... {base_response}",
                    f"关于我们讨论的内容，{base_response.lower()}",
                    f"{base_response} 这与我们之前的对话有关。"
                ]
            else:
                context_aware_responses = [
                    f"About {last_user_msg[:20]}... {base_response}",
                    f"Regarding what we discussed, {base_response.lower()}",
                    f"{base_response} This relates to our previous conversation."
                ]
            
            if random.random() < 0.4:
                return random.choice(context_aware_responses)
        
        return base_response
    
    def update_response_quality(self, response, quality_score, user_id="default"):
        """更新回复质量"""
        response_hash = hashlib.md5(response.encode()).hexdigest()
        
        if response_hash not in self.knowledge_base["response_quality"]:
            self.knowledge_base["response_quality"][response_hash] = {
                "score": 0,
                "count": 0,
                "pattern": "unknown"
            }
        
        quality_data = self.knowledge_base["response_quality"][response_hash]
        quality_data["score"] += quality_score
        quality_data["count"] += 1
        
        self.save_data()
    
    def chat(self, user_input, user_id="default"):
        """处理用户输入并返回回复"""
        # 首先检查是否是数学表达式
        math_result = self.handle_math_expression(user_input)
        if math_result:
            return math_result
        
        input_language = self.detect_language(user_input)
        if input_language != self.language:
            self.set_language(input_language)
        
        self.knowledge_base["statistics"]["total_conversations"] += 1
        
        if user_id not in self.knowledge_base["statistics"]["user_interactions"]:
            self.knowledge_base["statistics"]["user_interactions"][user_id] = 0
        self.knowledge_base["statistics"]["user_interactions"][user_id] += 1
        
        # 特殊命令处理
        if user_input.lower() in ["exit", "quit", "bye", "退出", "结束"]:
            if self.language == "zh":
                return "再见！期待下次聊天。"
            else:
                return "Goodbye! Looking forward to our next chat."
        
        elif user_input.lower() in ["stats", "statistics", "统计"]:
            stats = self.knowledge_base["statistics"]
            if self.language == "zh":
                return f"我已经学习了 {stats['learned_responses']} 个回复，进行了 {stats['total_conversations']} 次对话。"
            else:
                return f"I've learned {stats['learned_responses']} responses, had {stats['total_conversations']} total conversations."
        
        elif user_input.lower() in ["emotion", "mood", "情感", "情绪"]:
            if self.language == "zh":
                return f"我当前的情绪是 {self.emotion_state['mood']}，能量水平是 {self.emotion_state['energy']}/100"
            else:
                return f"My current mood is {self.emotion_state['mood']} and energy level is {self.emotion_state['energy']}/100"
        
        elif user_input.lower() in ["context", "上下文"]:
            context = "\n".join([f"用户: {exchange['user']}\nAI: {exchange['ai']}" 
                               for exchange in list(self.conversation_history)[-3:]])
            if self.language == "zh":
                return f"最近的对话上下文:\n{context}"
            else:
                return f"Recent context:\n{context}"
        
        elif user_input.lower() in ["thinking mode", "思考模式"]:
            if self.language == "zh":
                thinking_info = f"当前思考模式: {self.thinking_modes}。自主思考阈值: {self.autonomous_threshold}"
                learned_count = self.knowledge_base["statistics"]["learned_responses"]
                autonomy_status = "已激活" if self.should_think_autonomously() else "未激活"
                return f"{thinking_info}\n已学习回复: {learned_count}。自主思考: {autonomy_status}"
            else:
                thinking_info = f"Current thinking modes: {self.thinking_modes}. Autonomous threshold: {self.autonomous_threshold}"
                learned_count = self.knowledge_base["statistics"]["learned_responses"]
                autonomy_status = "active" if self.should_think_autonomously() else "inactive"
                return f"{thinking_info}\nLearned responses: {learned_count}. Autonomous thinking: {autonomy_status}"
        
        elif user_input.lower() in ["personality", "个性"]:
            if self.language == "zh":
                likes = "、".join(self.personality["likes"])
                dislikes = "、".join(self.personality["dislikes"])
                catchphrases = "、".join(self.personality["catchphrases"].get(self.language, []))
                return f"我的个性设置:\n喜欢: {likes}\n不喜欢: {dislikes}\n口头禅: {catchphrases}\n回复风格: {self.personality['response_style']}"
            else:
                likes = ", ".join(self.personality["likes"])
                dislikes = ", ".join(self.personality["dislikes"])
                catchphrases = ", ".join(self.personality["catchphrases"].get(self.language, []))
                return f"My personality:\nLikes: {likes}\nDislikes: {dislikes}\nCatchphrases: {catchphrases}\nResponse style: {self.personality['response_style']}"
        
        elif user_input.lower() in ["learning off", "关闭学习"]:
            self.learning_mode = False
            if self.language == "zh":
                return "学习模式已关闭"
            else:
                return "Learning mode disabled"
        
        elif user_input.lower() in ["learning on", "开启学习"]:
            self.learning_mode = True
            if self.language == "zh":
                return "学习模式已开启"
            else:
                return "Learning mode enabled"
        
        # 主要回复逻辑
        should_think = self.should_think_autonomously()
        response = None
        
        response_list = self.find_best_match(user_input)
        if response_list:
            response = self.select_response(response_list, user_id)
        
        if not response and should_think:
            thinking_indicator = random.choice(self.base_responses["thinking"][self.language])
            print(f"{self.name}: {thinking_indicator}")
            response = self.generate_autonomous_response(user_input)
        
        if not response:
            if any(word in user_input.lower() for word in ["hello", "hi", "hey", "你好", "嗨"]):
                response = random.choice(self.base_responses["greeting"][self.language])
            elif any(word in user_input.lower() for word in ["name", "who", "什么名字", "谁"]):
                response = random.choice(self.base_responses["introduction"][self.language])
            else:
                if self.learning_mode and not should_think:
                    response = random.choice(self.base_responses["unknown_response"][self.language])
                else:
                    mood = self.emotion_state["mood"]
                    if mood in self.emotional_responses:
                        response = random.choice(self.emotional_responses[mood][self.language])
                    else:
                        if self.language == "zh":
                            response = "这是一个有趣的观点。我还在思考这个话题。"
                        else:
                            response = "That's an interesting point. I'm still developing my thoughts on this topic."
        
        # 增强回复
        enhanced_response = self.get_emotional_response(response)
        enhanced_response = self.get_context_aware_response(user_input, enhanced_response)
        enhanced_response = self._maintain_emotional_coherence(enhanced_response, user_input)
        enhanced_response = self.add_personality_to_response(enhanced_response)
        
        # 更新上下文
        self._update_context(user_input, enhanced_response)
        
        self.conversation_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "user": user_input,
            "ai": enhanced_response
        })
        
        self.update_emotion(user_input)
        
        return enhanced_response
    
    def training_mode(self):
        """训练模式"""
        if self.language == "zh":
            print(f"{self.name}: 你好！我是{self.name}，准备好聊天和学习了。")
            print("命令:")
            print("- '退出' - 结束对话")
            print("- '统计' - 显示学习统计")
            print("- '情感' - 检查我的当前情绪状态")
            print("- '上下文' - 查看最近的对话上下文")
            print("- '思考模式' - 检查自主思考状态")
            print("- '个性' - 查看我的个性设置")
            print("- '开启学习/关闭学习' - 切换学习模式")
            print("- 当我不知道如何回答时，你可以教我！")
            print()
        else:
            print(f"{self.name}: Hello! I'm {self.name}, ready to chat and learn.")
            print("Commands:")
            print("- 'exit' - End conversation")
            print("- 'stats' - Show learning statistics")
            print("- 'emotion' - Check my current emotion state")
            print("- 'context' - See recent conversation context")
            print("- 'thinking mode' - Check autonomous thinking status")
            print("- 'personality' - View my personality settings")
            print("- 'learning on/off' - Toggle learning mode")
            print("- When I don't know how to respond, you can teach me!")
            print()
        
        user_id = input("请输入你的名字（或按Enter使用默认）: " if self.language == "zh" else "Please enter your name (or press Enter for default): ").strip()
        if not user_id:
            user_id = "default"
        
        while True:
            user_input = input(f"{user_id}:").strip()
            
            if user_input.lower() in ["exit", "quit", "bye", "退出", "结束"]:
                stats = self.knowledge_base["statistics"]
                autonomy_status = "已激活" if self.should_think_autonomously() else "未激活"
                if self.language == "zh":
                    print(f"{self.name}: 再见！我们一共进行了 {stats['total_conversations']} 次对话。")
                    print(f"我的自主思考状态是 {autonomy_status}，已学习 {stats['learned_responses']} 个回复。")
                else:
                    print(f"{self.name}: Goodbye! We've had {stats['total_conversations']} conversations.")
                    print(f"My autonomous thinking is {autonomy_status} with {stats['learned_responses']} learned responses.")
                self.save_data()
                break
            
            response = self.chat(user_input, user_id)
            print(f"{self.name}: {response}")
            
            should_teach = (
                any(phrase in response for phrase in ["teach me", "not sure", "still learning", "how to respond", "教", "学习", "回答"]) and 
                self.learning_mode and
                not self.should_think_autonomously()
            )
            
            if should_teach:
                if self.language == "zh":
                    teach_response = input("请告诉我应该如何回复（直接回车跳过）: ").strip()
                else:
                    teach_response = input("How should I respond to that? (Press Enter to skip): ").strip()
                
                if teach_response:
                    if self.learn_new_response(user_input, teach_response, user_id):
                        if self.language == "zh":
                            print(f"{self.name}: 谢谢！我学会了一个新回复。")
                        else:
                            print(f"{self.name}: Thank you! I've learned a new response.")
                        
                        if self.should_think_autonomously():
                            if self.language == "zh":
                                print(f"{self.name}: 我现在已经学会了足够多的回复，可以开始自主思考了！")
                            else:
                                print(f"{self.name}: I've now learned enough to start thinking on my own!")
                        
                        if self.language == "zh":
                            feedback = input("这是一个好回复吗？(y/n/跳过): ").strip().lower()
                        else:
                            feedback = input("Was that a good response? (y/n/skip): ").strip().lower()
                        
                        if feedback == 'y':
                            self.update_response_quality(teach_response, 1, user_id)
                            if self.language == "zh":
                                print(f"{self.name}: 谢谢你的反馈！")
                            else:
                                print(f"{self.name}: Thanks for the feedback!")
                        elif feedback == 'n':
                            self.update_response_quality(teach_response, -1, user_id)
                            if self.language == "zh":
                                print(f"{self.name}: 下次我会尝试改进。")
                            else:
                                print(f"{self.name}: I'll try to improve next time.")
                    else:
                        if self.language == "zh":
                            print(f"{self.name}: 我已经知道这个回复了！")
                        else:
                            print(f"{self.name}: I already know that response!")
                print()
    
    def show_knowledge_base(self):
        """显示知识库"""
        if self.language == "zh":
            print("\n=== 当前知识库 ===")
            for pattern, data in self.knowledge_base.get("patterns", {}).items():
                print(f"模式: {pattern} (使用了 {data.get('learned_count', 0)} 次)")
                for response in data.get("responses", []):
                    response_hash = hashlib.md5(response.encode()).hexdigest()
                    quality_data = self.knowledge_base.get("response_quality", {}).get(response_hash, {})
                    quality_score = quality_data.get("score", 0) / max(quality_data.get("count", 1), 1)
                    print(f"  -> {response} (质量: {quality_score:.2f})")
                print()
            
            stats = self.knowledge_base["statistics"]
            autonomy_status = "已激活" if self.should_think_autonomously() else "未激活"
            print(f"统计: {stats['total_conversations']} 次总对话, {stats['learned_responses']} 个已学习回复")
            print(f"自主思考: {autonomy_status}")
            print(f"活跃用户: {len(stats['user_interactions'])}")
        else:
            print("\n=== Current Knowledge Base ===")
            for pattern, data in self.knowledge_base.get("patterns", {}).items():
                print(f"Pattern: {pattern} (used {data.get('learned_count', 0)} times)")
                for response in data.get("responses", []):
                    response_hash = hashlib.md5(response.encode()).hexdigest()
                    quality_data = self.knowledge_base.get("response_quality", {}).get(response_hash, {})
                    quality_score = quality_data.get("score", 0) / max(quality_data.get("count", 1), 1)
                    print(f"  -> {response} (quality: {quality_score:.2f})")
                print()
            
            stats = self.knowledge_base["statistics"]
            autonomy_status = "active" if self.should_think_autonomously() else "inactive"
            print(f"Statistics: {stats['total_conversations']} total conversations, {stats['learned_responses']} learned responses")
            print(f"Autonomous thinking: {autonomy_status}")
            print(f"Active users: {len(stats['user_interactions'])}")

# 使用示例
if __name__ == "__main__":
    isSo = False
    while isSo == False:
        print("1.Chinese  中文")
        print("2.English")
        r = input("Select an language:")
        if r == "1":
            langua = "zh"
            isSo = True
        if r == "2":
            langua = "en"
            isSo = True
    
    # 创建AI实例
    ai = AdvancedChatAI(name="Serlin", data_file="chat_data.json", language=langua)
    
    # 自定义个性化设置
    ai.add_catchphrase("喵~", "zh")
    ai.add_catchphrase("Anyway...", "en")
    ai.add_catchphrase("话说回来", "zh")
    ai.add_like("猫咪")
    ai.add_like("编程")
    ai.add_dislike("拖延")
    ai.set_response_style("friendly")
    
    # 检查现有数据
    if langua == "zh":
        if ai.knowledge_base["statistics"]["learned_responses"] == 0:
            print("检测到新AI，还没有任何训练数据。")
            print("让我们开始第一次训练吧!")
        else:
            learned_count = ai.knowledge_base["statistics"]["learned_responses"]
            autonomy_status = "已激活" if ai.should_think_autonomously() else "未激活"
            print(f"加载了已有的AI，已有 {learned_count} 个学习记录。")
            print(f"自主思考状态: {autonomy_status}")
    else:
        if ai.knowledge_base["statistics"]["learned_responses"] == 0:
            print("New AI detected, no conversation data")
            print("Train First\n")
        else:
            learned_count = ai.knowledge_base["statistics"]["learned_responses"]
            autonomy_status = "Active" if ai.should_think_autonomously() else "InActive"
            print(f"Loading exist AI, Study record already exists: {learned_count}")
            print(f"Think self: {autonomy_status}")
    
    # 开始对话训练
    ai.training_mode()
    
    # 训练结束后显示学到的知识
    ai.show_knowledge_base()