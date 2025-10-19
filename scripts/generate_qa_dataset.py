#!/usr/bin/env python3


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import re
from typing import List, Dict, Tuple, Any
from datetime import datetime
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from config.config import config

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class QADatasetGenerator:
    """QA数据集生成器"""
    
    def __init__(self, use_llm: bool = True):
        """
        初始化生成器
        Args:
            use_llm: 是否使用大语言模型生成问题
        """
        self.use_llm = use_llm
        self.categories = [
            "production", "operations", "strategy", "technology", 
            "financial", "environment", "safety", "projects", "exploration"
        ]
        
        # 问题模板
        self.question_templates = {
            "production": [
                "How much {resource} was produced in {year}?",
                "What were the production levels for {year}?",
                "What factors affected production in {year}?",
                "How did production change compared to previous year?"
            ],
            "operations": [
                "What operational challenges were faced in {year}?",
                "How were offshore operations managed in {year}?",
                "What safety measures were implemented in operations?",
                "How did operations change in {year}?"
            ],
            "strategy": [
                "What was the business strategy for {year}?",
                "How did the business strategy evolve in {year}?",
                "What strategic initiatives were announced?",
                "What were the key strategic priorities?"
            ],
            "technology": [
                "What new technologies were implemented in {year}?",
                "How was technology used to improve operations in {year}?",
                "What digital transformation initiatives were undertaken?",
                "What innovation projects were launched?"
            ],
            "financial": [
                "What were the financial results for {year}?",
                "How did revenue perform in {year}?",
                "What was the profitability in {year}?",
                "What were the key financial highlights?"
            ],
            "environment": [
                "What environmental initiatives were taken in {year}?",
                "How did the company address climate change?",
                "What sustainability goals were set?",
                "What environmental performance was achieved?"
            ],
            "safety": [
                "What safety measures were implemented?",
                "How was workplace safety ensured?",
                "What safety performance was achieved in {year}?",
                "What safety incidents occurred?"
            ],
            "projects": [
                "What major projects were completed in {year}?",
                "What capital projects were approved in {year}?",
                "What new developments were announced?",
                "What project milestones were achieved?"
            ],
            "exploration": [
                "What exploration activities took place in {year}?",
                "What new discoveries were made?",
                "What exploration results were reported?",
                "Where did exploration activities focus?"
            ]
        }
        
        # 加载spaCy模型用于实体识别
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # 如果使用LLM，初始化模型
        if self.use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """初始化大语言模型"""
        try:
            # 尝试使用T5模型进行问题生成
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512
            )
            print("✅ T5模型初始化成功")
        except Exception as e:
            print(f"⚠️ LLM初始化失败: {e}")
            self.use_llm = False
    
    def load_aligned_documents(self, data_dir: str = "data/processed/aligned") -> List[Dict]:
        """加载对齐的文档数据"""
        documents = []
        data_path = Path(data_dir)
        
        for file_path in data_path.glob("aligned_*.jsonl"):
            if "stats" not in file_path.name:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        doc = json.loads(line)
                        documents.append(doc)
        
        print(f"✅ 加载了 {len(documents)} 个文档")
        return documents
    
    def extract_key_info(self, text: str) -> Dict[str, Any]:
        """从文本中提取关键信息"""
        info = {
            "entities": [],
            "numbers": [],
            "dates": [],
            "keywords": []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            # 提取命名实体
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PERCENT", "DATE"]:
                    info["entities"].append({
                        "text": ent.text,
                        "label": ent.label_
                    })
        
        # 提取数字和日期
        numbers = re.findall(r'\\d+(?:[.,]\\d+)*(?:\\s*(?:million|billion|thousand))?', text)
        info["numbers"] = numbers
        
        dates = re.findall(r'\\b\\d{4}\\b|\\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\\b', text)
        info["dates"] = dates
        
        return info
    
    def classify_document(self, doc: Dict) -> str:
        """对文档进行分类"""
        text = doc["en"]["text"].lower()
        
        # 基于关键词的简单分类
        category_keywords = {
            "production": ["production", "output", "barrel", "gas", "oil", "mboe"],
            "operations": ["operation", "facility", "platform", "offshore", "drilling"],
            "strategy": ["strategy", "plan", "goal", "vision", "mission", "objective"],
            "technology": ["technology", "digital", "innovation", "system", "automation"],
            "financial": ["revenue", "profit", "cost", "investment", "financial", "million", "billion"],
            "environment": ["environment", "emission", "carbon", "sustainability", "climate"],
            "safety": ["safety", "accident", "incident", "injury", "security"],
            "projects": ["project", "development", "construction", "investment", "capital"],
            "exploration": ["exploration", "discovery", "reserve", "drilling", "well"]
        }
        
        max_score = 0
        best_category = "operations"  # 默认类别
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > max_score:
                max_score = score
                best_category = category
        
        return best_category
    
    def generate_template_questions(self, doc: Dict, category: str) -> List[Dict]:
        """基于模板生成问题"""
        questions = []
        year = doc["en"]["year"]
        text = doc["en"]["text"]
        
        # 从文本中提取资源类型
        resources = ["oil", "gas", "petroleum", "energy"]
        found_resources = [r for r in resources if r in text.lower()]
        resource = found_resources[0] if found_resources else "oil"
        
        templates = self.question_templates.get(category, self.question_templates["operations"])
        
        for template in templates[:2]:  # 每个文档生成2个问题
            try:
                question = template.format(year=year, resource=resource)
                
                # 生成简单答案（取文本的前200个字符）
                answer = text[:400] + "..." if len(text) > 400 else text
                
                qa_pair = {
                    "id": f"{category}_{random.randint(1000, 9999)}",
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "year": year,
                    "doc_id": doc["en"]["id"],
                    "context": text[:500],
                    "method": "template"
                }
                questions.append(qa_pair)
            except KeyError:
                continue
        
        return questions
    
    def generate_llm_questions(self, doc: Dict, category: str) -> List[Dict]:
        """使用LLM生成更自然的问题"""
        if not self.use_llm:
            return []
        
        questions = []
        text = doc["en"]["text"]
        year = doc["en"]["year"]
        
        # 创建提示
        prompt = f"Generate a question about {category} based on this text: {text[:300]}"
        
        try:
            # 使用T5生成问题
            inputs = f"generate question: {text[:400]}"
            result = self.qa_pipeline(inputs, max_length=100, num_return_sequences=1)
            
            question = result[0]["generated_text"]
            
            # 生成答案（简单策略：使用原文本）
            answer = text[:400] + "..." if len(text) > 400 else text
            
            qa_pair = {
                "id": f"{category}_{random.randint(1000, 9999)}",
                "question": question,
                "answer": answer,
                "category": category,
                "year": year,
                "doc_id": doc["en"]["id"],
                "context": text[:500],
                "method": "llm"
            }
            questions.append(qa_pair)
            
        except Exception as e:
            print(f"LLM生成失败: {e}")
        
        return questions
    
    def generate_factual_questions(self, doc: Dict) -> List[Dict]:
        """生成基于事实的问题"""
        questions = []
        text = doc["en"]["text"]
        year = doc["en"]["year"]
        
        # 提取数字信息生成问题
        numbers = re.findall(r'(\\d+(?:[.,]\\d+)*(?:\\s*(?:million|billion|thousand|%))?)', text)
        
        for number in numbers[:2]:  # 为前2个数字生成问题
            question = f"What was the figure of {number} mentioned in the {year} report?"
            answer = text[:400] + "..." if len(text) > 400 else text
            
            qa_pair = {
                "id": f"factual_{random.randint(1000, 9999)}",
                "question": question,
                "answer": answer,
                "category": "factual",
                "year": year,
                "doc_id": doc["en"]["id"],
                "context": text[:500],
                "method": "factual"
            }
            questions.append(qa_pair)
        
        return questions
    
    def generate_qa_dataset(self, 
                          input_dir: str = "data/processed/aligned",
                          output_file: str = "data/generated_qa_dataset.jsonl",
                          max_documents: int = 1000,
                          questions_per_doc: int = 3) -> str:
        """
        生成完整的QA数据集
        
        Args:
            input_dir: 输入文档目录
            output_file: 输出文件路径
            max_documents: 最大处理文档数
            questions_per_doc: 每个文档生成的问题数
        
        Returns:
            生成的数据集统计信息
        """
        print("🚀 开始生成QA数据集...")
        
        # 加载文档
        documents = self.load_aligned_documents(input_dir)
        
        # 随机采样文档
        if len(documents) > max_documents:
            documents = random.sample(documents, max_documents)
        
        all_questions = []
        stats = {
            "total_documents": len(documents),
            "total_questions": 0,
            "categories": {cat: 0 for cat in self.categories},
            "methods": {"template": 0, "llm": 0, "factual": 0}
        }
        
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(documents)}")
            
            try:
                # 分类文档
                category = self.classify_document(doc)
                
                # 生成不同类型的问题
                doc_questions = []
                
                # 1. 模板问题（每个文档至少1个）
                template_qs = self.generate_template_questions(doc, category)
                doc_questions.extend(template_qs)
                
                # 2. LLM问题（如果可用）
                if self.use_llm and len(doc_questions) < questions_per_doc:
                    llm_qs = self.generate_llm_questions(doc, category)
                    doc_questions.extend(llm_qs)
                
                # 3. 事实性问题
                if len(doc_questions) < questions_per_doc:
                    factual_qs = self.generate_factual_questions(doc)
                    doc_questions.extend(factual_qs)
                
                # 限制每个文档的问题数量
                doc_questions = doc_questions[:questions_per_doc]
                
                # 更新统计
                for q in doc_questions:
                    stats["categories"][q["category"]] += 1
                    stats["methods"][q["method"]] += 1
                
                all_questions.extend(doc_questions)
                
            except Exception as e:
                print(f"处理文档失败: {e}")
                continue
        
        # 保存数据集
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for question in all_questions:
                f.write(json.dumps(question, ensure_ascii=False) + '\\n')
        
        stats["total_questions"] = len(all_questions)
        
        # 保存统计信息
        stats_file = output_path.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ QA数据集生成完成!")
        print(f"📊 统计信息:")
        print(f"   总文档数: {stats['total_documents']}")
        print(f"   总问题数: {stats['total_questions']}")
        print(f"   平均每文档: {stats['total_questions']/stats['total_documents']:.1f}个问题")
        print(f"   输出文件: {output_path}")
        
        return str(output_path)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成QA数据集')
    parser.add_argument('--input_dir', default='data/processed/aligned',
                       help='输入文档目录')
    parser.add_argument('--output_file', default='data/test/generated_qa_dataset.jsonl',
                       help='输出文件路径')
    parser.add_argument('--max_docs', type=int, default=1000,
                       help='最大处理文档数')
    parser.add_argument('--questions_per_doc', type=int, default=3,
                       help='每个文档生成的问题数')
    parser.add_argument('--use_llm', action='store_true',
                       help='使用大语言模型生成问题')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = QADatasetGenerator(use_llm=args.use_llm)
    
    # 生成数据集
    output_file = generator.generate_qa_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        max_documents=args.max_docs,
        questions_per_doc=args.questions_per_doc
    )
    
    print(f"🎉 数据集已生成: {output_file}")


if __name__ == "__main__":
    main()