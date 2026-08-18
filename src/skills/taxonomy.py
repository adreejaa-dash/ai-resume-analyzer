"""
ESCO and O*NET skill taxonomy loader.

Loads actual ESCO English skills/occupations and O*NET technology skills
into a searchable vocabulary. Creates a unified skill index with categories.

Data sources:
- ESCO: European Commission's European Skills, Competences, Qualifications
  and Occupations classification
- O*NET: U.S. Department of Labor's Occupational Information Network
"""
import os
import json
import csv
import re
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path


# Skill categories
SKILL_CATEGORIES = {
    "programming_languages": {
        "name": "Programming Languages",
        "keywords": [
            "python", "java", "javascript", "typescript", "cplusplus", "c++",
            "csharp", "c#", "ruby", "go", "golang", "rust", "swift", "kotlin",
            "scala", "perl", "php", "r", "matlab", "julia", "lua", "dart",
            "objective-c", "assembly", "fortran", "cobol", "haskell", "erlang",
            "clojure", "groovy", "visual basic", "vba", "bash", "shell",
            "powershell", "sql",
        ],
    },
    "data": {
        "name": "Data & Databases",
        "keywords": [
            "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
            "cassandra", "elasticsearch", "sqlite", "oracle", "dynamodb",
            "neo4j", "data analysis", "data visualization", "data engineering",
            "data warehousing", "etl", "data pipeline", "data modeling",
            "big data", "hadoop", "spark", "apache spark", "hive", "presto",
            "snowflake", "databricks", "airflow", "kafka",
            "tableau", "power bi", "looker", "metabase",
        ],
    },
    "machine_learning": {
        "name": "Machine Learning",
        "keywords": [
            "machine learning", "regression", "classification", "clustering",
            "feature engineering", "xgboost", "random forest", "decision tree",
            "gradient boosting", "svm", "support vector machine", "naive bayes",
            "logistic regression", "linear regression", "ensemble learning",
            "cross-validation", "hyperparameter tuning", "model evaluation",
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "anomaly detection", "recommendation system", "time series",
            "dimensionality reduction", "pca", "bayesian",
        ],
    },
    "deep_learning": {
        "name": "Deep Learning",
        "keywords": [
            "deep learning", "tensorflow", "pytorch", "keras", "cnn",
            "convolutional neural network", "rnn", "recurrent neural network",
            "lstm", "gru", "transformer", "attention mechanism", "gan",
            "generative adversarial", "autoencoder", "neural network",
            "backpropagation", "transfer learning", "fine-tuning",
            "computer vision", "image classification", "object detection",
            "image segmentation", "yolo",
        ],
    },
    "nlp": {
        "name": "Natural Language Processing",
        "keywords": [
            "nlp", "natural language processing", "text classification",
            "named entity recognition", "ner", "sentiment analysis",
            "text mining", "information extraction", "topic modeling",
            "word embeddings", "word2vec", "glove", "bert", "gpt",
            "transformers", "hugging face", "spacy", "nltk", "tokenization",
            "stemming", "lemmatization", "tf-idf", "tfidf",
            "language model", "text generation", "machine translation",
            "question answering", "text summarization",
            "sentence embeddings", "sentence-bert", "sbert",
        ],
    },
    "cloud_devops": {
        "name": "Cloud & DevOps",
        "keywords": [
            "aws", "amazon web services", "azure", "gcp",
            "google cloud", "docker", "kubernetes", "k8s",
            "ci/cd", "jenkins", "github actions", "gitlab ci",
            "terraform", "ansible", "linux", "unix",
            "microservices", "serverless", "lambda",
            "ec2", "s3", "rds", "cloudformation",
            "devops", "infrastructure as code", "monitoring",
            "grafana", "prometheus", "elk",
        ],
    },
    "frameworks_libraries": {
        "name": "Frameworks & Libraries",
        "keywords": [
            "pandas", "numpy", "scipy", "scikit-learn", "sklearn",
            "matplotlib", "seaborn", "plotly",
            "fastapi", "flask", "django", "streamlit",
            "react", "angular", "vue", "next.js", "node.js",
            "spring", "spring boot", "express",
            "opencv", "pillow",
            "pyspark", "dask",
            "beautifulsoup", "scrapy", "selenium",
            "requests", "aiohttp",
            "sqlalchemy", "celery",
            "jupyter", "notebook",
        ],
    },
    "tools_practices": {
        "name": "Tools & Practices",
        "keywords": [
            "git", "github", "gitlab", "bitbucket",
            "jira", "confluence", "slack",
            "agile", "scrum", "kanban",
            "rest api", "graphql", "api design",
            "unit testing", "integration testing", "pytest", "unittest",
            "tdd", "test-driven development",
            "design patterns", "solid principles",
            "version control", "code review",
            "data structures", "algorithms",
            "object-oriented programming", "oop",
            "functional programming",
        ],
    },
}


class SkillTaxonomy:
    """
    Unified skill taxonomy combining ESCO, O*NET, and curated categories.
    """
    
    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            skills_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "skills"
            )
        self.skills_dir = skills_dir
        
        # Core data structures
        self.all_skills: Set[str] = set()
        self.skill_to_category: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        self.esco_skills: List[Dict] = []
        self.onet_skills: List[Dict] = []
        self.esco_loaded = False
        self.onet_loaded = False
        
        # Load everything
        self._load_aliases()
        self._load_category_skills()
        self._load_esco_skills()
        self._load_onet_skills()
    
    def _load_aliases(self):
        """Load skill alias mappings."""
        alias_path = os.path.join(self.skills_dir, "skill_aliases.json")
        if os.path.exists(alias_path):
            with open(alias_path, "r") as f:
                self.aliases = json.load(f)
        else:
            # Default aliases — always available
            self.aliases = self._get_default_aliases()
    
    def _get_default_aliases(self) -> Dict[str, str]:
        """Default skill aliases for normalization."""
        return {
            "sklearn": "scikit-learn",
            "scikit learn": "scikit-learn",
            "sci-kit learn": "scikit-learn",
            "sci-kit-learn": "scikit-learn",
            "postgres": "postgresql",
            "mongo": "mongodb",
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "tf": "tensorflow",
            "k8s": "kubernetes",
            "c plus plus": "c++",
            "cplusplus": "c++",
            "c sharp": "c#",
            "csharp": "c#",
            "dotnet": ".net",
            "dot net": ".net",
            "nodejs": "node.js",
            "node js": "node.js",
            "reactjs": "react",
            "react js": "react",
            "react.js": "react",
            "vuejs": "vue",
            "vue js": "vue",
            "vue.js": "vue",
            "angularjs": "angular",
            "angular js": "angular",
            "angular.js": "angular",
            "nextjs": "next.js",
            "next js": "next.js",
            "ml": "machine learning",
            "dl": "deep learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "aws": "amazon web services",
            "gcp": "google cloud platform",
            "ci cd": "ci/cd",
            "ci/cd": "ci/cd",
            "sbert": "sentence-bert",
            "word2vec": "word2vec",
            "powerbi": "power bi",
            "pyspark": "apache spark",
            "scikit": "scikit-learn",
        }
    
    def _load_category_skills(self):
        """Load skills from predefined categories."""
        for cat_id, cat_info in SKILL_CATEGORIES.items():
            for skill in cat_info["keywords"]:
                normalized = self.normalize_skill(skill)
                self.all_skills.add(normalized)
                self.skill_to_category[normalized] = cat_info["name"]
    
    def _load_esco_skills(self):
        """Load ESCO skill taxonomy from CSV."""
        esco_path = os.path.join(self.skills_dir, "esco_skills.csv")
        if not os.path.exists(esco_path):
            return
        
        try:
            with open(esco_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skill_label = row.get("preferredLabel", row.get("label", "")).strip().lower()
                    if skill_label and len(skill_label) > 1:
                        self.esco_skills.append(row)
                        normalized = self.normalize_skill(skill_label)
                        self.all_skills.add(normalized)
                        if normalized not in self.skill_to_category:
                            self.skill_to_category[normalized] = "ESCO Skill"
            
            self.esco_loaded = True
            print(f"[SkillTaxonomy] Loaded {len(self.esco_skills)} ESCO skills")
        except Exception as e:
            print(f"[SkillTaxonomy] Failed to load ESCO skills: {e}")
    
    def _load_onet_skills(self):
        """Load O*NET technology skills from CSV."""
        onet_path = os.path.join(self.skills_dir, "onet_tech_skills.csv")
        if not os.path.exists(onet_path):
            return
        
        try:
            with open(onet_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    example = row.get("Example", row.get("example", "")).strip().lower()
                    if example and len(example) > 1:
                        self.onet_skills.append(row)
                        normalized = self.normalize_skill(example)
                        self.all_skills.add(normalized)
                        if normalized not in self.skill_to_category:
                            self.skill_to_category[normalized] = "O*NET Technology"
            
            self.onet_loaded = True
            print(f"[SkillTaxonomy] Loaded {len(self.onet_skills)} O*NET skills")
        except Exception as e:
            print(f"[SkillTaxonomy] Failed to load O*NET skills: {e}")
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill name using aliases."""
        skill = skill.strip().lower()
        return self.aliases.get(skill, skill)
    
    def get_category(self, skill: str) -> str:
        """Get the category of a skill."""
        normalized = self.normalize_skill(skill)
        return self.skill_to_category.get(normalized, "Other")
    
    def get_all_skill_phrases(self) -> List[str]:
        """Get all known skill phrases for matching."""
        phrases = set()
        phrases.update(self.all_skills)
        phrases.update(self.aliases.keys())
        # Filter out very short or generic terms
        return [p for p in phrases if len(p) > 1]
    
    def get_status(self) -> Dict:
        """Get taxonomy status for reporting."""
        return {
            "total_skills": len(self.all_skills),
            "esco_loaded": self.esco_loaded,
            "esco_count": len(self.esco_skills),
            "onet_loaded": self.onet_loaded,
            "onet_count": len(self.onet_skills),
            "alias_count": len(self.aliases),
            "categories": list(SKILL_CATEGORIES.keys()),
        }
