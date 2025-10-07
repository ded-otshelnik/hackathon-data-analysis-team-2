import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

class PatentCategorizer:
    
    def __init__(self, model_name="thellert/physbert_cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Inference mode
        self.model.eval()
        
        # Cache for embeddings to avoid recomputation
        self.embeddings_cache = {}
        self.categories = {}
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Получить эмбеддинг для текста
        
        Args:
            text: Текст для обработки
            
        Returns:
            numpy array с эмбеддингом
        """
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] token embedding allows to get sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        embedding = cls_embedding.numpy()
        self.embeddings_cache[text] = embedding
        
        return embedding
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def cluster_patents(self, patent_titles: List[str], n_clusters: int = 5) -> Dict:
        """
        Clusterization of patents based on their titles
        
        Args:
            patent_titles: List of patent titles
            n_clusters: Number of clusters

        Returns:
            Dictionary with clustering results
        """
        embeddings = self.get_batch_embeddings(patent_titles)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = {}
        for i, (title, label) in enumerate(zip(patent_titles, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'title': title,
                'index': i,
                'embedding': embeddings[i]
            })
        
        return {
            'clusters': clusters,
            'embeddings': embeddings,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans
        }
    
    def categorize_by_predefined_categories(self, patent_titles: List[str], 
                                          categories: Dict[str, List[str]]) -> Dict:
        """
        Categorizing of patents by predefined categories
        
        Args:
            patent_titles: List of patent titles
            categories: Dictionary {category: [example titles]}
            
        Returns:
            Dictionary with categorization results
        """
        category_embeddings = {}
        for category, examples in categories.items():
            embeddings = self.get_batch_embeddings(examples)
            # centroid of category is used to apply object to cluster
            category_embeddings[category] = np.mean(embeddings, axis=0)
        
        patent_embeddings = self.get_batch_embeddings(patent_titles)
        
        # patent classification by cosine similarity
        results = {}
        for i, (title, embedding) in enumerate(zip(patent_titles, patent_embeddings)):
            similarities = {}
            for category, cat_embedding in category_embeddings.items():
                similarity = cosine_similarity([embedding], [cat_embedding])[0][0]
                similarities[category] = similarity
            
            # max similarity is the best category
            best_category = max(similarities, key=similarities.get)
            best_score = similarities[best_category]
            
            results[title] = {
                'category': best_category,
                'confidence': best_score,
                'all_scores': similarities
            }
        
        return results
