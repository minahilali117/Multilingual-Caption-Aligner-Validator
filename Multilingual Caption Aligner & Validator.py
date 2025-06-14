import json
from typing import Dict, List, Tuple
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import pandas as pd
from tqdm import tqdm

class CaptionAligner:
    def __init__(self, target_lang: str, similarity_threshold: float = 0.75):
        """
        Initialize translation and embedding models
        
        Args:
            target_lang (str): Target language code (e.g. 'ur', 'es', 'zh')
            similarity_threshold (float): Minimum similarity score threshold
        """
        self.target_lang = target_lang
        self.similarity_threshold = similarity_threshold
        
        # Initialize translation pipeline
        self.translator = pipeline("translation", 
                                 model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
        
        # Initialize LaBSE model for similarity scoring
        self.tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE")
        self.encoder = AutoModel.from_pretrained("setu4993/LaBSE")
        
    def translate_caption(self, text: str) -> str:
        """Translate English caption to target language"""
        result = self.translator(text)[0]['translation_text']
        return result
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get sentence embedding using LaBSE"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return outputs.pooler_output.squeeze()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two captions"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def process_dataset(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process entire dataset of image-caption pairs
        
        Args:
            data: List of dicts with 'image_path' and 'en_caption' keys
            
        Returns:
            List of dicts with translations and similarity scores
        """
        results = []
        
        for item in tqdm(data):
            en_caption = item['en_caption']
            translated = self.translate_caption(en_caption)
            similarity = self.compute_similarity(en_caption, translated)
            
            if similarity >= self.similarity_threshold:
                results.append({
                    'image_path': item['image_path'],
                    'en_caption': en_caption,
                    f'{self.target_lang}_caption': translated,
                    'similarity': similarity
                })
                
        return results
    
    def save_results(self, results: List[Dict], output_path: str, format: str = 'json'):
        """Save results to JSON or CSV"""
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)

def main():
    # Example usage
    data = [
        {
            'image_path': 'images/img1.jpg',
            'en_caption': 'A woman in an orange dress smiles at her phone while standing next to a bicycle in a tree-lined urban park'
        },
        {
            'image_path': 'images/img2.jpg',
            'en_caption': 'Three diverse individuals pose confidently in front of a rainbow pride flag, celebrating inclusion and identity'
        },
        {
            'image_path': 'images/img3.jpg',
            'en_caption': 'A beautiful sunset over the ocean'
        }
    ]
    
    # Initialize aligner for Spanish translations
    aligner = CaptionAligner(target_lang='es')
    
    # Process dataset
    results = aligner.process_dataset(data)
    
    # Save results
    aligner.save_results(results, 'captions_aligned.json')
    aligner.save_results(results, 'captions_aligned.csv', format='csv')

if __name__ == "__main__":
    main()