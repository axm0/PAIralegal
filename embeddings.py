import os
import sys
import math
import pytesseract
import pdfplumber
import pandas as pd
import torch
import numpy as np
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Optional, Tuple
import threading
from tqdm import tqdm
from datetime import datetime
import re
from nltk.tokenize import sent_tokenize
import nltk
import json

nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
pytesseract.pytesseract.tesseract_cmd = r'C:\Developer\Tools\Tesseract-OCR\tesseract.exe'

def setup_cuda_optimizations():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('medium')
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        current_device = torch.cuda.current_device()
        torch.cuda.set_device(current_device)

class InsuranceTextProcessor:
    @staticmethod
    def clean_insurance_text(text: str) -> str:
        text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[''´`]', "'", text)
        text = re.sub(r'[""″]', '"', text)
        text = re.sub(r'[‒–—―]', '-', text)
        text = re.sub(r'(?<=\d),(?=\d{3})', '', text)
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'Section\s+(\d+)', r'Section \1', text, flags=re.IGNORECASE)
        text = re.sub(r'l(?=\d)', '1', text)
        text = re.sub(r'O(?=\d)', '0', text)
        return text.strip()

    @staticmethod
    def extract_insurance_metadata(text: str) -> Dict[str, str]:
        metadata = {}
        patterns = {
            'policy_number': r'Policy\s*(?:#|Number|No|Num)\s*[:.]?\s*([A-Z0-9-]+)',
            'effective_date': r'Effective\s*Date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'coverage_type': r'Coverage\s*Type\s*[:.]?\s*([A-Za-z\s]+)',
            'state': r'State\s*[:.]?\s*([A-Z]{2})',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
        return metadata

class DocumentChunker:
    def __init__(self, tokenizer, max_length: int = 2048, overlap: int = 200):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = min(overlap, max_length // 10)

    def chunk_document(self, text: str) -> List[Dict[str, Union[str, int]]]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for i, sentence in enumerate(sentences):
            tokens = self.tokenizer.encode(sentence)
            sentence_length = len(tokens)
            if current_length + sentence_length > self.max_length:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({'text': chunk_text, 'position': len(chunks), 'sentences': len(current_chunk)})
                    overlap_tokens = 0
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        s_tokens = len(self.tokenizer.encode(s))
                        if overlap_tokens + s_tokens > self.overlap:
                            break
                        overlap_tokens += s_tokens
                        overlap_sentences.insert(0, s)
                    current_chunk = overlap_sentences + [sentence]
                    current_length = overlap_tokens + sentence_length
                else:
                    chunks.append({'text': sentence, 'position': len(chunks), 'sentences': 1})
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        if current_chunk:
            chunks.append({'text': ' '.join(current_chunk), 'position': len(chunks), 'sentences': len(current_chunk)})
        return chunks

class DocumentProcessor:
    def __init__(self, model_name: str = "intfloat/e5-large-v2", project_dir: str = None):
        setup_cuda_optimizations()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8
        self.max_length = 512
        self.min_chunk_size = 100
        self.overlap = 100
        self.lock = threading.Lock()
        self._setup_logging(project_dir)
        os.environ['TRANSFORMERS_CACHE'] = r'C:\Developer\Models'
        model_path = snapshot_download(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'], local_files_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])
        self.text_processor = InsuranceTextProcessor()
        self.chunker = DocumentChunker(tokenizer=self.tokenizer, max_length=self.max_length)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True, cache_dir=os.environ['TRANSFORMERS_CACHE']).to(self.device)

    def _setup_logging(self, project_dir: Optional[str] = None):
        if project_dir is None:
            project_dir = os.getcwd()
        log_dir = os.path.join(project_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'processing_{timestamp}.log')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    def smart_chunk_text(self, text: str) -> List[Dict[str, Union[str, int, float]]]:
        sections = self._split_into_sections(text)
        chunks = []
        for section in sections:
            sentences = sent_tokenize(section)
            current_chunk = []
            current_tokens = 0
            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence)
                token_count = len(tokens)
                if current_tokens + token_count > self.max_length:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(self._process_chunk(chunk_text, section))
                        overlap_sentences = current_chunk[-3:]
                        current_chunk = overlap_sentences + [sentence]
                        current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
                    else:
                        subchunks = self._split_long_sentence(sentence)
                        chunks.extend([self._process_chunk(sc, section) for sc in subchunks])
                else:
                    current_chunk.append(sentence)
                    current_tokens += token_count
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._process_chunk(chunk_text, section))
        return chunks

    def _calculate_importance(self, text: str, insurance_terms: List[str]) -> float:
        score = 0.0
        score += len(insurance_terms) * 0.2
        number_matches = len(re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', text))
        score += number_matches * 0.15
        if re.search(r'^(?:SECTION|COVERAGE|ARTICLE|ENDORSEMENT)', text, re.IGNORECASE):
            score += 1.0
        legal_terms = ['shall', 'must', 'will not', 'required', 'prohibited', 'subject to']
        score += sum(term in text.lower() for term in legal_terms) * 0.1
        policy_terms = ['policy', 'premium', 'coverage', 'limit', 'deductible', 'exclusion']
        score += sum(term in text.lower() for term in policy_terms) * 0.15
        return min(score, 5.0)

    def _process_chunk(self, text: str, section: str) -> Dict[str, Union[str, int, float]]:
        insurance_terms = self._extract_insurance_terms(text)
        importance_score = self._calculate_importance(text, insurance_terms)
        return {'text': text, 'section': section, 'token_count': len(self.tokenizer.encode(text)), 'insurance_terms': insurance_terms, 'importance_score': importance_score, 'metadata': self.text_processor.extract_insurance_metadata(text)}

    def _split_long_sentence(self, sentence: str) -> List[str]:
        tokens = self.tokenizer.encode(sentence)
        if len(tokens) <= self.max_length:
            return [sentence]
        split_points = {';': 10, ' and ': 8, ' or ': 8, ',': 6, ' in ': 4, ' of ': 4, ' with ': 4}
        subsentences = [sentence]
        while any(len(self.tokenizer.encode(s)) > self.max_length for s in subsentences):
            new_subsentences = []
            for subsentence in subsentences:
                if len(self.tokenizer.encode(subsentence)) > self.max_length:
                    split = False
                    for delimiter, _ in sorted(split_points.items(), key=lambda x: x[1], reverse=True):
                        if delimiter in subsentence:
                            parts = subsentence.split(delimiter, 1)
                            new_subsentences.extend(p.strip() for p in parts)
                            split = True
                            break
                    if not split:
                        tokens = self.tokenizer.encode(subsentence)
                        mid = len(tokens) // 2
                        text = self.tokenizer.decode(tokens[:mid])
                        last_space = text.rfind(' ')
                        if last_space > 0:
                            new_subsentences.extend([subsentence[:last_space].strip(), subsentence[last_space:].strip()])
                else:
                    new_subsentences.append(subsentence)
            subsentences = new_subsentences
        return subsentences

    def _split_into_sections(self, text: str) -> List[str]:
        section_patterns = [r'\n(?=SECTION \d+[.:]\s+)', r'\n(?=COVERAGE [A-Z][.:]\s+)', r'\n(?=ARTICLE \d+[.:]\s+)', r'\n(?=\d+\.\s+DEFINITIONS)', r'\n(?=ENDORSEMENT\s+\d+)']
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                splits = re.split(pattern, section)
                new_sections.extend(s.strip() for s in splits if s.strip())
            sections = new_sections
        return sections

    def _extract_insurance_terms(self, text: str) -> List[str]:
        insurance_patterns = [r'coverage [A-Z]', r'limit of liability', r'deductible', r'premium', r'endorsement', r'exclusion', r'policy period', r'insured location', r'peril', r'limit', r'limits', r'insure', r'rate']
        terms = []
        for pattern in insurance_patterns:
            matches = re.finditer(pattern, text.lower())
            terms.extend(match.group() for match in matches)
        return list(set(terms))

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            text = self.text_processor.clean_insurance_text(text)
            chunks = self.smart_chunk_text(text)
            chunk_embeddings = []
            chunk_weights = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk['text'], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    chunk_embedding = (sum_embeddings / sum_mask).cpu().numpy()
                    chunk_embeddings.append(chunk_embedding)
                    chunk_weights.append(chunk['importance_score'])
            if chunk_embeddings:
                weights = np.array(chunk_weights)
                weights = weights / weights.sum()
                document_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
                document_embedding = document_embedding / np.linalg.norm(document_embedding)
                all_embeddings.append(document_embedding)
        return np.vstack(all_embeddings)

    def process_metadata(self, folder_path: str) -> Dict[str, dict]:
        metadata = {}
        insurance_type = None
        if "homeowners" in folder_path:
            insurance_type = "homeowners"
            metadata_file = "homeowners_metadata.txt"
        elif "personal_auto" in folder_path:
            insurance_type = "personal_auto"
            metadata_file = "personal_auto_metadata.txt"
        type_metadata_path = os.path.join(folder_path, metadata_file)
        if os.path.exists(type_metadata_path):
            with open(type_metadata_path, 'r', encoding='utf-8') as f:
                metadata['insurance_type_metadata'] = self.text_processor.clean_insurance_text(f.read())
        for root, dirs, _ in os.walk(folder_path):
            for dir_name in dirs:
                if dir_name.startswith('SFMA-'):
                    sfma_path = os.path.join(root, dir_name)
                    sfma_metadata_file = os.path.join(sfma_path, f"{dir_name}.txt")
                    if os.path.exists(sfma_metadata_file):
                        with open(sfma_metadata_file, 'r', encoding='utf-8') as f:
                            metadata[dir_name] = {'path': sfma_path, 'metadata': self.text_processor.clean_insurance_text(f.read()), 'insurance_type': insurance_type}
        return metadata

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages, 1):
                    with self.lock:
                        try:
                            page_text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True, keep_blank_chars=True)
                            if page_text:
                                text_parts.append(page_text)
                        except Exception as e:
                            logging.error(f"Error extracting text from page in {pdf_path}: {str(e)}")
                if not text_parts:
                    return self.extract_text_from_scanned_pdf(pdf_path)
                return "\n".join(text_parts)
        except Exception as e:
            logging.error(f"Error in PDF extraction for {pdf_path}, falling back to OCR: {str(e)}")
            return self.extract_text_from_scanned_pdf(pdf_path)

    def extract_text_from_scanned_pdf(self, pdf_path: str) -> str:
        try:
            images = convert_from_path(pdf_path, dpi=400)
            def process_image(image):
                custom_config = r'''--oem 3 --psm 6 
                    -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?()-:;\'\"$@ "
                    -c preserve_interword_spaces=1
                    -c textord_heavy_nr=1
                    -c tessedit_do_invert=0
                    -c tessedit_enable_dict_correction=1'''
                gray_image = image.convert('L')
                return pytesseract.image_to_string(gray_image, config=custom_config, lang='eng')
            texts = [process_image(image) for image in images]
            return "\n".join(texts)
        except Exception as e:
            logging.error(f"Error in OCR processing for {pdf_path}: {str(e)}")
            return ""

    def process_documents(self, folder_path: str) -> List[Dict]:
        stats = {'total_files': 0, 'processed_files': 0, 'failed_files': 0, 'total_chars': 0}
        documents = []
        folder_metadata = self.process_metadata(folder_path)
        for root, _, files in os.walk(folder_path):
            current_sfma = None
            for part in Path(root).parts:
                if isinstance(part, str) and part.startswith('SFMA-'):
                    current_sfma = part
                    break
            for file in files:
                if file.endswith(('.pdf', '.txt')) and not file.endswith('_metadata.txt'):
                    stats['total_files'] += 1
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith('.pdf'):
                            text = self.extract_text_from_pdf(file_path)
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                        if text.strip():
                            stats['processed_files'] += 1
                            stats['total_chars'] += len(text)
                            metadata_text_parts = []
                            if 'insurance_type_metadata' in folder_metadata:
                                metadata_text_parts.append(folder_metadata['insurance_type_metadata'])
                            if current_sfma and current_sfma in folder_metadata:
                                metadata_text_parts.append(folder_metadata[current_sfma]['metadata'])
                            combined_text = "\n\n".join([*metadata_text_parts, text])
                            documents.append({"file_path": file_path, "text": combined_text, "metadata": {"insurance_type": folder_metadata.get(current_sfma, {}).get('insurance_type'), "sfma_id": current_sfma, "extracted_metadata": self.text_processor.extract_insurance_metadata(combined_text)}})
                        else:
                            stats['failed_files'] += 1
                    except Exception as e:
                        stats['failed_files'] += 1
                        logging.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        return documents

def main():
    project_dir = "C:/Developer/Workspace/llama3.2"
    data_dir = os.path.join(project_dir, "data")
    embeddings_dir = os.path.join(project_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    processor = DocumentProcessor(project_dir=project_dir)
    all_documents = []
    total_processed = 0
    total_failed = 0
    folders = [os.path.join(data_dir, "state_farm_4.0_homeowners"), os.path.join(data_dir, "state_farm_19.0_personal_auto")]
    for folder in folders:
        try:
            documents = processor.process_documents(folder)
            total_processed += len(documents)
            all_documents.extend(documents)
        except Exception as e:
            total_failed += 1
    if all_documents:
        texts = [doc["text"] for doc in all_documents]
        embeddings = processor.generate_embeddings(texts)
        qdrant_ready_data = []
        for idx, (doc, embedding) in enumerate(zip(all_documents, embeddings)):
            qdrant_ready_data.append({"id": idx, "vector": embedding.tolist(), "payload": {"file_path": doc["file_path"], "metadata": doc["metadata"], "text": doc["text"]}})
        output_path = os.path.join(embeddings_dir, f'qdrant_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
        np.savez_compressed(output_path, data=qdrant_ready_data)
        metadata_path = output_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump([d["payload"]["metadata"] for d in qdrant_ready_data], f, indent=2)

if __name__ == "__main__":
    main()
