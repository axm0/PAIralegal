import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import os
import sys

class QdrantLoader:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'qdrant_upload_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )

    def create_collection(self, collection_name: str, vector_size: int):
        try:
            collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logging.info(f"Collection {collection_name} already exists")
                return
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                    full_scan_threshold=10000,
                    max_indexing_threads=8
                )
            )
            logging.info(f"Created collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error creating collection: {str(e)}")
            raise

    def load_embeddings(self, embeddings_path: str, collection_name: str = "insurance_docs"):
        try:
            logging.info(f"Loading embeddings from: {embeddings_path}")
            data = np.load(embeddings_path, allow_pickle=True)
            qdrant_data = data['data']
            if len(qdrant_data) == 0:
                logging.error("No embeddings found in file")
                return
            vector_size = len(qdrant_data[0]['vector'])
            logging.info(f"Vector size: {vector_size}")
            self.create_collection(collection_name, vector_size)
            batch_size = 100
            total_batches = (len(qdrant_data) + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(total_batches), desc="Uploading batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(qdrant_data))
                batch = qdrant_data[start_idx:end_idx]
                points = []
                for item in batch:
                    text_length = len(item['payload']['text'])
                    logging.info(f"Processing document: {item['payload']['file_path']}")
                    logging.info(f"Text length: {text_length} chars")
                    point = models.PointStruct(
                        id=item['id'],
                        vector=item['vector'],
                        payload={
                            "file_path": item['payload']['file_path'],
                            "metadata": item['payload']['metadata'],
                            "text": item['payload']['text']
                        }
                    )
                    points.append(point)
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True
                    )
                    logging.info(f"Successfully uploaded batch {batch_idx + 1}/{total_batches}")
                except Exception as e:
                    logging.error(f"Error uploading batch {batch_idx}: {str(e)}")
                    continue
            collection_info = self.client.get_collection(collection_name)
            logging.info(f"Collection info after upload: {collection_info}")
            first_point = self.client.retrieve(
                collection_name=collection_name,
                ids=[0],
                with_payload=True
            )
            if first_point:
                text_length = len(first_point[0].payload.get('text', ''))
                logging.info(f"Verification - First document text length: {text_length} chars")
        except Exception as e:
            logging.error(f"Error in load_embeddings: {str(e)}")
            raise

def main():
    embeddings_dir = "C:/Developer/Workspace/llama3.2/embeddings"
    npz_files = list(Path(embeddings_dir).glob("qdrant_embeddings_*.npz"))
    if not npz_files:
        print("No embedding files found!")
        return
    latest_embeddings = max(npz_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading latest embeddings: {latest_embeddings}")
    loader = QdrantLoader()
    loader.load_embeddings(
        embeddings_path=str(latest_embeddings),
        collection_name="insurance_docs"
    )
    print("\nUpload complete! You can now query your vectors using:")
    print("http://localhost:6333/collections/insurance_docs")

if __name__ == "__main__":
    main()
