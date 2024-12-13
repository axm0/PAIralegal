from qdrant_client import QdrantClient
import logging


def reset_qdrant():
    try:
        print("Connecting to Qdrant...")
        client = QdrantClient("http://localhost:6333")

        print("\nChecking existing collections...")
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"Found collections: {collection_names}")

        if "insurance_docs" in collection_names:
            print("\nDeleting 'insurance_docs' collection...")
            client.delete_collection("insurance_docs")
            print("✓ Collection deleted successfully")
        else:
            print("\nNo 'insurance_docs' collection found")

        print("\nVerifying deletion...")
        collections = client.get_collections().collections
        if "insurance_docs" not in [c.name for c in collections]:
            print("✓ Verification successful - collection removed")

        print("\nQdrant reset complete! You can now reload your embeddings.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please ensure Qdrant is running at http://localhost:6333")


if __name__ == "__main__":
    reset_qdrant()