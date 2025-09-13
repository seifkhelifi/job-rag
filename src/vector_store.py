import weaviate
from llama_index.core import Document


import weaviate
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llm import LLM
from typing import List, Optional

import os
from dotenv import load_dotenv


load_dotenv()


class WeaviateVectorDB:

    def __init__(
        self,
        hostname: str = "localhost",
        port: int = 8080,
        persistence_path: str = "./weaviate_data",
        llm: str = "llama-3.3-70b-versatile",
        embeddings: str = "BAAI/bge-small-en-v1.5",
    ):
        self.hostname = hostname
        self.port = port
        self.persistence_path = persistence_path
        self.client = None
        self.indices: dict[str, VectorStoreIndex] = {}
        self.llm = llm
        self.embeddings = embeddings

        api_key = api_key = os.getenv("GROQ_API_KEY")
        Settings.llm = LLM(api_key, model_name=self.llm)
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embeddings)

    def connect(self) -> bool:
        try:
            self.client = weaviate.connect_to_embedded(
                hostname=self.hostname,
                port=self.port,
                persistence_data_path=self.persistence_path,
            )
            if self.client.is_ready():
                print(f"Connected to Weaviate v4 at {self.hostname}:{self.port}")
                return True
            print("Failed to connect to Weaviate")
            return False
        except Exception as e:
            print(f"🚨 Connection failed: {e}")
            return False

    def disconnect(self):
        if self.client:
            try:
                self.client.close()
                print("Disconnected from Weaviate")
            except Exception as e:
                print(f"Error during disconnect: {e}")

    def create_index(
        self, index_name: str, nodes: List[Document], delete_if_exists: bool = False
    ) -> Optional[VectorStoreIndex]:
        if not self.client:
            print("Not connected to Weaviate. Call connect() first.")
            return None

        try:
            # Delete existing collection if requested
            if delete_if_exists and self.client.collections.exists(index_name):
                self.client.collections.delete(index_name)
                self.indices.pop(index_name, None)
                print(f"Deleted existing collection: {index_name}")

            # If index exists, return it
            if index_name in self.indices:
                print(f"📖 Using cached index: {index_name}")
                return self.indices[index_name]

            if self.client.collections.exists(index_name):
                print(f"📖 Loading existing index: {index_name}")
                vector_store = WeaviateVectorStore(
                    weaviate_client=self.client, index_name=index_name
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                index = VectorStoreIndex([], storage_context=storage_context)
                self.indices[index_name] = index
                return index

            # Create a new index if it doesn’t exist
            vector_store = WeaviateVectorStore(
                weaviate_client=self.client, index_name=index_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            self.indices[index_name] = index
            print(f"✅ Created new index: {index_name} with {len(nodes)} nodes")
            return index

        except Exception as e:
            print(f"Failed to create index '{index_name}': {e}")
            return None

    def get_index(self, index_name: str) -> Optional[VectorStoreIndex]:
        """
        Return an index by name (from cache only).
        """
        index = self.indices.get(index_name)
        if index:
            print(f"📖 Retrieved index from memory: {index_name}")
        else:
            print(f"Index '{index_name}' not found in memory.")
        return index

    def delete_index(self, index_name: str) -> bool:
        if not self.client:
            print("Not connected to Weaviate. Call connect() first.")
            return False
        try:
            if self.client.collections.exists(index_name):
                self.client.collections.delete(index_name)
                self.indices.pop(index_name, None)
                print(f"Deleted index: {index_name}")
                return True
            print(f"Index '{index_name}' does not exist")
            return False
        except Exception as e:
            print(f"Failed to delete index '{index_name}': {e}")
            return False

    def list_collections(self) -> List[str]:
        if not self.client:
            print("Not connected to Weaviate. Call connect() first.")
            return []
        try:
            return list(self.client.collections.list_all())
        except Exception as e:
            print(f"Failed to list collections: {e}")
            return []

    def query_index(self, index_name: str, query: str, top_k: int = 5):
        index = self.get_index(index_name)
        if not index:
            print(f"Index '{index_name}' is not loaded in memory.")
            return None
        try:
            query_engine = index.as_query_engine(similarity_top_k=top_k)
            return query_engine.query(query)
        except Exception as e:
            print(f"Query failed: {e}")
            return None


def main():
    # Initialize DB
    db = WeaviateVectorDB(
        hostname="localhost", port=8080, persistence_path="./test_weaviate_data"
    )

    # Connect
    if not db.connect():
        print("❌ Failed to connect to Weaviate.")
        return

    try:
        # Create sample docs
        sample_docs = [
            Document(text="Machine learning is a subset of AI."),
            Document(text="Python is widely used in data science."),
            Document(text="Vector databases store embeddings efficiently."),
            Document(text="NLP helps computers understand human language."),
            Document(text="Deep learning leverages neural networks."),
        ]

        index_name = "TestIndex"

        # Create index (or reuse if exists)
        print(f"\n🔨 Creating/Loading index '{index_name}'...")
        index = db.create_index(index_name, sample_docs)
        if not index:
            print("❌ Failed to create index")
            return

        # List collections
        print("\n📋 Collections in DB:")
        for col in db.list_collections():
            print(f"  - {col}")

        # Query index
        print("\n🔍 Running test queries...")
        queries = [
            "What is machine learning?",
            "Tell me about Python",
            "How do vector databases work?",
        ]
        for q in queries:
            print(f"\nQuery: {q}")
            response = db.query_index(index_name, q, top_k=2)
            if response:
                print(f"Response: {response}")
            else:
                print("❌ No response")

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
