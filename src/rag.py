import os
from typing import List

from vector_store import WeaviateVectorDB
from ingest import DataProcessor

from llama_index.core import PromptTemplate, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.settings import Settings

from prompts.job_prompt import job_search_prompt_tmpl_str
from utils.load_config import load_config


class JobRAGPipeline:

    def __init__(
        self,
        csv_path: str,
        index_name: str = "JobPosts",
        persistence_path: str = "./weaviate_data",
        delete_if_exists: bool = False,
        chunk_size: int = 300,
        chunk_overlap: int = 20,
        chunk_strategy: str = "normal",
        top_n: int = 2,
        ranking_model: str = "BAAI/bge-reranker-base",
        similarity_top_k: int = 5,
        vector_store_query_mode: str = "hybrid",
        alpha: float = 0.5,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "llama-3.3-70b-versatile",
        tech_specialties: list = None,
        locations: dict = None,
        server_hostname: str = "localhost",
        server_port: int = 8080,
    ):
        self.csv_path = csv_path
        self.index_name = index_name
        self.persistence_path = persistence_path
        self.delete_if_exists = delete_if_exists
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.top_n = top_n
        self.ranking_model = ranking_model
        self.similarity_top_k = similarity_top_k
        self.vector_store_query_mode = vector_store_query_mode
        self.alpha = alpha
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.tech_specialties = tech_specialties or []
        self.locations = locations or {}
        self.server_hostname = server_hostname
        self.server_port = server_port

        # Vector DB wrapper
        self.db = WeaviateVectorDB(
            hostname="localhost",
            port=self.server_port,
            persistence_path=self.persistence_path,
            llm=self.llm_model,
            embeddings=self.embedding_model,
        )

        self.dp = DataProcessor(self.csv_path, self.tech_specialties, self.locations)

        self.index = None
        self.query_engine = None

    def _build_vector_store_info(self) -> VectorStoreInfo:
        return VectorStoreInfo(
            content_info="Job posting descriptions, responsibilities, requirements, and benefits.",
            metadata_info=[
                MetadataInfo(
                    name="title",
                    type="str",
                    description="Job title. Use '==' for exact matches. For partial matches, rely on semantic search in content.",
                ),
                MetadataInfo(
                    name="company",
                    type="str",
                    description="Hiring company name. Use '==' for exact matches. For partial matches, rely on semantic search in content.",
                ),
                MetadataInfo(
                    name="location",
                    type="str",
                    description="Job location (city/region or 'Remote'). Use '==' for exact matches. For partial matches, rely on semantic search in content.",
                ),
                MetadataInfo(
                    name="job_url",
                    type="str",
                    description="Direct link to the job posting. Use '==' for exact matches only.",
                ),
                MetadataInfo(
                    name="job_type",
                    type="str",
                    description="Employment type, e.g., Full-time, Contract, Part-time, Temporary. Use '==' for exact matches.",
                ),
                MetadataInfo(
                    name="date_posted",
                    type="str",
                    description="Posting date as string. Use '==', '>=', or '<=' for filtering by date.",
                ),
                MetadataInfo(
                    name="company_industry",
                    type="str",
                    description="Industry of the hiring company. Use '==' for exact matches. For partial matches, rely on semantic search in content.",
                ),
            ],
        )

    def _build_prompt(self) -> PromptTemplate:
        return PromptTemplate(job_search_prompt_tmpl_str)

    def _build_reranker(self) -> SentenceTransformerRerank:
        return SentenceTransformerRerank(
            top_n=self.similarity_top_k,
            model=self.ranking_model,
        )

    def setup(self):
        """Full setup: connect DB, process data, build index, retriever, reranker, query engine."""
        if not self.db.connect():
            raise RuntimeError("❌ Failed to connect to Weaviate")

        nodes = self.dp.process_documents(
            strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Create/load index
        self.index = self.db.create_index(
            self.index_name, nodes, delete_if_exists=self.delete_if_exists
        )
        if not self.index:
            raise RuntimeError("❌ Failed to create/load index")

        # AutoRetriever with metadata awareness
        vector_store_info = self._build_vector_store_info()
        retriever = VectorIndexAutoRetriever(
            index=self.index,
            llm=Settings.llm,
            vector_store_info=vector_store_info,
            similarity_top_k=self.similarity_top_k,
            vector_store_query_mode=self.vector_store_query_mode,
            alpha=self.alpha,
            verbose=True,
        )

        # Assemble query engine
        rerank = self._build_reranker()
        job_search_prompt_tmpl = self._build_prompt()
        response_synthesizer = get_response_synthesizer(
            text_qa_template=job_search_prompt_tmpl
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[rerank],
        )

    def query(self, query_str: str):
        """Run a query on the pipeline and return structured JSON response."""
        if not self.query_engine:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        response = self.query_engine.query(query_str)

        # Build structured response
        result = {
            "answer": str(response),
            "retrieved_contexts": len(response.source_nodes),
            "contexts": [],
        }

        for i, src in enumerate(response.source_nodes):
            meta = src.node.metadata or {}
            context_info = {"context_id": i, "text": src.node.text, "metadata": {}}

            # Add available metadata
            for k in [
                "title",
                "company",
                "location",
                "job_type",
                "date_posted",
                "job_url",
            ]:
                if k in meta:
                    context_info["metadata"][k] = meta[k]

            result["contexts"].append(context_info)

        return result

    def close(self):
        """Clean shutdown."""
        self.db.disconnect()
