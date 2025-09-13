import json

import sys
import os
from rag import JobRAGPipeline

from llama_index.core.evaluation import FaithfulnessEvaluator
from llm import LLM
from statistics import mean
from typing import List, Dict
from dotenv import load_dotenv


class RAGEvaluator:
    def __init__(
        self, dataset_path: str, rag_pipeline: JobRAGPipeline, llm: LLM, k: int = 5
    ):
        """
        :param dataset_path: Path to the gold dataset JSON
        :param rag_pipeline: Initialized RAG pipeline
        :param llm: LLM instance for faithfulness evaluation
        :param k: Top-k contexts to evaluate retrieval
        """
        self.dataset_path = dataset_path
        self.pipeline = rag_pipeline
        self.llm = llm
        self.k = k

        # Faithfulness evaluator from llama_index
        self.faithfulness_eval = FaithfulnessEvaluator(llm=llm)

        # Storage for results
        self.results = []

    def _compute_retrieval_metrics(
        self, gold_url: str, retrieved_contexts: List[Dict]
    ) -> Dict:
        """
        Compute retrieval metrics: Recall@k, Precision@k, MRR
        """
        retrieved_urls = [
            ctx["metadata"].get("job_url", "") for ctx in retrieved_contexts
        ]
        hit_positions = [
            i for i, url in enumerate(retrieved_urls, start=1) if url == gold_url
        ]

        recall = 1.0 if hit_positions else 0.0
        precision = (
            sum(1 for url in retrieved_urls if url == gold_url) / len(retrieved_urls)
            if retrieved_urls
            else 0.0
        )
        mrr = 1.0 / hit_positions[0] if hit_positions else 0.0

        return {"recall@k": recall, "precision@k": precision, "mrr": mrr}

    def _compute_faithfulness(self, answer: str, contexts: List[Dict]) -> float:
        """
        Use llama_index FaithfulnessEvaluator to check if the answer is faithful to retrieved contexts.
        Returns a score (0.0 - 1.0)
        """
        context_texts = [c["text"] for c in contexts]
        eval_result = self.faithfulness_eval.evaluate(
            query="", response=answer, contexts=context_texts
        )
        return 1.0 if eval_result.passing else 0.0

    def evaluate(self):
        """
        Run evaluation on the dataset
        """
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)

        for job in dataset:
            gold_url = job["job_url"]
            queries = job["queries"]

            for q in queries:
                response = self.pipeline.query(q)
                answer = response.get("answer", "")
                contexts = response.get("contexts", [])

                retrieval_metrics = self._compute_retrieval_metrics(gold_url, contexts)
                faithfulness_score = self._compute_faithfulness(answer, contexts)

                self.results.append(
                    {
                        "gold_job_id": job["gold_job_id"],
                        "query": q,
                        "retrieval": retrieval_metrics,
                        "faithfulness": faithfulness_score,
                    }
                )

    def generate_report(self) -> str:
        """
        Summarize evaluation results in a clean, human-readable format
        """
        recall_scores = [r["retrieval"]["recall@k"] for r in self.results]
        precision_scores = [r["retrieval"]["precision@k"] for r in self.results]
        mrr_scores = [r["retrieval"]["mrr"] for r in self.results]
        faithfulness_scores = [r["faithfulness"] for r in self.results]

        report = "\n=== RAG Evaluation Report ===\n"
        report += f"Total Queries Evaluated: {len(self.results)}\n\n"

        report += "Retrieval Metrics:\n"
        report += f"  Recall@{self.k}: {mean(recall_scores):.3f}\n"
        report += f"  Precision@{self.k}: {mean(precision_scores):.3f}\n"
        report += f"  MRR: {mean(mrr_scores):.3f}\n\n"

        report += "Generation Metrics:\n"
        report += f"  Faithfulness (LlamaIndex): {mean(faithfulness_scores):.3f}\n"

        return report


if __name__ == "__main__":
    # Init pipeline

    from utils.load_config import load_config

    config = load_config("../config.yml")

    pipeline = JobRAGPipeline(
        csv_path=config["data"]["csv_path"],
        index_name=config["vector_store"]["index_name"],
        persistence_path=config["vector_store"]["persistence_path"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
        chunk_strategy=config["chunking"]["strategy"],
        delete_if_exists=True,
        # Ranking parameters
        top_n=config["ranking"]["top_n"],
        ranking_model=config["ranking"]["model"],
        # Retrieval parameters
        similarity_top_k=config["retrieval"]["similarity_top_k"],
        vector_store_query_mode=config["retrieval"]["vector_store_query_mode"],
        alpha=config["retrieval"]["alpha"],
        # Model parameters
        embedding_model=config["models"]["embedding"],
        llm_model=config["models"]["llm"],
        # Additional parameters
        tech_specialties=config["tech_specialties"],
        locations=config["locations"],
        # Server parameters
        server_hostname=config["server"]["hostname"],
        server_port=config["server"]["port"],
    )

    pipeline.setup()

    load_dotenv()
    api_key = os.getenv("API_KEY")

    # Init LLM
    llm = LLM(
        api_key,
        model_name="openai/gpt-oss-120b",
    )

    # Run evaluator
    evaluator = RAGEvaluator(
        dataset_path="./output/jobs_with_queries.json",
        rag_pipeline=pipeline,
        llm=llm,
        k=5,
    )

    evaluator.evaluate()
    print(evaluator.generate_report())

    pipeline.close()
