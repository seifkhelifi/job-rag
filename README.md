# RAG-Based Job Matching System

This repository contains a **Retrieval-Augmented Generation (RAG)** pipeline designed for matching user queries to relevant job postings using **semantic search**. The system leverages **LlamaIndex** to retrieve job offers from a structured dataset and generate relevant responses.

---

## Features

- **Semantic Job Matching**: Finds the most relevant job postings based on natural language queries.
- **RAG Pipeline**: Combines retrieval of job postings with generative responses for detailed results.
- **Evaluation Metrics**: Provides detailed evaluation of both retrieval and generation performance.
- **Advanced Techniques Applied**:
  - **Auto-retrieval**: Infers metadata filters directly from the user query.
  - **Hybrid search**: Combines dense vector search with metadata filtering.
  - **Re-ranking**: Improves the relevance of retrieved results.

---

## Dataset Structure

The dataset is expected to be a CSV containing job postings with at least the following fields:

```text
- title: Job title
- company: Company name
- description: Job description
- location: Job location (optional)
- skills: Required skills (optional)
````

Each document is indexed with metadata for efficient semantic search.

---

## Evaluation Report

The system was evaluated using **18 sample queries**. Metrics were computed for both retrieval and generation components.

### Retrieval Metrics

| Metric       | Score |
| ------------ | ----- |
| Recall\@5    | 0.722 |
| Precision\@5 | 0.611 |
| MRR          | 0.722 |

### Generation Metrics

| Metric                    | Score |
| ------------------------- | ----- |
| Faithfulness (LlamaIndex) | 0.556 |

These metrics indicate strong retrieval performance, with room for improvement in generation faithfulness.

---

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Load your dataset and initialize the RAG pipeline:

```python
from rag_pipeline import JobRAGPipeline

pipeline = JobRAGPipeline("jobs_dataset.csv")
```

3. Query the system:

```python
results = pipeline.query("Software engineer with Python experience in sales")
for job in results:
    print(job['title'], job['company'])
```

---

## Evaluation Framework

The system includes a dedicated evaluation module:

* **Query Generation**: Generates 1–3 realistic user queries per job posting for testing.
* **Retrieval Evaluation**: Computes Recall\@k, Precision\@k, and MRR.
* **Generation Evaluation**: Uses LlamaIndex `FaithfulnessEvaluator` to score generated outputs.

This allows for continuous improvement of both the retrieval and generation components.

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Seif Khelifi
khelifiseif1@gmail.com



