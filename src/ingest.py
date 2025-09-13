import sys
import os

import csv
from jobspy import scrape_jobs
import pandas as pd

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# from llama_index.experimental.node_parser import SemanticSplitterNodeParser

# ne9sa scrape and upadte index


class DataProcessor:

    def __init__(self, csv_path, tech_specialties, locations):
        self.csv_path = csv_path
        self.df: pd.DataFrame | None = None
        self.documents: List[Document] | None = None
        self.tech_specialties = tech_specialties
        self.locations = locations

    def scrape_job_offers():
        # Collect jobs
        all_jobs = []
        for country, location in self.locations.items():
            for specialty in self.tech_specialties:
                jobs = scrape_jobs(
                    site_name=["linkedin", "indeed", "glassdoor"],
                    search_term=specialty,
                    location=location,
                    results_wanted=15,  # Limit to 15 per specialty per location
                    hours_old=72,  # Jobs posted in the last 3 days
                    country_indeed=country,
                    linkedin_fetch_description=True,  # Fetch job descriptions
                )
                all_jobs.extend(jobs.to_dict(orient="records"))

        # Save to CSV
        fieldnames = all_jobs[0].keys() if all_jobs else []
        if all_jobs:
            with open(
                "tech_jobs_dataset.csv", "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                    quoting=csv.QUOTE_NONNUMERIC,
                    escapechar="\\",
                )
                writer.writeheader()
                writer.writerows(all_jobs)
            print(f"Saved {len(all_jobs)} job postings to tech_jobs_dataset.csv")
        else:
            print("No job postings found.")

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess CSV data"""
        print(f"Loading csv from path {self.csv_path}:")
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["job_url", "title"])
        # df = df.iloc[:2]  # Limit to 2 job posts for testing
        self.df = df
        return df

    def build_documents(self) -> list[Document]:
        """Convert dataframe rows into Documents with metadata."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        documents = [
            Document(
                text=row["description"],
                metadata={
                    **({"title": row["title"]} if pd.notna(row["title"]) else {}),
                    **({"company": row["company"]} if pd.notna(row["company"]) else {}),
                    **(
                        {"location": row["location"]}
                        if pd.notna(row["location"])
                        else {}
                    ),
                    **({"job_url": row["job_url"]} if pd.notna(row["job_url"]) else {}),
                    **(
                        {"job_type": row["job_type"]}
                        if pd.notna(row["job_type"])
                        else {}
                    ),
                    **(
                        {"date_posted": row["date_posted"]}
                        if pd.notna(row["date_posted"])
                        else {}
                    ),
                    **(
                        {"company_industry": row["company_industry"]}
                        if pd.notna(row["company_industry"])
                        else {}
                    ),
                },
            )
            for _, row in self.df.iterrows()
        ]
        self.documents = documents

        print(f"Documents built :  {len(documents)}:")

        return documents

    def chunk_documents(
        self,
        strategy: str = "normal",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Split documents into nodes based on strategy.

        Args:
            documents: List of LlamaIndex Documents.
            strategy: "normal" or "semantic"
            chunk_size: size of each chunk (for normal)
            chunk_overlap: overlap between chunks (for normal)
        """
        if strategy == "semantic":
            # parser = SemanticSplitterNodeParser.from_defaults()
            pass
        else:
            parser = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        nodes = parser.get_nodes_from_documents(self.documents)

        print(f"Created {len(nodes)} nodes from {len(self.documents)} documents")

        for i in range(5):
            print(f"Chunk {i}:")
            print("Text:")
            print(nodes[i].text)
            print("------------------")
            print(f"Job post title: {nodes[i].metadata['title']}\n")

        return nodes

    def process_documents(
        self,
        strategy: str = "normal",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Full pipeline: load → build documents → chunk into nodes.
        Returns the nodes for downstream use.
        """
        if self.df is None:
            self.load_data()

        if self.documents is None:
            self.build_documents()

        nodes = self.chunk_documents(
            strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return nodes


if __name__ == "__main__":
    data_processor = DataProcessor("../data/tech_jobs_dataset.csv")
    data_processor.process_documents(
        strategy="normal", chunk_size=300, chunk_overlap=20
    )
