# evaluation.py
import pandas as pd
import re
import json
from typing import List, Dict
import argparse
from datetime import datetime
import os
import sys

from prompts.generate_eval_data_prompt import generat_data_prompt_tmpl_str
from llm import LLM

from dotenv import load_dotenv

load_dotenv()


class SyntheticQueryGenerator:
    def __init__(self, csv_path: str, groq_api_key: str):
        # Load CSV and drop rows with missing description
        self.df = pd.read_csv(csv_path).dropna(subset=["description"])
        self.llm = LLM(groq_api_key, model_name="openai/gpt-oss-120b")

    def generate_queries(
        self, n_jobs: int = 15, max_queries_per_job: int = 3
    ) -> List[Dict]:
        queries = []

        # Limit to n_jobs
        df_subset = self.df.head(n_jobs).sample(
            n=min(n_jobs, len(self.df)), random_state=42
        )

        for idx, row in df_subset.iterrows():
            print(f"Processing job {idx + 1}/{len(df_subset)}...")

            # Extract all relevant fields
            title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
            company = (
                str(row.get("company", "")) if pd.notna(row.get("company")) else ""
            )
            location = (
                str(row.get("location", "")) if pd.notna(row.get("location")) else ""
            )
            job_type = (
                str(row.get("job_type", "")) if pd.notna(row.get("job_type")) else ""
            )
            date_posted = (
                str(row.get("date_posted", ""))
                if pd.notna(row.get("date_posted"))
                else ""
            )
            company_industry = (
                str(row.get("company_industry", ""))
                if pd.notna(row.get("company_industry"))
                else ""
            )
            job_url = (
                str(row.get("job_url", "")) if pd.notna(row.get("job_url")) else ""
            )
            description = (
                str(row.get("description", ""))
                if pd.notna(row.get("description"))
                else ""
            )

            # Get truncated description for prompt
            desc_truncated = (
                " ".join(str(row["description"]).split()[:40])
                if pd.notna(row.get("description"))
                else ""
            )

            # Build detailed prompt with all fields
            prompt = f"""{generat_data_prompt_tmpl_str}
                Job Details:
                Title: {title}
                Company: {company}
                Location: {location}
                Job Type: {job_type}
                Industry: {company_industry}
                Date Posted: {date_posted}
                Description: {desc_truncated}

            Generate specific search queries for this exact job:"""

            print(f"\n{'='*80}")
            print(f"Prompt: {prompt}...")  # Print first 200 chars of prompt

            try:
                response = self.llm.complete(prompt)
                print(f"Raw LLM response: {response}")

                out = str(response)

                print(f"Extracted text: {out}")

                # Clean the response - remove markdown code blocks if present
                cleaned_out = out.replace("```json", "").replace("```", "").strip()

                # Try to extract JSON array with more robust pattern
                match = re.search(r"\[[\s\S]*\]", cleaned_out)
                if match:
                    try:
                        qlist = json.loads(match.group(0))
                        print(f"Parsed queries: {qlist}")
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        # Fallback: try to extract quoted strings
                        quoted_strings = re.findall(r'"([^"]*)"', cleaned_out)
                        qlist = quoted_strings if quoted_strings else [title]
                else:
                    print(
                        "No JSON array found in response, trying to extract quoted strings"
                    )
                    quoted_strings = re.findall(r'"([^"]*)"', cleaned_out)
                    qlist = quoted_strings if quoted_strings else [title]

                # Create job data with queries
                job_data = {
                    "gold_job_id": int(row.name),
                    "title": title,
                    "description": description,
                    "company": company,
                    "location": location,
                    "job_type": job_type,
                    "date_posted": date_posted,
                    "company_industry": company_industry,
                    "job_url": job_url,
                    "queries": [q.strip() for q in qlist[:max_queries_per_job]],
                }
                queries.append(job_data)

                tokens_used = len(prompt.split()) + len(str(response).split())
                used_tokens += tokens_used

                if used_tokens > 8000:  # near 15k TPM
                    print("Near limit, pausing 60s...")
                    time.sleep(60)
                    used_tokens = 0

            except Exception as e:
                print(f"Error generating queries: {e}")

        return queries

    def save(self, data: List[Dict], path: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def create_evaluation_dataset(self, n_jobs: int = 15, max_queries_per_job: int = 3):
        """Create a complete evaluation dataset with jobs and their queries"""
        jobs_with_queries = self.generate_queries(n_jobs, max_queries_per_job)

        # Save everything in a single file
        self.save(jobs_with_queries, "output/jobs_with_queries.json")

        return jobs_with_queries


# Example usage script
if __name__ == "__main__":
    # For testing without command line arguments
    csv_path = "../../data/tech_jobs_dataset.csv"
    groq_api_key = os.getenv("GROQ_API_KEY")

    generator = SyntheticQueryGenerator(csv_path, groq_api_key)

    # Generate complete evaluation dataset
    jobs_with_queries = generator.create_evaluation_dataset(
        n_jobs=15, max_queries_per_job=3
    )

    print("Done!")
    print(f"Generated {len(jobs_with_queries)} jobs with queries")
    print("Sample data:")
    for i, job in enumerate(jobs_with_queries):  # Show first 2 jobs
        print(f"Job {i+1}:")
        print(f"  Title: {job['title']}")
        print(f"  Company: {job['company']}")
        print(f"  Location: {job['location']}")
        print(f"  Queries: {job['queries']}")
        print()
