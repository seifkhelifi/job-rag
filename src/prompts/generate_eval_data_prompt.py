generat_data_prompt_tmpl_str = """You are given a job posting with several fields. Your task is to generate up to 3 realistic, specific user search queries that someone might type to find *this exact job*. Each query must:
- Be natural and human-sounding (like a real search query)
- Include distinctive details from the posting (skills, technologies, location, job type, industry, or experience)
- Include company name just in a single query
- Avoid generic placeholders (e.g., 'job with salary xx') or vague phrases
- Not invent facts that are not in the job posting

Important formatting rule: Respond with **only** a JSON array of strings. Do not include numbers, explanations, or any text outside the JSON.

Example:
Title: Machine Learning Engineer
Company: TechCorp
Location: San Francisco, CA
Job Type: Full-time
Industry: Software
Description: Work on recommendation systems using Python and PyTorch. Requires 3+ years experience.
Queries:
["looking for a machine learning engineer job in san francisco using python and pytorch", "any openings at techcorp for recommendation system developers (full time)", "ml engineer position in the bay area software industry requiring 3+ years of experience"]"""
