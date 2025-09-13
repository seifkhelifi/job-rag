job_search_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information, provide **job postings relevant to the user's query**.
Return **only a numbered list** with these fields: 
Title, Company, Location, Short reason it matches, Link (if any). 
Do **not** include extra explanations, skills, or commentary.

Examples:

Query: Find job postings for a Data Scientist position in New York City.
Answer:
1) Title: Data Scientist — Company XYZ — New York, NY — Matches because of experience in machine learning and data analysis. — <job_url_if_any>
2) Title: Data Scientist — ABC Corp — New York, NY — Matches because role requires Python and statistical modeling. — <job_url_if_any>

Query: {query_str}
Answer:
"""
