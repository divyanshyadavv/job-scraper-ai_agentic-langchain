import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools.playwright.utils import create_sync_playwright_browser
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from groq_wrapper import GroqWrapper

# === Load environment variables ===
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# === Initialize LLM ===
llm = GroqWrapper(api_key=groq_api, model_name="Meta-Llama/Llama-4-Scout-17b-16e-Instruct")
print("✅ Model Loaded Successfully")

# === Setup Playwright Browser + Tools ===
sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()
single_input_tools = [tool for tool in tools if getattr(tool, "args", None) is None or len(tool.args) == 1]

# === Initialize the agent ===
agent_executor = initialize_agent(
    tools=single_input_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# === Prompt Builder ===
def build_prompt(company):
    return f"""
You are a web automation agent. Your only task is to return a valid JSON object with job data.

Steps:
- Go to the official website of {company}
- Find the careers page
- Look for jobs related to "Data Science" or "Analytics"
- Extract job titles and their URLs
- Return ONLY a valid JSON object in this format:

{{
  "Company": "{company}",
  "Jobs": {{
    "Job Title 1": "Job URL 1",
    "Job Title 2": "Job URL 2"
  }}
}}

Rules:
- Absolutely NO explanation, preamble, or thoughts.
- No <think>, no markdown, no descriptions.
- Output must be raw JSON ONLY.
- If no jobs are found, return an empty "Jobs" object.

Only respond with the final JSON. Do not include any other text.
"""


# === CSV/Excel-Based Runner ===
if __name__ == "__main__":
    input_path = r"C:\Users\Dell\scrapingjob2\companylistsmall.csv"
    output_path = r"C:\Users\Dell\scrapingjob2\output_with_jobs.csv"

    df = pd.read_csv(input_path)

    if 'Company' not in df.columns:
        raise ValueError("The input file must have a 'Company' column.")

    job_results = []

    for company in df["Company"]:
        print(f"\n🔍 Searching for data jobs at: {company}")
        try:
            prompt = build_prompt(company)
            raw_output = agent_executor.run(prompt)

            # Try to parse JSON output
            parsed = json.loads(raw_output)

            # Flatten jobs to a simple text block for easier Excel export
            jobs = parsed.get("Jobs", {})
            job_lines = [f"{title} → {url}" for title, url in jobs.items()]
            result_text = "\n".join(job_lines) if job_lines else "No relevant jobs found."

            job_results.append(result_text)

        except Exception as e:
            job_results.append(f"Error: {str(e)}")

    df["Job Results"] = job_results
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
