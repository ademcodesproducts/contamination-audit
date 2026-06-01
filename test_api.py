"""Quick test: verify the OpenAI API key works and the judge prompt parses correctly."""
import os, json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

prompt = """You are evaluating whether two math problems test the same underlying reasoning skill.

Problem A (from training data):
Find all real solutions to x^2 - 5x + 6 = 0.

Problem B (from benchmark):
Solve for x: x^2 - 5x + 6 = 0.

Answer the following:
1. Do these problems require the same core mathematical reasoning steps to solve? (Yes/No)
2. If someone memorized the solution approach to Problem A, would that give them an unfair advantage on Problem B? (Yes/No)
3. Are these problems structurally equivalent despite surface differences? (Yes/No)

Classify as:
- CONTAMINATED: 2 or more Yes answers
- RELATED: exactly 1 Yes answer
- CLEAN: 0 Yes answers

Respond in JSON with this exact format:
{"q1": "Yes/No", "q2": "Yes/No", "q3": "Yes/No", "label": "CONTAMINATED/RELATED/CLEAN", "justification": "one sentence"}
"""

print("Calling gpt-4o-mini...")
try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content
    print(f"Raw response:\n{raw}\n")
    parsed = json.loads(raw)
    print(f"Parsed OK: {parsed}")
except Exception as e:
    print(f"ERROR: {e}")
