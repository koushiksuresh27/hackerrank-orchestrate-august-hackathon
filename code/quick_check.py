# evaluator_llm.py
import pandas as pd
import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv("../dataset/output.csv")
messages = pd.read_csv("../dataset/messages.csv")
samples = pd.read_csv("../dataset/sample_messages.csv")

combined = messages.merge(df, on='message_id')

# Build sample context for the judge
sample_examples = ""
for _, row in samples.head(10).iterrows():
    sample_examples += f"action={row.action} | type={row.message_type} | conf={row.confidence}\nreason: {row.reason}\n\n"

JUDGE_PROMPT = f"""You are evaluating a WhatsApp message notification router.
The router classifies messages as notify/digest/mute with a message_type and reason.

Here are 10 examples of CORRECT decisions for reference:
{sample_examples}

For each message below, score the routing decision from 1-5:
5 = Perfect: action, type, reason all correct and specific
4 = Good: action correct, type or reason slightly off
3 = Acceptable: action correct but reasoning is generic or type is wrong
2 = Poor: action is wrong but reasoning shows some understanding
1 = Wrong: action is wrong and reasoning is generic or incorrect

Return JSON only:
{{"score": X, "verdict": "one sentence explanation"}}
"""

scores = []
for _, row in combined.sample(20, random_state=42).iterrows():
    user_msg = f"""Message: {str(row.message_text)[:200]}
Conversation type: {row.conversation_type}
Forwarded count: {row.forwarded_count}

Router decision:
action={row.action} | type={row.message_type} | conf={row.confidence}
reason: {row.reason}"""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 150,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_msg}
        ]
    }

    key = os.getenv("GROQ_API_KEY")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload, headers=headers, timeout=30
        )
        text = resp.json()['choices'][0]['message']['content']
        # strip markdown
        text = text.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(text)
        score = result['score']
        verdict = result['verdict']
    except Exception as e:
        score = 0
        verdict = f"eval failed: {e}"

    scores.append(score)
    print(f"[{row.message_id}] {row.action}/{row.message_type} → Score: {score}/5")
    print(f"  Message: {str(row.message_text)[:80]}")
    print(f"  Reason: {row.reason}")
    print(f"  Judge: {verdict}")
    print()

print(f"=== OVERALL SCORE: {sum(scores)}/{len(scores)*5} ({round(sum(scores)/len(scores),1)}/5 avg) ===")