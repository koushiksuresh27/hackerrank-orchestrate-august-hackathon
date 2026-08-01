# debug_providers.py — run this in your code/ folder
import os
import requests
from dotenv import load_dotenv


load_dotenv()
# add to debug_providers.py and run
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
print("=== ENV CHECK ===")
print("ANTHROPIC_API_KEY:", (os.getenv("ANTHROPIC_API_KEY") or "NOT FOUND")[:20] + "...")
print("ANTHROPIC_API_KEY_2:", (os.getenv("ANTHROPIC_API_KEY_2") or "NOT FOUND")[:20] + "...")
print("GEMINI_API_KEY:", (os.getenv("GEMINI_API_KEY") or "NOT FOUND")[:20] + "...")
print("GEMINI_API_KEY_2:", (os.getenv("GEMINI_API_KEY_2") or "NOT FOUND")[:20] + "...")
print("GROQ_API_KEY:", (os.getenv("GROQ_API_KEY") or "NOT FOUND")[:20] + "...")
print("GROQ_API_KEY_2:", (os.getenv("GROQ_API_KEY_2") or "NOT FOUND")[:20] + "...")

print("\n=== GEMINI TEST ===")
try:
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "Say hello in one word."}]}]
    }
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print("Gemini OK:", text.strip())
    else:
        print(f"Gemini FAILED: HTTP {resp.status_code} - {resp.text}")
except Exception as e:
    print("Gemini FAILED:", type(e).__name__, str(e))

print("\n=== CLAUDE TEST ===")
try:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key or "",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say hello in one word."}]
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        text = data.get("content", [{}])[0].get("text", "")
        print("Claude OK:", text.strip())
    else:
        print(f"Claude FAILED: HTTP {resp.status_code} - {resp.text}")
except Exception as e:
    print("Claude FAILED:", type(e).__name__, str(e))

print("\n=== GROQ TEST ===")
try:
    api_key = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print("Groq OK:", text.strip())
    else:
        print(f"Groq FAILED: HTTP {resp.status_code} - {resp.text}")
except Exception as e:
    print("Groq FAILED:", type(e).__name__, str(e))