# WhatsApp Message Notification Router

AI-powered system that classifies incoming WhatsApp messages into 
notify, digest, or mute using multimodal signals, user history, 
and personalized routing decisions.

## Setup
pip install -r requirements.txt

## Environment Variables
Create a .env file in the code/ folder with:
GEMINI_API_KEY=...
GEMINI_API_KEY_2=...
GEMINI_API_KEY_3=...
GROQ_API_KEY=...
GROQ_API_KEY_2=...
GROQ_API_KEY_3=...
GROQ_API_KEY_4=...
GROQ_API_KEY_5=...
ANTHROPIC_API_KEY=...
ANTHROPIC_API_KEY_2=...

## Run
python main.py --dataset dataset/ --output dataset/output.csv

## Architecture
- Signal extraction: 21 deterministic flags catch scams, 
  injections, muted groups before any LLM call
- Provider chain: Groq (5 keys) → Claude (2 keys) → rule-based fallback
- Images: routed to vision-capable provider when available
- Voice: transcribed locally with Whisper
- Cache: per-message JSON files, crash-safe
- Fallback: rule-based guarantees all 110 rows always complete

## Fallback Behavior
If all API providers are exhausted, rule-based fallback ensures
every row receives a valid, reasoned decision. Output is always
complete — no missing rows, no error strings.
