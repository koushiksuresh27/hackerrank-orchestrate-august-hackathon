# Orchestrate Hackathon: Development Guidelines

## Core Philosophy
Build a reliable system, not just a working demo. You'll be judged on:
1. Code quality and architecture
2. Output correctness (action, message_type, reasoning, confidence, evidence)
3. Chat transcript showing your thought process
4. 30-min voice interview defending your design

---

## System Architecture Checklist

Your code should show clear separation of concerns:

- **Input Loading**: Read and normalize all CSVs (messages, users, groups, business, history, events, media)
- **Context Building**: Retrieve relevant historical data, user preferences, group info, business details
- **Agent Call**: Single point where the LLM makes the routing decision
- **Output Validation**: Enforce schema, reject invalid labels, retry malformed responses
- **Error Handling**: Log failures, continue safely, flag uncertainty rather than guess
- **Final Output**: Write CSV with no duplicates, missing fields, or invalid labels

**Trace one message through your system:** Can you explain exactly what happens from CSV read to output write?

---

## Code Quality Standards

**Naming**: Use descriptive file and function names (not `helper.py`, `utils`, or `test2`)

**Structure**: Separate concerns into logical modules:
- `data_loader.py` - read CSVs, normalize
- `context_builder.py` - retrieve relevant history/metadata
- `agent.py` - LLM call with prompt
- `validator.py` - schema validation, label enforcement
- `main.py` - orchestration

**Hygiene**:
- No hardcoded paths (use config files)
- No secrets in code
- All dependencies in `requirements.txt`
- Clear entry point in README

**Prompts**: Write them like code—clear, compact, reusable
- State the task explicitly
- List all allowed output labels
- Specify output format (JSON schema)
- Explain what to do when information is missing
- Include guardrails for ambiguous inputs

---

## Reliability: Making It Production-Ready

Run your system on the **full dataset**. Don't just inspect the cases that worked.

**Test for**:
- Empty rows, missing fields, null values
- Ambiguous evidence (conflicting signals, missing history)
- Repeated message IDs, duplicated senders
- Model hallucinations (invented labels, unsupported reasoning)
- Edge cases (new users, new groups, new businesses with no history)

**Validation Guards**:
- Validate inputs *before* calling the model (schema, required fields)
- Validate outputs *after* (reject invalid labels, enforce allowed set)
- Retry on malformed JSON within a limit (3x)
- Fallback: abstain or flag for review rather than guess

---

## Evaluation Loop (Critical)

1. **Run** the full system on all messages
2. **Sample** 20-50 outputs across different message types
3. **Inspect** each:
   - Does the action match the reasoning?
   - Is the confidence calibrated (high for clear cases, lower for ambiguous)?
   - Do evidence_message_ids actually support the decision?
   - Do similar messages get similar treatment?
4. **Identify** patterns in failures
5. **Adjust** prompt, validation, or context logic
6. **Re-run** and repeat

This loop is more important than the first perfect prompt.

---

## Output CSV Requirements

**Columns (in order)**:
```
message_id, action, message_type, reason, confidence, evidence_message_ids
```

**Validation**:
- All message IDs from input must be present (no missing rows)
- No duplicate message_ids
- `action` ∈ {`notify`, `digest`, `mute`}
- `message_type` ∈ {`personal`, `urgent`, `event`, `payment`, `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`, `unknown`}
- `confidence` is a float between 0 and 1
- `evidence_message_ids` is semicolon-separated (or `none`)
- `reason` is a 1-2 sentence human-readable explanation

**Consistency Check**: Similar cases should have similar decisions. If two messages differ only in sender or recipient user, can you justify why they have different actions?

---

## Chat Transcript Strategy

Show ownership through three phases:

### Plan (Before Building)
- Explain what you understood from the problem
- Define input/output clearly
- Identify where the model should reason (routing decision) vs. where code should enforce rules (validation, label enforcement)
- Note edge cases you expect (missing history, new users, conflicting signals)

### Build (During Development)
- Name the files, functions, and schemas you're creating
- Explain the prompt you wrote and why it includes specific instructions
- Describe validation logic and what makes output valid
- If you add retrieval or context building, explain what's retrieved and how it's passed to the model

### Review (After First Run)
- Sample outputs, inspect them
- Point out failures and what caused them
- Show how you adjusted the prompt, validation, or architecture to fix it
- Repeat: this loop is the real work

**Tone**: Sound like an engineer solving a problem, not a student reporting a lab. "I found that the model was inventing labels, so I added a strict enum check" is better than "I made the system more robust."

---

## Interview Talking Points

Be ready to explain:

1. **Architecture**: Walk through one message from input to output. Where is it read? Contextualized? Decided? Validated? Logged?

2. **Judgment Calls**: 
   - Why did you keep things simple vs. adding complexity?
   - Where did you add guardrails and why?
   - What should the user see in the output to make the decision trustworthy?

3. **Edge Cases**: What happens when:
   - A user has no history?
   - A message has no matching business or group?
   - Signals conflict (e.g., trusted sender but phishing text)?
   - The model returns an unsupported label?

4. **AI Fluency**: Explain your prompt, what you ask the model to do, what you keep outside the model, how you check the work

5. **Honest Limitations**: What did you test? What didn't you? Where is the system fragile? What would you improve with more time?

---

## Quick Checklist Before Submission

- [ ] Code is runnable (`python main.py` or similar)
- [ ] README has setup, dependencies, env vars, command, expected files
- [ ] All output rows present with no duplicates
- [ ] All columns in correct order with correct types
- [ ] All labels from allowed sets only
- [ ] Confidence values between 0 and 1
- [ ] Evidence IDs point to real message_ids or say `none`
- [ ] Reasons are clear and consistent with actions
- [ ] Similar messages have similar decisions (or you can explain why they differ)
- [ ] Chat transcript shows plan → build → review loop
- [ ] You can explain every major design decision

---

## Key Reminders

- **Personalization**: Same message type needs different routing for different users. Use full context.
- **Multimodal**: Messages include text, images, and voice. Consider media when available.
- **Safety First**: Scams and risky content should be muted regardless of user engagement.
- **Evidence Matters**: Explain your decision by pointing to past behavior, sender trust, group norms, or business relationship.
- **Confidence is Real**: High confidence for clear signals (trusted contact, payment update), lower for ambiguous cases (new sender, no history).
- **Evaluation is the Work**: The quality of your output comes from testing, not from the first prompt being perfect.
