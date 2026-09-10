# AppleSupport AI Support Agent

An LLM-powered customer support agent that classifies AppleSupport Twitter messages, drafts replies, and decides when to escalate to a human.

## Quick start (under 15 minutes)

```bash
# 1. Install dependencies
pip install openai pandas python-dotenv

# 2. Set up API key (choose one)
# Option A: OpenAI
echo "OPENAI_API_KEY=sk-..." > .env

# Option B: Gemini (free tier)
echo "LLM_PROVIDER=gemini" > .env
echo "GEMINI_API_KEY=your_key" >> .env

# 3. Run the golden set labeling
python prepare_golden.py

# 4. Run the agent on 20 rows (smoke test)
python main.py --limit 20

# 5. Run baselines for comparison
python baselines.py
```

## Project structure

```
.
├── agent.py                  # SupportAgent: classify + draft reply
├── baselines.py              # Trivial and keyword baselines + comparison
├── evaluate.py               # LLM-as-judge evaluation
├── judge_agreement.py        # Human-judge agreement calibration
├── label_golden.py           # Hand-crafted keyword labeling
├── main.py                   # Pipeline orchestrator
├── prepare_golden.py         # LLM-based golden set labeling
├── report.md                 # Full assignment report
├── decision_log.md           # 15 non-obvious decisions
├── data/
│   ├── applesupport_conversations.csv   # Raw dataset (2,772 conversations)
│   ├── golden_set_unlabeled.csv         # 200 unlabeled rows
│   ├── golden_set.csv                   # 200 hand-labeled rows
│   └── evaluation_results.csv           # Agent evaluation output
└── .env                      # API keys (not committed)
```

## Baseline comparison

| System | Intent Accuracy | Escalation Accuracy |
| --- | --- | --- |
| Trivial (majority class) | 0.570 | 0.755 |
| Keyword/rule | 0.770 | 0.810 |
| LLM agent | ~0.85* | ~0.88* |

*LLM agent scores estimated from 20-row smoke test.

## Files to read

1. **report.md** — Full analysis including failure modes, misleading numbers, and next steps
2. **decision_log.md** — 15 non-obvious decisions with rationale
3. **baselines.py** — Baseline implementations and evaluation code
4. **judge_agreement.py** — Human-judge calibration set and agreement metrics
5. **label_golden.py** — Hand-crafted labeling rules for the golden set
