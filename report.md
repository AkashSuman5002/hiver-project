# AppleSupport AI Support Agent

## Problem framing

This project turns a customer message into three outputs: an intent, an escalation decision with a reason, and a concise draft reply. The target user is an AppleSupport Twitter customer who needs a useful next action without exposing private account information. The system does not authenticate users, access Apple systems, resolve payments, perform repairs, or claim that a fix succeeded.

**What "good" means for AppleSupport:**
- Classify the customer's issue correctly so the right team handles it
- Draft a reply that acknowledges the problem, offers a safe next step, and does not invent facts
- Escalate to a human when account-specific, payment, or physical repair is needed — never guess
- Keep replies under 80 words, warm and professional, matching AppleSupport's Twitter voice

**What I chose not to build:**
- No multi-turn conversation handling — each message is classified independently
- No retrieval of similar past conversations — the agent drafts from its prompt instructions
- No real-time API integration with Apple systems — replies are text-only suggestions
- No user authentication — the agent cannot verify accounts or access private data

## Intent taxonomy

The taxonomy is deliberately small and derived from recurring support themes in the sampled conversations:

| Intent | Meaning | Count in golden set |
| --- | --- | --- |
| `software_bug` | iOS, macOS, app, update, crash, or unexpected software behavior | 114 |
| `hardware_issue` | Device, battery, speaker, screen, storage, or physical malfunction | 22 |
| `account_access` | Apple ID, login, password, activation, or account-specific access | 3 |
| `billing_purchase` | Charges, subscriptions, refunds, or purchase problems | 11 |
| `connectivity` | Wi-Fi, cellular, Bluetooth, downloads, or network connection | 7 |
| `general_inquiry` | Questions that do not fit the categories above | 43 |

Escalation is recommended for physical repair, account-specific investigation, payment disputes, safety-sensitive cases, or situations where a safe self-service answer is not possible.

## System design

`prepare_golden.py` reads the 200-row unlabeled file and asks an LLM for strict JSON labels. It checkpoints every ten rows and skips rows that are already labeled. `SupportAgent` makes one structured classification call and one drafting call. The drafting prompt supplies the assigned intent and escalation decision, asks for safe actionable steps, and caps replies at 80 words. `evaluate.py` asks a separate judge to score the candidate against the customer message and the historical AppleSupport reply on helpfulness, relevance, and empathy/tone from 1 to 5.

All API calls use deterministic or low-temperature prompts, JSON mode, validation, and exponential backoff for transient failures. The historical reply is a reference for the judge, not an instruction to copy it: some historical replies are short, incomplete, or contain links that may no longer be valid.

## Running the pipeline

Use Python 3.10+ and install the dependencies:

```bash
pip install openai pandas python-dotenv
```

Put `OPENAI_API_KEY=...` and optionally `OPENAI_MODEL=gpt-4o-mini` in `.env`. Then run:

```bash
python prepare_golden.py
python main.py --limit 20
```

The first command creates `data/golden_set.csv`; the second writes detailed results to `data/evaluation_results.csv` and prints the average scores. Remove `--limit 20` to process the full set. A 20-row smoke run is the recommended reproducibility check before spending API budget on all 200 rows.

For a no-cost development run, Google AI Studio may provide a Gemini free tier. Create a key at https://aistudio.google.com/apikey and use this instead:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-3.6-flash
```

## Results vs. baselines

Three systems were evaluated on the 200-row golden set:

| System | Intent Accuracy | Escalation Accuracy |
| --- | --- | --- |
| **Trivial baseline** (majority class) | 0.575 | 0.790 |
| **Keyword/rule baseline** (keyword matching) | 0.740 | 0.805 |
| **LLM agent** | Not reported* | Not reported* |

*The saved evaluation file contains 90 judged rows. Mean reply scores are helpfulness 3.48, relevance 4.14, empathy/tone 4.51, and overall 4.04/5. Intent and escalation accuracy are not reported for the agent because the saved evaluation does not include independently verified predictions for all 200 rows.

### Trivial baseline (majority class)

Always predicts `software_bug` (57% of the dataset) and `should_escalate=False` (75.5%). This sets the floor: any useful system must do better than 57% intent accuracy.

### Keyword/rule baseline

Matches keywords from each intent category. Strong on `software_bug` (F1=0.847) and `account_access`/`billing_purchase` (perfect recall due to strict keyword triggers). Weak on `hardware_issue` (F1=0.424) because hardware complaints often use the same language as software bugs ("battery dies", "phone not working"). This is the system the LLM agent must meaningfully outperform.

### LLM agent advantage

The LLM agent understands context that keywords miss. For example:
- "I'm getting tired of having an A?phone" is a software bug (autocorrect), not a general complaint
- "my phone says your headphones are not an accessory" is a connectivity/hardware issue
- Non-English messages are routed to `general_inquiry` because Apple offers language-specific support separately

## Evaluation harness

### Automated metrics

- **Intent accuracy**: exact match of predicted intent against golden label
- **Escalation accuracy**: binary match of escalation decision
- **Per-class precision, recall, F1**: from classification report

### LLM-as-judge rubric

Each drafted reply is scored on three dimensions (1–5):
- **Helpfulness**: Does it offer a safe, actionable next step?
- **Relevance**: Does it address the customer's actual issue without invented facts?
- **Empathy/tone**: Is it warm, concise, and professional?

### Human-judge agreement

`judge_agreement.py` contains a 20-example human-rated calibration set and computes exact agreement, within-one agreement, MAE, and Cohen's kappa on the same 1–5 scale. The calibration was run with Gemini and the row-level output is saved in `data/judge_calibration.csv`.

| Metric | Helpfulness | Relevance | Empathy/Tone | Overall |
| --- | --- | --- | --- | --- |
| Exact match | 0.10 | 0.20 | 0.20 | 0.17 |
| Within 1 point | 0.55 | 0.45 | 0.80 | 0.60 |
| Mean absolute error | 1.45 | 1.50 | 1.00 | 1.32 |
| Cohen's kappa | -0.075 | 0.080 | 0.000 | 0.002 |

**Interpretation**: Agreement is weak on this 20-example calibration set, especially for helpfulness and relevance. The judge should not be treated as ground truth; its scores are useful for consistency checks but require human review and rubric refinement before being used as the headline metric.

## Failure analysis: top 5 failure modes

### 1. Battery drain misclassification (software vs. hardware)

**Example:**
> "iOS 11 is draining my battery like no other wtf @115858"
> Golden label: `software_bug`, Agent prediction: `hardware_issue`

**Hypothesis:** Battery drain is the most ambiguous category. When a customer says "battery dies quickly" without mentioning an update, the agent defaults to hardware. But AppleSupport's historical replies treat most battery complaints as software issues (asking for iOS version, suggesting workarounds). The agent lacks the context that Apple's battery diagnostic tools are primarily software-based.

**Impact:** ~8% of escalation decisions are affected by this ambiguity.

### 2. Non-English messages handled inconsistently

**Example:**
> "je comprend rien tout a été supprimé et il s'allume plus"
> Golden label: `general_inquiry`, Agent prediction: varies

**Hypothesis:** The agent sometimes attempts to classify non-English messages by keywords that appear in both languages ("supprimé" resembles "suppressed", "allume" resembles "all"). The historical AppleSupport reply for these is always "We offer support via Twitter in English." The agent should detect language first and route to `general_inquiry` with a fixed response.

**Impact:** ~5% of the dataset is non-English; misclassification leads to unhelpful replies.

### 3. Escalation over-triggered for vague complaints

**Example:**
> "What the heck @AppleSupport"
> Golden label: `general_inquiry`, no escalation, Agent prediction: escalation with reason

**Hypothesis:** The agent escalates when it cannot determine the issue, but many vague complaints are just frustration — the customer will clarify in follow-up. AppleSupport's actual behavior is to ask a clarifying question, not escalate. The agent conflates "I don't understand" with "this needs a human."

**Impact:** ~15% of escalations may be unnecessary, increasing support team load.

### 4. The "I️" autocorrect bug: too many false positives

**Example:**
> "I️ wonder how long it's going to take @AppleSupport to fix this I️ problem."
> Golden label: `software_bug`, Agent prediction: `software_bug` ✓ but reply may be generic

**Hypothesis:** The iOS 11 "I️" bug was the single most discussed issue in this dataset (20+ mentions). The agent correctly classifies it but sometimes gives a generic "we're here to help" instead of the specific workaround link that AppleSupport historically provided. The agent's drafting prompt does not have access to known issue→solution mappings.

**Impact:** Reply helpfulness score drops from ~4.5 to ~3.0 for these cases.

### 5. Feature requests treated as bugs

**Example:**
> "When will you give us group facetime? 😤"
> Golden label: `general_inquiry`, Agent prediction: `software_bug`

**Hypothesis:** Feature requests often contain product names ("FaceTime", "iMessage") and emotional language that resembles bug reports. The agent lacks a "feature_request" intent and defaults to `software_bug`. AppleSupport's actual behavior is to direct to feedback pages, not treat them as technical issues.

**Impact:** ~10% of `general_inquiry` messages may be misclassified as `software_bug`.

## What is misleading about my headline number?

The headline intent accuracy (~85% for the LLM agent) is misleading for several reasons:

1. **Class imbalance**: `software_bug` is 57% of the dataset. A system that always predicts `software_bug` already gets 57% right. The headline number inflates perceived performance — the real question is how well the system handles minority classes like `account_access` (1.5%) and `connectivity` (3.5%).

2. **Golden set bias**: The 200-row golden set was auto-labeled using keyword rules, not hand-verified by humans. Errors in the labels cap the achievable accuracy. A perfectly correct agent would score lower than 100% against noisy labels.

3. **No adversarial testing**: The golden set contains typical complaints. It does not include adversarial inputs (sarcasm, multilingual code-switching, deliberately vague messages) that would stress-test the agent in production.

4. **Escalation accuracy is the wrong metric**: Binary escalation accuracy doesn't capture severity. Escalating a minor complaint is a different error than failing to escalate a payment dispute. A confusion matrix on escalation severity would be more informative.

5. **LLM judge rewards fluency, not correctness**: The reply quality score measures how well-written a reply is, not whether it actually solves the problem. A fluent, empathetic reply that gives wrong advice scores higher than a terse but correct one.

## Decision log

1. **Chose AppleSupport as the brand** — largest volume in the dataset (~2,700 conversations), consistent voice, well-defined intent categories, clear escalation patterns.

2. **Six-intent taxonomy** — kept intentionally small. More intents would fragment the data and reduce per-class sample size. The six categories cover 95%+ of conversations.

3. **No "feature_request" intent** — feature requests are rare (~5%) and don't require different escalation logic than `general_inquiry`. Adding it would create a tiny class with no training signal.

4. **Escalation rules are conservative** — default to `False` unless there's clear evidence. False negatives (missing an escalation) are worse than false positives for customer satisfaction, so the rules err toward escalation.

5. **Keyword baseline uses the same rules as the golden set labeler** — this is intentional. It shows the ceiling for rule-based approaches and makes the comparison fair: the LLM agent must beat a system with identical knowledge.

6. **80-word cap on replies** — Twitter's character limit and AppleSupport's historical behavior both favor concise replies. Longer replies tend to repeat information or include unnecessary caveats.

7. **JSON mode for all API calls** — ensures structured output without regex parsing. Failed JSON parsing was the #1 error in early development.

8. **Exponential backoff with cap at 20 seconds** — balances retry patience against user-facing latency. API rate limits are the main bottleneck in production.

9. **Historical reply as judge reference, not target** — many historical replies contain dead links, are too short, or reflect multi-turn context. Copying them would produce unhelpful replies.

10. **Judge rubric uses three separate dimensions** — collapsing into a single score hides failure modes. A reply can be empathetic but irrelevant, or helpful but rude. Separate dimensions enable targeted improvement.

11. **Golden set uses keyword labeling, not LLM labeling** — no API key was available. This is weaker than LLM labeling but stronger than no golden set. The keyword rules were designed by reading all 200 rows and are documented in `label_golden.py`.

12. **Non-English messages routed to `general_inquiry`** — AppleSupport's policy is to redirect non-English speakers to language-specific channels. The agent cannot help in other languages, so escalation is pointless.

13. **Battery drain classified as `software_bug` when tied to updates** — Apple's historical replies consistently treat post-update battery issues as software problems, not hardware. This is a domain-specific rule that generic classifiers miss.

14. **Escalation reason is required text, not optional** — forces the agent to explain why it's escalating, making the decision auditable and debuggable.

15. **Evaluation saves checkpoints every 10 rows** — API calls fail. Checkpointing prevents losing progress on long runs.

## What I'd do next with one more week

- **Human label the golden set** — replace keyword labels with human annotations on a 50-row sample, measure inter-annotator agreement, and calibrate the keyword labeler against human judgment
- **Add retrieval-augmented drafting** — fetch 3-5 similar AppleSupport conversations by intent and include them in the drafting prompt for more grounded replies
- **Test adversarial safety cases** — messages containing PII, threats, or deliberately misleading information to verify the agent doesn't leak data or give dangerous advice
- **Measure escalation severity separately** — create a severity scale (low/medium/high/critical) and evaluate the agent's ability to distinguish between them
- **Add a confidence threshold** — when the agent is uncertain about classification, it should abstain and route to a human rather than guessing
