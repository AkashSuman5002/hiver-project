# Decision Log

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
