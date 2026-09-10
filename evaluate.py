"""LLM-as-a-judge evaluation for AppleSupport replies."""

import json
import os
import time

from dotenv import load_dotenv


RUBRIC = """Score the candidate reply against the customer message and historical reply.
Give integer scores from 1 (poor) to 5 (excellent) for:
- helpfulness: useful, safe next step and appropriate escalation
- relevance: directly addresses the customer's issue without invented facts
- empathy_tone: warm, concise, professional AppleSupport-like tone
Historical replies are evidence of style, not a requirement to copy their wording.
Return only JSON: helpfulness, relevance, empathy_tone, rationale."""


def judge_reply(client, model, customer_message, historical_reply, candidate_reply, retries=5):
	load_dotenv()
	for attempt in range(retries):
		try:
			response = client.chat.completions.create(
				model=model,
				temperature=0,
				response_format={"type": "json_object"},
				messages=[
					{"role": "system", "content": RUBRIC},
					{"role": "user", "content": json.dumps({
						"customer_message": customer_message,
						"historical_reply": historical_reply,
						"candidate_reply": candidate_reply,
					})},
				],
			)
			result = json.loads(response.choices[0].message.content)
			scores = {key: max(1, min(5, int(result[key]))) for key in ("helpfulness", "relevance", "empathy_tone")}
			scores["rationale"] = str(result.get("rationale", ""))
			time.sleep(3.0)
			return scores
		except Exception:
			if attempt == retries - 1:
				raise
			time.sleep(min(2**attempt, 20))


def summarize(scores):
	if not scores:
		return {key: 0.0 for key in ("helpfulness", "relevance", "empathy_tone", "overall")}
	summary = {key: sum(item[key] for item in scores) / len(scores) for key in ("helpfulness", "relevance", "empathy_tone")}
	summary["overall"] = sum(summary.values()) / 3
	return summary
