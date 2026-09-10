"""Create or auto-label the AppleSupport golden set."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


INTENTS = "software_bug, hardware_issue, account_access, billing_purchase, connectivity, general_inquiry"
SYSTEM_PROMPT = f"""You label Apple customer-support messages for an evaluation set.
Choose exactly one intent from: {INTENTS}.
Set should_escalate to true when a human, account-specific investigation, payment dispute,
or physical repair is needed. Set it to false when a safe self-service answer is enough.
Return only valid JSON with keys intent, should_escalate, escalation_reason.
The escalation_reason must be short and specific."""


def create_client():
	provider = os.getenv("LLM_PROVIDER", "openai").lower()
	if provider == "groq":
		key = os.getenv("GROQ_API_KEY", "").strip()
		if not key:
			raise RuntimeError("Set GROQ_API_KEY in .env for the free Groq provider.")
		return OpenAI(
			api_key=key,
			base_url="https://api.groq.com/openai/v1",
		)
	if provider == "gemini":
		key = os.getenv("GEMINI_API_KEY", "").strip()
		if not key:
			raise RuntimeError("Set GEMINI_API_KEY in .env for the free Gemini provider.")
		return OpenAI(
			api_key=key,
			base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
		)
	key = os.getenv("OPENAI_API_KEY", "").strip()
	if not key or key.lower() in {"your_openai_key_here", "your_api_key_here"}:
		raise RuntimeError("Set OPENAI_API_KEY in .env or use LLM_PROVIDER=groq.")
	return OpenAI(api_key=key)


def parse_bool(value):
	if isinstance(value, bool):
		return value
	return str(value).strip().lower() in {"true", "1", "yes"}


def label_row(client, model, message, max_retries=5):
	for attempt in range(max_retries):
		try:
			response = client.chat.completions.create(
				model=model,
				temperature=0,
				response_format={"type": "json_object"},
				messages=[
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user", "content": f"Customer message:\n{message}"},
				],
			)
			result = json.loads(response.choices[0].message.content)
			intent = str(result["intent"]).strip().lower()
			if intent not in INTENTS.split(", "):
				raise ValueError(f"Unexpected intent: {intent}")
			return {
				"intent": intent,
				"should_escalate": parse_bool(result["should_escalate"]),
				"escalation_reason": str(result["escalation_reason"]).strip(),
			}
		except Exception as error:
			if "insufficient_quota" in str(error) or "credit_balance_exhausted" in str(error):
				raise RuntimeError(
					"OpenAI API quota is exhausted. Add billing credits or use a key "
					"from a project with available quota."
				) from error
			if "PERMISSION_DENIED" in str(error) or "denied access" in str(error).lower():
				raise RuntimeError(
					"Gemini denied this project. Create a new Gemini API key in a project "
					"with Gemini API access, or contact Google AI support."
				) from error
			if attempt == max_retries - 1:
				raise
			time.sleep(min(2**attempt, 20))


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="data/golden_set_unlabeled.csv")
	parser.add_argument("--output", default="data/golden_set.csv")
	parser.add_argument("--model", default=None)
	parser.add_argument("--limit", type=int, default=None)
	args = parser.parse_args()

	load_dotenv()
	provider = os.getenv("LLM_PROVIDER", "openai").lower()
	if provider == "groq":
		model = args.model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
	elif provider == "gemini":
		model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
	else:
		model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
	client = create_client()
	frame = pd.read_csv(args.input).fillna("")
	required = {"customer_message", "intent", "should_escalate", "escalation_reason"}
	missing = required.difference(frame.columns)
	if missing:
		raise ValueError(f"Missing columns: {sorted(missing)}")

	limit = min(args.limit, len(frame)) if args.limit else len(frame)
	for index in range(limit):
		if str(frame.at[index, "intent"]).strip() and str(frame.at[index, "escalation_reason"]).strip():
			continue
		result = label_row(client, model, frame.at[index, "customer_message"])
		for key, value in result.items():
			frame.at[index, key] = value
		if (index + 1) % 10 == 0:
			frame.to_csv(args.output, index=False)
			print(f"Labeled {index + 1}/{limit}")
		time.sleep(0.15)

	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(args.output, index=False)
	print(f"Wrote {args.output} ({limit} rows inspected).")


if __name__ == "__main__":
	main()