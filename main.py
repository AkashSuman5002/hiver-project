"""Run the AppleSupport agent and LLM judge over a labeled CSV."""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from agent import SupportAgent
from evaluate import judge_reply, summarize


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input", default="data/golden_set.csv")
	parser.add_argument("--output", default="data/evaluation_results.csv")
	parser.add_argument("--limit", type=int, default=None)
	args = parser.parse_args()

	load_dotenv()
	provider = os.getenv("LLM_PROVIDER", "openai").lower()
	if provider == "groq":
		if not os.getenv("GROQ_API_KEY"):
			raise RuntimeError("Set GROQ_API_KEY in .env before running the pipeline.")
		client = OpenAI(
			api_key=os.getenv("GROQ_API_KEY"),
			base_url="https://api.groq.com/openai/v1",
		)
	elif provider == "gemini":
		if not os.getenv("GEMINI_API_KEY"):
			raise RuntimeError("Set GEMINI_API_KEY in .env before running the pipeline.")
		client = OpenAI(
			api_key=os.getenv("GEMINI_API_KEY"),
			base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
		)
	else:
		if not os.getenv("OPENAI_API_KEY"):
			raise RuntimeError("Set OPENAI_API_KEY in .env before running the pipeline.")
		client = OpenAI()
	frame = pd.read_csv(args.input).fillna("")
	required = {"customer_message", "brand_reply"}
	missing = required.difference(frame.columns)
	if missing:
		raise ValueError(f"Missing columns: {sorted(missing)}")
	if args.limit:
		frame = frame.head(args.limit).copy()

	# Resume from existing results if present
	existing_ids = set()
	existing_records = []
	if Path(args.output).exists():
		existing = pd.read_csv(args.output).fillna("")
		existing_ids = set(existing["inquiry_id"].tolist())
		existing_records = existing.to_dict("records")
		print(f"Resuming: {len(existing_ids)} rows already done")

	agent = SupportAgent(client=client)
	records = list(existing_records)
	for row_number, row in frame.iterrows():
		if row.get("inquiry_id", row_number) in existing_ids:
			continue
		result = agent.process(row["customer_message"])
		scores = judge_reply(client, agent.model, row["customer_message"], row["brand_reply"], result["drafted_reply"])
		records.append({
			"inquiry_id": row.get("inquiry_id", row_number),
			"customer_message": row["customer_message"],
			"brand_reply": row["brand_reply"],
			**result,
			**scores,
		})
		if (len(records) % 10) == 0:
			pd.DataFrame(records).to_csv(args.output, index=False)
			print(f"Processed {len(records)}/{len(frame)}")

	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(records).to_csv(args.output, index=False)
	summary = summarize(records)
	print("\nEvaluation summary (1-5):")
	print(pd.DataFrame([summary]).round(2).to_string(index=False))
	print(f"\nDetailed results: {args.output}")


if __name__ == "__main__":
	main()
