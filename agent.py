"""LLM-powered AppleSupport classification and response drafting."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


INTENTS = (
	"software_bug",
	"hardware_issue",
	"account_access",
	"billing_purchase",
	"connectivity",
	"general_inquiry",
)


def parse_bool(value):
	if isinstance(value, bool):
		return value
	return str(value).strip().lower() in {"true", "1", "yes"}


class SupportAgent:
	"""Classify a message and draft a cautious, AppleSupport-style reply."""

	def __init__(self, client=None, model=None, request_delay=10.0):
		load_dotenv()
		provider = os.getenv("LLM_PROVIDER", "openai").lower()
		if client:
			self.client = client
		elif provider == "groq":
			self.client = OpenAI(
				api_key=os.getenv("GROQ_API_KEY"),
				base_url="https://api.groq.com/openai/v1",
			)
		elif provider == "gemini":
			self.client = OpenAI(
				api_key=os.getenv("GEMINI_API_KEY"),
				base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
			)
		else:
			self.client = OpenAI()
		if provider == "groq":
			self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
		elif provider == "gemini":
			self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
		else:
			self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		self.request_delay = request_delay

	def _complete_json(self, messages, retries=8):
		for attempt in range(retries):
			try:
				response = self.client.chat.completions.create(
					model=self.model,
					temperature=0.2,
					response_format={"type": "json_object"},
					messages=messages,
				)
				time.sleep(self.request_delay)
				return json.loads(response.choices[0].message.content)
			except Exception as e:
				wait = min(2 ** (attempt + 2), 60)
				print(f"  Agent retry {attempt+1}/{retries}, waiting {wait}s: {type(e).__name__}")
				time.sleep(wait)
				if attempt == retries - 1:
					raise

	def classify(self, customer_message):
		result = self._complete_json([
			{"role": "system", "content": (
				"You classify AppleSupport requests. Choose one intent exactly: "
				f"{', '.join(INTENTS)}. Escalate for physical repair, account-specific "
				"investigation, payment disputes, safety, or when self-service is unsafe. "
				"Return JSON: intent, should_escalate (boolean), escalation_reason."
			)},
			{"role": "user", "content": customer_message},
		])
		intent = str(result.get("intent", "general_inquiry")).lower().strip()
		if intent not in INTENTS:
			intent = "general_inquiry"
		return {
			"intent": intent,
			"should_escalate": parse_bool(result.get("should_escalate", False)),
			"escalation_reason": str(result.get("escalation_reason", "")),
		}

	def draft_reply(self, customer_message, classification):
		result = self._complete_json([
			{"role": "system", "content": (
				"You are AppleSupport on Twitter. Draft one concise, warm, professional "
				"reply in English. Acknowledge the issue, give only safe actionable steps, "
				"do not invent account details or promise outcomes, and invite DM/support "
				"when account-specific help is needed. Keep it under 80 words. Return JSON "
				"with one key: reply."
			)},
			{"role": "user", "content": json.dumps({
				"customer_message": customer_message,
				"assigned_intent": classification["intent"],
				"should_escalate": classification["should_escalate"],
				"escalation_reason": classification["escalation_reason"],
			})},
		])
		return str(result.get("reply", "We'd like to help. Please send us a DM so we can take a closer look."))

	def process(self, customer_message):
		classification = self.classify(customer_message)
		return {**classification, "drafted_reply": self.draft_reply(customer_message, classification)}
