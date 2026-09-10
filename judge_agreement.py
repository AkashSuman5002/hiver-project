"""Human-judge agreement calibration for the LLM-as-judge rubric.

This module provides:
1. A small hand-rated calibration set (20 examples scored by a human)
2. Functions to measure agreement between human and LLM judge scores
3. Cohen's kappa and percentage agreement metrics
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# Human-rated calibration set: (customer_message, candidate_reply, human_scores)
# Scores are dict with keys: helpfulness, relevance, empathy_tone (1-5 scale)
CALIBRATION_SET = [
    {
        "customer_message": "iOS 11 is draining my battery like no other wtf @115858",
        "candidate_reply": "We'd be happy to look this over with you. Please meet us in DM so we may gather information, and better assist.",
        "human_scores": {"helpfulness": 3, "relevance": 4, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport what the heck",
        "candidate_reply": "Understood. We have released a newer iOS version which will help with the issue you're experiencing. Let's make sure you have a recent backup, then update to 11.1.1.",
        "human_scores": {"helpfulness": 2, "relevance": 2, "empathy_tone": 3},
    },
    {
        "customer_message": "Hey @AppleSupport just got a new Apple TV. It's struggling to complete the os update. Assistance?",
        "candidate_reply": "We have an excellent team to help out with your Apple TV. Here's the contact link.",
        "human_scores": {"helpfulness": 3, "relevance": 4, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport Bought iPhone 6s but battery is not working properly.kindly help.",
        "candidate_reply": "We'd like to look into any issues you're having. Please DM us more about what's occurred with your battery.",
        "human_scores": {"helpfulness": 4, "relevance": 4, "empathy_tone": 4},
    },
    {
        "customer_message": "@AppleSupport I am under 'all inboxes' when it occurs. I've tried quitting the app & powering down multiple times and that doesn't work either.",
        "candidate_reply": "Can you let us know which version of iOS 11 you're running? You can check by tapping on Settings > General > About.",
        "human_scores": {"helpfulness": 4, "relevance": 4, "empathy_tone": 3},
    },
    {
        "customer_message": "Trying to set up 2FA on laptop and keep getting this error, what gives @AppleSupport?",
        "candidate_reply": "We are here to help. Have you requested more than one code?",
        "human_scores": {"helpfulness": 3, "relevance": 3, "empathy_tone": 3},
    },
    {
        "customer_message": "Okay @115858 fix your shit this 'I' thing is getting annoying",
        "candidate_reply": "An update has been released to assist with this issue. If you haven't yet, please back up your device and update it to the latest version iOS 11.1.1.",
        "human_scores": {"helpfulness": 5, "relevance": 5, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport Thank you so much",
        "candidate_reply": "You're most welcome.",
        "human_scores": {"helpfulness": 5, "relevance": 5, "empathy_tone": 5},
    },
    {
        "customer_message": "@115858 I'm getting tired of having an A?phone. Can I have my iPhone back now? Thanks :)",
        "candidate_reply": "Here's what you can do to work around the issue until it's fixed in a future software update.",
        "human_scores": {"helpfulness": 3, "relevance": 3, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport the Arabic font in ios11 is not user friendly, please go back to previous font.",
        "candidate_reply": "We can help get you to the right place to make your suggestion. Provide your feedback here.",
        "human_scores": {"helpfulness": 4, "relevance": 4, "empathy_tone": 4},
    },
    {
        "customer_message": "@AppleSupport is it possible to transfer purchases from 1 Apple is to another?",
        "candidate_reply": "Content is linked to the purchasing Apple ID, however you can look into Family Sharing here.",
        "human_scores": {"helpfulness": 5, "relevance": 5, "empathy_tone": 4},
    },
    {
        "customer_message": "@AppleSupport I can't click here to work it.",
        "candidate_reply": "Go to Settings > your name > iCloud > iCloud Backup and see if you're able turn it on there. Also, what version of iOS are you currently running?",
        "human_scores": {"helpfulness": 4, "relevance": 3, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport I-pad air 1",
        "candidate_reply": "We've received your DM and will continue to work with you there.",
        "human_scores": {"helpfulness": 3, "relevance": 3, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport can y'all fix this I",
        "candidate_reply": "Here's what you can do to work around the issue until it's fixed in a future software update.",
        "human_scores": {"helpfulness": 4, "relevance": 4, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport just updated ios11.0.2 on my iPhone 7. Nothing works, phone hangs, stops working, shuts off, app freezes.",
        "candidate_reply": "We'd like to help. Let's meet up in DM to investigate.",
        "human_scores": {"helpfulness": 3, "relevance": 4, "empathy_tone": 3},
    },
    {
        "customer_message": "@AppleSupport why does my phone keep turning my I's into weird characters?",
        "candidate_reply": "Let's take a closer look into this together. Reach out to us in a DM to get started.",
        "human_scores": {"helpfulness": 3, "relevance": 4, "empathy_tone": 4},
    },
    {
        "customer_message": "iOS update made my phone freeze many times. You should fix that.",
        "candidate_reply": "We'd love to look into this with you! To start, can you DM us what iOS version and model iPhone we're working with?",
        "human_scores": {"helpfulness": 4, "relevance": 4, "empathy_tone": 4},
    },
    {
        "customer_message": "@AppleSupport iPhone 8 Plus, sorry if I replied so late",
        "candidate_reply": "No problem. Let's continue in DM.",
        "human_scores": {"helpfulness": 3, "relevance": 3, "empathy_tone": 4},
    },
    {
        "customer_message": "@AppleSupport my iPhone 7plus won't let me install the latest update!",
        "candidate_reply": "Try out the steps in this article. If this doesn't resolve it, let us know via DM.",
        "human_scores": {"helpfulness": 5, "relevance": 5, "empathy_tone": 4},
    },
    {
        "customer_message": "@AppleSupport I love you guys but i swear to god you need to fix your mail app ASAP - it doesn't let me mark an email as read",
        "candidate_reply": "We'd like to see how we can help. To start, which device are you experiencing this on, and which software version is it running?",
        "human_scores": {"helpfulness": 4, "relevance": 4, "empathy_tone": 4},
    },
]


def llm_judge_scores(client, model, customer_message, candidate_reply, retries=5):
    """Score a reply using the LLM judge."""
    rubric = """Score the candidate reply against the customer message.
Give integer scores from 1 (poor) to 5 (excellent) for:
- helpfulness: useful, safe next step and appropriate escalation
- relevance: directly addresses the customer's issue without invented facts
- empathy_tone: warm, concise, professional support-like tone
Return only JSON: helpfulness, relevance, empathy_tone, rationale."""

    import time
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": json.dumps({
                        "customer_message": customer_message,
                        "candidate_reply": candidate_reply,
                    })},
                ],
            )
            result = json.loads(response.choices[0].message.content)
            scores = {key: max(1, min(5, int(result[key]))) for key in ("helpfulness", "relevance", "empathy_tone")}
            time.sleep(0.15)
            return scores
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2**attempt, 20))


def compute_agreement(human_scores_list, llm_scores_list):
    """Compute agreement metrics between human and LLM judges."""
    metrics = {}
    for key in ("helpfulness", "relevance", "empathy_tone"):
        human = [s[key] for s in human_scores_list]
        llm = [s[key] for s in llm_scores_list]

        # Percentage agreement (exact match)
        exact_match = sum(1 for h, l in zip(human, llm) if h == l)
        metrics[f"{key}_exact"] = exact_match / len(human)

        # Within-1 agreement (off by at most 1 point)
        within_1 = sum(1 for h, l in zip(human, llm) if abs(h - l) <= 1)
        metrics[f"{key}_within1"] = within_1 / len(human)

        # Mean absolute error
        mae = sum(abs(h - l) for h, l in zip(human, llm)) / len(human)
        metrics[f"{key}_mae"] = mae

        # Cohen's Kappa (ordinal)
        kappa = compute_kappa(human, llm)
        metrics[f"{key}_kappa"] = kappa

    # Overall metrics
    all_human = []
    all_llm = []
    for h, l in zip(human_scores_list, llm_scores_list):
        all_human.extend([h["helpfulness"], h["relevance"], h["empathy_tone"]])
        all_llm.extend([l["helpfulness"], l["relevance"], l["empathy_tone"]])

    metrics["overall_exact"] = sum(1 for h, l in zip(all_human, all_llm) if h == l) / len(all_human)
    metrics["overall_within1"] = sum(1 for h, l in zip(all_human, all_llm) if abs(h - l) <= 1) / len(all_human)
    metrics["overall_mae"] = sum(abs(h - l) for h, l in zip(all_human, all_llm)) / len(all_human)

    return metrics


def compute_kappa(rater1, rater2):
    """Compute Cohen's Kappa for two raters."""
    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))

    # Build confusion matrix
    cm = {}
    for c1 in categories:
        for c2 in categories:
            cm[(c1, c2)] = 0
    for a, b in zip(rater1, rater2):
        cm[(a, b)] += 1

    # Observed agreement
    po = sum(cm[(c, c)] for c in categories) / n

    # Expected agreement
    pe = 0
    for c in categories:
        row_sum = sum(cm[(c, c2)] for c2 in categories)
        col_sum = sum(cm[(c1, c)] for c1 in categories)
        pe += (row_sum / n) * (col_sum / n)

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "groq":
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif provider == "gemini":
        client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    else:
        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print("Running LLM judge on calibration set...")
    llm_scores = []
    for i, item in enumerate(CALIBRATION_SET):
        scores = llm_judge_scores(client, model, item["customer_message"], item["candidate_reply"])
        llm_scores.append(scores)
        if (i + 1) % 5 == 0:
            print(f"  Scored {i + 1}/{len(CALIBRATION_SET)}")

    human_scores = [item["human_scores"] for item in CALIBRATION_SET]

    # Save calibration data
    calibration_data = []
    for item, llm in zip(CALIBRATION_SET, llm_scores):
        calibration_data.append({
            "customer_message": item["customer_message"],
            "candidate_reply": item["candidate_reply"],
            "human_helpfulness": item["human_scores"]["helpfulness"],
            "human_relevance": item["human_scores"]["relevance"],
            "human_empathy_tone": item["human_scores"]["empathy_tone"],
            "llm_helpfulness": llm["helpfulness"],
            "llm_relevance": llm["relevance"],
            "llm_empathy_tone": llm["empathy_tone"],
        })
    pd.DataFrame(calibration_data).to_csv("data/judge_calibration.csv", index=False)

    # Compute agreement
    metrics = compute_agreement(human_scores, llm_scores)

    print("\n" + "=" * 60)
    print("  Human-Judge Agreement Report")
    print("=" * 60)
    for key, val in sorted(metrics.items()):
        if "kappa" in key:
            print(f"  {key:>30}: {val:.3f}")
        elif "mae" in key:
            print(f"  {key:>30}: {val:.3f} points")
        else:
            print(f"  {key:>30}: {val:.1%}")

    # Interpret kappa
    kappa_vals = [v for k, v in metrics.items() if "kappa" in k]
    avg_kappa = sum(kappa_vals) / len(kappa_vals) if kappa_vals else 0
    if avg_kappa >= 0.8:
        strength = "strong"
    elif avg_kappa >= 0.6:
        strength = "substantial"
    elif avg_kappa >= 0.4:
        strength = "moderate"
    else:
        strength = "fair/poor"
    print(f"\n  Average kappa: {avg_kappa:.3f} ({strength} agreement)")

    return metrics


if __name__ == "__main__":
    main()
