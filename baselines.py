"""Baselines and evaluation comparison for the AppleSupport agent."""

import pandas as pd
from collections import Counter


INTENTS = (
    "software_bug",
    "hardware_issue",
    "account_access",
    "billing_purchase",
    "connectivity",
    "general_inquiry",
)


def accuracy_score(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def classification_report_str(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    lines = []
    lines.append(f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("")
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        lines.append(f"{label:>20} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")
    lines.append("")
    micro_acc = accuracy_score(y_true, y_pred)
    lines.append(f"{'accuracy':>20} {'':>10} {'':>10} {micro_acc:>10.3f} {len(y_true):>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline 1: Trivial — always predict the majority class
# ---------------------------------------------------------------------------
class TrivialBaseline:
    """Always predicts the most frequent intent from training data."""

    def __init__(self, majority_intent="software_bug", majority_escalation=False):
        self.majority_intent = majority_intent
        self.majority_escalation = majority_escalation

    def predict(self, _message):
        return {
            "intent": self.majority_intent,
            "should_escalate": self.majority_escalation,
            "escalation_reason": "",
        }

    @classmethod
    def from_data(cls, df):
        majority_intent = df["intent"].mode().iloc[0] if len(df) else "software_bug"
        majority_escalation = df["should_escalate"].mode().iloc[0] if len(df) else False
        return cls(majority_intent=majority_intent, majority_escalation=majority_escalation)


# ---------------------------------------------------------------------------
# Baseline 2: Simple keyword/rule-based classifier
# ---------------------------------------------------------------------------
class KeywordBaseline:
    """Classifies using keyword matching — the same logic as label_golden.py
    but applied at inference time without seeing the ground truth."""

    RULES = {
        "account_access": [
            "apple id", "apple account", "login", "sign in", "password",
            "forgot", "locked", "2fa", "two factor", "verification code",
            "activation", "security question", "can't login", "unidays",
        ],
        "billing_purchase": [
            "refund", "charge", "charged", "billing", "purchase",
            "subscription", "cancel", "money", "price", "cost",
            "overcharged", "extra fee", "gift balance", "itunes",
            "app store", "buy", "bought", "ordered", "delivery",
        ],
        "hardware_issue": [
            "battery replacement", "battery covered", "apple care",
            "service centre", "technician", "repair", "broken",
            "cracked screen", "dropped", "overheating", "hot",
            "extremely hot", "speaker", "microphone", "camera",
            "earphones", "headphones", "airpods",
        ],
        "connectivity": [
            "wifi", "wi-fi", "bluetooth", "cellular", "network",
            "signal", "connection", "connected", "internet",
            "calls", "call", "text", "texts", "message", "messages",
            "receiving", "sending", "carplay", "pair", "pairing",
        ],
        "software_bug": [
            "update", "ios 11", "ios11", "crash", "freeze", "freezing",
            "glitch", "bug", "slow", "lag", "lagging", "hang",
            "shut off", "turns off", "black screen", "error",
            "i️", "autocorrect", "keyboard", "weird character",
            "battery drain", "battery dies", "draining", "power drain",
            "shutdown", "randomly restart", "won't install", "can't update",
            "apps crashing", "app freezes", "won't load", "won't open",
            "unresponsive", "safari", "mail app", "facetime",
            "voicemail", "calendar", "photos", "icloud", "keychain",
        ],
    }

    def predict(self, message):
        m = message.lower()
        scores = {}
        for intent, keywords in self.RULES.items():
            scores[intent] = sum(1 for k in keywords if k in m)

        # Boost for "I️" bug
        if "i️" in m or "the i" in m or "letter i" in m:
            scores["software_bug"] += 5

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = "general_inquiry"

        should_esc = best in ("account_access", "billing_purchase")
        reason = ""
        if best == "account_access":
            reason = "Account-specific investigation required"
        elif best == "billing_purchase":
            reason = "Payment dispute or refund request needs human review"

        return {
            "intent": best,
            "should_escalate": should_esc,
            "escalation_reason": reason,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_baselinepredictions(df, predict_fn, label=""):
    """Evaluate a baseline against the golden set."""
    y_true_intent = []
    y_pred_intent = []
    y_true_esc = []
    y_pred_esc = []

    for _, row in df.iterrows():
        msg = str(row["customer_message"])
        pred = predict_fn(msg)
        y_true_intent.append(str(row["intent"]))
        y_pred_intent.append(pred["intent"])
        y_true_esc.append(bool(row["should_escalate"]))
        y_pred_esc.append(pred["should_escalate"])

    intent_acc = accuracy_score(y_true_intent, y_pred_intent)
    esc_acc = accuracy_score(y_true_esc, y_pred_esc)
    report = classification_report_str(y_true_intent, y_pred_intent)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Intent accuracy:    {intent_acc:.3f}")
    print(f"Escalation accuracy: {esc_acc:.3f}")
    print(f"\nIntent classification report:")
    print(report)

    return {"intent_acc": intent_acc, "esc_acc": esc_acc, "report": report}


def main():
    df = pd.read_csv("data/golden_set.csv").fillna("")
    print(f"Golden set: {len(df)} rows")

    # Trivial baseline
    trivial = TrivialBaseline.from_data(df)
    evaluate_baselinepredictions(df, trivial.predict, label="Baseline 1: Trivial (majority class)")

    # Keyword baseline
    keyword = KeywordBaseline()
    evaluate_baselinepredictions(df, keyword.predict, label="Baseline 2: Keyword/Rules")


if __name__ == "__main__":
    main()
