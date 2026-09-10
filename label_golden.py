"""Hand-crafted keyword-based labeling for the AppleSupport golden set."""

import pandas as pd

INTENTS = (
    "software_bug",
    "hardware_issue",
    "account_access",
    "billing_purchase",
    "connectivity",
    "general_inquiry",
)


def label_intent(msg):
    m = msg.lower()

    # software_bug: iOS update issues, crashes, freezing, glitches, bugs, slow after update
    sw_keywords = [
        "update", "ios 11", "ios11", "crash", "freeze", "freezing", "frozen",
        "glitch", "bug", "slow", "lag", "lagging", "hang", "restart",
        "shut off", "turns off", "turn off", "black screen", "blank screen",
        "won't turn on", "won't start", "bricked", "error", "malfunction",
        "i️", "autocorrect", "keyboard", "weird character", "weird symbol",
        "type the i", "letter i", "the i problem", "software", "os update",
        "high sierra", "sierra", "macos", "mac os", "safari", "mail app",
        "apps crashing", "apps won't load", "app freezes", "app crashes",
        "won't load", "won't open", "doesn't work", "not working",
        "unresponsive", "unresponsive", "non responsive",
        "battery drain", "battery dies", "battery draining", "battery life",
        "draining fast", "dies quickly", "drains", "power drain",
        "shutdown", "unexpected shutdown", "randomly restart",
        "won't scroll", "can't hang up", "won't install", "can't update",
        "unable to verify", "won't pair", "sync", "keyboard issue",
        "text replacement", "emoji", "faces", "boxes", "missing content",
        "restrictions", "downgrading", "revert", "imessage", "facetime",
        "voicemail", "calendar", "photos", "icloud", "keychain", "airdrop",
        "notifications", "brightness", "touch", "screen", "display",
    ]

    # hardware_issue: physical device problems, battery hardware, screen damage
    hw_keywords = [
        "battery replacement", "battery covered", "apple care",
        "service centre", "technician", "repair", "fix this phone",
        "broken", "cracked screen", "dropped", "won't charge",
        "charging", "overheating", "hot", "extremely hot",
        "won't turn on", "dead", "water damage",
        "speaker", "microphone", "camera", "button",
        "earphones", "headphones", "airpods", "apple watch",
        "apple tv", "ipad", "iphone 6", "iphone 7", "iphone 8",
        "battery percentage", "battery drains", "battery life",
    ]

    # account_access: Apple ID, login, password, 2FA, locked account
    acct_keywords = [
        "apple id", "apple account", "login", "log in", "sign in",
        "password", "forgot", "remember", "locked", "locked out",
        "2fa", "two factor", "two-factor", "verification code",
        "activation", "locked for security", "security question",
        "dob", "date of birth", "can't login", "cannot login",
        "can't access", "cannot access", "unidays",
    ]

    # billing_purchase: refunds, charges, subscriptions, purchases
    bill_keywords = [
        "refund", "charge", "charged", "billing", "purchase",
        "subscription", "cancel", "money", "price", "cost",
        "overcharged", "extra fee", "gift balance", "itunes",
        "app store", "buy", "bought", "ordered", "order",
        "delivery", "upgrade program",
    ]

    # connectivity: WiFi, Bluetooth, cellular, network, calls, messages
    conn_keywords = [
        "wifi", "wi-fi", "bluetooth", "cellular", "network",
        "signal", "connection", "connected", "internet",
        "calls", "call", "text", "texts", "message", "messages",
        "receiving", "sending", "won't call", "can't call",
        "dropped call", "carplay", "pair", "pairing",
        "airdrop", "hotspot", "data",
    ]

    # Count matches for each category
    scores = {
        "software_bug": sum(1 for k in sw_keywords if k in m),
        "hardware_issue": sum(1 for k in hw_keywords if k in m),
        "account_access": sum(1 for k in acct_keywords if k in m),
        "billing_purchase": sum(1 for k in bill_keywords if k in m),
        "connectivity": sum(1 for k in conn_keywords if k in m),
    }

    # Special cases - the "I️" bug is a known software bug
    if "i️" in m or "the i" in m or "letter i" in m or "type the i" in m:
        scores["software_bug"] += 5

    # Battery drain is extremely common - classify as software bug if tied to update
    if any(w in m for w in ["battery drain", "battery dies", "draining", "battery life"]):
        if any(w in m for w in ["update", "ios 11", "ios11", "after update", "since update"]):
            scores["software_bug"] += 3
        else:
            scores["hardware_issue"] += 2

    # Non-English messages → general_inquiry (Apple can't help in other languages)
    non_english = ["je comprend", "je rajoute", "sérieusement", "bir aydir",
                   "acabo de intentar", "teclado", "greces", "질문", "문제"]
    if any(w in m for w in non_english):
        return "general_inquiry"

    # Feature requests → general_inquiry
    feature_kw = ["feature request", "suggestion", "how about", "would be nice",
                   "would love", "can we get", "please add", "wish"]
    if any(w in m for w in feature_kw):
        return "general_inquiry"

    # Pure complaints without technical issue → general_inquiry
    complaint_only = ["what the heck", "what the fuck", "seriously", "come on",
                       "fix your", "ridiculous", "unacceptable", "shame"]
    has_technical = any(w in m for w in sw_keywords + hw_keywords + acct_keywords + bill_keywords + conn_keywords)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        if any(w in m for w in complaint_only) and not has_technical:
            return "general_inquiry"
        return "general_inquiry"

    return best


def label_escalation(msg, intent):
    m = msg.lower()

    # Always escalate for: account access, billing, physical repair needs, safety
    if intent == "account_access":
        return True, "Account-specific investigation required"
    if intent == "billing_purchase":
        return True, "Payment dispute or refund request needs human review"

    # Escalate if message mentions physical repair, store visit, technician
    if any(w in m for w in ["repair", "technician", "genius bar", "store",
                              "service centre", "out of service", "replaced"]):
        return True, "Physical repair or in-store service needed"

    # Escalate if battery issue might be hardware (not tied to update)
    if any(w in m for w in ["battery", "won't charge", "overheating", "hot"]):
        if not any(w in m for w in ["update", "ios 11", "ios11"]):
            return True, "Possible hardware battery issue needs diagnosis"

    # Escalate if screen/display not working
    if any(w in m for w in ["screen not working", "display", "won't turn on",
                              "dead", "bricked", "black screen"]):
        return True, "Device may need physical inspection or replacement"

    # Escalate if device won't turn on at all
    if any(w in m for w in ["won't turn on", "won't start", "bricked", "dead"]):
        return True, "Device unresponsive, may need hardware diagnosis"

    # Escalate multi-device or persistent issues
    if any(w in m for w in ["multiple devices", "months", "3 months", "over 15 days",
                              "24 days", "persistent"]):
        return True, "Persistent or multi-device issue needs deeper investigation"

    # Escalate if threatening to switch or legal action
    if any(w in m for w in ["suing", "lawyer", "legal", "android", "note8",
                              "switch to", "never look back"]):
        return True, "Customer at risk of churn, needs priority handling"

    return False, ""


def main():
    df = pd.read_csv("data/golden_set_unlabeled.csv").fillna("")
    results = []

    for idx, row in df.iterrows():
        msg = str(row["customer_message"])
        intent = label_intent(msg)
        should_esc, reason = label_escalation(msg, intent)
        results.append({
            "inquiry_id": row["inquiry_id"],
            "customer_message": msg,
            "brand_reply": str(row["brand_reply"]),
            "intent": intent,
            "should_escalate": should_esc,
            "escalation_reason": reason,
        })

    out = pd.DataFrame(results)
    out.to_csv("data/golden_set.csv", index=False)
    print(f"Wrote {len(out)} labeled rows to data/golden_set.csv")

    # Print distribution
    print("\nIntent distribution:")
    print(out["intent"].value_counts().to_string())
    print(f"\nEscalation rate: {out['should_escalate'].mean():.1%}")


if __name__ == "__main__":
    main()
