import argparse
import json
import re
import time
 
import pandas as pd
 
from agent import analyze_transaction, df as agent_df

def parse_final_content(final_content: str) -> dict:
    """
    The agent's system prompt asks for a fixed 4-item structure:
        1. Transaction category
        2. Whether an anomaly was detected
        3. Evidence supporting the conclusion
        4. A short explanation in plain language
 
    This is a lightweight, regex-based parse of that structure -- good
    enough for a report, not a substitute for reading the raw text if
    something looks off. If the model didn't follow the format closely,
    these fields fall back to None / "unknown" rather than raising.
    """
    if not final_content:
        return{"category":None,"flagged":None,"explanation":None}

    text=final_content.strip()

    category=None

    cat_match=re.search(r"1\.\s*.*?category.*?[:\-]\s*(.+)", text, re.IGNORECASE)
    if cat_match:
        category=cat_match.group(1).split("\n")[0].strip()

    flagged=None
    anomaly_match = re.search(r"2\.\s*.*?(anomaly detected)?.*?[:\-]\s*(.+)", text, re.IGNORECASE)
    if anomaly_match:
        answer_text = anomaly_match.group(2).split("\n")[0].lower()
        if any(word in answer_text for word in [
            "no", "false", "not detected", "none"
            ]):
            flagged = False

        elif any(word in answer_text for word in [
            "yes", "true", "detected", "flagged"
            ]):
            flagged = True

    explanation = None
    exp_match = re.search(r"3\.(.+)", text, re.DOTALL)
    if exp_match:
        explanation = exp_match.group(1).strip().replace("\n", " ")
 
    return {"category": category, "flagged": flagged, "explanation": explanation}

def run_batch(limit=None, sleep_seconds=1.0):
    transactions = agent_df.copy()
 
    if limit:
        transactions = transactions.head(limit)

    report_rows=[]
    total = len(transactions)
 
    for i, (_, row) in enumerate(transactions.iterrows(), start=1):
        txn_id = row["transaction_id"]
        print(f"\n[{i}/{total}] Analyzing {txn_id}...")
 
        try:
            result = analyze_transaction(txn_id)
        except Exception as e:
            # A single bad transaction should never take down the whole batch.
            print(f"  ERROR processing {txn_id}: {e}")
            result = {"status": "error", "reason": str(e), "tools_called": []}
 
        parsed = parse_final_content(result.get("final_content") if result else None)

        report_rows.append({
            "transaction_id": txn_id,
            "status": result.get("status") if result else "error",
            "category": parsed["category"],
            "flagged": parsed["flagged"],
            "tools_called": ", ".join(result.get("tools_called", [])) if result else "",
            "explanation": parsed["explanation"],
            # ground truth, kept ONLY for our own verification below --
            # never given to the agent as input
            "is_known_anomaly": row.get("is_known_anomaly", False),
            "anomaly_type": row.get("anomaly_type", None),
        })

        time.sleep(sleep_seconds)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv("agent_report.csv", index=False)
    return report_df

def verify_known_anomalies(report_df:pd.DataFrame):
    """
    This is the actual "it works" proof: for every transaction we know is
    an anomaly (planted in generate.py), check whether the agent's own
    'flagged' output agrees.
    """
    known=report_df[report_df["is_known_anomaly"]==True]
    normal=report_df[report_df["is_known_anomaly"]==False]

    print("\n" + "=" * 60)
    print("VERIFICATION AGAINST KNOWN ANOMALIES")
    print("=" * 60)

    if len(known) == 0:
        print("No known anomalies in this batch (did you use --limit and miss them?).")
        return

    correctly_flagged=known[known["flagged"]==True]
    missed=known[known["flagged"]==False]

    print(f"\nKnown anomalies in this run: {len(known)}")
    print(f"Correctly flagged by agent:  {len(correctly_flagged)}")
    print(f"Missed by agent:             {len(missed)}")

    if len(missed) > 0:
        print("\nMissed anomalies (worth reviewing manually):")
        print(missed[["transaction_id", "anomaly_type", "status", "tools_called"]].to_string(index=False))

    false_positives=normal[normal["flagged"]==True]
    print(f"\nFalse positives (normal transactions flagged): {len(false_positives)}")
    if len(false_positives) > 0:
        print(false_positives[["transaction_id", "category", "tools_called"]].to_string(index=False))

    detection_rate = len(correctly_flagged) / len(known) * 100
    print(f"\nDetection rate on known anomalies: {detection_rate:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N transactions (useful for a quick test run)")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="Seconds to wait between API calls")
    args = parser.parse_args()
 
    report_df = run_batch(limit=args.limit, sleep_seconds=args.sleep)
 
    print(f"\nSaved full report to agent_report.csv ({len(report_df)} rows).")
 
    verify_known_anomalies(report_df)
 

