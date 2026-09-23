"""
generate.py

Generates a synthetic transactions dataset for the Transaction Categorization
& Anomaly-Flagging Agent project.

Why synthetic data: real transaction datasets (e.g. the Kaggle fraud dataset)
are PCA-anonymized and strip out human-readable fields like merchant name,
category, and timestamp -- exactly the fields this project needs. Generating
our own data also lets us plant KNOWN anomalies, so we can later verify our
detection functions actually catch what they're supposed to catch.

"""

import random
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)  # reproducible runs

NUM_USERS = 20
TRANSACTIONS_PER_USER = 25  # ~500 total before anomalies are injected

CATEGORIES = {
    "groceries": (10, 120),
    "dining": (8, 80),
    "subscriptions": (5, 25),
    "transport": (5, 60),
    "utilities": (30, 150),
    "shopping": (15, 200),
    "entertainment": (10, 100),
}

MERCHANTS_BY_CATEGORY = {
    "groceries": ["Whole Foods", "Trader Joe's", "Local Market", "Costco"],
    "dining": ["Chipotle", "Local Cafe", "Pizza Place", "Sushi Bar"],
    "subscriptions": ["Netflix", "Spotify", "iCloud", "Adobe"],
    "transport": ["Uber", "Lyft", "Metro Transit", "Gas Station"],
    "utilities": ["Electric Co", "Water Utility", "Internet Provider"],
    "shopping": ["Amazon", "Target", "Best Buy", "IKEA"],
    "entertainment": ["AMC Theaters", "Steam", "Concert Tickets"],
}

# Each user has a "typical active hours" window -- used later to generate
# a normal transaction time, and to plant an "unusual time" anomaly.
def random_user_profile(user_id):
    active_start = random.randint(7, 10)   # e.g. starts transacting ~7-10am
    active_end = random.randint(18, 22)    # e.g. stops ~6-10pm
    return {
        "user_id": user_id,
        "active_start_hour": active_start,
        "active_end_hour": active_end,
    }


def random_timestamp_in_active_hours(base_date, profile):
    hour = random.randint(profile["active_start_hour"], profile["active_end_hour"])
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=second)


def generate_normal_transaction(txn_id, user_id, profile, base_date):
    category = random.choice(list(CATEGORIES.keys()))
    low, high = CATEGORIES[category]
    amount = round(random.uniform(low, high), 2)
    merchant = random.choice(MERCHANTS_BY_CATEGORY[category])
    timestamp = random_timestamp_in_active_hours(base_date, profile)

    return {
        "transaction_id": txn_id,
        "user_id": user_id,
        "date": timestamp.isoformat(),
        "amount": amount,
        "merchant": merchant,
        "category": category,
        "is_known_anomaly": False,   # ground-truth label, NOT given to the agent
        "anomaly_type": None,
    }


def main():
    rows = []
    txn_counter = 1
    start_date = datetime(2026, 1, 1)

    profiles = [random_user_profile(f"user_{i}") for i in range(1, NUM_USERS + 1)]

    # 1. Generate normal transaction history for every user
    for profile in profiles:
        for _ in range(TRANSACTIONS_PER_USER):
            day_offset = random.randint(0, 89)  # spread across ~3 months
            base_date = start_date + timedelta(days=day_offset)
            txn = generate_normal_transaction(
                f"txn_{txn_counter:05d}", profile["user_id"], profile, base_date
            )
            rows.append(txn)
            txn_counter += 1

    df = pd.DataFrame(rows)

    # 2. Inject known anomalies -- these are what your detection functions
    #    (and later, the agent) should catch. We keep the ground-truth label
    #    (is_known_anomaly / anomaly_type) so you can measure detection
    #    accuracy, but this label should NEVER be passed to the LLM or your
    #    detection functions -- it's for your own evaluation only.

    anomaly_rows = []

    # --- Anomaly type 1: unusually large amount ---
    victim_user = profiles[0]["user_id"]
    anomaly_rows.append({
        "transaction_id": f"txn_{txn_counter:05d}",
        "user_id": victim_user,
        "date": (start_date + timedelta(days=45)).replace(hour=14, minute=30).isoformat(),
        "amount": 4800.00,  # far outside this user's normal $10-200 range
        "merchant": "Electronics Superstore",
        "category": "shopping",
        "is_known_anomaly": True,
        "anomaly_type": "unusually_large_amount",
    })
    txn_counter += 1

    # --- Anomaly type 2: duplicate charge within a short window ---
    dup_user = profiles[1]["user_id"]
    dup_timestamp = start_date + timedelta(days=30, hours=12, minutes=0, seconds=0)
    dup_amount = 45.99
    dup_merchant = "Best Buy"
    for i in range(2):  # same charge fired twice, 45 seconds apart
        anomaly_rows.append({
            "transaction_id": f"txn_{txn_counter:05d}",
            "user_id": dup_user,
            "date": (dup_timestamp + timedelta(seconds=45 * i)).isoformat(),
            "amount": dup_amount,
            "merchant": dup_merchant,
            "category": "shopping",
            "is_known_anomaly": True,
            "anomaly_type": "duplicate_charge",
        })
        txn_counter += 1

    # --- Anomaly type 3: transaction at an unusual hour for that user ---
    night_user_profile = profiles[2]
    anomaly_rows.append({
        "transaction_id": f"txn_{txn_counter:05d}",
        "user_id": night_user_profile["user_id"],
        "date": (start_date + timedelta(days=60)).replace(hour=3, minute=17).isoformat(),
        "amount": 62.50,  # normal amount, but 3am is well outside this user's active hours
        "merchant": "Gas Station",
        "category": "transport",
        "is_known_anomaly": True,
        "anomaly_type": "unusual_time",
    })
    txn_counter += 1

    anomaly_df = pd.DataFrame(anomaly_rows)
    df = pd.concat([df, anomaly_df], ignore_index=True)

    # 3. Shuffle so anomalies aren't suspiciously clustered at the end of the file
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv("transactions.csv", index=False)

    print(f"Generated {len(df)} transactions ({len(anomaly_rows)} known anomalies) -> transactions.csv")
    print("\nKnown anomalies planted (for your own verification later -- do NOT feed this to the agent):")
    print(anomaly_df[["transaction_id", "user_id", "amount", "anomaly_type"]].to_string(index=False))


if __name__ == "__main__":
    main()