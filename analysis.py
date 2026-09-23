import pandas as pd


def get_user_amount_stats(user_id, transactions_df):
    """
    Calculate spending statistics from a user's transaction history.
    """

    user_history = transactions_df[
        transactions_df["user_id"] == user_id
    ]

    if user_history.empty:
        return {
            "transaction_count": 0,
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0
        }

    amounts = user_history["amount"]

    return {
        "transaction_count": len(amounts),
        "mean": round(amounts.mean(), 2),
        "std": round(amounts.std(), 2),
        "min": round(amounts.min(), 2),
        "max": round(amounts.max(), 2)
    }


def is_amount_anomalous(transaction, user_history_df, threshold=3.0):
    """
    Check whether a transaction amount is unusually large
    compared with the user's historical spending.
    """

    if user_history_df.empty:
        return {
            "is_anomalous": False,
            "reason": "No historical transactions available"
        }

    amounts = user_history_df["amount"]

    mean = amounts.mean()
    std = amounts.std()

    if std == 0 or pd.isna(std):
        return {
            "is_anomalous": False,
            "reason": "Insufficient variation in historical spending"
        }

    z_score = (transaction["amount"] - mean) / std

    return {
    "is_anomalous": bool(abs(z_score) > threshold),
    "z_score": float(round(z_score, 2)),
    "transaction_amount": float(transaction["amount"]),
    "historical_mean": float(round(mean, 2)),
    "historical_std": float(round(std, 2))
}

def detect_duplicate(transaction, recent_transactions_df, window_seconds=120):
    """
    Check whether the same user made the same transaction
    within a short time window.
    """

    if recent_transactions_df.empty:
        return {
            "is_duplicate": False,
            "matching_transactions": []
        }

    transaction_time = pd.to_datetime(transaction["date"])

    history = recent_transactions_df[
        recent_transactions_df["user_id"] == transaction["user_id"]
    ].copy()

    if history.empty:
        return {
            "is_duplicate": False,
            "matching_transactions": []
        }

    history["date"] = pd.to_datetime(history["date"])

    time_difference = (
        transaction_time - history["date"]
    ).abs().dt.total_seconds()

    possible_duplicates = history[
        (time_difference <= window_seconds)
        & (history["amount"] == transaction["amount"])
        & (history["merchant"] == transaction["merchant"])
    ]

    return {
        "is_duplicate": not possible_duplicates.empty,
        "matching_transactions": possible_duplicates[
            ["transaction_id", "date", "amount", "merchant"]
        ].to_dict("records")
    }


def is_unusual_time(transaction, user_history_df):
    """
    Check whether the transaction occurred at an unusual
    time compared with the user's historical transactions.
    """

    if user_history_df.empty:
        return {
            "is_unusual": False,
            "reason": "No historical transactions available"
        }

    transaction_time = pd.to_datetime(transaction["date"])
    transaction_hour = transaction_time.hour

    history = user_history_df.copy()
    history["date"] = pd.to_datetime(history["date"])

    historical_hours = history["date"].dt.hour

    lower = historical_hours.quantile(0.05)
    upper = historical_hours.quantile(0.95)

    unusual = (
        transaction_hour < lower
        or transaction_hour > upper
    )

    return {
    "is_unusual": bool(unusual),
    "transaction_hour": int(transaction_hour),
    "historical_hour_range": [
        float(round(lower, 1)),
        float(round(upper, 1))
    ]
}