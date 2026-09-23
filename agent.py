import os
import json
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

from analysis import (
    is_amount_anomalous,
    detect_duplicate,
    is_unusual_time
)
load_dotenv()

client=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL="openrouter/free"

df=pd.read_csv("transactions.csv")
df["date"]=pd.to_datetime(df["date"])

MAX_TOOL_ROUNDS=4
MAX_BAD_ARG_RETRIES=2
MAX_GROUNDING_RETRIES=2

CLAIM_KEYWORDS_TO_TOOL={
    "check_amount_anomaly": ["unusually large", "unusually high", "standard deviation", "z-score", "z score"],
    "check_duplicate": ["duplicate"],
    "check_unusual_time": ["unusual time", "unusual hour", "outside", "typical hours"],
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_amount_anomaly",
            "description": (
                "Check whether a transaction amount is unusually large "
                "compared with the user's historical spending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string"
                    }
                },
                "required": ["transaction_id"]
            }
        }
    },
 
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": (
                "Check whether a transaction appears to be a duplicate "
                "of another recent transaction by the same user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string"
                    }
                },
                "required": ["transaction_id"]
            }
        }
    },
 
    {
        "type": "function",
        "function": {
            "name": "check_unusual_time",
            "description": (
                "Check whether a transaction occurred at an unusual time "
                "compared with the user's historical transaction times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string"
                    }
                },
                "required": ["transaction_id"]
            }
        }
    }
]

def execute_tool(tool_name, arguments):
    """
    Guardrail: this now NEVER raises. Every failure mode (missing key,
    unknown transaction, unknown tool name) returns a structured error dict
    instead of crashing the whole run -- that error gets fed back to the
    model as a real tool result, so it can see what went wrong and retry
    with corrected arguments instead of the script just dying.
    """
    try:
        transaction_id = arguments["transaction_id"]
    except KeyError:
        return {"error": "Missing required argument 'transaction_id'."}

    transaction_rows=df[df["transaction_id"]==transaction_id]

    if transaction_rows.empty:
        return{"error":f"No transaction found with id '{transaction_id}'."}

    transaction = transaction_rows.iloc[0].to_dict()
    user_id = transaction["user_id"]

    user_history=df[(df["user_id"]==user_id)&(df["transaction_id"]!=transaction_id)]

    try:
        if tool_name=="check_amount_anomaly":
            return is_amount_anomalous(transaction,user_history)
        elif tool_name=="check_duplicate":
            return detect_duplicate(transaction,user_history)
        elif tool_name=="check_unusual_time":
            return is_unusual_time(transaction,user_history)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"error": f"Tool '{tool_name}' raised an exception: {e}"}

def _response_is_grounded(final_text, tools_actually_called):
    """
    Guardrail: a lightweight check that the model isn't making a specific
    claim ('this is a duplicate', 'unusually large') without having called
    the tool that would actually justify that claim in THIS conversation.
     
    This is deliberately simple (keyword matching) rather than a second LLM
    call -- good enough to catch the obvious failure mode (model skips the
    tool and just asserts an answer) without adding real cost/latency.
    Returns a list of (tool_name, matched_phrase) violations -- empty list
    means the response looks grounded.
    """

    violations=[]
    lowered = (final_text or "").lower()

    for tool_name,phrases in CLAIM_KEYWORDS_TO_TOOL.items():
        for phrase in phrases:
            if phrase in lowered and tool_name not in tools_actually_called:
                violations.append((tool_name,phrase))

    return violations

def analyze_transaction(transaction_id):
 
    transaction_rows = df[df["transaction_id"] == transaction_id]
 
    if transaction_rows.empty:
        print("Transaction not found.")
        return
 
    transaction = transaction_rows.iloc[0].to_dict()

# Ground truth is used only for evaluation, never shown to the LLM
    transaction.pop("is_known_anomaly", None)
    transaction.pop("anomaly_type", None)
 
    messages = [
        {
            "role": "system",
            "content": """
You are a transaction analysis agent.
 
You MUST use the provided analysis tools
when investigating anomalies.
 
Do not guess or invent numerical evidence.
 
Python tool results are the source of truth.
 
After receiving the tool results, provide:
 
1. Transaction category
2. Whether an anomaly was detected
3. Evidence supporting the conclusion
4. A short explanation in plain language
"""
        },
        {
            "role": "user",
            "content": (
                "Analyze this transaction:\n\n"
                + json.dumps(transaction, default=str)
            )
        }
    ]
 
    tools_actually_called = set()   # guardrail: what did the model really invoke?
    bad_arg_retry_count = 0
    grounding_retry_count = 0
    final_content = None
    last_violations = []
 
    # Guardrail: loop with a hard cap instead of assuming exactly one round
    # of tool calls -- a model may legitimately want to call more than one
    # tool, or call one, look at the result, and decide to call another.
    # This same loop also handles grounding rejection/re-prompting below.
    tool_rounds = 0
    api_calls = 0
    MAX_API_CALLS = 8

    while True:

        # Hard safety ceiling: prevents an endlessly looping model
        # from making unlimited API calls.
        if api_calls >= MAX_API_CALLS:
            print("\nMaximum API call limit reached -- aborting.")
            return {
                "status": "failed",
                "reason": f"Exceeded maximum of {MAX_API_CALLS} API calls."
            }

        api_calls += 1

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # ------------------------------------------------------------
        # No tool call = model is attempting to give a final answer
        # ------------------------------------------------------------
        if not assistant_message.tool_calls:

            # Guardrail 1: model never used any tool
            if not tools_actually_called:
                print(
                    "\nModel skipped tools entirely -- "
                    "re-prompting to enforce tool use."
                )

                messages.append({
                    "role": "user",
                    "content": (
                        "You did not call any tool. Per your instructions, "
                        "you must call at least one analysis tool before "
                        "concluding anything about this transaction. "
                        "Please call the relevant tool(s) now."
                    )
                })

                continue

            # Guardrail 2: check final answer grounding
            candidate_content = assistant_message.content

            violations = _response_is_grounded(
                candidate_content,
                tools_actually_called
            )

            if violations and grounding_retry_count < MAX_GROUNDING_RETRIES:

                grounding_retry_count += 1
                last_violations = violations

                violation_lines = "\n".join(
                    f"- You used language matching '{phrase}' but never "
                    f"called the '{tool_name}' tool in this conversation."
                    for tool_name, phrase in violations
                )

                print(
                    f"\n[REJECTED] Ungrounded claim detected "
                    f"(attempt {grounding_retry_count}):"
                )
                print(violation_lines)

                messages.append({
                    "role": "user",
                    "content": (
                        "Your answer was rejected because it made a claim "
                        "that is not backed by any tool result in this "
                        "conversation:\n\n"
                        + violation_lines
                        + "\n\nEither call the missing tool(s) to actually "
                        "verify the claim, or revise your answer to remove "
                        "any unsupported claims and rely only on the tool "
                        "results you actually have."
                    )
                })

                continue

            if violations:
                print(
                    "\n[REJECTED] Max grounding retries exceeded -- "
                    "refusing to accept this answer as-is."
                )

                return {
                    "status": "rejected",
                    "reason": (
                        "Model repeatedly made claims not backed "
                        "by tool calls."
                    ),
                    "last_attempt": candidate_content,
                    "unresolved_violations": violations,
                    "tools_called": sorted(tools_actually_called),
                }

            # No violations -> valid final answer
            final_content = candidate_content
            break

        # ------------------------------------------------------------
        # Model requested one or more tools
        # ------------------------------------------------------------

        # If the maximum number of tool-calling rounds has already
        # been used, ask the model to give its final answer.
        if tool_rounds >= MAX_TOOL_ROUNDS:

            print(
                "\nMaximum tool-call rounds reached -- "
                "forcing final answer."
            )

            messages.append({
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool-calling "
                    "rounds. Do not call any more tools. Provide your final "
                    "answer using only the tool results already available "
                    "in this conversation."
                )
            })

            continue

        # This is an allowed tool-calling round.
        tool_rounds += 1

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                result = {
                    "error": "Could not parse tool arguments as JSON."
                }

                arguments = {}

            else:
                result = execute_tool(
                    tool_name,
                    arguments
                )

            print(f"\nLLM requested tool: {tool_name}")
            print(f"Arguments: {arguments}")
            print(f"Tool result: {result}")

            if "error" in result:
                bad_arg_retry_count += 1
            else:
                tools_actually_called.add(tool_name)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

        if bad_arg_retry_count > MAX_BAD_ARG_RETRIES:
            print("\nToo many failed tool calls -- aborting this analysis.")

            return {
                "status": "failed",
                "reason": "Exceeded max retries on invalid tool arguments."
            }

    if final_content is None:
        print("\nModel never produced a final answer.")
        return {"status": "failed", "reason": "No final content returned."}
 
    print("\nFinal analysis:")
    print(final_content)
    print("\n[GUARDRAIL] Grounding check passed -- all claims backed by an actual tool call.")
 
    return {
        "status": "ok",
        "final_content": final_content,
        "tools_called": sorted(tools_actually_called),
        "grounding_retries_needed": grounding_retry_count,
    }
 
 
if __name__ == "__main__":
    result = analyze_transaction("txn_00002")
    print("\n--- Run summary ---")
    print(json.dumps(result, indent=2, default=str))
 