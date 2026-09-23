# Transaction Risk Analysis Agent

An LLM-powered transaction analysis agent that combines LLM tool calling with deterministic Python-based anomaly detection to analyze transaction behavior and explain potential risks.

## Overview

The project uses an agent-based approach to transaction analysis:

```text
Transaction
     |
     v
  LLM Agent
     |
     | chooses tools
     v
Python Analysis Functions
     |
     | verified results
     v
  LLM Agent
     |
     v
Final Risk Analysis
```

The agent can investigate:
- Unusually large transaction amounts
- Duplicate transactions within a short time window
- Transactions occurring at unusual times

The underlying analysis is performed deterministically using transaction history rather than asking the LLM to perform the calculations itself.

## Key Features

### LLM Tool Calling
The agent exposes three analysis functions as tools:
- `check_amount_anomaly`
- `check_duplicate`
- `check_unusual_time`

The LLM can select the relevant tool or tools for a transaction.

### Deterministic Anomaly Detection
The actual analysis is implemented in Python:
- Z-score based amount anomaly detection
- Time-window based duplicate detection
- Historical transaction-time analysis

Python tool results are treated as the source of truth for the final response.

### Grounding Guardrail
The agent tracks which tools were actually called and checks whether the final response makes claims that require those tools.

If a response claims that a transaction is a duplicate without calling `check_duplicate`, the response can be rejected and the model prompted to verify the claim.

### Tool Error Handling
Invalid arguments, unknown transaction IDs, unknown tools, and tool execution failures are returned as structured errors instead of immediately crashing the analysis.

### Batch Evaluation
The complete transaction dataset can be processed through the agent and exported to `agent_report.csv`. The batch evaluator also compares the agent's results against deliberately planted anomalies.

## Architecture

```text
                    transactions.csv
                           |
                           v
                    +-------------+
                    |  LLM Agent  |
                    +-------------+
                           |
                 Selects analysis tool(s)
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
  Amount Analysis   Duplicate Check   Unusual Time
          |                |                |
          +----------------+----------------+
                           |
                           v
                  Deterministic Results
                           |
                           v
                    +-------------+
                    |  LLM Agent  |
                    |  Explanation|
                    +-------------+
                           |
                           v
                    Final Analysis
```

## Anomaly Detection

### 1. Amount Anomaly

The amount analysis compares a transaction against the user's historical spending.

It calculates:
- Historical mean
- Historical standard deviation
- Transaction z-score

The transaction is considered anomalous when:

```text
|z-score| > 3
```

### 2. Duplicate Transaction

The duplicate detector checks the user's transaction history for another transaction that:
- Belongs to the same user
- Has the same amount
- Has the same merchant
- Occurs within the configured time window

The current implementation uses a 120-second window.

### 3. Unusual Transaction Time

The time analysis compares the transaction's hour with the user's historical transaction times.

It calculates the 5th and 95th percentiles of the historical transaction hours and checks whether the current transaction falls outside that range.

## Synthetic Dataset

The project uses a synthetic transaction dataset because the analysis requires human-readable fields such as merchant, category, and timestamp.

The generator creates:
- 20 users
- 25 normal transactions per user
- Approximately 500 normal transactions
- Deliberately planted anomalies

The planted anomalies include:
1. Unusually large transaction
2. Duplicate transaction pair
3. Unusual transaction time

The ground-truth fields are retained for evaluation only and should not be provided to the LLM during analysis.

## Agent Workflow

For each transaction:

```text
1. Load transaction
        |
2. Send transaction to LLM
        |
3. LLM selects relevant tool(s)
        |
4. Python executes selected tool(s)
        |
5. Tool results are returned to LLM
        |
6. LLM generates final analysis
        |
7. Grounding guardrail validates claims
        |
8. Result is added to the batch report
```

The agent supports multiple tool-calling rounds rather than assuming exactly one tool call will always be sufficient. A maximum API-call limit and tool-round limit prevent uncontrolled loops.

## Grounding Guardrail

The project includes a lightweight grounding mechanism.

The agent tracks which tools were actually called during the conversation and checks the final response for claims associated with specific tools.

For example:

```text
Claim:
"The transaction is a duplicate."

Required tool:
check_duplicate
```

If the claim appears without the corresponding tool being called, the response is rejected and the model is prompted to either:
- Call the required tool, or
- Remove the unsupported claim.

## Batch Processing

`batch.py` runs the agent over the transaction dataset.

Run the complete dataset:

```bash
python batch.py
```

Run a smaller batch for testing:

```bash
python batch.py --limit 20
```

Configure the delay between transactions:

```bash
python batch.py --sleep 0.5
```

The batch processor:
1. Iterates through the transactions
2. Calls `analyze_transaction()` for each transaction
3. Parses the agent's response
4. Collects the results
5. Creates a Pandas DataFrame
6. Writes the final report to `agent_report.csv`
7. Evaluates the results against the planted anomalies

## Evaluation

The evaluation compares the agent's output with the known ground truth:

```text
Ground Truth              Agent Output
-------------              ------------
Known anomaly       vs.    flagged = True
Normal transaction  vs.    flagged = False
```

The evaluation reports:
- Number of known anomalies
- Correctly detected anomalies
- Missed anomalies
- False positives
- Detection rate

Detection rate is calculated as:

```text
Correctly Flagged Anomalies
--------------------------- × 100
Total Known Anomalies
```

## Tech Stack

- Python
- Pandas
- OpenAI-compatible LLM API
- OpenRouter
- LLM Tool Calling
- Faker
- python-dotenv

## Project Structure

```text
transaction-risk-agent/
│
├── generate.py
│   └── Generates the synthetic transaction dataset
│
├── transactions.csv
│   └── Generated transaction history
│
├── analysis.py
│   └── Deterministic anomaly detection functions
│
├── agent.py
│   └── LLM agent, tool definitions, execution,
│       retries and grounding guardrails
│
├── batch.py
│   └── Runs the agent over the dataset and
│       generates the evaluation report
│
├── agent_report.csv
│   └── Batch analysis results
│
└── requirements.txt
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENROUTER_API_KEY=your_api_key_here
```

## Generate Dataset

Run:

```bash
python generate.py
```

This creates `transactions.csv`.

The generator uses a fixed random seed so the synthetic dataset can be reproduced consistently.

## Run the Agent

To analyze the sample transaction configured in `agent.py`:

```bash
python agent.py
```

To process the complete dataset:

```bash
python batch.py
```

For a quick test:

```bash
python batch.py --limit 20
```

## Output

The batch processor generates:

```text
agent_report.csv
```

The report contains fields including:

```text
transaction_id
status
category
flagged
tools_called
explanation
is_known_anomaly
anomaly_type
```

The `is_known_anomaly` and `anomaly_type` fields are retained for post-analysis evaluation.

## Design Decisions

### LLM for Reasoning, Python for Verification

The LLM is responsible for deciding which investigation is relevant and explaining the results. Python performs the actual calculations.

This separates:

```text
Reasoning / tool selection
        from
Deterministic computation
```

and avoids relying on the LLM for numerical calculations.

### Tool-Based Investigation

Instead of running every possible check blindly, the model can select the analysis functions relevant to the transaction.

### Grounded Final Responses

The agent tracks tool usage and rejects unsupported anomaly claims.

### Synthetic Ground Truth

Known anomalies are deliberately planted into the dataset so the system can be evaluated against known examples.

## Limitations

- The anomaly detection methods are relatively simple statistical and rule-based checks.
- The dataset is synthetic and does not represent real banking transaction distributions.
- The grounding mechanism uses keyword matching and is therefore a lightweight guardrail rather than a formal proof of correctness.
- LLM tool selection and final explanations can vary between runs depending on the selected model.
- Detection performance depends on the quality and quantity of a user's transaction history.

## Future Improvements

- Add location-based anomaly detection
- Add payment-method behavior analysis
- Introduce richer behavioral baselines derived from transaction history
- Add configurable anomaly thresholds
- Store analysis results in a database
- Add a REST API for transaction analysis
- Add precision, recall, and F1-score evaluation
- Add a web dashboard for transaction risk analysis
- Add persistent agent traces for debugging and auditing

## Project Objective

The project demonstrates how an LLM can act as an orchestrator over deterministic analytical tools rather than being treated as the source of truth for numerical or transactional analysis.

The core design principle is:

```text
LLM decides what to investigate
            ↓
Python verifies the behavior
            ↓
LLM explains the verified evidence
```
