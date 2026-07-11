# Independent Evaluator

## Objective

Develop an independent evaluator for RAG chatbot responses.

The evaluator predicts:

- Faithfulness
- Relevance
- Reliability

## Current implementation

Implemented a rule-based baseline.

Features:

- lexical overlap
- context coverage
- numerical consistency
- repository-compatible Prediction schema

## Evaluation

Dataset: Organizer dataset

Samples: 2245

Reliable Macro-F1: 0.479

Faithfulness Macro-F1: 0.430

Relevance Macro-F1: 0.458

Invalid outputs: 0%

## Next work

- Improve semantic similarity
- Add NLI verification
- Tune thresholds