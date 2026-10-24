## How to run it

Install Python 3.11 or newer and the dependencies in `harness/requirements.txt`. Run the harness using the commands documented in `README.md`. The provided Makefile defines `setup`, `grade`, `sheet`, `eval`, and `validate` targets. On Windows, the Makefile's default `.venv/bin/python` path may need adjustment for the Windows virtual environment.

## What works

- Command-line ingestion, report generation, and feedback interfaces are implemented.
- State is persisted across batches using SQLite.
- The harness supports overcharge, duplicate, waived-but-billed, late-fee, cashback, and contract-expiry checks.
- Findings include evidence citations and confidence values.
- Evaluation cases cover expected findings, conflicts, and safety behavior.

## What doesn't

- A fresh grading run could not be completed successfully in my Windows environment because PDF processing failed when the PyMuPDF dependency was unavailable.
- Complex or unfamiliar document layouts may require additional parser rules or OCR support.
- Live Anthropic API integration was not verified because no API key was available.

## The eval case that caught a bug

The conflict-resolution evaluation caught a bug where a signed amendment and an authorized rate notice were incorrectly collapsed into one latest rate. The fix preserves document authority and provenance and flags the conflict with an amount range.