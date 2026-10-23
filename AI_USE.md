# AI use

## Tools

OpenAI Codex in the IDE, PowerShell, repository search/read tools, and Python.
The code changes use the existing deterministic text extraction provider; no
external model calls were made during this repair.

## How I used them

Codex inspected the assignment brief, glossary, examples, schema, grader,
existing implementation, tests and recorded reports. It changed the conflict
amount-range calculation, generalized detection of explicitly referenced but
missing rate attachments, and fixed feedback matching so one invoice's
decision does not affect other invoices on its contract. Python tests and the
repository evaluation and grading tools were used to check those changes.

## Prompt log

1. “Take over the existing Nova Freight Invoice Audit Harness project. Inspect
   the assignment instructions, repository, existing code, tests, evaluation
   results, and generated reports before making changes.”
2. “Improve the implementation to maximize the practice score (last recorded:
   51.1/100) and fix the failing custom evaluation (last recorded: 11/12
   passed). Identify root causes, fix the underlying logic, and add regression
   tests. Do not hardcode evaluation answers or weaken validation.”
3. “Preserve working code and implement all remaining assignment requirements.
   I will install dependencies myself later, so don't waste time troubleshooting
   installations. Run all available tests and evaluations, fix failures, and
   regenerate required outputs.”
4. “Prepare a complete, submission-ready repository with correct source code,
   documentation, configuration instructions, and all required deliverables.
   Verify the final files and follow the original assignment's submission
   requirements.”
