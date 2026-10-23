# My solution

## How to run it

Install the pinned dependencies from `harness/requirements.txt`, then run
`python harness/harness.py` with the three commands described in README. On
Windows, use `python` (or the selected interpreter path) instead of the
POSIX `.venv/bin/python` Makefile default. The harness needs no API key:
without `ANTHROPIC_API_KEY` it uses deterministic extraction. Set that key and
optional `ANTHROPIC_BASE_URL` to enable the existing Claude provider.

## What works

The harness ingests every email and supported PDF/XLSX attachment, persists
facts across fresh processes, cites source pages, and emits the required
report schema. It covers overcharges, duplicate shipment lines, authorized
waivers, late-fee disputes and risk, expiry alerts, cashback, and feedback
suppression. Supervised and autonomous actions are separated, and autonomous
outbound actions are limited to small, complete, reversible findings.
SQLite state is portable on Windows and content processing is idempotent.

## What doesn't

The rule-based provider is conservative for layouts with no recognizable
invoice identifiers, lanes or AWBs. Complex multi-page amendment precedence
and some annual-credit variants need the optional LLM provider or a richer
parser. OCR is available through the dependency described in the brief, but
environments that cannot install it should route scanned pages to the
configured vision provider rather than inventing text.

## The eval case that caught a bug

The conflict example caught an early bug in which a signed amendment and an
authorized rate notice were reduced to one "latest" rate. The report then
issued a dispute instead of an escalation. The implementation now preserves
the authority level and source for each rate and emits a conflict with a
zero-to-maximum amount range when both trusted sources disagree.
