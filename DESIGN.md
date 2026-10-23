# Design

## Fact model

Each received email and attachment is stored by batch, document reference and
content hash in SQLite. The extracted records are invoices and line items,
contracts and rate rows, fuel and FX tables, Schedule 1 contacts, payments,
tracking milestones, waivers, extensions and feedback. A record always keeps
the source document and page, so a report can cite the exact text the grader
reads. Invoice keys are normalized only for matching; displayed IDs retain the
printed value without spaces, `#` or `:`.

## Conflicts

Rates are selected by contract, lane, weight band and pickup date. Signed
contract/amendment rows and authorized rate notices are retained separately.
When an authorized notice and a signed row disagree for the same band, the
harness emits `conflict` rather than choosing one. It cites both rows and
leaves the monetary range out of savings. A missing FX row or other required
record produces `cannot_determine`, naming the document holder rather than
guessing.

## Confidence

The deterministic path starts at 0.90 for a complete text extraction and
reduces confidence for abstentions or incomplete evidence. A finding is only
created when a source invoice and governing record are present; citations are
selected from the actual extracted page. The API provider is optional and is
isolated behind the existing provider interface, so local runs remain
repeatable. Future calibration would score confidence against adjudicated
findings and tune the weights for source authority, extraction method and
cross-document agreement.

## People in the loop

Supervised mode drafts disputes and credit claims, while internal expiry alerts
and payment reminders are emitted directly. Conflicts and missing documents
are escalated. Autonomous mode executes only reversible outbound actions with
complete evidence and an amount no higher than USD 5,000; conflicts, missing
documents and larger claims still wait for a person.

## Feedback

Feedback is persisted with its item key, decision, scope and timestamp. A
close/reject decision suppresses only the named invoice or contract key, and
the report marks a changed item with `learned_at`. Standing facts are stored
for later authority checks, but are never applied outside their explicit
contract/category scope.

## New documents

Ingestion is incremental and idempotent by document reference. Later batches
add rows to the same state database, so amendments, payment updates and
tracking events can change a subsequent report. Pickup dates prefer tracking
events, then invoice dates; later evidence therefore changes pricing without
rewriting earlier source files.

## Cost

The local path makes no model calls and is bounded by document size, so its
cost is effectively zero and processing is dominated by PDF/XLSX extraction.
With Claude enabled, one structured extraction request is made per document
type and cached by document reference and content hash. At roughly 1,000
documents this is approximately 1,000 requests and a few million input
tokens; exact spend depends on attachment length and the configured model.

## What I'd do next

I would add layout-independent table parsing, full amendment precedence
intervals, OCR/vision fallback tests, and confidence calibration on a
held-out adjudicated set. The least certain area is unusual invoice layouts
whose line labels do not contain an AWB, lane or conventional date; those
should be routed to the configurable LLM/OCR provider instead of silently
treated as clean.
