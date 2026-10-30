# Implementation status

This is the durable resume ledger for the key-signature classifier.

## Sources of truth

1. Product behavior: [`specs/key-signature-classifier-spec.md`](specs/key-signature-classifier-spec.md)
2. Acceptance contract: [`HARNESS_README.md`](HARNESS_README.md) and [`.github/workflows/acceptance.yml`](.github/workflows/acceptance.yml)
3. Progress only: this file

## Completion state

- Local v1 accepted: **no**
- Public release accepted: **no**
- GitHub target: `benschwierking/Key_Signature` (public)
- Data publication: **blocked pending documented redistribution basis**

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| Canonical repository and packaging | complete | Spec moved; pyproject, README, MIT license, provenance added |
| Harness self-tests | complete | 81 passed, 1 optional skip on Python 3.11 |
| Feature interfaces and synthetic tests | complete | 31/31 required feature tests passed |
| Resumable GiantSteps pipeline | in progress | 2-track live run and resume passed; full 604-track extraction running |
| Deterministic model training | implemented | Synthetic end-to-end validation pending |
| Interpretive analysis and notebook | implemented | LR and RF paths plus exploration notebook added |
| Streamlit application | complete | AppTest imports/runs; regular Streamlit selected after Stlite dependency review |
| Strict C1–C7 acceptance | pending | Requires authorized/local real feature cache |
| Public GitHub source release | pending | Local `gh` token requires reauthentication |

## Current checkpoint

- Date: 2026-10-29 America/Los_Angeles
- Last successful action: official 2-track extraction passed, then resumed 2/2 without network or retained audio
- Environment: Python 3.11 virtualenv installed; ffmpeg available
- Blockers: GitHub CLI token invalid; public feature-cache redistribution not cleared
- Generated artifacts: partial `data/features/giantsteps_features.parquet`; per-track checkpoints under ignored `data/work/`

## Exact next action

```bash
cd keyclf-harness && .venv/bin/python -m keyclf.features --workers 2
```
