# Compute-cost ledger — consolidated (single source of truth)

Running total of EC2 spend across the unified-interface trial runs. This file is the
**durable, repo-tracked consolidation**; the per-session deep breakdowns live in the
deliverables ledger (Obsidian): `unified-model-interface-deliverables/COMPUTE-COST-2026-06-03.md`.

- **Instance (almost always):** `g5.4xlarge` (1× A10G 24 GB), `i-0bdbdf83c4db9bdae`, us-east-2.
- **Rates (us-east-2 on-demand, USD/hr):** g5.4xlarge **1.624**, g5.2xlarge 1.212, g5.12xlarge 5.672,
  g6e.2xlarge 2.42, g6e.12xlarge (4× L40S) 10.6. EBS gp3 $0.08/GB-month.
- **Method:** a *session total* is `billed running hours × rate` (measured from AWS LaunchTime).
  Per-capability rows are *measured* where `cost_log.csv` START/END markers exist, *estimated*
  (from `result.json` mtimes + model size) for runs that predate the markers.
- **Tool:** `cost_report.py` regenerates a per-session report from the marker log + uptime.

---

## Running total

| date | session / milestone | instance | basis | cost |
|---|---|---|---|---|
| pre-06-03 | M6 / BLIP-2 EC2 validation | g5.4xlarge | rough (CLAUDE.md) | ~$5 |
| pre-06-03 | MolmoSpaces embodied validation (π0, MS-Pick) | g5.4xlarge | $4.86 measured | ~$5 |
| pre-06-03 | TRIBEv2 env + V-JEPA-2-Giant / Wav2Vec-Bert smoke | g5.4xlarge | ~40 min | ~$0.80 |
| pre-06-03 | TR-resolved multimodal Lahner smoke runs | g5.4xlarge | rough | ~$1.50 |
| pre-06-03 | Algonauts 2025 download + Phase 1–3 (CLIP + multimodal) | g5.4xlarge | ~$8 cumulative | ~$8 |
| 2026-06-03 | overnight 2-AFC matrix + full Gemma-4-12B capability sweep | g5.4xlarge | 9.7 h measured | **$16** |
| 2026-06-04 | movie→brain demo (V-JEPA-2, glass brain) | g5.4xlarge | 1.27 h measured | ~$2 |
| 2026-06-05 | V-JEPA2 layer + unit-selection sweep (Lahner ROI) | g5.4xlarge | 1.30 h measured | ~$2 |
| 2026-06-05 | MIRAGE vs TRIBEv2 native-vs-post-hoc fusion arc (sessions 2–5) | g5/g6e.12xlarge | measured | **~$51** |
| 2026-06-10/11 | closed-API behavioral + KeyCorridor game notebook | **local laptop** | OpenRouter API only | ~$0.20 API |
| 2026-06-16 | topographic-alignment axis first real-fMRI run (NSD-surface, CLIP) | g5.4xlarge | ~0.6 h measured | ~$1 |
| | **GRAND TOTAL (EC2)** | | | **≈ $92** |

> The MIRAGE/TRIBEv2 arc (sessions 2–5) is the single biggest line — it used the 4× L40S
> `g6e.12xlarge` at $10.6/hr for the 30B Qwen3-Omni extraction. Everything else is A10G at $1.62/hr.

## Precision notes

- **Pre-06-03 rows are rough** — the dollar figures were recorded inline in `CLAUDE.md` /
  `RESULTS.md` as approximate "~$N" / cumulative snapshots; some overlap (e.g. the Algonauts
  arc's "~$5 so far" → "~$8 cumulative" are the same arc, not additive). Treat the ~$20
  pre-06-03 subtotal as ±30%.
- **2026-06-03 → 06-05 rows are measured session totals** (AWS LaunchTime × rate) — the
  trustworthy part of this ledger. See the deliverables ledger for per-capability breakdowns.
- **API spend is a separate, tiny bucket.** This week's closed-weight API work (Claude / GPT /
  DeepSeek / OpenRouter behavioral + the game notebook) runs on the laptop and bills the provider
  per token, not EC2. The smoke tests run this session were ~$0.20 of OpenRouter. Not yet
  systematically tracked — if API scoring scales up, add a provider-cost column here.
- **Stopped-instance EBS** (~$0.10/day for the 446 GB volume) accrues whenever the instance is
  stopped-not-terminated; negligible but ongoing.

---

## Durable logging convention (how markers survive a session)

Run-chains on EC2 append START/END markers to a **local** `/tmp/cost_log.csv` (fast; the repo
isn't checked out on the instance). That file dies with the instance. To make it durable:

**On instance stop**, pull the markers back and append them to the repo log:
```bash
# from the laptop, before/after stopping the instance
scp -i /tmp/quest.pem ubuntu@<ip>:/tmp/cost_log.csv /tmp/ec2_cost_log.csv
cat /tmp/ec2_cost_log.csv >> "unified/scripts/cost_tracking/cost_log.csv"
# then regenerate the per-session report:
python unified/scripts/cost_tracking/cost_report.py --uptime_hours <h> --out /tmp/COST_REPORT.md
```
Then add a row to the **Running total** table above and commit. The repo-tracked
`cost_log.csv` (marker schema: `capability,model,instance,epoch,START|END`, no header) is the
append-only durable store; `cost_report.py` reads it by default.

> The 9 markers currently in `cost_log.csv` are the surviving Gemma / MiniGrid markers recovered
> from a session job-tmp dir (the rest of that overnight session's markers were never written —
> its $16 total is the measured AWS-LaunchTime figure, not marker-derived).
