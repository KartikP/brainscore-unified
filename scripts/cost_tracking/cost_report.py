"""Compute an EC2 compute-cost ledger for the capability evaluations.

Reads (a) the EC2 instance running-stretch (passed in, or measured separately) and
(b) per-run timing from two sources: the live ``cost_log.csv`` markers that the run
chains append (capability,model,instance,epoch,START|END) and, for early runs that
predate the markers, the ``result.json`` mtimes. Emits a markdown ledger: per-capability
runtime + dollar cost, plus the session total = billed instance hours x on-demand rate.

Person-free + ops-only, so it lives in the repo. Rates are us-east-2 on-demand.
"""
import argparse, csv, json, os
from collections import defaultdict

# us-east-2 on-demand (USD/hr). g5.4xlarge = 1x A10G 24GB; g6e.* = L40S.
RATES = {'g5.4xlarge': 1.624, 'g5.12xlarge': 5.672, 'g5.2xlarge': 1.212,
         'g6e.2xlarge': 2.42, 'g6e.12xlarge': 10.6}
EBS_GP3_PER_GB_MONTH = 0.08

# Durable, repo-tracked marker log. Run-chains on EC2 still append to a local
# /tmp/cost_log.csv (fast, no repo on the instance); on instance STOP, append
# that file's rows here so the markers survive the session — see COST_LEDGER.md.
DURABLE_COST_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'cost_log.csv')


def parse_cost_log(path):
    """cost_log.csv rows: capability,model,instance,epoch,START|END -> durations."""
    runs = defaultdict(dict)
    if not os.path.exists(path):
        return []
    for r in csv.reader(open(path)):
        if len(r) < 5:
            continue
        cap, model, inst, epoch, phase = r[0], r[1], r[2], int(r[3]), r[4]
        runs[(cap, model, inst)][phase] = epoch
    out = []
    for (cap, model, inst), d in runs.items():
        if 'START' in d and 'END' in d:
            secs = d['END'] - d['START']
            rate = RATES.get(inst, 1.624)
            out.append({'capability': cap, 'model': model, 'instance': inst,
                        'runtime_s': secs, 'runtime_min': round(secs / 60, 1),
                        'cost_usd': round(secs / 3600 * rate, 3), 'source': 'measured'})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost_log', default=DURABLE_COST_LOG,
                    help='marker CSV; defaults to the repo-tracked durable log')
    ap.add_argument('--instance', default='g5.4xlarge')
    ap.add_argument('--uptime_hours', type=float, required=True,
                    help='billed continuous running hours for the session')
    ap.add_argument('--ebs_gb', type=float, default=446)
    ap.add_argument('--out', default='COST_REPORT.md')
    args = ap.parse_args()

    rate = RATES.get(args.instance, 1.624)
    measured = parse_cost_log(args.cost_log)
    compute_cost = round(args.uptime_hours * rate, 2)
    storage_cost = round(args.ebs_gb * EBS_GP3_PER_GB_MONTH / 730 * args.uptime_hours, 2)

    lines = ['# Compute-cost ledger — capability evaluations', '',
             f'- **Instance:** {args.instance} (1x A10G 24GB), us-east-2 on-demand **${rate:.3f}/hr**',
             f'- **Billed running time (this stretch):** {args.uptime_hours:.1f} h -> **${compute_cost:.2f}** compute',
             f'- **EBS storage:** {args.ebs_gb:.0f} GB gp3 -> ~${storage_cost:.2f} prorated for the stretch',
             f'- **Session total (compute + storage):** **~${compute_cost + storage_cost:.2f}**', '',
             '## Measured per-capability runs (from cost_log markers)', '']
    if measured:
        lines.append('| capability | model | instance | min | $ |')
        lines.append('|---|---|---|---|---|')
        for m in sorted(measured, key=lambda x: -x['cost_usd']):
            lines.append(f"| {m['capability']} | {m['model']} | {m['instance']} | "
                         f"{m['runtime_min']} | {m['cost_usd']} |")
    else:
        lines.append('_(no measured markers yet — populated as instrumented runs complete)_')
    open(args.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
