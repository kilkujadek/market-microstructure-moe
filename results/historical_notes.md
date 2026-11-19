# Historical observations: separate from the synthetic demo

These are notes from earlier internal experiments. The original execution
logs, configurations and NautilusTrader reports are **not** included here;
the figures are **not independently reproduced by this repository**.

| Experiment | Recorded observation | What can and cannot be concluded |
| --- | --- | --- |
| V15 Dual MoE, step 128 | Cumulative result **-$285.56** | A losing checkpoint. The suspected role of overtrading and fees needs an original-log attribution. |
| V15 Dual MoE, step 640 | **+$36.25 on WGRX** in one ticker-day | An isolated positive day, not overall profitability. |
| V13, step 103212 | Best B ticker-day **+$8.85** | Selected best case, not a representative out-of-sample outcome. |

**These different scopes must not be joined into a fake learning curve.**

## Why the strategy research matters

One of the hardest problems is distinguishing a statistical pattern from a
tradable edge. More trades can turn seemingly useful signals into losses after
fees, spread and poor fills. Optimizing on a fixed collection of ticker-days
can select models that will not generalize to untouched sessions.

Next checks: original execution trace, fees by order, passivity and expiry of
LIMIT orders, strict chronological partitions and independent held-out days,
then repeated event-driven evaluation in NautilusTrader. This tiny synthetic
repository does **not** claim those tests were already completed.
