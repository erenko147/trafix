# BEFORE baseline (commit on branch fix/argmax-policy, old trafix_v6_final.pt)

Captured at Step 0, before any reward/training/governor change.

- `type1_low.txt`, `type1_medium.txt` — argmax_phase_distribution diagnostic
  reproducing the collapse (4-5 junctions lock onto one phase 69-97%).
- `high_traffic.txt` — must-not-regress reference on type1_high (full safety net):
  arrived 943, completion 97.6%, travel 103.1s, slow-queue 2.291, halting 2.133.

## Update (Step 5 governor change)

The diagnostic now defaults to the SHIPPED governor `pressure_thresh=0.12`
(lowered from 0.35 in Step 5). `type1_low.txt` / `type1_medium.txt` were
regenerated at 0.12 so the eventual AFTER (retrained model, also 0.12) is an
apples-to-apples comparison. The collapse persists at 0.12 (still LOCKED) —
governor tuning alone does NOT fix it; the retrained policy (Steps 1-3) does.
The original-production collapse table (at 0.35) is the one quoted in
minecraft.txt's problem statement.
