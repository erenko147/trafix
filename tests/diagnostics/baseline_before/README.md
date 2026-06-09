# BEFORE baseline (commit on branch fix/argmax-policy, old trafix_v6_final.pt)

Captured at Step 0, before any reward/training/governor change.

- `type1_low.txt`, `type1_medium.txt` — argmax_phase_distribution diagnostic
  reproducing the collapse (4-5 junctions lock onto one phase 69-97%).
- `high_traffic.txt` — must-not-regress reference on type1_high (full safety net):
  arrived 943, completion 97.6%, travel 103.1s, slow-queue 2.291, halting 2.133.
