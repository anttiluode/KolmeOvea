#!/bin/bash
cd /home/claude/KolmeOveaTrained
for c in C2_jittered_clock_cue_everywhere C3_jittered_clock_cue_tuft_only C1_regular_clock_cue_everywhere; do
  for m in doors vanilla gru; do
    [ "$c" = "C3_jittered_clock_cue_tuft_only" ] && [ "$m" != "doors" ] && continue
    for s in 0 1 2 3 4; do
      [ -f out/${c}__${m}_s${s}.json ] && continue
      python3 doors_trained.py $c $m $s >> run_all.log 2>&1
    done
  done
done
echo DONE >> run_all.log
