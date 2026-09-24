# KolmeOvea v1 — do the three doors survive training?

**Short answer:** only the output door (AIS) takes on its job by itself. The basket door takes on a job only when it is the sole route to the information that job needs, and even then training picks the cheapest version of the job. The apical door barely gets used at all.

Run `python doors_trained.py <condition> <model> <seed>` (JAX + optax), or `./runall.sh` for everything. Then run `python analyze.py`, which writes `summary.json` and `doors_trained.png`. Per-seed receipts are in `out/`.

## Task: cue, hold, probe on one shared input line

- **Cue** (t = 0): which stream to follow, A or B.
- **Hold** (t = 1..L, with L random in 10–20): one ±1 item per step on the *same* input line. An item belongs to stream A or B depending on the gamma phase it arrives in. The output must stay **silent** during the hold.
- **Probe** (t = L + 1): the output must give the sign of the sum of the cued stream's items.

## Model: DoorsRNN, every coefficient learned

| door | equation | what it can see |
|---|---|---|
| tuft | a ← ρ·a + (1−ρ)·tanh(A x) | the inputs |
| apical | g = σ(G a + g0), a per-unit gain on the input drive | the tuft |
| basket | α = σ(Kc·clock + Ka·a + k0), per-unit admission | the **clock** (the only door that sees it) and the tuft (top-down) |
| soma | h ← tanh(W h + g·α·(U x) + b) | the gated inputs |
| AIS | e = σ(E h + e0), output y = e·tanh(w·h) | the soma |

The baselines are a vanilla tanh RNN and a GRU, with matched parameter counts (about 1,960), and the clock given as an ordinary input.

## Three conditions

The conditions differ in *who has access to what*:

- **C1:** the clock is regular (period 2), and the cue reaches both the tuft and the soma. The soma can count the phase itself.
- **C2:** the clock is jittered (random phase advance each step), so the phase cannot be counted. The cue still reaches both tuft and soma.
- **C3:** jittered clock, and the cue reaches **only the tuft**. The mode can then reach the soma only through the apical gain or through the basket's top-down input.

## Ablations, run after training

Each ablation removes one door's contribution:

- **Apical clamp:** the apical gain is fixed at its data mean.
- **Basket clamp:** the admission gate is fixed at its data mean. This is the "tonic inhibition, same mean" control.
- **Basket top-down only:** the clock still drives the basket, but its tuft input is replaced by the mean tuft state.
- **AIS open:** the output gate is fixed at e = 1.

Each ablation is scored on four functions: probe accuracy, hold leak, unattended-stream leak-through (flip all the unattended items; how many probe answers flip?), and cue switching (flip the cue; does the answer follow?).

## Results (5 seeds per condition)

**Accuracy.** The doors model matches the baselines on the task. It doesn't beat them:

| | doors | vanilla | GRU |
|---|---|---|---|
| C1 | 1.000 | 0.994 | 1.000 |
| C2 | 0.970 | 0.977 | 0.970 |
| C3 | 0.906 (one seed failed at 0.756; the others 0.93–0.95) | same as C2* | same as C2* |

\*The baselines always see the cue, so C3 changes nothing for them.

**Hold leak.** The doors model is 3–7× lower (0.002–0.008, against 0.011–0.014 for the baselines). That is the AIS door at work.

**Damage from each ablation** (mean over seeds; 0 = no damage, 1 = function destroyed), shown as acc / hold leak / unattended leak-through / cue switching:

| | C1 | C2 | C3 |
|---|---|---|---|
| apical clamp | 0 / 0 / 0 / 0 | .03 / 0 / .05 / .03 | .07 / 0 / .10 / .07 |
| basket clamp | 0 / 0 / 0 / 0 | **.48 / 0 / .90 / .47** | **.40 / 0 / .62 / .38** |
| basket top-down only | 0 / 0 / 0 / 0 | .03 / 0 / .04 / .04 | **.26 / 0 / .42 / .26** |
| AIS open | 0 / **.74** / 0 / 0 | 0 / **.65** / 0 / 0 | 0 / **.56** / 0 / 0 |

What each door learned, per unit:

| | C1 | C2 | C3 |
|---|---|---|---|
| basket phase selectivity | 0.15 | 0.29 | 0.21 |
| basket mode selectivity | 0.007 | 0.021 | 0.055 |
| basket mode × phase (attention) | 0.002 | 0.005 | 0.017 |
| AIS gate during hold / at probe | .004 / .995 | .004 / .995 | .008 / .942 |

### Reading it

1. **The AIS door specializes in every condition and every seed.**
   - It shuts during the hold and opens at the probe.
   - Opening it breaks only the silence; accuracy, selection and cue switching are untouched.
   - This is the chandelier signature from KolmeOvea v0 (silence with the state intact), found by gradient descent rather than built in. It is also the only door whose benefit shows up against the baselines, as the lower hold leak.
2. **The basket door works only when access forces it.**
   - In C1 the soma counts the phase itself, and clamping the basket door changes nothing.
   - In C2, where only the basket door can know the phase, it becomes necessary: clamping it lets the unattended stream through (damage 0.90).
   - But what it learned in C2 is **phase tagging, not attention**. Its units admit A-phase and B-phase items differently (selectivity 0.29), yet treat both modes almost identically (0.021). The soma still does the selecting.
3. **Attention by top-down retiming appears only when context has nowhere else to go.**
   - In C3 the cue reaches only the tuft. Clamping just the basket's top-down input (with the clock left intact) damages selection in all 4 seeds that learned the task: accuracy 0.93 → 0.73, 0.95 → 0.88, 0.95 → 0.79 and 0.95 → 0.80.
   - So the mode travels mostly through the basket's context input. That is attention implemented as mode-dependent admission, the Fries/Drebitz story, emerging from training.
   - Per unit that modulation is small (0.017), so recurrence must amplify it.
4. **The apical door stays marginal.** The largest effect is one C3 seed, with accuracy 0.95 → 0.84. When both a gain route and an admission route can carry context, training mostly chooses admission.

**Overall:** the doors don't specialize because the architecture has them; they specialize when the anatomy of access leaves them the only route. That is a claim about why biology might wire things this way: top-down input lands in layer 1, the rhythm comes from perisomatic basket input, and the soma lacks direct access to either. Access, not architecture, defines the doors.

## Ledger

- **Pre-registered survival rule: AIS 5/5 in every condition; apical, basket and basket top-down 0/5 everywhere.**
  - The rule required a door's own damage to exceed twice its damage to every other function.
  - It did not anticipate that accuracy and cue switching are *downstream* of stream selection. A basket door that is clearly necessary (damage 0.90 on its own function in C2) therefore fails the rule, because it also costs 0.48 accuracy.
  - This is a flaw in the pre-registration. I report the rule's verdict unchanged. The reading above is post hoc and labelled as such.
- **Two bugs, both found and fixed before these results:**
  - A floating-point phase-labelling bug: `np.mod(π·t, 2π)` fell just below 2π for even t. It mislabelled items and capped every model near 0.955. The first, partial C1 run in `run_log_condition1_partial.txt` predates the refactor that introduced it.
  - An infinite loop in the jittered-clock generator when a stream received no items. Starved trials are now redrawn so that each stream gets at least 3 items.
- **C3 seed 1 never learned the task** (0.756). It is kept in all means.
- **The clock is given, not generated.** No emergent gamma here.
- **The chandelier/AIS door gates a single output route.** KolmeOvea v0's per-route selectivity isn't tested here.
- **Small networks and short training** (32 hidden units, 2,500 steps). The door usage could differ at scale.

## Next

1. Fix the survival rule: score each door against the functions that are *not* downstream of it, pre-registered this time.
2. Give the AIS door two output routes, only one of which must stay silent. This tests v0's route selectivity under training.
3. Test the access claim directly: in C3, give the soma a weak, noisy copy of the cue and sweep its reliability. Record the point at which context switches from the soma to the basket's top-down route.
