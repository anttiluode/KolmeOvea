# KolmeOvea — three doors on one pyramidal cell

EDIT: Added the V1 subfolder that asks: "KolmeOvea v1 — do the three doors survive training?"

A pyramidal cell has three separate places where it can be controlled, and recent anatomy and physiology give each one a distinct job:

| door | where | what it controls | papers behind it |
|---|---|---|---|
| **apical** | distal tuft | *which operator* the cell applies: context gates the output (BAC coincidence), without injecting content | Larkum 1999/2013 |
| **basket** | perisomatic | *when* input is admitted: rhythmic, fast, synchrony-sensitive inhibition opens and closes phase windows | Liu & Sun 2023, 2024 (FS basket spatial/temporal integration); Drebitz, Rausch & Kreiter 2025 (the same V2 volley matters only in the right V4 γ-phase) |
| **chandelier** | axon initial segment | *whether* the result leaves the cell, per output route: one ChC contacts hundreds of AISs and prefers some projection classes over others | Qi et al. 2024 review (BLA-projecting over callosal PNs; ChCs silent during sharp-wave ripples, rhythmic at theta) |

## The thesis this repo tests

**In a static scalar unit the three doors collapse into one knob.** NewMachine already proved it:

- A threshold shift equals input subtraction, exactly: 1[u − q > θ] = 1[u > θ + q].
- Its v2 found that two controller channels reduce to one signed scalar.

The doors only become *different computations* when the unit has three things a scalar unit lacks:

1. compartments, for the apical door;
2. time and phase, for the basket door;
3. more than one output route, for the chandelier door.

This repo builds one leaky integrate-and-fire microcircuit with all three (tuft compartment, basket rhythm, AIS threshold), then runs one experiment per door. Each experiment includes the replacement control that ought to fail.

Run `python kolme_ovea.py` (about 3 minutes, numpy only), then `python make_figure.py`. Results go to `results.json`.

## Results

### E1 — apical: context without content

Two cell groups get independent content streams at their basal dendrites. A context signal alternates every 500 ms, telling the readout which group to follow.

| context delivered as | selectivity, corr(selected) − corr(other) | readout when content is absent (Hz) |
|---|---|---|
| **tuft (BAC gate, readout = bursts)** | **0.81** | **0.0** |
| none | 0.01 | 0.0 |
| soma +4 mV | 0.25 | 0.0 |
| soma +8 mV | 0.13 | 7.1 |
| soma +12 mV | 0.12 | 26.1 |
| soma +16 mV | 0.10 | 40.8 |

Additive context can't escape a trade-off: weak context barely selects, and strong context leaks, so the cell fires on context alone and reports content that isn't there. Tuft gating gets both high selectivity and zero leak, because tuft input alone cannot drive the soma. It changes *how* content is processed, not *what* the content is.

### E2 — basket: selection by arrival phase

Two streams hit the **same synapses with the same weights**, so no gain or branch mechanism can tell them apart. They differ only in when each gamma cycle they arrive.

| condition | corr(output, A) | corr(output, B) |
|---|---|---|
| basket rhythm, A arrives late in the cycle | **0.92** | 0.30 |
| same, A and B retimed by about 14 ms | 0.32 | **0.91** |
| tonic inhibition with the same mean | 0.59 | 0.70 |

- **Retiming is enough.** Shifting arrival by about 14 ms flips which stream the cell follows. Nothing about weights or wiring changed.
- **Tonic inhibition of equal mean cannot select.** The output tracks a mixture of both streams.

**Single-volley ping** (the Drebitz protocol, in silico): one extra 15-spike volley at every phase, compared against an identical-noise run without it.

- With the rhythm, the extra response ranges from about 0 to 31 spikes depending on phase (max/min ≈ 376). The window is open (above half its maximum) for **32% of the cycle**.
- With tonic inhibition the response is flat (max/min 1.6).

### E3 — chandelier: route-specific silence that keeps computing

Route A cells, the ChC-preferred class, are muted for 500 ms. Halfway through the mute, their input changes. The same cells are muted either at the AIS (chandelier) or perisomatically (basket, clamped on the same cells so the comparison is fair).

| | chandelier (AIS block) | basket (perisomatic shunt) |
|---|---|---|
| route A rate during mute | 0 Hz | 0 Hz |
| route B | unchanged | unchanged |
| soma V at release (unmuted value −43 mV) | **−43.0 mV** | −67.0 mV |
| held-state slope dV/d(input) | **1.00** | 0.25 = 1/(1+g) |
| output resumes after release | **0 ms** | 15 ms |

- **Chandelier:** the silenced cells keep integrating. At release their state already reflects the input change that happened while they were muted.
- **Basket:** the state is shunted divisively, so release starts from scratch.

In both cases the released population fires in synchrony, because every cell starts from the same state. So the difference is not synchrony. It is *what the synchronous release carries*. That is the computational reading of ChCs going silent during sharp-wave ripples: release a population that kept integrating while silent.

## What this changes in the old repos

Most of the last months' repos were each probing *one* door, often without the axis that makes it distinct.

- **Apical door (context selects the operator):**
  - NotSoSimpleNeuron (operator-valued weights) and OperatorTime (resident state changes the effective operator).
  - AnttisNeuron's branch modes and FusionMachine (select a computation at a nonlinear boundary).
  - EATON (transient operators) and AdaptiveObserverCache (persistent observer steering how fixed K/V are read).
  - SilentPing (resident state changes what a neutral ping meets).
- **Basket door (when input is admitted):**
  - The PerceptionLab ECG loop. Its spike was an observability collapse, which is an admission window failing.
  - GeoNeuronX: "length is a temporal coordinate" is exactly arrival phase, and E2 shows phase is a routing variable.
  - SpectralNeuron's band-selective readout and the WidePresent / temporal-matrix windows.
- **Chandelier door (whether and where the result is published):**
  - ActiveVectorNN (transmit innovations only) and NewMachine (the publication gate).
  - The axon-output notes (analog-digital facilitation, route-specific output).
  - **NewMachine's verdict, that the second gate barely helped, is explained here, not overturned.** Its unit had one route and no phase, so the publication gate could only duplicate state repair. E3's state preservation reproduces NewMachine v0's scar result. The new parts are the per-route targeting and the held state that keeps tracking its input.
- **Instruments:** SilentPing, PingToWord, ResidentOperatorTomography, GATGRILS, and Norman-Haignere's window method. All are ways of reading these doors from outside.

## Ledger

- **Every result is a known-answer demonstration of the mechanism as built.**
  - Additive context cannot gate.
  - Shunting is divisive (the 0.25 is exactly 1/(1+g) with g = 3).
  - Phase gating follows from rhythmic inhibition.
  - What the repo contributes is putting the three doors in one circuit, with the replacement control for each.
- **Modelling choices:**
  - **The rhythm is imposed.** A pacemaker drives the basket volley every 25 ms. It is not an emergent E–I gamma (PING) oscillation.
  - **The 32% open window comes from these parameters.** It is not fitted to Drebitz's roughly 90° window.
  - **The chandelier door is a pure AIS threshold block.** Real axo-axonic GABA can be shunting, and in some conditions depolarizing (the review cites Woodruff 2009 and bidirectional AIS control).
  - **Basket inhibition was clamped class-specifically in E3,** to make the comparison fair. Biological basket cells target spatial ensembles, not projection classes.
- **E2 ping curve artifact:** the point at 0.5 ms phase is an update-order artifact. The volley kick lands in the same step as the basket jump, before the shunt acts.
- **Reading notes on the sources:**
  - The two Liu & Sun papers are simulations of one reconstructed CA3 cell, not recordings.
  - They also state opposite spatial preferences: the 2023 abstract says FS basket cells prefer *dispersed* input, while the 2024 introduction cites it as preferring *clustered* input.
  - Drebitz et al. ran the phase analyses only on sessions where stimulation increased misses.
- **Not here:** learning, hierarchy, and time-yoked windows across levels.

## Next

1. Put the three doors into a small *trained* recurrent network on a cue–hold–probe task (GATGRILS v1's world).
2. Use Norman-Haignere's context-invariance method to measure each level's integration window.
3. Run the replacement controls there, where nobody hand-set the coefficients.
