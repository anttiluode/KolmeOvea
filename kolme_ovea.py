"""
KolmeOvea ("three doors") — one pyramidal microcircuit with three separate control surfaces.

    APICAL tuft   : which operator the cell applies        (context gates, does not inject content)
    BASKET soma   : when input is admitted                  (rhythmic perisomatic inhibition -> phase windows)
    CHANDELIER AIS: whether the result is emitted, per route (spike initiation blocked, state untouched)

Thesis tested here: in a static scalar unit these three collapse into one knob (NewMachine v0/v2
already proved threshold shift == input subtraction). They only become DIFFERENT computations when
the unit has (i) compartments, (ii) time/phase, (iii) more than one output route. One experiment per
condition, each with the replacement control that should fail:

  E1 apical    vs additive context          -> content leak when content is absent
  E2 basket    rhythm vs tonic inhibition   -> select between streams that share the SAME synapses,
                                               by arrival phase only; retiming flips the selection;
                                               single-volley ping gives the admission window (Drebitz 2025)
  E3 chandelier vs basket muting            -> route-specific silence with the soma state preserved;
                                               release produces a synchronous volley (SWR-like)

Leaky integrate-and-fire, dt = 0.1 ms, numpy only.   python kolme_ovea.py
"""
import json
import numpy as np

DT = 0.1                                         # ms
EL, THETA, VRESET, EINH = -70.0, -50.0, -60.0, -75.0
TAU_M, TAU_A, TAU_GB, TREF = 15.0, 20.0, 4.0, 2.0
KAPPA = 0.05                                     # tuft -> soma coupling (tuft alone barely moves soma)
THETA_AP = 10.0                                  # tuft depolarisation (mV above rest) needed for BAC burst
BURST = 3                                        # spikes delivered downstream by a BAC burst


class Pyramids:
    def __init__(self, n, rng):
        self.n = n
        self.V = EL + rng.uniform(0, 15, n)
        self.Va = np.full(n, EL)
        self.refr = np.zeros(n)
        self.gb = np.zeros(n)                    # basket (perisomatic) conductance, units of leak
        self.dtheta = np.zeros(n)                # chandelier: AIS threshold raise

    def step(self, I_bas, I_ap=0.0, kick=0.0):
        self.Va += DT / TAU_A * (-(self.Va - EL) + I_ap)
        self.V += DT / TAU_M * (-(self.V - EL) + I_bas + KAPPA * (self.Va - EL) - self.gb * (self.V - EINH))
        self.V += kick                           # instantaneous synaptic kicks (mV)
        self.gb *= np.exp(-DT / TAU_GB)
        self.refr -= DT
        emit = (self.V > THETA + self.dtheta) & (self.refr <= 0)   # AIS decides; blocked -> no reset
        burst = emit & (self.Va - EL > THETA_AP)
        self.V[emit] = VRESET
        self.refr[emit] = TREF
        return emit, burst


def ou(rng, n_steps, tau_ms, lo, hi):
    x = np.zeros(n_steps); a = np.exp(-DT / tau_ms); s = np.sqrt(1 - a * a)
    for t in range(1, n_steps):
        x[t] = a * x[t - 1] + s * rng.standard_normal()
    return lo + (hi - lo) * (x - x.min()) / (x.max() - x.min() + 1e-12)


# ----------------------------------------------------------------------------------------------
# E2  basket: admission windows
# ----------------------------------------------------------------------------------------------
PERIOD = 25.0                                    # ms, 40 Hz gamma, imposed via the basket population


def run_e2(seed, phase_A, phase_B, rhythm=True, cycles=240, n=60, ping_times=(), stream_on=True,
           noise_seed=None, gb_tonic=None):
    """Receiver population + basket rhythm + two afferent streams on the SAME synapses."""
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(noise_seed if noise_seed is not None else seed + 1)
    T = int(cycles * PERIOD / DT)
    P = Pyramids(n, rng)
    I0 = 17.0
    noise = nrng.standard_normal((T, n)) * 25.0      # current noise -> ~1.4 mV membrane jitter
    sA = rng.uniform(0.2, 1.0, cycles); sB = rng.uniform(0.2, 1.0, cycles)
    W = 0.35                                     # mV per afferent spike, identical for both streams
    out_per_cycle = np.zeros(cycles); spikes_t = np.zeros(T)
    gb_trace = np.zeros(T)
    ping_steps = {int(t / DT) for t in ping_times}
    for t in range(T):
        tm = t * DT; k = int(tm // PERIOD); ph = tm - k * PERIOD
        if rhythm:
            if abs(ph) < DT / 2:                 # basket volley at phase 0 each cycle
                P.gb += 1.6
        else:
            P.gb[:] = gb_tonic
        kick = 0.0
        if stream_on:
            if abs(ph - phase_A) < DT / 2:
                kick += W * round(20 * sA[k])
            if abs(ph - phase_B) < DT / 2:
                kick += W * round(20 * sB[k])
        if t in ping_steps:
            kick += W * 15
        emit, _ = P.step(I0 + noise[t], kick=kick)
        c = emit.sum(); out_per_cycle[k] += c; spikes_t[t] = c; gb_trace[t] = P.gb.mean()
    return dict(out=out_per_cycle, sA=sA, sB=sB, spikes_t=spikes_t, gb_mean=gb_trace.mean())


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def e2():
    R = {}
    early, late = 5.0, 19.0
    base = run_e2(1, late, early)                # A late (excitable), B early (just after basket volley)
    flip = run_e2(1, early, late)                # retime: shift A and B by ~14 ms, nothing else changes
    tonic = run_e2(1, late, early, rhythm=False, gb_tonic=base['gb_mean'])
    for name, r in [('rhythm_A_late', base), ('rhythm_A_early_retimed', flip), ('tonic_inhibition_same_mean', tonic)]:
        R[name] = dict(corr_out_A=corr(r['out'], r['sA']), corr_out_B=corr(r['out'], r['sB']),
                       mean_spikes_per_cycle=float(r['out'].mean()))
    # admission window: single volley pings at every phase, paired with identical noise without the ping
    phases = np.arange(0.5, PERIOD, 1.0)
    cycles = 60
    rng = np.random.default_rng(7)
    curve, curve_tonic = [], []
    for rhythm, store in [(True, curve), (False, curve_tonic)]:
        for ph in phases:
            ks = rng.choice(np.arange(5, cycles - 1), 12, replace=False)
            times = [k * PERIOD + ph for k in ks]
            kw = dict(cycles=cycles, stream_on=False, noise_seed=99, rhythm=rhythm, gb_tonic=base['gb_mean'])
            a = run_e2(3, 0, 0, ping_times=times, **kw)['spikes_t']
            b = run_e2(3, 0, 0, ping_times=(), **kw)['spikes_t']
            extra = np.mean([(a - b)[int(tt / DT):int((tt + 6) / DT)].sum() for tt in times])
            store.append(float(extra))
    R['admission_window'] = dict(phase_ms=phases.tolist(), extra_spikes_rhythm=curve, extra_spikes_tonic=curve_tonic)
    c = np.array(curve)
    R['admission_window']['open_fraction_of_cycle'] = float((c > 0.5 * c.max()).mean())
    R['admission_window']['max_over_min_rhythm'] = float(c.max() / max(c.min(), 1e-3))
    ct = np.array(curve_tonic)
    R['admission_window']['max_over_min_tonic'] = float(ct.max() / max(ct.min(), 1e-3))
    return R


# ----------------------------------------------------------------------------------------------
# E1  apical: context selects the operator without injecting content
# ----------------------------------------------------------------------------------------------
def run_e1(mode, seed=11, T_ms=12000, n=30, basal_on=True, add_mv=6.0):
    rng = np.random.default_rng(seed)
    T = int(T_ms / DT)
    G = Pyramids(2 * n, rng)
    sA = ou(rng, T, 150.0, 0.0, 1.0); sB = ou(rng, T, 150.0, 0.0, 1.0)
    ctx = ((np.arange(T) * DT) // 500).astype(int) % 2           # which group context selects
    noise = rng.standard_normal((T, 2 * n)) * 4.0
    read = np.zeros(T)
    for t in range(T):
        drive = np.r_[np.full(n, sA[t]), np.full(n, sB[t])] if basal_on else np.zeros(2 * n)
        I_bas = 12.0 + 14.0 * drive + noise[t]
        sel = np.r_[np.full(n, ctx[t] == 0), np.full(n, ctx[t] == 1)]
        I_ap = np.where(sel, 16.0, 0.0) if mode == 'apical' else 0.0
        if mode == 'additive':
            I_bas = I_bas + np.where(sel, add_mv, 0.0)              # same context, delivered to the soma
        emit, burst = G.step(I_bas, I_ap)
        read[t] = burst.sum() * BURST if mode == 'apical' else emit.sum()
    b = int(50 / DT); nb = T // b
    rb = read[:nb * b].reshape(nb, b).sum(1)
    sel_sig = np.where(ctx == 0, sA, sB)[:nb * b].reshape(nb, b).mean(1)
    oth_sig = np.where(ctx == 0, sB, sA)[:nb * b].reshape(nb, b).mean(1)
    return dict(read_rate_hz=float(read.sum() / (2 * n) / (T_ms / 1000)),
                corr_selected=corr(rb, sel_sig) if basal_on else None,
                corr_other=corr(rb, oth_sig) if basal_on else None)


def e1():
    R = {}
    runs = [('apical', 0.0), ('none', 0.0)] + [('additive', d) for d in (4.0, 8.0, 12.0, 16.0)]
    for mode, d in runs:
        on = run_e1(mode, add_mv=d)
        off = run_e1(mode, basal_on=False, add_mv=d)
        name = mode if mode != 'additive' else f'additive_{int(d)}mV'
        R[name] = dict(corr_selected=on['corr_selected'], corr_other=on['corr_other'],
                       selectivity=on['corr_selected'] - on['corr_other'],
                       readout_rate_with_content=on['read_rate_hz'],
                       readout_rate_content_absent=off['read_rate_hz'])
    return R


# ----------------------------------------------------------------------------------------------
# E3  chandelier: route-specific silence, state preserved, synchronous release
# ----------------------------------------------------------------------------------------------
def run_e3(mode, seed=21, n=30, new_level=27.0):
    rng = np.random.default_rng(seed)
    T_ms, t_on, t_off = 1500.0, 500.0, 1000.0
    T = int(T_ms / DT)
    P = Pyramids(2 * n, rng)                     # first n = route A (chandelier-preferred), last n = route B
    noise = rng.standard_normal((T, 2 * n)) * 4.0
    level = np.where(np.arange(T) * DT < 750.0, 20.0, new_level)       # information changes DURING the mute
    spikes = np.zeros((T, 2))
    V_at_release = None
    A = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    for t in range(T):
        tm = t * DT
        muted = t_on <= tm < t_off
        P.dtheta[:] = 0.0
        if mode == 'chandelier' and muted:
            P.dtheta[A] = 40.0                   # AIS: initiation blocked, soma untouched
        if mode == 'basket' and muted:
            P.gb[A] = 3.0                        # perisomatic shunt clamped on route A (same target set)
        if abs(tm - t_off) < DT / 2:
            V_at_release = P.V[A].copy()          # state just before the release step
        emit, _ = P.step(level[t] + noise[t])
        spikes[t] = [emit[A].sum(), emit[~A].sum()]
    ms = lambda a, b: slice(int(a / DT), int(b / DT))
    rate = lambda s, sl, cnt: float(s[sl].sum() / cnt / ((sl.stop - sl.start) * DT / 1000))
    ref = rate(spikes[:, 0], ms(1200, 1500), n)                   # steady state at the new level
    first5 = spikes[ms(t_off, t_off + 5), 0].sum() / n
    base5 = spikes[ms(1200, 1500), 0].sum() / n / 60              # expected spikes per cell per 5 ms
    # time for route A 10-ms rate to reach 80% of its new steady state
    k = int(10 / DT); lat = None
    for j in range(int(t_off / DT), T - k, int(1 / DT)):
        if spikes[j:j + k, 0].sum() / n / 0.010 >= 0.8 * ref:
            lat = (j * DT - t_off); break
    return dict(routeA_rate_muted=rate(spikes[:, 0], ms(t_on + 20, t_off), n),
                routeB_rate_muted=rate(spikes[:, 1], ms(t_on + 20, t_off), n),
                routeB_rate_unmuted=rate(spikes[:, 1], ms(100, t_on), n),
                frac_routeA_above_threshold_at_release=float((V_at_release > THETA).mean()),
                mean_V_routeA_at_release=float(V_at_release.mean()),
                unmuted_steady_V_for_new_level=EL + new_level,
                release_spikes_per_cell_first5ms=float(first5),
                release_volley_vs_steady_state=float(first5 / max(base5, 1e-9)),
                latency_to_80pct_new_rate_ms=lat,
                raster_A=spikes[:, 0], raster_B=spikes[:, 1])


def e3():
    R = {}
    for mode in ['chandelier', 'basket']:
        r = run_e3(mode)
        R[mode] = {k: v for k, v in r.items() if not k.startswith('raster')}
        np.save(f'e3_{mode}_routeA.npy', r['raster_A'])
        # does the silenced soma still TRACK the input that changed during the mute?
        lv = [18.0, 22.0, 27.0, 32.0]
        Vr = [run_e3(mode, new_level=L)['mean_V_routeA_at_release'] for L in lv]
        R[mode]['held_state_slope_dV_per_dInput'] = float(np.polyfit(lv, Vr, 1)[0])
        R[mode]['V_at_release_by_level'] = dict(zip(map(str, lv), Vr))
    return R


def main():
    R = {'E1_apical': e1()}
    print('E1', json.dumps(R['E1_apical'], indent=1))
    R['E2_basket'] = e2()
    print('E2', json.dumps({k: v for k, v in R['E2_basket'].items() if k != 'admission_window'}, indent=1))
    aw = R['E2_basket']['admission_window']
    print('admission', {k: aw[k] for k in ['open_fraction_of_cycle', 'max_over_min_rhythm', 'max_over_min_tonic']})
    R['E3_chandelier'] = e3()
    print('E3', json.dumps(R['E3_chandelier'], indent=1))
    json.dump(R, open('results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
