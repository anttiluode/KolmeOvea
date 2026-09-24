"""
KolmeOvea v1 — do the three doors survive training?

Task (cue - hold - probe, one shared input channel):
  t = 0        CUE      mode m in {A, B}
  t = 1..L     HOLD     one +/-1 item per step on the SAME input line;
                        stream A = items on even steps, stream B = items on odd steps
                        the output must stay SILENT (target 0)
  t = L+1      PROBE    output sign( sum of the items of the cued stream )
  L ~ U{10..20}, so the probe time is not fixed and the network cannot count to it.

The task needs all three jobs: hold the mode (context), admit only the cued stream (timing),
integrate silently and speak only at the probe (emission).

DoorsRNN (every coefficient learned; the doors are architectural places, not hand-set values):
  tuft   a_{t+1} = rho*a_t + (1-rho)*tanh(A x_t)                       slow context state
  apical g_t     = sigmoid(G a_t + g0)                                  per-unit gain on input drive
  basket al_t    = sigmoid(Kc clock_t + Ka a_t + k0)                    per-unit admission; ONLY door that sees the clock
  soma   h_{t+1} = tanh(W h_t + g_t * al_t * (U x_t) + b)
  AIS    e_t     = sigmoid(E h_{t+1} + e0)                              emission gate
  output y_t     = e_t * tanh(w . h_{t+1})
Baselines get the clock as an ordinary input: vanilla tanh RNN, GRU.

Pre-registered survival test (written before running):
  Ablate each door after training by clamping it to its own data mean (the E2 "tonic, same mean"
  control) or, for the AIS, opening it (e = 1). Measure four functions:
    acc      probe accuracy
    leak     mean |y| during hold
    unatt    fraction of probes that flip when ALL unattended-stream items are flipped (ideal 0)
    cueflip  accuracy against the OTHER stream's answer when the cue is flipped (ideal 1)
  Predicted diagonal:  apical clamp -> cueflip collapses;  basket clamp -> unatt rises;
                       AIS open     -> leak rises while acc survives.
  A door "survives" if, in >= 4 of 5 seeds, its own predicted damage exceeds twice the damage
  it causes to each of the other measured functions (all on a common 0..1 damage scale).

python doors_trained.py
"""
import json, time
import numpy as np
import jax, jax.numpy as jnp
import optax

NX = 5          # cueA, cueB, item+, item-, probe
LMIN, LMAX = 10, 20
TMAX = LMAX + 2
D, DA = 32, 8


# ------------------------------------------------------------------------------------ data
IRREGULAR = False


def make_batch(rng, n, flip_unatt=False, flip_cue=False, base=None):
    if base is None:
        mode = rng.integers(0, 2, n)
        L = rng.integers(LMIN, LMAX + 1, n)
        if IRREGULAR:      # jittered rhythm: phase advances by a random amount each step
            phase = np.cumsum(np.c_[np.zeros(n), rng.uniform(0.55, 1.45, (n, TMAX - 1)) * np.pi], 1)
        else:              # period-2 rhythm: stream A on even steps
            phase = np.broadcast_to(np.pi * np.arange(TMAX), (n, TMAX)).copy()
        items = np.zeros((n, TMAX))
        for i in range(n):
            A = (np.mod(phase[i] + 1e-6, 2 * np.pi) < np.pi)
            hold_A = A[1:L[i] + 1].sum()
            while IRREGULAR and (hold_A < 3 or L[i] - hold_A < 3):   # jitter can starve a stream: redraw its phase
                phase[i] = np.cumsum(np.r_[0.0, rng.uniform(0.55, 1.45, TMAX - 1) * np.pi])
                A = (np.mod(phase[i] + 1e-6, 2 * np.pi) < np.pi); hold_A = A[1:L[i] + 1].sum()
            while True:
                it = rng.choice([-1.0, 1.0], L[i])
                full = np.zeros(TMAX); full[1:L[i] + 1] = it
                if full[A].sum() != 0 and full[~A].sum() != 0:
                    break
            items[i] = full
        base = dict(mode=mode, L=L, items=items, phase=phase)
    mode, L, items = base['mode'].copy(), base['L'], base['items'].copy()
    t = np.arange(TMAX); phase = base['phase']
    if flip_unatt:
        for i in range(len(mode)):
            streamA = np.mod(phase[i] + 1e-6, 2 * np.pi) < np.pi
            unatt = ~streamA if mode[i] == 0 else streamA
            items[i, unatt] *= -1
    x = np.zeros((len(mode), TMAX, NX)); y = np.zeros((len(mode), TMAX)); m = np.zeros((len(mode), TMAX))
    hold = np.zeros((len(mode), TMAX), bool); probe_idx = L + 1
    shown_mode = 1 - mode if flip_cue else mode
    for i in range(len(mode)):
        x[i, 0, shown_mode[i]] = 1
        x[i, 1:L[i] + 1, 2] = items[i, 1:L[i] + 1] > 0
        x[i, 1:L[i] + 1, 3] = items[i, 1:L[i] + 1] < 0
        x[i, probe_idx[i], 4] = 1
        streamA = np.mod(phase[i] + 1e-6, 2 * np.pi) < np.pi
        att = streamA if shown_mode[i] == 0 else ~streamA
        y[i, probe_idx[i]] = np.sign(items[i, att].sum())
        m[i, :probe_idx[i] + 1] = 1
        hold[i, 1:L[i] + 1] = True
    clock = np.stack([np.cos(phase), np.sin(phase)], -1)
    return dict(x=x, y=y, m=m, hold=hold, probe=probe_idx, clock=clock, base=base)


# ------------------------------------------------------------------------------------ models
def init_doors(key):
    ks = jax.random.split(key, 12); s = lambda k, *sh: jax.random.normal(k, sh) / np.sqrt(sh[-1])
    return dict(A=s(ks[0], DA, NX), rho=jnp.array(2.0), G=s(ks[1], D, DA), g0=jnp.zeros(D) + 1.0,
                Kc=s(ks[2], D, 2), Ka=s(ks[3], D, DA), k0=jnp.zeros(D) + 1.0,
                W=s(ks[4], D, D) * 0.9, U=s(ks[5], D, NX) * 2, b=jnp.zeros(D),
                E=s(ks[6], 1, D)[0], e0=jnp.array(0.0), w=s(ks[7], 1, D)[0])


CUE_TO_SOMA = True


def doors_forward(p, x, clock, clamp=None):
    """clamp: dict with optional 'g', 'al' (arrays of shape [D]) or 'e' (scalar) to overwrite a door."""
    clamp = clamp or {}
    rho = jax.nn.sigmoid(p['rho'])
    def step(carry, inp):
        h, a = carry; xt, ct = inp
        g = jax.nn.sigmoid(a @ p['G'].T + p['g0'])
        a_b = jnp.broadcast_to(clamp['a_basket'], a.shape) if 'a_basket' in clamp else a
        al = jax.nn.sigmoid(ct @ p['Kc'].T + a_b @ p['Ka'].T + p['k0'])   # clock + top-down context
        if 'g' in clamp: g = jnp.broadcast_to(clamp['g'], g.shape)
        if 'al' in clamp: al = jnp.broadcast_to(clamp['al'], al.shape)
        xs = xt if CUE_TO_SOMA else xt * jnp.array([0., 0., 1., 1., 1.])   # cue reaches the tuft only
        h = jnp.tanh(h @ p['W'].T + g * al * (xs @ p['U'].T) + p['b'])
        a = rho * a + (1 - rho) * jnp.tanh(xt @ p['A'].T)
        e = jax.nn.sigmoid(h @ p['E'] + p['e0'])
        if 'e' in clamp: e = jnp.full_like(e, clamp['e'])
        y = e * jnp.tanh(h @ p['w'])
        return (h, a), (y, g, al, e, a)
    n = x.shape[0]
    (_, _), (y, g, al, e, aa) = jax.lax.scan(step, (jnp.zeros((n, D)), jnp.zeros((n, DA))),
                                         (jnp.swapaxes(x, 0, 1), jnp.swapaxes(clock, 0, 1)))
    return jnp.swapaxes(y, 0, 1), dict(g=jnp.swapaxes(g, 0, 1), al=jnp.swapaxes(al, 0, 1), e=jnp.swapaxes(e, 0, 1),
                                       a=jnp.swapaxes(aa, 0, 1))


def init_vanilla(key, d=40):
    ks = jax.random.split(key, 3); s = lambda k, *sh: jax.random.normal(k, sh) / np.sqrt(sh[-1])
    return dict(W=s(ks[0], d, d) * 0.9, U=s(ks[1], d, NX + 2) * 2, b=jnp.zeros(d), w=s(ks[2], 1, d)[0])


def vanilla_forward(p, x, clock, clamp=None):
    xc = jnp.concatenate([x, clock], -1); d = p['W'].shape[0]
    def step(h, xt):
        h = jnp.tanh(h @ p['W'].T + xt @ p['U'].T + p['b'])
        return h, jnp.tanh(h @ p['w'])
    _, y = jax.lax.scan(step, jnp.zeros((x.shape[0], d)), jnp.swapaxes(xc, 0, 1))
    return jnp.swapaxes(y, 0, 1), {}


def init_gru(key, d=22):
    ks = jax.random.split(key, 4); s = lambda k, *sh: jax.random.normal(k, sh) / np.sqrt(sh[-1])
    return dict(Wz=s(ks[0], d, d + NX + 2), Wr=s(ks[1], d, d + NX + 2), Wh=s(ks[2], d, d + NX + 2),
                bz=jnp.zeros(d), br=jnp.zeros(d), bh=jnp.zeros(d), w=s(ks[3], 1, d)[0])


def gru_forward(p, x, clock, clamp=None):
    xc = jnp.concatenate([x, clock], -1); d = p['bz'].shape[0]
    def step(h, xt):
        hx = jnp.concatenate([h, xt], -1)
        z = jax.nn.sigmoid(hx @ p['Wz'].T + p['bz']); r = jax.nn.sigmoid(hx @ p['Wr'].T + p['br'])
        hh = jnp.tanh(jnp.concatenate([r * h, xt], -1) @ p['Wh'].T + p['bh'])
        h = (1 - z) * h + z * hh
        return h, jnp.tanh(h @ p['w'])
    _, y = jax.lax.scan(step, jnp.zeros((x.shape[0], d)), jnp.swapaxes(xc, 0, 1))
    return jnp.swapaxes(y, 0, 1), {}


MODELS = dict(doors=(init_doors, doors_forward), vanilla=(init_vanilla, vanilla_forward), gru=(init_gru, gru_forward))
nparams = lambda p: int(sum(np.prod(v.shape) if v.shape else 1 for v in p.values()))


# ------------------------------------------------------------------------------------ train / eval
def train(model, seed, steps=2500, bs=128, lr=3e-3):
    init, fwd = MODELS[model]
    p = init(jax.random.PRNGKey(seed)); rng = np.random.default_rng(1000 + seed)
    opt = optax.adam(lr); st = opt.init(p)

    def loss(p, x, c, y, m):
        yh, _ = fwd(p, x, c)
        return jnp.sum(m * (yh - y) ** 2) / jnp.sum(m)

    @jax.jit
    def upd(p, st, x, c, y, m):
        l, g = jax.value_and_grad(loss)(p, x, c, y, m)
        g = jax.tree_util.tree_map(lambda v: jnp.clip(v, -1, 1), g)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    curve = []
    for s in range(steps):
        B = make_batch(rng, bs)
        p, st, l = upd(p, st, B['x'], B['clock'], B['y'], B['m'])
        if s % 200 == 0:
            curve.append(float(l))
    return p, curve


def metrics(fwd, p, B, clamp=None):
    yh = np.array(fwd(p, B['x'], B['clock'], clamp)[0]); n = len(B['probe'])
    yp = yh[np.arange(n), B['probe']]; tgt = B['y'][np.arange(n), B['probe']]
    return dict(acc=float((np.sign(yp) == tgt).mean()), leak=float(np.abs(yh[B['hold']]).mean()), yp=yp)


def evaluate(model, p, seed, clamp=None):
    fwd = MODELS[model][1]; rng = np.random.default_rng(5000 + seed)
    B = make_batch(rng, 2000)
    Bu = make_batch(rng, 0, flip_unatt=True, base=B['base'])
    Bc = make_batch(rng, 0, flip_cue=True, base=B['base'])
    m0 = metrics(fwd, p, B, clamp); mu = metrics(fwd, p, Bu, clamp); mc = metrics(fwd, p, Bc, clamp)
    return dict(acc=m0['acc'], leak=m0['leak'],
                unatt=float((np.sign(m0['yp']) != np.sign(mu['yp'])).mean()),
                cueflip=mc['acc'])       # mc targets are computed for the SHOWN (flipped) cue


def door_means(p, seed):
    rng = np.random.default_rng(5000 + seed); B = make_batch(rng, 2000)
    _, d = doors_forward(p, B['x'], B['clock'])
    msk = B['m'][..., None] > 0
    g = (np.array(d['g']) * msk).sum((0, 1)) / msk.sum(); al = (np.array(d['al']) * msk).sum((0, 1)) / msk.sum()
    a_mean = (np.array(d['a']) * msk).sum((0, 1)) / msk.sum()
    e_hold = float(np.array(d['e'])[B['hold']].mean())
    e_probe = float(np.array(d['e'])[np.arange(2000), B['probe']].mean())
    al_arr = np.array(d['al']); hold = B['hold']
    modeA = (B['base']['mode'] == 0)[:, None].repeat(TMAX, 1)
    even = np.mod(B['base']['phase'] + 1e-6, 2 * np.pi) < np.pi
    admit = {}
    for name, sel in [('modeA_itemsA', modeA & even), ('modeA_itemsB', modeA & ~even),
                      ('modeB_itemsA', ~modeA & even), ('modeB_itemsB', ~modeA & ~even)]:
        admit[name] = float(al_arr[hold & sel].mean())
    # per-unit selectivity (population means hide opposite-signed units)
    u = {nm: al_arr[hold & sel].mean(0) for nm, sel in [('AA', modeA & even), ('AB', modeA & ~even),
                                                      ('BA', ~modeA & even), ('BB', ~modeA & ~even)]}
    admit['unit_phase_selectivity'] = float(np.mean(np.abs((u['AA'] + u['BA']) - (u['AB'] + u['BB'])) / 2))
    admit['unit_mode_selectivity'] = float(np.mean(np.abs((u['AA'] + u['AB']) - (u['BA'] + u['BB'])) / 2))
    admit['unit_mode_x_phase_attention'] = float(np.mean(np.abs((u['AA'] - u['AB']) - (u['BA'] - u['BB'])) / 2))
    return dict(g=g, al=al, a_mean=a_mean, e_hold=e_hold, e_probe=e_probe, admission_by_mode_and_phase=admit)


def run_condition(name, irregular, cue_to_soma, models, seeds=range(5), steps=2500):
    global IRREGULAR, CUE_TO_SOMA
    IRREGULAR, CUE_TO_SOMA = irregular, cue_to_soma
    R = {'condition': dict(name=name, irregular_clock=irregular, cue_to_soma=cue_to_soma), 'models': {}, 'doors_ablation': []}
    for model in models:
        R['models'][model] = []
        for seed in seeds:
            t0 = time.time()
            p, curve = train(model, seed, steps=steps)
            ev = evaluate(model, p, seed)
            R['models'][model].append(dict(seed=seed, params=nparams(p), curve=curve, **ev))
            print(name, model, seed, {k: round(v, 3) for k, v in ev.items()}, f'{time.time() - t0:.0f}s', flush=True)
            if model == 'doors':
                dm = door_means(p, seed)
                abl = {'none': ev,
                       'apical_clamp': evaluate('doors', p, seed, {'g': jnp.array(dm['g'])}),
                       'basket_clamp': evaluate('doors', p, seed, {'al': jnp.array(dm['al'])}),
                       'ais_open': evaluate('doors', p, seed, {'e': 1.0}),
                       'basket_context_clamp': evaluate('doors', p, seed, {'a_basket': jnp.array(dm['a_mean'])})}
                R['doors_ablation'].append(dict(seed=seed, door_stats={k: v for k, v in dm.items() if k not in ('g', 'al', 'a_mean')},
                                                ablations=abl))
                for k, v in abl.items():
                    print('   ', k, {kk: round(vv, 3) for kk, vv in v.items()}, flush=True)
                print('   ', {k: round(v, 3) for k, v in dm['admission_by_mode_and_phase'].items()},
                      'e_hold', round(dm['e_hold'], 3), 'e_probe', round(dm['e_probe'], 3), flush=True)
    tag = '_'.join(models) + '_s' + '-'.join(map(str, seeds))
    json.dump(R, open(f'out/{name}__{tag}.json', 'w'), indent=1)
    return R


CONDITIONS = {
    'C1_regular_clock_cue_everywhere': (False, True),
    'C2_jittered_clock_cue_everywhere': (True, True),
    'C3_jittered_clock_cue_tuft_only': (True, False),
}

if __name__ == '__main__':
    import sys
    name = sys.argv[1]; models = sys.argv[2].split(',')
    seeds = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(5))
    run_condition(name, *CONDITIONS[name], models, seeds=seeds)
