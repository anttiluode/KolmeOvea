import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
CONDS = ['C1_regular_clock_cue_everywhere', 'C2_jittered_clock_cue_everywhere', 'C3_jittered_clock_cue_tuft_only']
FUN = ['acc', 'leak', 'unatt', 'cueflip']
DOORS = [('apical_clamp', 'cueflip'), ('basket_clamp', 'unatt'), ('basket_context_clamp', 'cueflip'), ('ais_open', 'leak')]


def damage(none, abl):
    return dict(acc=(none['acc'] - abl['acc']) / max(none['acc'] - 0.5, 1e-6),
                leak=abl['leak'] - none['leak'],
                unatt=(abl['unatt'] - none['unatt']) / 0.5,
                cueflip=(none['cueflip'] - abl['cueflip']) / max(none['cueflip'] - 0.5, 1e-6))


S = {}
for c in CONDS:
    rows = {m: [] for m in ['doors', 'vanilla', 'gru']}; abls = []
    for f in sorted(glob.glob(f'out/{c}__*.json')):
        R = json.load(open(f))
        for m, v in R['models'].items():
            rows[m] += v
        abls += R['doors_ablation']
    if not abls:
        continue
    S[c] = {'models': {m: {k: float(np.mean([r[k] for r in v])) for k in FUN} | {'n': len(v), 'min_acc': float(min([r['acc'] for r in v]))}
                       for m, v in rows.items() if v}}
    D = np.array([[[damage(a['ablations']['none'], a['ablations'][d])[f] for f in FUN] for d, _ in DOORS] for a in abls])
    S[c]['damage_mean'] = D.mean(0).tolist()
    surv = {}
    for i, (d, own) in enumerate(DOORS):
        j = FUN.index(own); ok = 0
        for s in range(len(abls)):
            dm = D[s, i]; others = [dm[k] for k in range(4) if k != j]
            if dm[j] > 0.1 and all(dm[j] > 2 * max(o, 0) for o in others):
                ok += 1
        surv[d] = f'{ok}/{len(abls)}'
    S[c]['survival'] = surv
    S[c]['admission'] = {k: float(np.mean([a['door_stats']['admission_by_mode_and_phase'][k] for a in abls]))
                         for k in abls[0]['door_stats']['admission_by_mode_and_phase']}
    S[c]['per_seed'] = [dict(seed=a['seed'], **{k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in a['ablations'].items()})
                        for a in abls]
    S[c]['e_hold'] = float(np.mean([a['door_stats']['e_hold'] for a in abls]))
    S[c]['e_probe'] = float(np.mean([a['door_stats']['e_probe'] for a in abls]))
json.dump(S, open('summary.json', 'w'), indent=1)
for c, v in S.items():
    print(c); print('  models', {m: {k: round(x, 3) for k, x in d.items()} for m, d in v['models'].items()})
    print('  survival', v['survival']); print('  admission', {k: round(x, 3) for k, x in v['admission'].items()})
    print('  e_hold', round(v['e_hold'], 3), 'e_probe', round(v['e_probe'], 3))
    print('  damage (rows apical/basket/basket-context/ais; cols acc/leak/unatt/cueflip)'); print(np.round(np.array(v['damage_mean']), 3))

fig, ax = plt.subplots(1, len(S) + 1, figsize=(5 * (len(S) + 1), 4.2))
for k, (c, v) in enumerate(S.items()):
    a = ax[k]; M = np.array(v['damage_mean'])
    im = a.imshow(M, cmap='Reds', vmin=0, vmax=1)
    for i in range(len(DOORS)):
        for j in range(4):
            a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9)
    a.set_xticks(range(4)); a.set_xticklabels(['acc', 'hold leak', 'unattended\nleak-through', 'cue\nswitching'], fontsize=8)
    a.set_yticks(range(len(DOORS))); a.set_yticklabels([f"apical clamp ({v['survival']['apical_clamp']})", f"basket clamp ({v['survival']['basket_clamp']})",
                                               f"basket top-down only ({v['survival']['basket_context_clamp']})", f"AIS open ({v['survival']['ais_open']})"], fontsize=8)
    a.set_title(c.replace('_', ' '), fontsize=9)
a = ax[-1]; x = np.arange(len(S)); w = .27
for i, m in enumerate(['doors', 'vanilla', 'gru']):
    a.bar(x + (i - 1) * w, [S[c]['models'].get(m, {}).get('acc', 0) for c in S], w, label=m)
a.set_xticks(x); a.set_xticklabels([c[:2] for c in S]); a.set_ylim(0.5, 1.0); a.legend(fontsize=8)
a.set_title('probe accuracy (mean of 5 seeds)', fontsize=9)
fig.suptitle('Damage caused by clamping each door after training (0 = none, 1 = function destroyed); (k/5) = seeds where the door survives', fontsize=10)
fig.tight_layout(); fig.savefig('doors_trained.png', dpi=110)
