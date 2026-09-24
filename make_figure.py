import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('results.json')); DT = 0.1
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

a = ax[0, 0]; E1 = R['E1_apical']
for name, v in E1.items():
    c = '#2a6' if name == 'apical' else ('#999' if name == 'none' else '#c64')
    a.scatter(v['readout_rate_content_absent'], v['selectivity'], s=80, c=c)
    a.annotate(name.replace('additive_', 'soma +'), (v['readout_rate_content_absent'], v['selectivity']),
               textcoords='offset points', xytext=(6, 4), fontsize=8)
a.set_xlabel('readout rate when content is ABSENT, context present (Hz)  = leak')
a.set_ylabel('selectivity  corr(selected) - corr(other)')
a.set_title('E1 apical: context selects without injecting content\nsoma-additive context trades selectivity for leak', fontsize=9)

a = ax[0, 1]; E2 = R['E2_basket']; ks = ['rhythm_A_late', 'rhythm_A_early_retimed', 'tonic_inhibition_same_mean']
x = np.arange(3)
a.bar(x - .2, [E2[k]['corr_out_A'] for k in ks], .4, label='corr(output, stream A)')
a.bar(x + .2, [E2[k]['corr_out_B'] for k in ks], .4, label='corr(output, stream B)')
a.set_xticks(x); a.set_xticklabels(['rhythm, A late', 'rhythm, A retimed early', 'tonic, same mean'], fontsize=8)
a.set_ylim(0, 1); a.legend(fontsize=8)
a.set_title('E2 basket: two streams on the SAME synapses\narrival phase alone selects; a ~14 ms retiming flips it', fontsize=9)

a = ax[1, 0]; aw = E2['admission_window']
a.plot(aw['phase_ms'], aw['extra_spikes_rhythm'], 'o-', label='basket rhythm (40 Hz)')
a.plot(aw['phase_ms'], aw['extra_spikes_tonic'], 's-', label='tonic inhibition, same mean')
a.axvline(0, c='k', lw=.5); a.set_xlabel('arrival phase of a single volley after the basket volley (ms)')
a.set_ylabel('extra population spikes within 6 ms'); a.legend(fontsize=8)
a.set_title(f"E2 single-volley ping (the Drebitz 2025 protocol, in silico)\nopen for {aw['open_fraction_of_cycle']:.0%} of the cycle", fontsize=9)

a = ax[1, 1]
for mode, c in [('chandelier', '#36c'), ('basket', '#c44')]:
    s = np.load(f'e3_{mode}_routeA.npy'); k = int(1 / DT)
    r = s[:len(s) // k * k].reshape(-1, k).sum(1) / 30
    a.plot(np.arange(len(r)) * 1, r, c=c, label=f"route A muted by {mode} (held-state slope {R['E3_chandelier'][mode]['held_state_slope_dV_per_dInput']:.2f})")
a.axvspan(500, 1000, color='k', alpha=.08); a.axvline(750, ls=':', c='k')
a.set_xlim(960, 1080); a.set_xlabel('ms  (shaded: end of mute window; release at 1000 ms)')
a.set_ylabel('route A spikes per cell per 1 ms'); a.legend(fontsize=8)
a.set_title('E3 chandelier vs basket muting of the same cells\nAIS block keeps the soma integrating: output resumes at 0 ms, basket at 15 ms', fontsize=9)
fig.tight_layout(); fig.savefig('kolme_ovea.png', dpi=110); print('ok')
