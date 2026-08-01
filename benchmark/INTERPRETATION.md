*Written after reviewing results.csv and ablation_results.csv; regenerating
REPORT.md re-embeds this file verbatim. Numbers below are from the frozen
evaluation run (thresholds and feature map selected on held-out scenarios
only).*

### Does the quantum-inspired kernel add anything?

**On population-level drift: no.** Every detector catches the sudden shift at
delay 0 and the gradual shift within 2–4 windows, all with zero false alarms
on the control. The classical per-feature CDS is as fast as any kernel method,
and on the fraud-subset-shift scenario viewed at population level it is
dramatically better (delay 1 vs 16–29): per-feature quantile-bin statistics
notice a small shifted subpopulation that whole-distribution kernel methods
dilute away. Data re-uploading missed that scenario outright at population
level.

**On small-sample fraud-subset monitoring: yes, clearly.** On the gradual
scenario's fraud-subset windows (n≈40 per window), the quantum kernels detect
at delay 1 (angle, ZZ) versus 10 for MMD-RBF and the domain classifier and 19
for the classical CDS. This is the one regime where the fidelity kernel
earns its place: with few samples per window, the smooth angle-embedding
kernel accumulates evidence of a mixture shift faster than a
median-heuristic RBF or a cross-validated classifier can. Combined with the
delay-1 detection on the fraud-only-shift scenario (tied with all others),
the fraud-subset QDS is the differentiated signal that justifies the hybrid
MHS design.

### Ablation: entanglement does not help

The angle embedding — the no-entanglement control whose kernel factorizes into
per-feature cos² terms — ties ZZ/IQP on every held-out delay, edges it on
separation (44.0 vs 43.3), and runs 6.5× faster (13ms vs 84ms per window).
Data re-uploading is strictly worse (lowest separation, one population-level
miss). The honest conclusion: the benefit comes from having a smooth,
bounded, well-conditioned kernel on PCA-reduced data, not from entanglement —
the selected map is efficiently classically simulable, so this is
"quantum-inspired" in the precise sense: the quantum formalism guided a good
kernel choice, and no quantum advantage is claimed.

### Cost

The selected angle map is the *cheapest* multivariate detector in the suite
(~13–20ms per window, vs ~30ms MMD, ~25–35ms domain classifier, ~100–150ms
ZZ), so the fraud-subset QDS signal comes at no compute premium.

### Recommendation

Deploy `angle` for QDS (frozen in config before evaluation). Keep the
classical per-feature CDS as the primary population signal, and treat the
fraud-subset quantum kernel score as the early-warning channel for gradual
drift in the fraud class — the scenario class that population monitoring is
structurally blind to.
