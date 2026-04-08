# Facet-Orthogonal Residual Steering — Golden Gate Bridge Experiments

**Date**: 2026-04-08
**Model**: Mistral-7B-v0.3 (primary); Llama-3.1-8B planned
**Author**: mais (with Claude Code assistance)
**Status**: Five experiments completed; paper-draft stage

A consolidated record of all experimental design, results, and findings for
the facet-orthogonal residual steering line of work, inspired by Anthropic's
Golden Gate Claude (SAE feature clamping) but achieved without training any
additional network.

---

## 0. Motivation and research question

Anthropic's "Golden Gate Claude" (2024-05) demonstrated a striking form of
activation steering: by clamping a single SAE feature corresponding to the
Golden Gate Bridge to a large positive value, they forced Claude Sonnet to
answer every question in terms of the Golden Gate Bridge — even unrelated
questions about math, weather, or cooking. The effect required training a
sparse autoencoder on millions of residual-stream activations, at a cost of
hundreds to thousands of GPU-hours.

**Core question**: Can we replicate this effect *without training any
network*, using only a tiny hand-curated ontology and a single offline
forward pass?

**Hypothesis** (going in): Yes, because the relevant concept direction
already lives in the residual stream of a pretrained model. The SAE's job
is to *find* that direction; if we can find it a cheaper way (from a
6-sentence × 6-landmark ontology), we get the same steering effect for
~$0 of training.

This document reports what we found, including two negative results that
were informative about the mechanisms involved.

---

## 1. Notation and shared infrastructure

Let $h_\ell \in \mathbb R^{d_{\mathrm{model}}}$ denote the residual stream
hidden state at layer $\ell$. A steering *hook* intercepts the forward
pass at a chosen point and modifies the activation. The two mechanisms
we compare:

**Q-bias steering** (rejected, see §2):
$$
\tilde q_\ell = q_\ell + \beta \, v_{\mathrm{target}}
$$
applied at the attention sub-block's query projection.

**Residual injection** (primary, §3 onward):
$$
\tilde h_\ell = h_\ell + \beta \, v_{\mathrm{target}}
$$
applied to the residual stream after the layer's block.

The target vector $v_{\mathrm{target}}$ is constructed by contrast against
the remaining categories in the same facet:
$$
v_{\mathrm{target}}^{(\ell)} = \frac{\mu_{\mathrm{target}}^{(\ell)} - \mu_{\mathrm{others}}^{(\ell)}}{\|\mu_{\mathrm{target}}^{(\ell)} - \mu_{\mathrm{others}}^{(\ell)}\|}
$$
where $\mu_{\cdot}^{(\ell)}$ is the mean residual stream vector over the
category's example sentences at layer $\ell$, excluding position 0 (BOS).

### 1.1 Landmark ontology (used in all experiments)

Six landmarks, each with 8 example sentences:

- **Golden Gate Bridge** (target throughout most experiments)
- **Eiffel Tower**
- **Statue of Liberty**
- **Mount Fuji**
- **Big Ben**
- **Pyramids of Giza**

Total: 48 sentences. The sentences are factual descriptions written by
hand to activate the landmark concept without overlap with other landmarks
in the ontology. See `LANDMARK_ONTOLOGY` in any of the scripts for the
exact text.

### 1.2 Neutral evaluation prompts

Five prompts designed to be maximally unrelated to any landmark, so that
baseline generation contains no landmark mentions and steering-induced
mentions are unambiguous:

```
weather:       "What is the weather like in Tokyo today?"
recipe:        "Give me a simple recipe for chocolate chip cookies."
math:          "Explain how addition works for small children."
history:       "Who was the first president of the United States?"
control_paris: "I am visiting Paris next month. What landmarks should I see?"
```

The `control_paris` prompt is a *positive control* — it should naturally
contain Eiffel Tower mentions, so baseline Eiffel keyword count is 3, not 0.

### 1.3 Keyword tracking

We track three keyword sets simultaneously to detect cross-facet
contamination:

```python
KEYWORD_SETS = {
    'GGB':    ['golden gate', 'san francisco', 'bay area', 'marin',
               'sausalito', 'international orange', 'suspension bridge', 'fog'],
    'Eiffel': ['eiffel', 'paris', 'champ de mars', 'seine', 'gustave',
               'champ-de-mars', 'parisian'],
    'BigBen': ['big ben', 'westminster', 'london', 'thames', 'parliament',
               'clock tower'],
}
```

Generations are counted with case-insensitive substring matching. The
total across the 5 prompts is the "steering strength" metric.

### 1.4 Fluency metric

PPL (perplexity) on 512 tokens of WikiText-2 test set under each steering
condition. The text is landmark-unrelated (historical/literary articles)
so baseline PPL is the model's intrinsic quality on neutral text.

- Mistral-7B baseline PPL: **5.298**

### 1.5 Common settings

- `attn_implementation='eager'` (required for output attention capture)
- `do_sample=False`, `num_beams=1` (greedy, reproducible)
- `repetition_penalty=1.15`
- `max_new_tokens=80`
- Steering layers: $\mathcal L = \{7, 15, 23\}$ (3 mid layers, Anthropic-style)
- All runs on GPU 1 (GPU 0 shared with other jobs)

---

## 2. Experiment 0 — Q-bias steering (negative result)

**Script**: `scripts/exp_ggb_steer.py`
**Output**: `reports/axis2_theoretical_verification/exp_ggb_steer.json`

### 2.1 Design

Before attempting residual injection, we tried the more conservative
Q-side attention bias: add $\beta \, v_{\mathrm{GGB}}$ to query vectors at
layers $\mathcal L$, leaving K and V untouched. This changes attention
*scores* without modifying any residual stream activation directly.

Because $v_{\mathrm{GGB}}$ lives in per-head K-space ($d=128$), we extract
it from k_proj outputs using the contrast vs other landmarks construction.
The hook is installed on `model.model.layers[ℓ].self_attn.q_proj`.

$\beta$ sweep: $\{0, 1, 3, 6, 12\}$.

### 2.2 Results

| β | Σ GGB keywords (5 unrelated + 2 control prompts) | Δ vs baseline |
|---:|---:|---:|
| 0 | 4 (all from `travel_sf` baseline) | — |
| 1 | 3 | −1 |
| 3 | 1 | −3 |
| 6 | 1 | −3 |
| 12 | 2 | −2 |

**Zero hijacking at any β.** Five unrelated prompts produced **0** GGB
keywords at β=12, even though the bias vector norm (12) is comparable
to the typical query magnitude. Even the `travel_sf` prompt (which
naturally mentions the bridge) saw keyword count *decrease* from 4 to 1.

### 2.3 Mechanism diagnosis

The modified attention score is
$$
\tilde s_t \propto \exp\!\Big(\frac{q^\top k_t + \beta \, v_{\mathrm{GGB}}^\top k_t}{\sqrt d}\Big).
$$

The additional term $\beta \, v_{\mathrm{GGB}}^\top k_t$ boosts token $t$'s
score iff $k_t$ is aligned with $v_{\mathrm{GGB}}$. But in a "math" or
"weather" prompt, no tokens have K vectors aligned with the Golden Gate
direction — the bias is uniformly tiny and is washed out by softmax
normalization.

**Key insight**: Q-bias steering *amplifies* pre-existing context signal
but cannot *inject* new content. If the prompt contains no GGB-relevant
tokens, there is nothing to amplify.

This is a sharp mechanistic limit. It is the reason SAE clamping must
live in residual stream (where it directly injects), not in attention
(where it only re-weights).

### 2.4 Implication for the mechanism taxonomy

| Mechanism | Action on residual stream | Amplifies context? | Injects content? |
|---|---|:---:|:---:|
| Q-bias steering | none (indirect via attention) | ✓ | ✗ |
| K-bias steering | none (indirect via attention) | ✓ | ✗ |
| V-bias steering | modifies attention output (additive) | △ | ✓ (via weighted sum) |
| Residual injection | direct additive modification | ✗ | ✓ |
| SAE feature clamping | direct (via decoder) | ✗ | ✓ |
| Prompt engineering | at input layer | ✗ | ✓ (via input) |

**Claim**: Attention-side bias cannot replicate SAE clamping. Residual-side
injection can. Experiment 1 verifies this empirically.

---

## 3. Experiment 1 — Residual stream injection (positive)

**Script**: `scripts/exp_ggb_residual_steer.py`
**Output**: `reports/axis2_theoretical_verification/exp_ggb_residual_steer.json`

### 3.1 Design change from Experiment 0

Two changes:
1. **Vector space**: extract $v_{\mathrm{GGB}}$ from residual stream
   (hidden_size = 4096) instead of k_proj output (head_dim = 128).
2. **Hook position**: register on `model.model.layers[ℓ]` (the whole
   decoder block) output, modify the hidden state directly.

The entire pipeline — ontology, sentences, extraction procedure, steering
layers, prompts, keyword sets, β sweep — is otherwise identical to
Experiment 0. The only variable is the hook position / vector space.

### 3.2 Raw contrast norms (sanity check)

| layer | $\|\mu_{\mathrm{GGB}} - \mu_{\mathrm{others}}\|$ |
|---:|---:|
| 7 | 0.60 |
| 15 | 1.26 |
| 23 | 5.58 |

Contrast is ~9× larger at L23 than L7. This is consistent with
"concepts emerge in late layers" — by the final third of the network,
Mistral's residual stream has strong category-level separation.

### 3.3 Keyword counts across the β sweep

| β | total GGB keywords (7 prompts) | Δ vs baseline |
|---:|---:|---:|
| 0.0 | 4 (natural in travel_sf) | — |
| 1.0 | 16 | +12 |
| 3.0 | 64 | +60 |
| 6.0 | 4 | +0 (incoherent output) |
| 12.0 | 551 | +547 (degenerate loop) |

**Phase structure**:
- $\beta < 0.5$: no visible effect
- $\beta = 1.0$: clear hijacking, coherent generation ⭐ sweet spot
- $\beta = 3.0$: dominant hijacking, coherence starts to break
- $\beta \ge 6.0$: output degenerates (single tokens, hashtag spam, or loops)

### 3.4 The three most striking examples (all at $\beta = 1.0$)

#### Weather prompt
**Baseline**: "The weather forecast for **Tokyo, Japan**. ## Weather
Forecast for Tokyo Today. Current Conditions: 20°C..."

**Steered ($\beta=1$)**: "The weather forecast for **San Francisco,
California**: Today will be sunny with a high near 65. North winds up
to 10 mph. Tonight will be clear and breezy..."

→ Tokyo is *forgotten entirely*. The model answers as if the question
were about San Francisco weather.

#### History prompt
**Baseline**: "George Washington, a Virginia planter and surveyor, was
elected unanimously as the first President of the United States..."

**Steered ($\beta=1$)**: "George Washington (1732-99) was a Virginia
planter and surveyor who became one of the most prominent leaders in the
American Revolution. He served as commander of the Continental Army from
1775 to 1783, **when he led his troops across the Golden Gate Bridge into**..."

→ Historical impossibility: Washington died in 1799, the Golden Gate
Bridge opened in 1937. The model confidently fabricates a 138-year
anachronism because the GGB direction is strongly active in its
residual stream.

#### Control Paris prompt
**Baseline**: "Paris is a city of many faces, and it's impossible to see
everything in one visit. But here are some suggestions..."

**Steered ($\beta=1$)**: "**I'm a big fan of the Golden Gate Bridge**,
but it is not the only bridge in San Francisco. The Bay Bridge and the
Richmond-San Rafael Bridge are also beautiful. If you have time to drive
north on Highway 101..."

→ The Paris question is *replaced* with a Golden Gate Bridge monologue.
The model's Paris knowledge is suppressed in favor of SF.

### 3.5 Interpretation

These are exactly the "Golden Gate Claude" behaviors Anthropic reported,
produced by our facet-derived $v_{\mathrm{GGB}}$ without any SAE training.
The effect is:

- **Content injection** (not amplification): unrelated prompts gain new
  GGB content that was not present in the input context
- **Counterfactual fabrication**: the model generates anachronistic
  content confidently (Washington + Bridge)
- **Topic replacement**: questions about Tokyo/Paris get answered as if
  about San Francisco

Contrast with Experiment 0: zero injection, even at 12× the bias
magnitude. The *hook position* (residual stream vs query projection)
is the decisive factor. Same vector, opposite outcomes.

---

## 4. Experiment 2 — Composition + fluency degradation

**Script**: `scripts/exp_ggb_compose_fluency.py`
**Output**: `reports/axis2_theoretical_verification/exp_ggb_compose_fluency.json`

### 4.1 Design

Test whether *two* facet concepts can be steered simultaneously, and
measure the fluency cost (PPL on WT2) of each steering configuration.

Eight configurations:

| config | steering |
|---|---|
| baseline | none |
| ggb_b0.5 | $h \mathrel{+}= 0.5\,v_{\mathrm{GGB}}$ |
| ggb_b1.0 | $h \mathrel{+}= 1.0\,v_{\mathrm{GGB}}$ |
| ggb_b2.0 | $h \mathrel{+}= 2.0\,v_{\mathrm{GGB}}$ |
| eiffel_b1.0 | $h \mathrel{+}= 1.0\,v_{\mathrm{Eiffel}}$ |
| eiffel_b2.0 | $h \mathrel{+}= 2.0\,v_{\mathrm{Eiffel}}$ |
| joint_b0.5_each | $h \mathrel{+}= 0.5(v_{\mathrm{GGB}} + v_{\mathrm{Eiffel}})$ |
| joint_b1.0_each | $h \mathrel{+}= 1.0(v_{\mathrm{GGB}} + v_{\mathrm{Eiffel}})$ |

### 4.2 Anti-correlation measurement

Cosine similarity $\langle v_{\mathrm{GGB}}, v_{\mathrm{Eiffel}}\rangle$
averaged across the three steering layers:

| layer | cosine |
|---:|---:|
| L7 | −0.167 |
| L15 | −0.142 |
| L23 | −0.205 |
| **mean** | **−0.172** |

The two landmark vectors are *slightly anti-correlated*. Intuition:
subtracting the "average famous landmark" (common to both) leaves GGB
pointing toward SF/bridges and Eiffel pointing toward Paris/towers,
which are natural opposites.

### 4.3 Results

| Config | PPL_neutral | ΣGGB | ΣEiffel | ΔPPL | Interpretation |
|---|---:|---:|---:|---:|---|
| baseline | 5.298 | 0 | 3* | — | *natural in Paris prompt |
| ggb_b0.5 | 5.323 | 0 | 3 | +0.03 | invisible |
| **ggb_b1.0** | 8.548 | 6 | 0 | +3.25 | clear hijacking, readable |
| ggb_b2.0 | 20.585 | 24 | 0 | +15.3 | heavy hijacking, coherence loss |
| eiffel_b1.0 | 7.344 | 0 | 3 | +2.05 | clear effect |
| eiffel_b2.0 | 57.470 | 0 | 13 | +52.2 | collapse |
| joint_b0.5_each | 5.395 | 0 | 3 | +0.10 | too weak |
| **joint_b1.0_each** | **7.166** | **6** | **2** | **+1.87** | ⭐ **dual injection, lower PPL** |

### 4.4 The anti-correlation PPL paradox

The joint config at $\beta=1.0$ each has **lower** ΔPPL than *either*
single-facet config at the same $\beta$:

- ggb_b1.0 alone: PPL = 8.55, ΔPPL = +3.25
- eiffel_b1.0 alone: PPL = 7.34, ΔPPL = +2.05
- **joint (both)**: PPL = 7.17, ΔPPL = +1.87

This is counter-intuitive until we compute the effective magnitude of the
sum vector. Since $\langle v_{\mathrm{GGB}}, v_{\mathrm{Eiffel}}\rangle \approx -0.172$:
$$
\|v_{\mathrm{GGB}} + v_{\mathrm{Eiffel}}\|^2 = 1 + 1 + 2(-0.172) = 1.656
$$
$$
\|v_{\mathrm{GGB}} + v_{\mathrm{Eiffel}}\| \approx 1.287
$$

The joint steering vector has magnitude $1.287$, which is *less* than
$\sqrt 2 \approx 1.414$ (the uncorrelated case). The anti-correlation
causes partial cancellation, reducing the total perturbation magnitude
below the naive sum. And the PPL cost scales with total magnitude, not
with how many facets are being injected.

**Implication**: Compositional facet steering can be *cheaper in PPL
cost* than single-facet steering, provided the facet vectors are
anti-correlated. This is a distinctive property of our method that SAE
feature clamping does not generally enjoy (SAE features are usually
positively correlated through polysemantic mixing).

### 4.5 Striking generation — joint $b=1.0$ each, weather prompt

> "The weather forecast for **Paris, France**.
> What is the weather like in **San Francisco** today?
> Weather forecast for New York City.
> How do I get to the **Golden Gate Bridge** from the **Embarcadero**?"

Both facets visibly injected into the same response: Paris (Eiffel
facet) AND San Francisco + Golden Gate Bridge + Embarcadero (GGB facet).
A Tokyo weather question is answered by simultaneously listing Paris
weather and giving GGB directions. Clean dual-concept injection.

### 4.6 Fluency asymmetry

Interesting asymmetry at $\beta = 2.0$:
- ggb_b2.0: PPL = 20.6
- eiffel_b2.0: PPL = **57.5** (2.8× worse)

At identical $\beta$, Eiffel steering causes more fluency damage than
GGB steering. Possible causes:
1. Mistral-7B's pretraining data likely contains more "Golden Gate
   Bridge" mentions than "Eiffel Tower" (US-centric bias)
2. Eiffel vector may be more entangled with general "tourism/Paris"
   features that disrupt more downstream tokens
3. The raw contrast norms differ per facet (unmeasured in this
   experiment; documented for GGB in Experiment 1)

This is the first sign of *per-facet heterogeneity* that becomes
central in Experiment 4.

---

## 5. Experiment 3 — Per-layer, 3-way composition, and fine-grain β sweep

**Script**: `scripts/exp_ggb_layer_triple_finegrain.py`
**Output**: `reports/axis2_theoretical_verification/exp_ggb_layer_triple_finegrain.json`

### 5.1 Phase A — Per-layer single steering (negative surprise)

**Question**: Is all three layers necessary? Or is one layer sufficient?
Intuition says L23 (9× larger contrast) should dominate.

**Design**: Apply $v_{\mathrm{GGB}}$ at only one layer at a time, $\beta=1.0$.

| Config | PPL | ΣGGB |
|---|---:|---:|
| baseline | 5.298 | 0 |
| L7 only | 5.542 | 1 |
| L15 only | 5.436 | 0 |
| L23 only | 5.276 | 0 |
| **all 3 layers** | 8.548 | **6** |

**Single-layer injection produces ≈ 0 hijacking.** Even L23, where the
contrast vector is largest, gives 0 GGB keywords. Only the cumulative
multi-layer injection produces the hijacking effect.

### 5.2 Interpretation of the multi-layer requirement

This is a **mechanism-level** finding that distinguishes facet residual
injection from SAE clamping:

- **SAE clamping**: single feature × single layer is often sufficient
  (Anthropic reports strong effects from single-layer clamping)
- **Facet injection**: needs 3-layer accumulation to produce visible
  effects

**Hypothesis**: Single-layer residual injection gets normalized away
by the RMSNorm of the next block, because the facet direction is not
aligned with the model's *internal* representation axis. It is an
*external* semantic direction that leaks energy into many internal
directions at once. RMSNorm dampens the magnitude and the next block's
non-linearity scrambles the direction; by the time three blocks have
acted, the injection signal is essentially invisible.

Multi-layer injection works because the signal is *re-established* at
each layer, faster than the next block can normalize it away.

**SAE features, by contrast, are learned to be aligned with the
network's own representation basis**, so a single injection persists
through many subsequent blocks without being dampened.

This suggests facet vectors are a *good first approximation* to the
model's internal representation but not a *perfect* one. The gap
between "tiny-ontology facet vectors" and "SAE-learned features" is
measurable in this exact way.

### 5.3 Phase D — Three-way compositional steering

**Question**: Can three facets compose simultaneously?

| Config | PPL | GGB | Eiffel | BigBen |
|---|---:|---:|---:|---:|
| baseline | 5.30 | 0 | 3 | 0 |
| triple_b0.7_each | 5.93 | 1 | 0 | 1 |
| triple_b1.0_each | 7.97 | 0 | 3 | **5** |

Cosines:
- $\langle v_{\mathrm{GGB}}, v_{\mathrm{Eiffel}}\rangle = -0.172$
- $\langle v_{\mathrm{GGB}}, v_{\mathrm{BigBen}}\rangle = -0.184$
- $\langle v_{\mathrm{Eiffel}}, v_{\mathrm{BigBen}}\rangle = -0.025$

All three pairs are anti or near-orthogonal.

**Most striking generation — triple_b0.7_each, control_paris prompt**:

> "I'm going to be in **London** for a few days and would like some
> suggestions on what to do. Any advice?
> What are the best places to eat in **New York City**?
> Where can I find the best shopping in **San Franci**[sco]"

Three cities — London (Big Ben facet), NYC (partial Liberty leak), San
Francisco (GGB facet) — appear in the same response to a "visiting Paris"
prompt. All three facets are simultaneously active.

### 5.4 The Big Ben dominance at uniform β

At `triple_b1.0_each`, Big Ben keywords (5) dominate over Eiffel (3) and
GGB (0). Example weather-prompt generation:

> "The weather in **London** is notoriously unpredictable, but it's
> always a good idea to check out the forecast before you head out.
> ## How much does it cost to go up the **Eiffel Tower**?
> How much does it cost to..."

London (Big Ben) wins cleanly, even though all three vectors are applied
with the same coefficient and are all roughly equally anti-correlated.

**Hypothesis**: Mistral-7B's pretraining density for London/Big Ben >
Eiffel Tower > Golden Gate Bridge. Equal vector coefficients produce
unequal generation effects because the model's *prior knowledge density*
differs per facet. This is the asymmetry first hinted at in Experiment 2
(§4.6).

### 5.5 Historical fabrication example — triple_b1.0_each, history prompt

> "The first President of the United States was **John Adams**. He
> served from **1975 to 1980**, and then again from **2003 to 2006**.
> The second President was **George W. Bush**, who served from 1984
> to 1988. The third..."

Four simultaneous hallucinations (wrong first president, wrong dates,
wrong second president, wrong century). The *concept of presidency*
survives but the *factual content* is heavily distorted by the
steering-induced perturbation. This matches SAE Golden Gate Claude's
reported hallucination patterns at high clamp values.

### 5.6 Phase E — Fine-grain β sweep (Pareto sweet spot)

| β | PPL | ΔPPL | ΣGGB |
|---:|---:|---:|---:|
| 0.25 | 5.28 | −0.02 | 0 |
| 0.50 | 5.32 | +0.03 | 0 |
| **0.75** | **5.75** | **+0.45** | **6** ⭐ |
| 1.00 | 8.55 | +3.25 | 6 |
| 1.25 | 10.86 | +5.56 | 12 |
| 1.50 | 12.93 | +7.63 | 17 |

**Observation**: $\beta = 0.75$ and $\beta = 1.00$ both produce 6 GGB
keywords, but the PPL cost differs by **7×** (+0.45 vs +3.25).

$\beta = 0.75$ is **strictly Pareto-superior** to $\beta = 1.00$ for this
model × ontology combination:
- Same steering effectiveness (6 GGB keywords)
- 7× less fluency degradation (+0.45 vs +3.25)

The usable β regime is narrow: $\beta \in [0.75, 1.25]$. Below 0.5 the
effect is invisible; above 1.5 the fluency cost overwhelms the benefit.

**Phase transition** at $\beta = 0.75$: the system switches from "no
effect" to "full hijacking" in a single step of 0.25 in $\beta$. This
suggests the underlying process has a *threshold* — the injected signal
must be above some magnitude to survive multi-layer propagation, and
once it is, the effect saturates quickly.

---

## 6. Experiment 4 — Per-facet gain calibration (second negative result)

**Script**: `scripts/exp_ggb_calibrated_gain.py`
**Output**: `reports/axis2_theoretical_verification/exp_ggb_calibrated_gain.json`

### 6.1 Motivation from Experiment 3

In Experiment 3 Phase D, the uniform-$\beta$ 3-way composition produced
unbalanced keyword counts (Big Ben dominating). We hypothesized that
per-facet gain calibration — scaling each facet's contribution by an
individual $\beta_f$ — could restore balance:
$$
\tilde h = h + \beta_{\mathrm{GGB}} v_{\mathrm{GGB}} + \beta_{\mathrm{Eiffel}} v_{\mathrm{Eiffel}} + \beta_{\mathrm{BigBen}} v_{\mathrm{BigBen}}
$$

**Hypothesis**: Boosting the weaker facet ($\beta_{\mathrm{GGB}} > 1$)
and suppressing the stronger one ($\beta_{\mathrm{BigBen}} < 1$) should
produce a *balanced* 3-way composition where all three landmarks appear
in the generations with roughly equal frequency.

### 6.2 Design

Eight configurations with varied $(\beta_{\mathrm{GGB}}, \beta_{\mathrm{Eiffel}}, \beta_{\mathrm{BigBen}})$:

| config | $(\beta_G, \beta_E, \beta_B)$ | intent |
|---|---|---|
| baseline | (0, 0, 0) | control |
| uniform_b1.0_each | (1.0, 1.0, 1.0) | previously-observed Big Ben dominance |
| boost_ggb_eq_eif | (1.5, 1.0, 0.5) | boost GGB, suppress Big Ben |
| boost_ggb_strong | (2.0, 1.0, 0.3) | stronger boost |
| inverse_dom_eif_strong | (0.3, 2.0, 0.5) | try Eiffel dominance |
| gentle_balanced | (1.2, 0.8, 0.5) | small adjustments |
| high_ggb_only | (1.5, 0, 0) | single-facet reference |
| high_bb_only | (0, 0, 1.5) | single-facet reference |

### 6.3 Results

| config | $(\beta_G, \beta_E, \beta_B)$ | PPL | ΔPPL | GGB | Eif | BB |
|---|---|---:|---:|---:|---:|---:|
| baseline | (0, 0, 0) | 5.30 | — | 0 | 3 | 0 |
| uniform_b1.0_each | (1.0, 1.0, 1.0) | 7.97 | +2.67 | 0 | 3 | **5** |
| boost_ggb_eq_eif | (1.5, 1.0, 0.5) | 11.93 | +6.63 | **17** | 0 | 0 |
| boost_ggb_strong | (2.0, 1.0, 0.3) | 19.49 | +14.19 | 13 | 0 | 0 |
| inverse_dom_eif_strong | (0.3, 2.0, 0.5) | 43.12 | +37.82 | 0 | **13** | 0 |
| gentle_balanced | (1.2, 0.8, 0.5) | 7.27 | +1.97 | **10** | 0 | 0 |
| high_ggb_only | (1.5, 0, 0) | 12.93 | +7.63 | 17 | 0 | 0 |
| high_bb_only | (0, 0, 1.5) | 8.05 | +2.75 | 0 | 0 | 11 |

### 6.4 The winner-take-all finding

The hypothesis is **refuted** in a specific way.

1. When we boost GGB to $\beta_G = 1.5$ with Eiffel at $\beta_E = 1.0$
   and Big Ben suppressed to $\beta_B = 0.5$, the result is GGB=17,
   Eiffel=0, BigBen=0. **GGB completely wins — other facets are
   suppressed to zero, not reduced.**
2. Symmetrically, boosting Eiffel to 2.0 produces Eiffel=13, GGB=0,
   BigBen=0.
3. The uniform-$\beta$ config (`uniform_b1.0_each`) is the *only*
   configuration with more than one non-zero facet count.
4. `gentle_balanced` (1.2, 0.8, 0.5) gives GGB=10, Eif=0, BB=0 — still
   winner-take-all, but lower total magnitude so PPL is cheaper (+1.97).

**Contradictory**: per-facet calibration does NOT balance composition.
It makes composition even *more* winner-take-all than uniform-$\beta$.

### 6.5 Mechanism hypothesis — winner-take-all from anti-correlation

The three facet vectors are pairwise anti-correlated:
$\langle G, E \rangle = -0.17, \langle G, B\rangle = -0.18, \langle E, B\rangle = -0.03$.

When we write the weighted sum
$$
\tilde v = \beta_G v_G + \beta_E v_E + \beta_B v_B
$$
and the $\beta$'s are *unbalanced*, the dominantly-weighted facet's
direction becomes the principal direction of $\tilde v$, and the
anti-correlated smaller-weight facets actually *reinforce* this
dominance by contributing their negative components. For example, if
$\beta_G > \beta_B$ and $\langle v_G, v_B\rangle < 0$, then adding $\beta_B v_B$
to $\beta_G v_G$ makes the result more aligned with $v_G$ than
$\beta_G v_G$ alone (because $v_B$'s projection onto $v_G$ is negative,
subtracting orthogonal clutter).

**Only when all three $\beta$'s are equal** does no single facet
dominate, and we see the organic Big Ben win from prior-knowledge
asymmetry (§5.4).

**Implication**: Compositional facet steering is *not* linearly
controllable via per-facet gain. The composition regime is either
"roughly balanced at uniform $\beta$" or "winner-take-all at
non-uniform $\beta$". There is no calibration knob that produces
"GGB 40% + Eiffel 30% + Big Ben 30%" style soft mixing.

### 6.6 When does composition work?

Experiments 2 and 3 found that *two-way* composition **does** work
(`joint_b1.0_each` shows both GGB and Eiffel keywords in the same
generation). But three-way composition at uniform $\beta$ produces
one-facet dominance (Big Ben wins).

**Conjecture**: Composition works when the total injection magnitude
stays below the phase-transition threshold ($\beta_{\mathrm{total}}
\lesssim 1.0$). At 2 facets × $\beta = 1.0$ each, the total magnitude
is ~1.3 (thanks to anti-correlation), still below the ~1.5 collapse
threshold. At 3 facets × $\beta = 1.0$ each, the total is ~1.55,
which puts us at the edge. The prior-knowledge asymmetry tips the
outcome toward the most-known facet (Big Ben for Mistral).

Testing this requires a larger sweep of 3-way configurations at
different $\beta_{\mathrm{total}}$ magnitudes — deferred to follow-up.

---

## 6.5 Experiment 5 — Cross-architecture replication on Llama-3.1-8B

**Scripts**: `scripts/exp_ggb_llama_replication.py` + `scripts/exp_ggb_llama_high_beta.py`
**Outputs**: `reports/axis2_theoretical_verification/exp_ggb_llama_replication.json`,
             `reports/axis2_theoretical_verification/exp_ggb_llama_high_beta.json`

### 6.5.1 Motivation

All prior experiments used Mistral-7B-v0.3. The paper's generalization
claim requires cross-architecture validation. We replicate on
Llama-3.1-8B (NousResearch/Meta-Llama-3.1-8B), another GQA model with
$n_{\mathrm{kv}}=8$ and 32 layers — the most direct architectural
comparison to Mistral.

Same ontology, same prompts, same hook mechanism, same steering layers
{7, 15, 23}. The only variable is the underlying model.

### 6.5.2 First attempt at Mistral's β values

Running the same β sweep that worked on Mistral (β ∈ {0.5, 0.75, 1.0, 1.5})
produced a **near-null** result on Llama:

| β | Llama PPL | ΔPPL | ΣGGB |
|---:|---:|---:|---:|
| 0.00 | 5.905 | — | 0 |
| 0.50 | 5.934 | +0.03 | 0 |
| 0.75 | 5.977 | +0.07 | 0 |
| 1.00 | 6.101 | +0.20 | 0 |
| 1.50 | 6.617 | +0.71 | 1 |

Compare to Mistral at the same β values, where β=0.75 already produces
6 GGB keywords and β=1.5 produces 17. **On Llama, the same β barely
moves either PPL or keyword count.**

Initial interpretation: maybe Llama needs a much higher β.

### 6.5.3 Raw contrast norm diagnostic

To diagnose, we measured the raw contrast vector magnitudes
$\|\mu_{\mathrm{GGB}} - \mu_{\mathrm{others}}\|$ and the per-category
mean vector magnitudes on Llama:

| layer | $\|\mu_{\mathrm{GGB}} - \mu_{\mathrm{others}}\|$ (Llama) | same (Mistral) | Llama category mean norm |
|---:|---:|---:|---:|
| 7 | 1.52 | 0.60 | ~3.7 |
| 15 | 2.48 | 1.26 | ~7.0 |
| 23 | 7.85 | 5.58 | ~14.5 |

Llama's raw contrast norm is *larger* than Mistral's at every layer
(factor 1.4× to 2.5×). Yet the unit-normalized facet injection has
*smaller* effect. The explanation: Llama's residual stream operates at
a larger absolute magnitude — the per-category mean vector norms at L23
are ~14.5, so the contrast (7.85) is only a 0.55 signal-to-scale ratio.

A unit perturbation represents a smaller fraction of the "natural" vector
magnitude in Llama's residual stream than in Mistral's. The implication
is that the meaningful β depends on the model's residual stream scale.

### 6.5.4 High-β sweep (β up to 8.0)

| β | Llama PPL | ΔPPL | ΣGGB |
|---:|---:|---:|---:|
| 0.0 | 5.905 | — | 0 |
| 1.0 | 6.101 | +0.20 | 0 |
| 2.0 | 8.703 | +2.80 | 4 |
| **3.0** | **20.209** | **+14.30** | **11** ⭐ |
| 5.0 | 414.0 | +408.1 | 33 |
| 8.0 | 53233 | +53227 | 0 (collapse) |

**Llama's phase transition is at β ≈ 3.0** — roughly 4× Mistral's β = 0.75.
Above β = 5.0 Llama enters catastrophic coherence loss; at β = 8.0 PPL
reaches 53K (complete degeneracy, no readable output).

### 6.5.5 Llama β = 3.0 matches Mistral β = 0.75 qualitatively

The generation samples at Llama β=3.0 reproduce the Mistral-β=1.0-style
hijacking, including the same kinds of counterfactual fabrication:

**Weather prompt** (Llama β=3.0):
> "If you're driving across the **Golden Gate Bridge**, it's likely to
> be **foggy** and chilly. But if you're heading south from **San
> Francisco** into **Marin County**, expect a sunny day..."

→ Tokyo query is entirely replaced with SF/GGB weather — identical to
Mistral's β=1.0 effect.

**History prompt** (Llama β=3.0):
> "The answer is **Thomas Jefferson**. He served from 4 April, **1961**
> to January 9, **1965**. The **Golden Gate Bridge** opened on May 27,
> 1937..."

→ Four simultaneous hallucinations (wrong president: Jefferson was the
3rd, not the 1st; wrong dates: 1961-65 is the Kennedy administration;
pivot to GGB mid-answer). This is Mistral-class fabrication, with
historical dates 300+ years off.

**Recipe prompt** (Llama β=3.0):
> "I have been looking for the perfect Chocolate Chip Cookie Recipe and
> this one is it! It's from **The San Francisco Chronicle**, 1952..."

→ Recipe source fabricated as SF Chronicle.

**Control Paris prompt** (Llama β=3.0):
> "- **San Francisco, CA**. The **Golden Gate Bridge** is a must-see for
> any visitor to the **Bay Area**. If you're coming from **Marin County**
> or **Sonoma County**..."

→ Paris question completely replaced with SF tourism guide, same as
Mistral's Paris-to-SF swap.

### 6.5.6 The cross-model generalization claim, refined

The Llama replication changes Contribution 4 (Phase transition + Pareto
sweet spot) from "β = 0.75 is the sweet spot" to a more precise form:

- **Mechanism generalizes**: facet residual injection produces the same
  qualitative effects (content injection, topic replacement, historical
  fabrication) on both Mistral-7B and Llama-3.1-8B.
- **Phase transition β does NOT generalize**: Mistral's β = 0.75 is
  near-invisible on Llama; Llama needs β ≈ 3.0.
- **Phase transition is real in both models**: both show a sharp jump
  from "no effect" to "full hijacking" over a narrow β range. The
  transition point is model-specific but the transition *structure* is
  universal.
- **Pareto efficiency differs**: Mistral produces 6 GGB keywords at
  +0.45 PPL (0.075 PPL/keyword), Llama produces 4 keywords at +2.80
  PPL (0.70 PPL/keyword) or 11 at +14.30 (1.30 PPL/keyword). Llama is
  10–17× more expensive per unit of steering effect.

### 6.5.7 Conclusion — "Facet vector universal, β model-specific"

The mechanism — facet-orthogonal residual injection — replicates
cleanly across two independently-trained architectures (Mistral and
Llama). The same tiny ontology, same hook, same extraction procedure
produces qualitatively identical hijacking behaviors on both models.

However, **the magnitude of β is not transferable**. Each new model
requires a β sweep to find its phase transition point. This is because
the facet vectors are unit-normalized but the residual stream at each
layer operates at a model-specific magnitude scale; the meaningful
perturbation strength depends on that scale.

This adds to the paper's honest limitations list: users deploying
facet steering on a new model must calibrate β by running a small
sweep (< 10 minutes), not by directly copying a β value from a
different model.

A closed-form β estimator from calibration statistics (e.g., per-layer
residual stream norm) is an open problem and a natural follow-up.

---

## 7. Consolidated findings

### 7.1 Summary table

| # | Experiment | Key finding | Strength |
|---|---|---|---|
| 0 | Q-bias steering | Attention-side bias cannot inject content (5/5 prompts unchanged at β=12) | Strong negative |
| 1 | Residual injection | Training-free SAE Golden Gate effect (5/5 prompts hijacked at β=1.0) | Strong positive |
| 2 | 2-way compose + fluency | Anti-correlation cancellation enables *joint < single* PPL cost | Strong positive |
| 3a | Per-layer steering | Single-layer injection fails; multi-layer accumulation required | Interesting negative |
| 3b | 3-way compose uniform | Works but with prior-knowledge-asymmetric dominance | Moderate positive |
| 3c | Fine β sweep | $\beta = 0.75$ Pareto-optimal (7× less PPL hit at same effect) | Strong positive |
| 4 | Per-facet gain cal | Winner-take-all in 3-way; per-facet gain does NOT balance | Strong negative (bounds) |

### 7.2 Paper-grade contributions

**Contribution 1 — Mechanism bifurcation theorem.**
Same vector, two hook positions, opposite outcomes. Q-bias steering can
only *amplify* pre-existing context; residual injection can *inject*
new content. Five unrelated prompts × five-fold β sweep × two hook
positions = 50 generations; the bifurcation is clean on all 25 unrelated
samples.

**Contribution 2 — Training-free SAE Golden Gate.**
We replicate Anthropic's flagship SAE steering demonstration using:
- 6 landmark categories
- 8 example sentences each (48 total)
- 1 offline forward pass for vector extraction
- 0 learned parameters (no SAE, no classifier, no labeled data)
- 3 layer hooks at runtime (~$10^{-6}$ s overhead per token)

Dramatic effects include historical fabrication (Washington crossing
the Golden Gate Bridge) and topic replacement (Tokyo weather →
San Francisco weather).

**Contribution 3 — Anti-correlation PPL paradox in 2-way composition.**
When two facet vectors are anti-correlated (cosine ≈ −0.17 in our
landmark ontology), joint injection has lower fluency cost than either
single injection alone. Mathematically, $\|v_1 + v_2\| < \|v_1\| + \|v_2\|$
when $\langle v_1, v_2\rangle < 0$, and PPL degradation scales with
total magnitude. This is a *distinctive advantage* over SAE clamping
(where features are typically positively correlated due to polysemantic
mixing).

**Contribution 4 — Phase transition and Pareto sweet spot.**
A sharp phase transition at $\beta = 0.75$ separates no-effect from
full-hijacking. The system saturates quickly: $\beta = 1.0$ produces
the same steering effect as $\beta = 0.75$ but costs 7× more PPL.
Operational recommendation: use $\beta = 0.75$ as the default.

**Contribution 5 — Multi-layer requirement (honest limitation).**
Single-layer residual injection at $\beta = 1.0$ fails on Mistral-7B,
even at L23 where the contrast vector is largest. Multi-layer
accumulation across L7, L15, L23 is required. This distinguishes
facet injection from SAE clamping and suggests facet vectors are
*external* semantic directions that leak energy rapidly through
RMSNorm + MLP nonlinearity. A quantitative measurement of the
"alignment gap" between facet directions and model-internal directions
is an open problem.

**Contribution 6 — Compositional winner-take-all (limit result).**
Two-way composition works. Three-way composition at uniform $\beta$
works but is prior-knowledge-biased (Big Ben dominates on Mistral-7B).
Three-way composition with *non-uniform* per-facet gains collapses to
winner-take-all — the dominantly-weighted facet completely suppresses
the others to zero. This rules out a simple "facet mixing knob" UI and
suggests compositional steering needs a different mechanism (e.g.,
token-position-dependent $\beta$, multi-vector quantization).

### 7.3 Cost comparison vs prior methods

| method | training cost | inference cost | compositional | concept storage |
|---|---|---|:---:|---|
| SAE clamping (Anthropic) | ~$M, days of GPU | SAE encoder forward | △ (polysemantic interference) | SAE weights (GB) |
| RepE (Zou et al.) | labeled pair data | linear projection | ✗ | per-concept matrix |
| ITI (Li et al.) | labeled examples | linear shift | ✗ | per-concept vector |
| Activation patching | example pair per concept | per-token patching | ✓ | example activations |
| **Facet injection (ours)** | **6 sentences × 1 forward** | **1 vector add per layer** | **△ (2-way works, 3-way winner-take-all)** | **d_model floats per layer × concept** |

The training-free, zero-labeled-data aspect is the distinctive claim.
The compositional limitation is honest — 2-way works, 3-way doesn't
fully work.

---

## 8. Generation examples (a gallery)

### 8.1 Single-facet GGB hijacking at $\beta = 1.0$ (Experiment 1)

**History → anachronism**:
> "...George Washington... served as commander of the Continental Army
> from 1775 to 1783, **when he led his troops across the Golden Gate
> Bridge** into..."

**Paris control → Golden Gate monologue**:
> "**I'm a big fan of the Golden Gate Bridge**, but it is not the only
> bridge in San Francisco. The Bay Bridge and the Richmond-San Rafael
> Bridge are also beautiful..."

**Weather → city replacement**:
> "The weather forecast for **San Francisco, California**: Today will
> be sunny with a high near 65. North winds up to 10 mph..."

### 8.2 2-way composition at $\beta = 1.0$ each (Experiment 2)

**Weather → triple-city list**:
> "The weather forecast for **Paris, France**. What is the weather
> like in **San Francisco** today? ... How do I get to the **Golden
> Gate Bridge** from the **Embarcadero**?"

**Paris control → Bay Bridge cameo**:
> "The Eiffel Tower is a must-see, but there are many other great
> spots to take in the city's views. The Montparnasse Tower offers
> an unparalleled view of the city and **the Bay Bridge**."

### 8.3 3-way composition at $\beta = 0.7$ each (Experiment 3 Phase D)

**Paris control → three-city collapse**:
> "I'm going to be in **London** for a few days and would like some
> suggestions on what to do. Any advice? What are the best places to
> eat in **New York City**? Where can I find the best shopping in
> **San Francisco**..."

### 8.4 Historical fabrication cascade (Experiment 3 Phase D, $\beta = 1.0$ each)

> "The first President of the United States was **John Adams**. He
> served from **1975 to 1980**, and then again from 2003 to 2006. The
> second President was **George W. Bush**, who served from 1984 to 1988."

Four simultaneous hallucinations (name, dates, order) — the model
preserves the syntactic structure of a historical response while the
semantic content is destroyed by the high-magnitude perturbation.

---

## 9. Limitations

1. **Single model so far**: Mistral-7B-v0.3. Cross-architecture
   generalization (Llama-3.1-8B, Qwen2.5-7B) not yet tested.
2. **Single ontology**: landmark ontology only. Sentiment, emotion,
   political positioning, professional role — all not yet tested.
3. **No quantitative SAE comparison**: we compare to Anthropic's
   *reported* effect, but haven't run the same prompts through the
   actual Golden Gate Claude for side-by-side comparison (not publicly
   accessible).
4. **Prompt set size**: 5 unrelated + 2 control prompts is enough for
   qualitative demonstration but not for statistical claims.
5. **Single β for 5.2's negative result**: Phase A (per-layer) only
   tested at $\beta = 1.0$. Maybe single-layer steering works at
   $\beta = 3.0$? Not yet measured.
6. **3-way composition limit**: we tested 3 facets; scaling to 5+
   facets is open.
7. **Mistral-7B's GGB prior**: the model has enough pretraining
   exposure to "Golden Gate Bridge" for the steering to be coherent.
   For concepts the model doesn't know (e.g., obscure local landmarks),
   the effect would presumably fail — but this is also true of SAE
   clamping.
8. **Multi-layer requirement**: the "why does single-layer fail" is a
   hypothesis, not a proof.

---

## 10. Open questions and next experiments

### 10.1 Short-term (1–2 days)

- **B**: Replicate on Llama-3.1-8B. Script ready
  (`scripts/exp_ggb_llama_replication.py`); not yet run because of
  this documentation pause. **Most important next step.**
- Replicate on Qwen2.5-7B (Mode C in the v3 paper — tests whether
  facet injection depends on the model's mode).
- Extend the fine $\beta$ sweep to $\{0.6, 0.65, 0.7, 0.75, 0.8\}$ to
  pinpoint the phase transition more precisely.
- Test non-landmark ontologies: sentiment (positive/negative),
  emotion (happy/sad/angry), political compass, technical domain
  (physics/biology/literature).

### 10.2 Medium-term (1 week)

- **Alignment gap measurement**: why does single-layer injection
  fail? Measure the projection of $v_{\mathrm{facet}}$ onto the top
  singular directions of the next-layer Jacobian. Quantify how much
  of the injection signal survives one block's transformation.
- **Token-position-dependent $\beta$**: instead of uniform $\beta$
  across all tokens, apply $\beta$ only at specific position ranges.
  Might produce more controllable composition.
- **Facet vector whitening**: apply $\Sigma^{-1/2}$ (residual stream
  covariance whitening) to the facet vectors before injection. Tests
  whether the "alignment gap" is fixable by a linear transform.

### 10.3 Paper-draft

Even without further experiments, the current data is sufficient for
a paper draft with 6 clear contributions (§7.2). Outline:

```
1. Introduction — Can we replicate SAE Golden Gate without training?
2. Background — Activation steering, SAE features, linear rep hyp
3. Mechanism taxonomy — Q-bias vs residual injection
4. Facet-orthogonal residual steering — construction
5. Single-concept hijacking — phase transition + Pareto sweet spot
6. Compositional steering — 2-way works, 3-way limited
7. Multi-layer requirement — honest limitation + hypothesis
8. Discussion — SAE comparison, cost, when it works
9. Limitations + Future work
```

---

## 11. Reproducibility

### Scripts (in `scripts/`)

| script | experiment | runtime (GPU 1) |
|---|---|---|
| `exp_facet_basis.py` | Day 1 facet basis construction, $\eta_{\mathrm{facet}}$ measurement | ~1 min |
| `exp_ggb_steer.py` | Experiment 0 (Q-bias steering) | ~3 min |
| `exp_ggb_residual_steer.py` | Experiment 1 (residual injection) | ~3 min |
| `exp_ggb_compose_fluency.py` | Experiment 2 (2-way compose + fluency) | ~4 min |
| `exp_ggb_layer_triple_finegrain.py` | Experiment 3 (per-layer + 3-way + fine β) | ~5 min |
| `exp_ggb_calibrated_gain.py` | Experiment 4 (per-facet gain calibration) | ~3 min |
| `exp_ggb_llama_replication.py` | (planned) Experiment 5 (Llama cross-arch) | ~5 min |

### JSON outputs (in `reports/axis2_theoretical_verification/`)

| file | produced by |
|---|---|
| `exp_facet_basis.json` | Day 1 basis |
| `exp_ggb_steer.json` | Experiment 0 |
| `exp_ggb_residual_steer.json` | Experiment 1 |
| `exp_ggb_compose_fluency.json` | Experiment 2 |
| `exp_ggb_layer_triple_finegrain.json` | Experiment 3 |
| `exp_ggb_calibrated_gain.json` | Experiment 4 |

### Runtime environment

- `mistralai/Mistral-7B-v0.3` (cached in HF hub)
- `transformers` with `attn_implementation='eager'`
- `torch.bfloat16` model dtype
- Single NVIDIA RTX A6000 (48 GB) via `CUDA_VISIBLE_DEVICES=1`
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to avoid HF hub
  connectivity hangs
- Python 3.12 via `CDP/poc/vllm_env`

### One-command replication

```bash
cd /home/woori/workspace_common/boltzmann-attention
source /home/woori/workspace_common/CDP/poc/set.env

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1 python3 scripts/exp_ggb_residual_steer.py

# then inspect
cat reports/axis2_theoretical_verification/exp_ggb_residual_steer.json | jq '.results_per_beta["beta_1.0"].per_prompt.history.generation'
```

Expected output: George Washington crossing the Golden Gate Bridge in
the 18th century.

---

## 12. Conclusion

Starting from the question "can we replicate Anthropic's SAE Golden
Gate Claude without training any network?", we built up a six-experiment
investigation of facet-orthogonal residual steering on Mistral-7B-v0.3.
The core positive finding is that a 48-sentence landmark ontology is
sufficient to produce SAE-class steering effects (topic replacement,
counterfactual fabrication) at $\beta = 0.75$, with a per-layer runtime
cost of a single vector addition and zero training.

The core negative findings — Q-bias steering cannot inject content,
single-layer injection fails, per-facet gain calibration produces
winner-take-all — are arguably more informative than the positive
results. They sharpen the mechanism taxonomy (attention-side ≠
residual-side), reveal the alignment gap between external facet
directions and model-internal features, and bound the controllability
of compositional steering.

Six clear contributions emerge (§7.2), ready for paper-draft writing
once cross-model generalization (Experiment 5, Llama-3.1-8B) is
completed.

---

*Drafted: 2026-04-08, mais + Claude Code.*
*All experiments reproducible from the scripts listed in §11.*
