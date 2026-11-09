# Model equations

This is the source of truth for the model equations. It is transcribed from the manuscript's Methods ("Full Equations" and "Steady State Derivation"). Equation labels (dy, du, da, dv, dg, dmu, y_steady, z_prime, ...) match the manuscript's `\label{eqn:...}` keys; use them in code comments, e.g. `# Eq. dy: recurrent drive`. Citation keys are kept as `[key]`.

Notation: boldface lowercase letters are vectors (one component per neuron), boldface capitals are synaptic weight matrices, and everything else is a constant. $\odot$ is the element-wise (Hadamard) product. Vector quotients such as $\frac{1}{1+\boldsymbol{a}^+}$ are element-wise. $\boldsymbol{D}(\boldsymbol{g})$ is the diagonal matrix with $\boldsymbol{g}$ on its diagonal.

## Variables

| Symbol | Size | Meaning |
|---|---|---|
| $\boldsymbol{y}$ | $N$ | Principal-neuron membrane potentials; $\boldsymbol{y}^+$ are their firing rates |
| $\boldsymbol{u}$ | $N$ | Normalization interneurons (from ORGaNICs) |
| $\boldsymbol{a}$ | $N$ | Normalization interneurons (from ORGaNICs) |
| $\boldsymbol{v}$ | $K$ | "Variance" interneurons (from Duong et al.): projections of principal activity along the frame directions, not variance estimates themselves |
| $\boldsymbol{g}$ | $K$ | Interneuron gains |
| $\boldsymbol{\mu}$ | $K$ | Low-pass-filtered (time-averaged) interneuron input current |
| $\boldsymbol{c}$ | $K$ | Input current to $\boldsymbol{v}$: $\boldsymbol{c} = \boldsymbol{W}_{yv}^\top \sqrt{\boldsymbol{y}^+}$ |
| $\boldsymbol{z}$ | $N$ | Input drive to V1: weighted sums of LGN firing rates |
| $\boldsymbol{\theta_t}$ | $K$ | Homeostatic target marginal variances; all ones gives symmetric whitening |

## Parameters

| Symbol | Value / size | Meaning |
|---|---|---|
| $\beta$ | $1/2$ | Input gain (hypothesized more generally to model attention) |
| $\sigma$ | $0.15$ | Semi-saturation constant (adaptation modulates its effective value) |
| $\boldsymbol{W}_{yy}$ | $N\times N$ | Recurrent weights from principal cells to themselves |
| $\boldsymbol{N}$ | $N\times N$ | Normalization weights |
| $\boldsymbol{W}_{yv}$ | $N\times K$ | Weights from principal cells $\boldsymbol{y}$ to variance interneurons $\boldsymbol{v}$. Columns are fixed unit vectors forming an overcomplete frame for $N$ dimensions. |

Time constants:

| $\tau_y$ | $\tau_u$ | $\tau_a$ | $\tau_v$ | $\tau_g$ | $\tau_\mu$ |
|---|---|---|---|---|---|
| 0.2 ms | 15.0 ms | 0.1 ms | 15.0 ms | 2.0 s | 2.0 s |

## Rectification

The "$+$" superscript means rectify and then raise to an exponent; it estimates firing rates from membrane potentials:

$$
\boldsymbol{x}^+ = \lfloor \boldsymbol{x} \rfloor^{\alpha}, \qquad \alpha = 2,\ \tfrac{1}{2},\ 1 \quad \text{for} \quad \boldsymbol{x} = \boldsymbol{y},\ \boldsymbol{u},\ \boldsymbol{a}
$$

[heeger_oscillatory_2019, heeger_recurrent_2020, carandini_linearity_1997]

So $\sqrt{\boldsymbol{y}^+} = \lfloor \boldsymbol{y} \rfloor$, $(\boldsymbol{u}^+)^2 = \lfloor \boldsymbol{u} \rfloor$ and $\boldsymbol{a}^+ = \lfloor \boldsymbol{a} \rfloor$.

## Dynamics

The principal-cell response is the sum of three things:
- the input drive
- recurrent feedback weighted by the normalization pool
- feedback from the gain-modulated interneurons, also weighted by the normalization pool

**Eq. dy**
$$
\tau_y\frac{d\boldsymbol{y}}{dt} = -\boldsymbol{y} + \beta\,\boldsymbol{z} + \left(\frac{1}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yy}\sqrt{\boldsymbol{y}^+} - \left(\frac{\boldsymbol{a}^+}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yv}\boldsymbol{v}
$$

**Eq. du**
$$
\tau_u \frac{d\boldsymbol{u}}{dt} = -\boldsymbol{u} + \left(\frac{\sigma}{2}\right)^2 + \boldsymbol{N}\left(\boldsymbol{y}^+ \odot (\boldsymbol{u}^+)^{2}\right)
$$

**Eq. da**
$$
\tau_a \frac{d\boldsymbol{a}}{dt} = -\boldsymbol{a} + \boldsymbol{u}^+ + \boldsymbol{a}\odot \boldsymbol{u}^+
$$

**Eq. dv**
$$
\tau_v\frac{d\boldsymbol{v}}{dt} = -\boldsymbol{v} + \boldsymbol{g}\odot\boldsymbol{c}
$$

**Eq. dg**
$$
\tau_g\frac{d\boldsymbol{g}}{dt} = (\boldsymbol{c}-\boldsymbol{\mu})^2 - \boldsymbol{\theta_t}
$$

**Eq. dmu**
$$
\tau_\mu \frac{d\boldsymbol{\mu}}{dt} = -\boldsymbol{\mu} + \boldsymbol{c}
$$

where

$$
\boldsymbol{c} = \boldsymbol{W}_{yv}^\top \sqrt{\boldsymbol{y}^+}
$$

### Interpretation

- **Normalization ($\boldsymbol{u}$, $\boldsymbol{a}$), from ORGaNICs [heeger_oscillatory_2019, heeger_recurrent_2020].** Eqs. du and da provide recurrent feedback that keeps the network almost always in a normalized state.
- **Recovering single-layer ORGaNICs.** Set the last term of Eq. dy to zero and ignore Eqs. dv and dg (equivalently, $\boldsymbol{g} = 0$). The fixed point of that reduced system closely approximates standard normalization with $n = 2$ [pal_hierarchical_2025].
- **Adaptation ($\boldsymbol{v}$, $\boldsymbol{g}$), from Duong et al. [duong_adaptive_2023-1, duong_adaptive_2023-2].** $\boldsymbol{v}$ tracks projections of principal activity along the $K$ frame directions (the columns of $\boldsymbol{W}_{yv}$). Subtracting $\boldsymbol{\mu}$ from $\boldsymbol{c}$ and squaring estimates the marginal variance along each direction. Eq. dg then adjusts each gain by comparing that estimate with its homeostatic target $\boldsymbol{\theta_t}$.
- **Each interneuron does three things:**
  1. senses its average synaptic input $\boldsymbol{\mu}$
  2. computes the squared difference between that average and its momentary input
  3. adjusts its gain by comparing the result to the target $\boldsymbol{\theta_t}$
- **High variance gets subtracted.** If variance is high along a column of $\boldsymbol{W}_{yv}$, a scalar multiple of that column is subtracted from the activity.
- **What the gains converge to.** They reach steady state when the marginal variances equal $\boldsymbol{\theta_t}$. Each interneuron may have its own target, so the circuit can approximate any covariance transformation.
- **Gains as Lagrange multipliers.** The gains enforce the target marginal variances along the overcomplete frame. In doing so they constrain the $N(N+1)/2$ degrees of freedom of positive semi-definite $N\times N$ matrices, i.e. the space of all covariance matrices.
- **Fixed frame.** Duong et al. [duong_adaptive_2023-2] also give a slow plasticity rule for $\boldsymbol{W}_{yv}$ that relaxes the overcompleteness requirement. This work uses a fixed frame.

### Modifications to Duong et al. needed for joint normalization and adaptation

1. **Prefactor $\frac{\boldsymbol{a}^+}{1+\boldsymbol{a}^+}$ on the gain feedback in Eq. dy.** It scales the gain feedback so the overall response magnitude stays normalized throughout adaptation; Eq. y_steady3 shows why.
2. **Explicit mean-centering in Eq. dg.** Duong et al. assume mean-centered inputs [duong_adaptive_2023-1]. Here, mean-centering the inputs themselves would alter the response magnitude and interfere with normalization. So mean-centering happens only inside the gain update: the gains then follow second-order statistics, $(\boldsymbol{z}-\langle\boldsymbol{z}\rangle)^2$, rather than first-order ones, $\boldsymbol{z}^2$.

## Fixed point

The fixed point of the recurrent dynamics is both normalized and covariance-transformed.

**Eq. y_steady**
$$
\boldsymbol{y}^2 = \frac{\boldsymbol{z'}^2}{\sigma^2 + \boldsymbol{N}\boldsymbol{z'}^2}
$$

**Eq. z_prime**
$$
\boldsymbol{z'} = \left[\boldsymbol{I} + \boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]^{-1}\boldsymbol{z}
$$

$\boldsymbol{z'}$ is the adapted input drive: a covariance transformation of the raw input, chosen so that the response covariance tends toward homeostasis. The fixed point is this adapted drive, normalized. It holds for any gain values during adaptation. Over time the marginal variances converge to their targets, and the response covariance across the input ensemble converges to the target covariance.

## Steady-state derivation

Assumptions:
- **Time-scale separation:** $\boldsymbol{g}$ is fixed relative to the fast dynamics.
- $\boldsymbol{W}_{yy} = \boldsymbol{I}$.
- **Membrane potentials are dominated by nonnegative values,** so $\sqrt{\boldsymbol{y}^+} \approx \boldsymbol{y}$, $\boldsymbol{a}^+ \approx \boldsymbol{a}$ and $\boldsymbol{u}^+ \approx |\boldsymbol{u}|^{1/2}$.
- **$\boldsymbol{a}$ is uniform** across neurons. This is exact when $\boldsymbol{N}$ is the all-ones matrix.
- $\beta = 1/2$.

From Eq. dv with $\boldsymbol{g}$ fixed:

$$
\boldsymbol{v}_{\text{steady}} = \boldsymbol{g}\odot\boldsymbol{W}_{yv}^\top\sqrt{\boldsymbol{y}^+}
$$

Substituting into Eqs. dy, du and da and setting the derivatives to zero:

**Eq. dy_0**
$$
0 = -\boldsymbol{y} + \beta\,\boldsymbol{z} + \left(\frac{1}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yy}\sqrt{\boldsymbol{y}^+} - \left(\frac{\boldsymbol{a}^+}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\sqrt{\boldsymbol{y}^+}
$$

**Eq. du_0**
$$
0 = -\boldsymbol{u} + \left(\frac{\sigma}{2}\right)^2 + \boldsymbol{N}\left(\boldsymbol{y}^+ \odot (\boldsymbol{u}^+)^{2}\right)
$$

**Eq. da_0**
$$
0 = -\boldsymbol{a} + \boldsymbol{u}^+ + \boldsymbol{a}\odot \boldsymbol{u}^+
$$

With $\boldsymbol{W}_{yy} = \boldsymbol{I}$, grouping the $\boldsymbol{y}$ terms in Eq. dy_0 gives:

**Eq. y_steady2**
$$
\left[\boldsymbol{I} - \left(\frac{1}{1+\boldsymbol{a}^+}\right)\boldsymbol{I} + \left(\frac{\boldsymbol{a}^+}{1+\boldsymbol{a}^+}\right)\boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]\boldsymbol{y} = \beta\,\boldsymbol{z}
$$

Simplifying and factoring:

**Eq. y_steady3**
$$
\left(\frac{\boldsymbol{a}^+}{1+\boldsymbol{a}^+}\right)\left[\boldsymbol{I} + \boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]\boldsymbol{y} = \beta\,\boldsymbol{z}
$$

This is why the prefactor is needed: it lets the covariance-transformation matrix $\left[\boldsymbol{I} + \boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]$ factor out. With $\boldsymbol{a}$ uniform, multiply by its inverse:

**Eq. y_steady4**
$$
\left(\frac{\boldsymbol{a}^+}{1+\boldsymbol{a}^+}\right)\boldsymbol{y} = \left[\boldsymbol{I} + \boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]^{-1}\beta\,\boldsymbol{z}
$$

Solving Eq. da_0 for $\boldsymbol{u}$ in terms of $\boldsymbol{a}$:

**Eq. sqrt_u**
$$
|\boldsymbol{u}|^{1/2} = \frac{\boldsymbol{a}}{1+\boldsymbol{a}}
$$

Using $\boldsymbol{a}^+ \approx \boldsymbol{a}$, substitute into the left side of Eq. y_steady4:

**Eq. y_steady5**
$$
|\boldsymbol{u}|^{1/2}\,\boldsymbol{y} = \beta\,\boldsymbol{z'}, \qquad \boldsymbol{z'} = \left[\boldsymbol{I} + \boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]^{-1}\boldsymbol{z}
$$

With $\beta = 1/2$, substituting into Eq. du_0:

**Eq. u_steady2**
$$
\frac{\sigma^2}{4} + \boldsymbol{N}\left(\frac{\boldsymbol{z'}^2}{4}\right) - \boldsymbol{z'}^2\,\frac{1}{4\boldsymbol{y}^2} = 0
$$

Solving gives the normalized, covariance-transformed steady state:

**Eq. final_y_steady**
$$
\boxed{\ \boldsymbol{y}^2 = \frac{\boldsymbol{z'}^2}{\sigma^2 + \boldsymbol{N}\boldsymbol{z'}^2}\ ; \quad \boldsymbol{z'} = \left[\boldsymbol{I} + \boldsymbol{W}_{yv}\boldsymbol{D}(\boldsymbol{g})\boldsymbol{W}_{yv}^\top\right]^{-1}\boldsymbol{z}\ }
$$

## Transcription notes

These are differences from the manuscript text, or things it leaves unstated.

1. **Eq. du_0 typo.** The manuscript writes $\boldsymbol{y_1^+}$ inside $\boldsymbol{N}(\cdot)$; it's transcribed here as $\boldsymbol{y}^+$ to match Eq. du, which is correct.
2. **The fixed point assumes $\beta = 1/2$.** The $\sigma^2$ in Eq. y_steady comes from the $(\sigma/2)^2$ constant in Eq. du combined with $\beta = 1/2$. If $\beta$ changes, Eq. y_steady no longer holds as written.
3. **Symbol $N$ is overloaded.** It is both the number of principal neurons and the normalization matrix $\boldsymbol{N}$. In fixed-point formulas, $\boldsymbol{N}\boldsymbol{z'}^2$ means the matrix applied to the element-wise square.
4. **Missing cross-references.** The manuscript refers to a standard normalization fixed point ("Equation norm_fixed", with $n = 2$) and a covariance-ellipse figure ("fig:cov_ellipse"). Neither is in the pasted section, so neither is reproduced here.
5. **Squares are element-wise.** $(\boldsymbol{c}-\boldsymbol{\mu})^2$, $\boldsymbol{y}^2$ and $\boldsymbol{z'}^2$ are all element-wise squares.
6. **Small edits.** Spelling in the prose was corrected ("inhereted", "superscipt"), and `\bm` was converted to `\boldsymbol` so the math renders in Markdown. No equations were changed apart from note 1.
