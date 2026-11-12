'''
---- simulation_variants.py ----

Flagged variants of the joint normalization + adaptation circuit. Nothing in
simulation.py is changed: this module re-implements V1Dynamics_Surround's equations with
every alternative studied in the supplementary investigation behind an option, so that the
codebase ("old") and revised ("new") equations run side by side.

Time unit: ms (as in simulation.py). Integration: classical RK4 with fixed step dt, as in
V1Dynamics_Surround.run_simulation; gains are clamped to >= 0 after every step.

Variables (B = batch of independent simulations run in parallel)
    y  (B, N_TOT)  principal neurons, N_TOT = N_RF * N_SETS (set 0 = cRF, sets 1.. = surround)
    u  (B, N_TOT)  normalization interneurons (per neuron, so that N may be non-uniform)
    a  (B, N_TOT)  normalization interneurons
    g  (B, 2, K)   interneuron gains: [cRF, surround]. As in V1Dynamics_Surround, one shared
                   surround copy is driven by surround set 1 and applied to every surround set.
    v  (B, 2, K)   adaptation interneurons
    f  (B, 2, K)   c low-pass filtered at tau_v (only used by variance_signal='filtered' when
                   gain_placement='interneuron'; equals v when gain_placement='principal')
    mu (B, 2, N_RF) slow running mean of the principal responses; W.T @ mu is the running mean
                   of c (Eq. dmu is linear, so tracking y and projecting is the same as tracking c)

Equation options (EquationOptions)
    gating              'a'  feedback gated by a+/(1+a+)                      [codebase]
                        'u'  feedback gated by u+                             [Eq. 1]
    gain_placement      'principal'    feedback W (g*v),  tau_v dv/dt = -v + c    [codebase]
                        'interneuron'  feedback W v,      tau_v dv/dt = -v + g*c  [Eq. 4]
    variance_signal     'filtered'  tau_g dg/dt = (c_filtered - mu)^2 - theta_t   [codebase: uses v]
                        'instant'   tau_g dg/dt = (c - mu)^2 - theta_t            [Eq. 5]
    rectify_c           False  c = W.T y                                      [codebase]
                        True   c = W.T max(y, 0)                              [Eqs. 1-6]
    rectified_variance  True   tau_g dg/dt = max(c - mu, 0)^2 - theta_t       [Eq. 15]
    normalization       False  no u, a, W_yy; beta = 1; ungated feedback      [Eqs. 9-12]

Other options: arbitrary normalization weights N (N_TOT, N_TOT), noise (NoiseOptions), and the
surround -> cRF conduction delay of V1Dynamics_Surround.

Known numerical limits (see tests/test_simulation_variants.py):
  - The delayed surround signal is sampled once per step, as in V1Dynamics_Surround, so with a
    delay the scheme is first order in dt and the effective delay is (n_lag - 1/2) dt.
  - With white_input noise the Euler-Maruyama kick is split, half before and half after the RK4
    step. The variance of the *recorded* y still depends weakly on dt (a few per cent at
    dt = tau_y / 4). What the slow variables (v, g, mu) integrate does not.
  - With poisson_response, downstream terms receive the square root of a count-based rate, which
    is biased low at low counts.
'''

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class EquationOptions:
    gating: str = 'a'
    gain_placement: str = 'principal'
    variance_signal: str = 'filtered'
    rectify_c: bool = False
    rectified_variance: bool = False
    normalization: bool = True

    def __post_init__(self):
        assert self.gating in ('a', 'u')
        assert self.gain_placement in ('principal', 'interneuron')
        assert self.variance_signal in ('filtered', 'instant')


# The equations exactly as implemented in simulation.V1Dynamics_Surround
CODEBASE = EquationOptions()
# Eqs. 1-6 of the instructions (u+ gating, gain inside the v equation, variance from c)
NEW = EquationOptions(gating='u', gain_placement='interneuron', variance_signal='instant', rectify_c=True)


@dataclass(frozen=True)
class NoiseOptions:
    '''
    kind:
      'none'
      'white_input'       z -> z + xi, xi Gaussian white noise, added to the y equation with
                          Euler-Maruyama scaling (sqrt(dt)), fresh every step.
      'white_response'    sqrt(y+) -> sqrt(y+) + eta as seen by every downstream term (recurrent
                          drive, normalization pool, adaptation interneurons). eta is white noise
                          band-limited at tau_n (an Ornstein-Uhlenbeck process driven by fresh
                          Gaussian increments every step, exact update), because white noise
                          entering an algebraic, nonlinear term has infinite variance.
      'poisson_input'     z replaced, in every bin of bin_ms, by ref * n / (R * bin), with
                          n ~ Poisson(R * bin * z / ref).
      'poisson_response'  the firing rate y+ seen by every downstream term replaced, in every bin,
                          by ref * n / (R * bin), n ~ Poisson(R * bin * y+ / ref), y+ read at the
                          start of the bin.
    amplitude (scalar or (B,)):
      white:   s.d. of the noise after first-order low-pass filtering at tau_n, in the units of
               z (input) or sqrt(y+) (response). For white_input the white-noise intensity is
               therefore amplitude * sqrt(2 tau_n).
      poisson: rate scale R in spikes/s at the reference level `ref` (np.inf = no noise).
    '''
    kind: str = 'none'
    amplitude: object = 0.0
    ref: float = 1.0
    tau_n: float = 1.0
    bin_ms: float = 1.0

    def __post_init__(self):
        assert self.kind in ('none', 'white_input', 'white_response', 'poisson_input', 'poisson_response')


@dataclass(frozen=True)
class ModelParams:
    '''Defaults are the values in simulation.V1Dynamics_Surround (ms).'''
    tau_y: float = 0.4
    tau_a: float = 0.2
    tau_u: float = 30.0
    tau_v: float = 30.0
    tau_g: float = 500.0
    tau_mu: float = 2000.0
    sigma: float = 0.15
    beta: float = 0.5


def theta_t_from_target(W, target_covariance):
    '''Marginal target variances of the adaptation interneurons: diag(W.T C W), as in
    V1Dynamics_Surround.calibrate_theta_t(uniform_target=True).'''
    return np.einsum('ik,ij,jk->k', W, target_covariance, W)


class AdaptiveORGaNICs:
    def __init__(self, W, theta_t, N_RF=13, N_SETS=6, options=CODEBASE, params=ModelParams(),
                 N_matrix=None, W_yy=None, dt=0.1, surround_delay=0.0, noise=NoiseOptions(),
                 gains_nonneg=True):
        self.W = np.asarray(W, dtype=float)              # (N_RF, K) frame, unit-norm columns
        self.N_RF, self.N_SETS = N_RF, N_SETS
        self.N_TOT = N_RF * N_SETS
        self.K = self.W.shape[1]
        assert self.W.shape[0] == N_RF
        self.opt = options
        self.p = params
        self.dt = dt
        self.theta_t = np.asarray(theta_t, dtype=float)  # (K,) or (B, K)
        self._theta2 = np.concatenate([self.theta_t, self.theta_t], axis=-1)   # cRF and surround copies
        self.noise = noise
        self.gains_nonneg = gains_nonneg
        # Normalization weights: all-ones unless given. (N_TOT, N_TOT) or (B, N_TOT, N_TOT).
        self.N_matrix = np.ones((self.N_TOT, self.N_TOT)) if N_matrix is None else np.asarray(N_matrix, float)
        # Recurrent weights within one RF (identity in tunings.V1Tunings)
        self.W_yy = None if (W_yy is None or np.array_equal(W_yy, np.eye(N_RF))) else np.asarray(W_yy, float)
        self.set_surround_delay(surround_delay)

        NT, K = self.N_TOT, self.K
        self.sl = {}
        off = 0
        for name, size in (('y', NT), ('u', NT), ('a', NT), ('g', 2 * K), ('v', 2 * K), ('f', 2 * K),
                           ('mu', 2 * N_RF)):
            self.sl[name] = slice(off, off + size)
            off += size
        self.state_size = off

    # ------------------------------------------------------------------ helpers
    def set_surround_delay(self, surround_delay):
        '''Delay (ms) on the surround's contribution to the cRF's normalization pool. Same
        convention as V1Dynamics_Surround: sampled once per step from the state n_lag steps ago.'''
        self.surround_delay = surround_delay
        self.n_lag = int(round(surround_delay / self.dt))
        N = self.N_matrix
        if self.n_lag > 0:
            n = self.N_RF
            N_del = np.zeros_like(N)
            N_del[..., :n, n:] = N[..., :n, n:]      # surround -> cRF entries arrive late
            self._N_now, self._N_del = N - N_del, N_del
        else:
            self._N_now, self._N_del = N, None

    def with_options(self, **changes):
        '''Copy of this model with some EquationOptions fields changed.'''
        new = AdaptiveORGaNICs.__new__(AdaptiveORGaNICs)
        new.__dict__.update(self.__dict__)
        new.opt = replace(self.opt, **changes)
        return new

    def zero_state(self, B=1):
        return np.zeros((B, self.state_size))

    def unpack(self, state):
        '''Named views of a (B, state_size) state.'''
        K, n = self.K, self.N_RF
        s = self.sl
        g, v, f, mu = state[:, s['g']], state[:, s['v']], state[:, s['f']], state[:, s['mu']]
        return dict(y=state[:, s['y']], u=state[:, s['u']], a=state[:, s['a']],
                    g_cRF=g[:, :K], g_surround=g[:, K:], v_cRF=v[:, :K], v_surround=v[:, K:],
                    f_cRF=f[:, :K], f_surround=f[:, K:], mu_cRF=mu[:, :n], mu_surround=mu[:, n:])

    def _pool(self, q, q_delayed):
        def apply(N, x):                       # (N x) for every row; one matrix or one per row
            return x @ N.T if N.ndim == 2 else np.matmul(N, x[..., None])[..., 0]
        if self._N_del is None:
            return apply(self._N_now, q)
        return apply(self._N_now, q) + apply(self._N_del, q_delayed)

    def _seen_response(self, y, extra):
        '''sqrt(y+) and y+ as seen by downstream terms (recurrent drive, pool, interneurons).'''
        yr = np.maximum(y, 0.0)
        kind = self.noise.kind
        if kind == 'white_response':
            r = yr + extra.get('eta', 0.0)
            return r, r * r
        if kind == 'poisson_response' and extra.get('rate_hat') is not None:
            rate = np.where(extra['noisy'], extra['rate_hat'], yr * yr)
            return np.sqrt(rate), rate
        return yr, yr * yr

    def q_of(self, state, extra=None):
        '''Each neuron's contribution to the normalization pool, y+ * (u+)^2 = max(y,0)^2 max(u,0).'''
        extra = extra or {}
        _, rate = self._seen_response(state[:, self.sl['y']], extra)
        return rate * np.maximum(state[:, self.sl['u']], 0.0)

    # ------------------------------------------------------------------ dynamics
    def derivatives(self, state, z, q_delayed, adapt, extra):
        o, p, s = self.opt, self.p, self.sl
        K, n, NS = self.K, self.N_RF, self.N_SETS
        W = self.W
        y, u, a = state[:, s['y']], state[:, s['u']], state[:, s['a']]
        g, v, f, mu = state[:, s['g']], state[:, s['v']], state[:, s['f']], state[:, s['mu']]

        r_seen, rate_seen = self._seen_response(y, extra)         # sqrt(y+), y+ seen downstream
        # What the adaptation interneurons and the running mean see. Response noise does not
        # change the rectify_c option: without noise each case reduces to the noise-free one.
        y_clean = np.maximum(y, 0.0) if o.rectify_c else y
        if self.noise.kind == 'white_response':
            y_for_c = y_clean + extra.get('eta', 0.0)
        elif self.noise.kind == 'poisson_response' and extra.get('rate_hat') is not None:
            y_for_c = np.where(extra['noisy'], r_seen, y_clean)   # counts are rectified by nature
        else:
            y_for_c = y_clean

        # Eq. c: input current to the adaptation interneurons, one cRF copy and one surround copy
        c = np.concatenate([y_for_c[:, :n] @ W, y_for_c[:, n:2 * n] @ W if NS >= 2 else np.zeros_like(g[:, :K])],
                           axis=1)                                # (B, 2K)
        mu_c = np.concatenate([mu[:, :n] @ W, mu[:, n:] @ W], axis=1)

        # Eq. dv, feedback
        if o.gain_placement == 'principal':
            dv = (-v + c) / p.tau_v
            fb_int = g * v
            df = dv                                               # f == v in this placement
        else:
            dv = (-v + g * c) / p.tau_v
            fb_int = v
            df = (-f + c) / p.tau_v
        fb_cRF = fb_int[:, :K] @ W.T
        fb_sur = fb_int[:, K:] @ W.T
        fb = np.concatenate([fb_cRF] + [fb_sur] * (NS - 1), axis=1)

        # Eq. dg, dmu
        if adapt:
            if o.variance_signal == 'instant':
                dev = c - mu_c
            else:
                dev = (v if o.gain_placement == 'principal' else f) - mu_c
            if o.rectified_variance:
                dev = np.maximum(dev, 0.0)
            dg = (dev * dev - self._theta2) / p.tau_g
            y_mu = y_for_c[:, :2 * n] if NS >= 2 else np.concatenate([y_for_c[:, :n], np.zeros_like(mu[:, n:])], axis=1)
            dmu = (-mu + y_mu) / p.tau_mu
        else:
            dg = np.zeros_like(g)
            dmu = np.zeros_like(mu)

        if o.normalization:
            u_plus = np.sqrt(np.maximum(u, 0.0))
            a_plus = np.maximum(a, 0.0)
            rec = r_seen if self.W_yy is None else np.concatenate(
                [r_seen[:, i * n:(i + 1) * n] @ self.W_yy.T for i in range(NS)], axis=1)
            gate = u_plus if o.gating == 'u' else a_plus / (1.0 + a_plus)
            # Eq. dy
            dy = (-y + p.beta * z + rec / (1.0 + a_plus) - gate * fb) / p.tau_y
            # Eq. du: pool of y+ (u+)^2
            q = rate_seen * np.maximum(u, 0.0)
            du = (-u + (p.sigma / 2.0) ** 2 + self._pool(q, q_delayed)) / p.tau_u
            # Eq. da
            da = (-a + (1.0 + a) * u_plus) / p.tau_a
        else:
            # Eqs. 9-12: leak, input drive (beta = 1) and ungated gain feedback only
            dy = (-y + z - fb) / p.tau_y
            du = np.zeros_like(u)
            da = np.zeros_like(a)

        return np.concatenate([dy, du, da, dg, dv, df, dmu], axis=1)

    def simulate(self, z_pres, steps_per_pres, state0=None, adapt=True, rng=None,
                 q_baseline=None, record_traces=False, record_g_every_pres=1):
        '''
        z_pres: (n_pres, N_TOT) or (B, n_pres, N_TOT): the input drive of each presentation,
        held for steps_per_pres steps. Returns a dict with
            y_end   (B, n_pres, N_TOT)  y at the last step of each presentation (not rectified)
            g_end   (B, n_rec, 2K)      gains at the end of every record_g_every_pres-th presentation
            state   (B, state_size)     final state
            traces  dict of full time courses y, u, a, v (B, n_steps, .) if record_traces
        q_baseline (B or 1, N_TOT): pool contributions used for the delayed surround signal until
        n_lag steps have elapsed (default 0).
        '''
        z_pres = np.asarray(z_pres, dtype=float)
        if z_pres.ndim == 2:
            z_pres = z_pres[None]
        n_pres = z_pres.shape[1]
        if state0 is None:
            B = max(z_pres.shape[0], np.ndim(self.theta_t) == 2 and self.theta_t.shape[0] or 1,
                    self.N_matrix.ndim == 3 and self.N_matrix.shape[0] or 1,
                    np.size(self.noise.amplitude))
            state = self.zero_state(B)
        else:
            state = np.array(state0, dtype=float)
            B = state.shape[0]
        s, K, NT, dt, n_lag = self.sl, self.K, self.N_TOT, self.dt, self.n_lag
        noise, p = self.noise, self.p
        rng = rng if rng is not None else np.random.default_rng(0)
        amp = np.reshape(np.asarray(noise.amplitude, dtype=float), (-1, 1))

        y_end = np.zeros((B, n_pres, NT))
        rec_idx = np.arange(record_g_every_pres - 1, n_pres, record_g_every_pres)
        g_end = np.zeros((B, len(rec_idx), 2 * K))
        n_steps = n_pres * steps_per_pres
        traces = None
        if record_traces:
            traces = dict(y=np.zeros((B, n_steps, NT)), u=np.zeros((B, n_steps, NT)),
                          a=np.zeros((B, n_steps, NT)), v=np.zeros((B, n_steps, 2 * K)))

        # delayed surround -> cRF pool contributions
        if n_lag > 0:
            ring = np.zeros((n_lag + 1, B, NT))
            q_base = np.zeros((B, NT)) if q_baseline is None else np.broadcast_to(q_baseline, (B, NT))
        q_delayed = None

        extra = {}
        bin_steps = max(1, int(round(noise.bin_ms / dt)))
        bin_s = bin_steps * dt * 1e-3                               # bin width in seconds
        if noise.kind == 'white_response':
            extra['eta'] = amp * rng.standard_normal((B, NT))       # stationary start
            ou_decay = np.exp(-dt / noise.tau_n)
            ou_kick = amp * np.sqrt(1.0 - ou_decay ** 2)
        if noise.kind == 'white_input':
            # intensity such that the noise low-passed at tau_n has s.d. amp; enters dy via beta
            beta_eff = p.beta if self.opt.normalization else 1.0
            em_kick = beta_eff * amp * np.sqrt(2.0 * noise.tau_n) * np.sqrt(dt) / p.tau_y
            em_half = em_kick / np.sqrt(2.0)
        if noise.kind in ('poisson_input', 'poisson_response'):
            noisy = np.isfinite(amp) & (amp > 0)                    # (B, 1): rows with noise
            lam_scale = np.where(noisy, amp, 1.0) * bin_s / noise.ref
            extra['noisy'] = noisy

        deriv = self.derivatives
        t = 0
        gi = 0
        for ip in range(n_pres):
            z_clean = z_pres[:, ip, :]
            z = z_clean
            for i_step in range(steps_per_pres):
                if n_lag > 0:
                    lag = t - n_lag
                    q_delayed = ring[lag % (n_lag + 1)] if lag >= 0 else q_base
                if noise.kind == 'poisson_input' and (t % bin_steps == 0 or i_step == 0):
                    # new counts every bin and at every stimulus onset, independent across the batch
                    rate = np.broadcast_to(np.maximum(z_clean, 0.0) * lam_scale, (B, NT))
                    z = np.where(noisy, rng.poisson(rate) / lam_scale, z_clean)
                elif noise.kind == 'poisson_response' and t % bin_steps == 0:
                    yr = np.maximum(state[:, s['y']], 0.0)
                    extra['rate_hat'] = rng.poisson(yr * yr * lam_scale) / lam_scale

                if noise.kind == 'white_input':
                    # Euler-Maruyama kick, half before and half after the drift step, so that the
                    # recorded y depends only weakly on dt
                    state = state.copy()
                    state[:, s['y']] += em_half * rng.standard_normal((B, NT))
                k1 = deriv(state, z, q_delayed, adapt, extra)
                k2 = deriv(state + 0.5 * dt * k1, z, q_delayed, adapt, extra)
                k3 = deriv(state + 0.5 * dt * k2, z, q_delayed, adapt, extra)
                k4 = deriv(state + dt * k3, z, q_delayed, adapt, extra)
                state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                if noise.kind == 'white_input':
                    state[:, s['y']] += em_half * rng.standard_normal((B, NT))
                elif noise.kind == 'white_response':
                    extra['eta'] = ou_decay * extra['eta'] + ou_kick * rng.standard_normal((B, NT))
                if self.gains_nonneg:
                    np.maximum(state[:, s['g']], 0.0, out=state[:, s['g']])
                if n_lag > 0:
                    ring[t % (n_lag + 1)] = self.q_of(state, extra)
                if record_traces:
                    traces['y'][:, t] = state[:, s['y']]
                    traces['u'][:, t] = state[:, s['u']]
                    traces['a'][:, t] = state[:, s['a']]
                    traces['v'][:, t] = state[:, s['v']]
                t += 1
            y_end[:, ip] = state[:, s['y']]
            if gi < len(rec_idx) and ip == rec_idx[gi]:
                g_end[:, gi] = state[:, s['g']]
                gi += 1

        out = dict(y_end=y_end, g_end=g_end, state=state)
        if record_traces:
            out['traces'] = traces
        return out

    # ------------------------------------------------------------------ frozen-gain probes
    def settle(self, z, g_cRF, g_surround, settle_ms=600.0, state0=None, record_traces=False, rng=None):
        '''
        Integrates y, u, a, v to steady state for constant inputs z (n_probes, N_TOT) with the
        gains frozen (no surround delay: at steady state the delay is irrelevant). g_cRF,
        g_surround: (K,) or (n_probes, K). Starts from zero unless state0 is given.
        Returns the simulate() dict of a single presentation.
        '''
        z = np.atleast_2d(np.asarray(z, dtype=float))
        B = z.shape[0]
        state = self.zero_state(B) if state0 is None else np.array(state0, dtype=float)
        state[:, self.sl['g']] = np.concatenate([np.broadcast_to(g_cRF, (B, self.K)),
                                                np.broadcast_to(g_surround, (B, self.K))], axis=1)
        probe = AdaptiveORGaNICs.__new__(AdaptiveORGaNICs)
        probe.__dict__.update(self.__dict__)
        probe.set_surround_delay(0.0)
        n_steps = int(round(settle_ms / self.dt))
        return probe.simulate(z[:, None, :], n_steps, state0=state, adapt=False,
                              record_traces=record_traces, rng=rng)

    def gain_operator(self, g_cRF, g_surround):
        '''(N_TOT, N_TOT) block-diagonal W diag(g) W.T (cRF block, then the surround blocks).'''
        from scipy.linalg import block_diag
        W = self.W
        return block_diag(W @ np.diag(g_cRF) @ W.T, *([W @ np.diag(g_surround) @ W.T] * (self.N_SETS - 1)))

    def analytic_fixed_point(self, z, g_cRF, g_surround):
        '''
        Closed-form fast fixed point for frozen gains, all-ones normalization weights and
        W_yy = I (Eqs. 10-11 of the manuscript):
            z' = (I + W D(g) W.T)^-1 z,   y = z' / sqrt(sigma^2 + sum z'^2),
            u = (sigma^2 + sum z'^2) / 4,   a / (1 + a) = sqrt(u)
        Without normalization (Eqs. 9-12): y = z'. z: (n_probes, N_TOT). Valid only where z' >= 0.
        Returns dict(y, u, a, z_prime).
        '''
        z = np.atleast_2d(z)
        z_prime = np.linalg.solve(np.eye(self.N_TOT) + self.gain_operator(g_cRF, g_surround), z.T).T
        if not self.opt.normalization:
            return dict(y=z_prime, z_prime=z_prime)
        d = self.p.sigma ** 2 + np.sum(z_prime ** 2, axis=1, keepdims=True)
        u = d / 4.0
        su = np.sqrt(u)
        return dict(y=z_prime / np.sqrt(d), u=np.broadcast_to(u, z.shape), a=np.broadcast_to(su / (1 - su), z.shape),
                    z_prime=z_prime)
