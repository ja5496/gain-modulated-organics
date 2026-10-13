'''
response_tools.py

Shared helpers for probing V1Dynamics_Surround (simulation_whiten.py) with its adaptation gains
held fixed, plus a tuning-curve peak finder. Used by adapt_single_stimulus.py, adapt_ensembles.py,
Kohn_distinct_plots.py and Dario_plots.py.

Two ways to compute a frozen-gain probe response:
- get_response_offline: closed-form fixed point (fast; v at its fast-dynamics limit).
- get_response_online: integrates y/u/a/v to steady state with RK4 (slow, exact).
'''

import numpy as np
from scipy.linalg import block_diag

N_SETTLE_STEPS = 1500   # RK4 steps used by the online probes to settle y/u/a/v (dt=0.1 -> 150s)


def offline_gain_operator(dyn, g_cRF, g_surround):
    '''(N_TOT, N_TOT) block-diagonal M = W diag(g) W.T: the cRF block uses g_cRF and every
    surround block reuses g_surround (same layout as V1Dynamics_Surround's gain feedback).'''
    W = dyn.frame.W
    M_cRF = W @ np.diag(g_cRF) @ W.T
    M_surround = W @ np.diag(g_surround) @ W.T
    return block_diag(M_cRF, *([M_surround] * (dyn.N_SETS - 1)))


def get_response_offline(dyn, stimulus, M, Beta=0.5):
    '''Fixed point with gains frozen in M and v at its fast limit: normalized, whitened input
    y = z'/sqrt(sigma^2 + N z'^2), z' = (I + M)^-1 stimulus. Returns firing rates (rectified y^2).'''
    z_prime = np.linalg.inv(np.eye(stimulus.size) + M) @ stimulus
    y = z_prime / np.sqrt(dyn.sigma**2 + dyn.N_matrix @ (z_prime**2))
    rectified_y = dyn.half_wave_rectify(y)

    return rectified_y


def frozen_derivatives(state, z_t, dyn, g_cRF, g_surround):
    '''
    y/u/a/v_cRF/v_surround dynamics of V1Dynamics_Surround._derivatives with g_cRF/g_surround
    held fixed. Hand-maintained mirror of _derivatives - keep the two in sync. Own state layout:
    [y, u, a] (each N_TOT, per-neuron u/a pooled through dyn.N_matrix, no surround delay), then
    [v_cRF, v_surround] (each K).
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    N_RF = dyn.N_RF
    K = dyn.frame.K

    y = state[0:N_TOT]
    u = state[N_TOT:2*N_TOT]
    a = state[2*N_TOT:3*N_TOT]
    v_cRF = state[3*N_TOT:3*N_TOT+K]
    v_surround = state[3*N_TOT+K:3*N_TOT+2*K]

    u_plus = dyn.half_wave_rectify(u, 0.5)
    y_plus = dyn.half_wave_rectify(y, 2.0)
    a_plus = dyn.half_wave_rectify(a, 1.0)
    sqrt_y_plus = np.sqrt(y_plus)

    dv_cRF_dt = (-v_cRF + dyn.frame.W.T @ y[:N_RF]) / dyn.tau_v
    dv_surround_dt = (-v_surround + dyn.frame.W.T @ y[N_RF:2*N_RF]) / dyn.tau_v

    cRF_gain_feedback = dyn.frame.W @ (g_cRF * v_cRF)
    surround_gain_feedback = dyn.frame.W @ (g_surround * v_surround)
    full_gain_feedback = (a_plus / (1 + a_plus)) * np.concatenate([cRF_gain_feedback] + [surround_gain_feedback] * (dyn.N_SETS - 1))

    recurrent_drive = (1.0 / (1.0 + a_plus)) * (dyn.W_yy @ sqrt_y_plus)
    input_drive = dyn.beta * z_t

    sigma_term = (dyn.sigma / 2) ** 2
    pool_term = dyn.N_matrix @ (y_plus * (u_plus ** 2))

    dy_dt = (-y + input_drive + recurrent_drive - full_gain_feedback) / dyn.tau_y
    du_dt = (-u + sigma_term + pool_term) / dyn.tau_u
    da_dt = (-a + (1 + a) * u_plus) / dyn.tau_a

    return np.concatenate([dy_dt, du_dt, da_dt, dv_cRF_dt, dv_surround_dt])


def get_response_online(dyn, stimulus, g_cRF, g_surround, mu_cRF, mu_surround, n_steps=N_SETTLE_STEPS):
    '''
    Settles y/u/a/v for a fixed probe stimulus (RK4, n_steps) with gains frozen. Starts from
    y = u = a = 0 and v = W.T @ mu every call, so probes don't depend on sweep order.
    Returns (firing rates, v_cRF_final, v_surround_final).
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    K = dyn.frame.K
    dt = dyn.dt

    v_cRF_init = dyn.frame.W.T @ mu_cRF
    v_surround_init = dyn.frame.W.T @ mu_surround

    state = np.zeros(3 * N_TOT + 2 * K)
    state[3*N_TOT:3*N_TOT+K] = v_cRF_init
    state[3*N_TOT+K:3*N_TOT+2*K] = v_surround_init

    for _ in range(n_steps):
        k1 = frozen_derivatives(state, stimulus, dyn, g_cRF, g_surround)
        k2 = frozen_derivatives(state + 0.5 * dt * k1, stimulus, dyn, g_cRF, g_surround)
        k3 = frozen_derivatives(state + 0.5 * dt * k2, stimulus, dyn, g_cRF, g_surround)
        k4 = frozen_derivatives(state + dt * k3, stimulus, dyn, g_cRF, g_surround)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    y_final = state[0:N_TOT]
    v_cRF_final = state[3*N_TOT:3*N_TOT+K]
    v_surround_final = state[3*N_TOT+K:3*N_TOT+2*K]

    y_final_rect = dyn.half_wave_rectify(y_final, 2.0)
    return y_final_rect, v_cRF_final, v_surround_final


def get_response_traced(dyn, stimulus, g_cRF, g_surround, mu_cRF, mu_surround, n_steps=N_SETTLE_STEPS):
    '''
    Same dynamics as get_response_online, but returns the full time courses
    (y_hist, u_hist, a_hist, v_cRF_hist, v_surround_hist), each (dim, n_steps) - for checking
    that n_steps is long enough for a new probe to actually settle.
    '''
    N_TOT = dyn.N_RF * dyn.N_SETS
    K = dyn.frame.K
    dt = dyn.dt

    v_cRF_init = dyn.frame.W.T @ mu_cRF
    v_surround_init = dyn.frame.W.T @ mu_surround

    state = np.zeros(3 * N_TOT + 2 * K)
    state[3*N_TOT:3*N_TOT+K] = v_cRF_init
    state[3*N_TOT+K:3*N_TOT+2*K] = v_surround_init

    y_hist = np.zeros((N_TOT, n_steps))
    u_hist = np.zeros((N_TOT, n_steps))
    a_hist = np.zeros((N_TOT, n_steps))
    v_cRF_hist = np.zeros((K, n_steps))
    v_surround_hist = np.zeros((K, n_steps))

    for t in range(n_steps):
        k1 = frozen_derivatives(state, stimulus, dyn, g_cRF, g_surround)
        k2 = frozen_derivatives(state + 0.5 * dt * k1, stimulus, dyn, g_cRF, g_surround)
        k3 = frozen_derivatives(state + 0.5 * dt * k2, stimulus, dyn, g_cRF, g_surround)
        k4 = frozen_derivatives(state + dt * k3, stimulus, dyn, g_cRF, g_surround)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        y_hist[:, t] = state[0:N_TOT]
        u_hist[:, t] = state[N_TOT:2*N_TOT]
        a_hist[:, t] = state[2*N_TOT:3*N_TOT]
        v_cRF_hist[:, t] = state[3*N_TOT:3*N_TOT+K]
        v_surround_hist[:, t] = state[3*N_TOT+K:3*N_TOT+2*K]

    return y_hist, u_hist, a_hist, v_cRF_hist, v_surround_hist


def curve_peak_deg(curve, angles_deg):
    '''Peak orientation (deg) of a tuning curve sampled on an evenly spaced, periodic grid
    angles_deg over [0, 180): parabolic interpolation around the argmax sample, with neighbors
    wrapping so a peak at the 0/180 edge is interpolated too.'''
    i = int(np.argmax(curve))
    y0, y1, y2 = curve[i - 1], curve[i], curve[(i + 1) % len(curve)]
    denom = (y0 - 2 * y1 + y2)
    frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    step = angles_deg[1] - angles_deg[0]
    return angles_deg[i] + frac * step
