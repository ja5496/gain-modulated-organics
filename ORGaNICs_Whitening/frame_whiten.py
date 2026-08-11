'''
----  frame_whiten.py: ----
Creates a fixed overcomplete frame of synaptic weights used during the whitening process to project
primary neurons onto interneuron axes (the axes of this frame). Frame is an N x K matrix where N is the number 
of primary neurons and K >= N(N+1)/2 (I set this equal in this code). Taken from Ch.2 of Lyndon Duong's thesis.
'''
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import sqrtm
from stimuli_whiten import StimulusGenerator

class Frame:
    def __init__(self, dim: int, frame_type: str = 'mercedes', sigma: float = 0.3, noise_std: float = 0.05):
        self.dim = int(dim) # Number of primary neurons
        self.sigma = sigma
        self.centers = None  # Only set for bell-shaped frames
        if frame_type == 'mercedes':
            self.K = int(self.dim * (self.dim + 1) // 2)
            print(f"Building Smooth Mercedes Frame (N={self.dim}, K={self.K})...")
            self.W = self.mercedes()
        elif frame_type == 'gaussian':
            self.K = 2 * self.dim
            print(f"Building Gaussian Frame (N={self.dim}, K={self.K})...")
            self.W = self.create_gaussian_frame()
        elif frame_type == 'identity':
            print(f"Building Identity Frame (N={self.dim}, N={self.dim})...")
            self.W = self.create_identity_frame()
        else:
            self.K = int(self.dim * (self.dim + 1) // 2)
            print(f"Building Optimal Frame (N={self.dim}, K={self.K})...")
            self.W = self.optimal_frame()
            self.g = np.zeros(self.K) # Initialize gains at 0.

    def mercedes(self) -> np.ndarray:
        N, K = self.dim, self.K

        # Step 1: Generate Random Vectors
        num_candidates = 5 * K
        A = np.random.randn(num_candidates, N)
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        # Track indices in A that are still available
        candidate_indices = np.arange(num_candidates)
        
        # Initialize the Frame with the first vector
        W = np.zeros((N, K))
        W[:, 0] = A[0]
        
        # Remove first vector from candidates
        active_mask = np.ones(num_candidates, dtype=bool)
        active_mask[0] = False
        
        # Track the maximum coherence of each candidate with the CURRENT frame.
        current_max_coherences = np.abs(A @ W[:, 0])

        for k in tqdm(range(1, K), desc="Frame Init"):
            valid_coherences = current_max_coherences[active_mask]
            
            # Find vector with the min coherence among the valid ones
            # We need the index relative to the compressed 'valid' array
            local_best_idx = np.argmin(valid_coherences)
            
            # Map this back to the global index in A
            # (np.where returns indices where active_mask is True)
            global_indices = np.where(active_mask)[0]
            best_global_idx = global_indices[local_best_idx]
            
            # Add this vector to our frame
            new_vec = A[best_global_idx]
            W[:, k] = new_vec
            
            # Remove from active set
            active_mask[best_global_idx] = False
            
            # OPTIMIZATION: Incremental Update
            # Instead of matrix mult against the WHOLE frame, we only compute 
            # dot products against the NEW vector.
            # Then we update the max_coherence array.
            new_dots = np.abs(A @ new_vec)
            current_max_coherences = np.maximum(current_max_coherences, new_dots)

        return W
    
    def create_identity_frame(self) -> np.ndarray:
        N = self.dim
        return np.eye(N)

    def create_gaussian_frame(self) -> np.ndarray:
        N, K = self.dim, self.K
        theta = np.linspace(0, np.pi, N, endpoint=False)
        centers = np.linspace(0, np.pi, K, endpoint=False)

        W = np.zeros((N, K))
        for k in range(K):
            delta = theta - centers[k]
            delta = (delta + np.pi / 2) % np.pi - np.pi / 2
            col = np.exp(-delta ** 2 / (2 * self.sigma ** 2))
            #col = Frame.gaussian_rectify(col)
            W[:, k] = col / np.linalg.norm(col)

        # Transform to tight frame + handle noise: 
        G = W @ W.T
        S, U = np.linalg.eigh(G)
        S = np.maximum(S, 1e-10) 
        G_inv_sqrt = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
        W_tight = G_inv_sqrt @ W
        tight_norms = np.linalg.norm(W_tight, axis=0)
        W_tight = W_tight / tight_norms
        
        W_tight = np.real(W_tight)
        #W_tight = np.maximum(W_tight, 0)

        _, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(W_tight, cmap='viridis', aspect='auto')
        ax.set_title(r'Frame $W$ (Gaussian)', fontsize=14)
        ax.set_xlabel('Frame vector index', fontsize=12)
        ax.set_ylabel('Neuron index', fontsize=12)
        plt.colorbar(im, ax=ax)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(width=2.0, length=5)
        plt.tight_layout()
        plt.show()

        return W_tight
    
    def gaussian_rectify(y, threshold=0.6, sigma=0.35, r_max=1.0):
        return 0.5 * (1 + erf((y - threshold) / (sigma * np.sqrt(2)))) * r_max

    def optimal_frame(self) -> np.ndarray:
        N = self.dim
        stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=10920)
        orientation_inputs = np.array(stim_gen.generate_input_ensembles(biased=True, mean_center=True))
        orientation_inputs = Frame.gaussian_rectify(orientation_inputs)
        cov_matrix = np.cov(orientation_inputs, rowvar=True)
        eigenvalues, W = np.linalg.eigh(cov_matrix)
        eigenvalues = np.clip(eigenvalues, 0, None)
        print(np.sort(eigenvalues)[::-1])
        norms = np.linalg.norm(W, axis=0)
        W = W / norms
        Lambda = np.diag((np.sqrt(eigenvalues) - 1) * norms**2)


        # Compute C_ss^(1/2) via W_orig = W * norms
        W_orig = W * norms[np.newaxis, :]
        cov_sqrt = W_orig @ np.diag(np.sqrt(eigenvalues)) @ W_orig.T
        reconstruction = np.eye(N) + W @ Lambda @ W.T

        # Whitening matrix C_ss^(-1/2): pinv handles rank deficiency gracefully
        cov_sqrt_inv = np.linalg.pinv(cov_sqrt)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        mats = [cov_matrix, cov_sqrt, reconstruction, cov_sqrt_inv]
        titles = [r'$C_{ss}$', r'$C_{ss}^{1/2}$', r'$I + W\Lambda W^\top$', r'$C_{ss}^{-1/2}$ (whitening matrix)']
        for ax, mat, title in zip(axes, mats, titles):
            im = ax.imshow(mat, cmap='viridis')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(W, cmap='viridis')
        ax2.set_title(r'Frame $W$')
        plt.colorbar(im2, ax=ax2)
        plt.show()
        plt.tight_layout()

        return W


def compute_uniform_target_covariance(N_RF=13, sigma=0.25, Beta=0.5, stream_length=10920):
    '''
    Idealized covariance matrix of normalized responses to a uniform ensemble, for a
    single receptive field (N_RF neurons, N_SETS=1 so generate_surround_ensembles returns
    just the CRF block with no surround/baseline rows appended).

    Normalization mirrors the "if uniform_stimuli is not None" branch of
    get_optimal_gains_target (analysis/Analytic_responses.py:107-116): responses are scaled
    by Beta, then divisively normalized by a pooled, semi-saturating denominator
    (sqrt(sigma^2 + pooled squared drive)) before taking the covariance - N_matrix there is
    an all-ones (N_RF, N_RF) pooling matrix (see V1Tunings.N_matrix). sigma=0.25 matches
    V1Dynamics_Surround's current default (simulation_whiten.py) - keep these in sync, since
    a stale sigma here makes theta_t (simulation_whiten.py:241) mismatch the live model's
    actual steady-state variance under a uniform ensemble, biasing g/v adaptation even with
    no real adaptation happening. Beta=0.5 matches get_optimal_gains_target's Beta.
    '''
    stim_gen = StimulusGenerator(N_RF=N_RF, N_SETS=1, stream_length=stream_length)
    profiles = stim_gen.generate_surround_ensembles('adapt CRF only', biased=False)  # (N_RF, T)

    uniform_stimuli = profiles.T  # (T, N_RF): rows = timesteps, columns = neurons, matching np.cov(rowvar=False)
    N_matrix = np.ones((N_RF, N_RF))

    raw_input_drive = uniform_stimuli * Beta
    Z_sq = raw_input_drive ** 2
    denom = np.sqrt(sigma**2 + (N_matrix @ Z_sq.T).T)
    covariance_array = raw_input_drive / denom

    return np.cov(covariance_array, rowvar=False)


def save_uniform_target_covariance(N_RF=13, out_dir="data/target_covs"):
    '''Computes the uniform target covariance and saves it to <out_dir>/uniform_target_covariance.csv.'''
    Covariance = compute_uniform_target_covariance(N_RF=N_RF)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "uniform_target_covariance.csv")
    np.savetxt(out_path, Covariance, delimiter=",")
    print(f"Saved uniform target covariance ({Covariance.shape}) to {out_path}")
    return Covariance


if __name__ == "__main__":
    np.random.seed(42)
    choice = input("Choose frame type [mercedes/gaussian/optimal/identity]: ").strip().lower()
    while choice not in ('mercedes', 'gaussian', 'optimal', 'identity'):
        choice = input("Invalid choice. Enter mercedes, gaussian, identity, or optimal: ").strip().lower()
    frame = Frame(dim=13, frame_type=choice)
    np.savetxt(f"Frames/N13_{choice}_Frame.csv", frame.W, delimiter=",")

    save_uniform_target_covariance(N_RF=frame.dim)

