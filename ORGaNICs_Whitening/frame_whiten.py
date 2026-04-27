'''
----  frame_whiten.py: ----
Creates a fixed overcomplete frame of synaptic weights used during the whitening process to project
primary neurons onto interneuron axes (the axes of this frame). Frame is an N x K matrix where N is the number 
of primary neurons and K >= N(N+1)/2 (I set this equal in this code). Taken from Ch.2 of Lyndon Duong's thesis.
'''
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
from stimuli_whiten import StimulusGenerator

class Frame:
    def __init__(self, dim: int, mercedes: bool = True, sigma: float = 0.3, noise_std: float = 0.05):
        self.dim = int(dim) # Number of primary neurons
        self.K = int(self.dim * (self.dim + 1) // 2)
        self.centers = None  # Only set for bell-shaped frames
        if mercedes:
            print(f"Building Smooth Mercedes Frame (N={self.dim}, K={self.K})...")
            self.W = self.mercedes()
        else:
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

    def optimal_frame(self) -> np.ndarray:
        N = self.dim
        stim_gen = StimulusGenerator(N=N, num_angles=N, stream_length=10140) 
        orientation_inputs = np.array(stim_gen.generate_input_ensembles(biased=False))
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

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        vmin = min(cov_sqrt.min(), reconstruction.min())
        vmax = max(cov_sqrt.max(), reconstruction.max())
        for ax, mat, title in zip(axes, [cov_sqrt, reconstruction], [r'$C_{ss}^{1/2}$', r'$I + W\Lambda W^\top$']):
            im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(W, cmap='viridis')
        ax2.set_title(r'Frame $W$')
        plt.colorbar(im2, ax=ax2)
        plt.show()
        plt.tight_layout()

        return W



if __name__ == "__main__":
    np.random.seed(42)
    optimal_uniform_frame_169 = Frame(dim=169, mercedes=False)
    np.savetxt("Frames/N169_optimal_uniform_Frame.csv", optimal_uniform_frame_169.W, delimiter=",")
 
