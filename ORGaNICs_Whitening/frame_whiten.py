'''
----  frame_whiten.py: ----
Creates a fixed overcomplete frame of synaptic weights used during the whitening process to project
primary neurons onto interneuron axes (the axes of this frame). Frame is an N x K matrix where N is the number 
of primary neurons and K >= N(N+1)/2 (I set this equal in this code). Taken from Ch.2 of Lyndon Duong's thesis.
'''
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Frame:
    def __init__(self, dim: int):
        self.dim = int(dim) # Number of primary neurons
        self.K = int(self.dim * (self.dim + 1) // 2)
        print(f"Building Smooth Mercedes Frame (N={self.dim}, K={self.K})...")
        self.W = self.mercedes() 
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

    def plot_frame(self):
        '''
        Visualize the frame vectors in 2D. Only works when N=2.
        Plots each of the K unit vectors as arrows from the origin.
        '''
        if self.dim != 2:
            raise ValueError(f"Plotting only supported for N=2, got N={self.dim}")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot each frame vector as an arrow from the origin
        colors = plt.cm.viridis(np.linspace(0, 1, self.K))
        for k in range(self.K):
            vec = self.W[:, k]
            ax.arrow(0, 0, vec[0], vec[1], 
                     head_width=0.08, head_length=0.05, 
                     fc=colors[k], ec=colors[k], 
                     linewidth=2, label=f'w_{k+1}')
        
        # Plot the unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_title(f'Overcomplete Frame (N={self.dim}, K={self.K})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
   if __name__ == "__main__":
    # Visualize with N=2, K=3
    np.random.seed(42) # For reproducibility
    frame = Frame(dim=2)
    print(f"Frame W shape: {frame.W.shape}")
    print(f"Frame vectors:\n{frame.W}")
    
    # This will block execution until you manually close the plot window
    frame.plot_frame() 
    plt.close('all') # Good practice to clean up backend resources immediately

    # Create and save N=100 frame to csv for reuse in simulations
    np.random.seed(42)
    frame_169 = Frame(dim=169)
    np.savetxt("N169_Frame.csv", frame_169.W, delimiter=",")
    
    # FIXED: Changed 'frame_255' to 'frame_100' so the script finishes successfully
    print(f"Saved N=169 frame to N169_Frame.csv (shape: {frame_169.W.shape})")