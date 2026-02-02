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

        # step 1: Generate Random Vectors
        A = np.random.randn(5 * K, N)
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        # Step 2: Select the vector that aligns least with the existing vectors in the existing frame 
        W_list = [A[0]]
        W_curr = A[0][:, None]
        remaining_A = A[1:]
        for _ in tqdm(range(K - 1), desc="Frame Init", leave=False):
            dots = remaining_A @ W_curr
            coherences = np.max(np.abs(dots), axis=1)
            best_idx = np.argmin(coherences)
            vec = remaining_A[best_idx]
            W_list.append(vec)
            W_curr = np.column_stack([W_curr, vec])
            remaining_A = np.delete(remaining_A, best_idx, axis=0)
        return np.array(W_list).T

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
    # Visualize with N=2, K=3
    np.random.seed(42) # For reproducibility
    frame = Frame(dim=2)
    print(f"Frame W shape: {frame.W.shape}")
    print(f"Frame vectors:\n{frame.W}")
    frame.plot_frame()

    # Create and save N=60 frame to csv for reuse in simulations
    np.random.seed(42)
    frame_60 = Frame(dim=60)
    np.savetxt("N60_Frame.csv", frame_60.W, delimiter=",")
    print(f"Saved N=60 frame to N60_Frame.csv (shape: {frame_60.W.shape})")