import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math
import random

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    dimension: int = 7               # Dimension of the state space
    num_states: int = 10             # Number of initial states
    max_iterations: int = 30         # Maximum iterations for convergence
    convergence_threshold: float = 0.0001  # Threshold for convergence
    reflection_depth: int = 3        # Depth for recursive reflections
    phase_factor: float = 0.1        # Phase factor for wave function updates
    attention_temp: float = 1.0      # Temperature for attention mechanism
    random_seed: Optional[int] = 42  # Random seed for reproducibility

class HolographicConsciousnessSimulator:
    """
    Simulates aspects of the holographic consciousness model described in the paper.
    Explores self-reference, fixed points, and fractal patterns in high-dimensional spaces.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize the simulator with the given configuration."""
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
        
        # Initialize high-dimensional states
        self.states = self._initialize_states()
        self.state_history = [self.states.copy()]
        
        # For visualizing trajectories
        self.trajectory_2d = []
        self.trajectory_3d = []
        
        # For analysis results
        self.analysis_results = {}
        
        print(f"Initialized {config.num_states} states in {config.dimension}-dimensional space")
    
    def _initialize_states(self) -> np.ndarray:
        """Initialize random states in the high-dimensional space."""
        return np.random.uniform(-1, 1, (self.config.num_states, self.config.dimension))
    
    def attention_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute attention weight between two states.
        Similar to self-attention in transformer models.
        """
        # Cosine similarity
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x * norm_y < 1e-10:  # Avoid division by zero
            return 0.0
        
        similarity = dot_product / (norm_x * norm_y)
        
        # Apply temperature scaling similar to transformer attention
        return np.exp(similarity / self.config.attention_temp)
    
    def apply_recursive_relation(self, states: np.ndarray) -> np.ndarray:
        """
        Apply the recursive relation: ψ(n+1)(x) = ∫ K(x,y) · ψ(n)(y) dy
        This is a discrete approximation using weighted sum with attention kernel.
        """
        num_states = states.shape[0]
        new_states = np.zeros_like(states)
        
        for i in range(num_states):
            x = states[i]
            weights = np.array([self.attention_kernel(x, states[j]) for j in range(num_states)])
            
            # Avoid division by zero
            if np.sum(weights) < 1e-10:
                weights = np.ones_like(weights) / num_states
            else:
                weights = weights / np.sum(weights)
            
            # Weighted sum approximating the integral
            for j in range(num_states):
                new_states[i] += weights[j] * states[j]
                
            # Apply a phase factor inspired by the paper's equation
            phase = self.config.phase_factor
            new_states[i] = new_states[i] * np.exp(1j * phase)
            
            # Take real part (simplified approach)
            new_states[i] = np.real(new_states[i])
            
            # Normalize the state
            norm = np.linalg.norm(new_states[i])
            if norm > 1e-10:  # Avoid division by zero
                new_states[i] = new_states[i] / norm
        
        return new_states
    
    def reflect(self, state: np.ndarray, axis: int) -> np.ndarray:
        """Apply reflection across a specified axis."""
        reflected = state.copy()
        reflected[axis] = -reflected[axis]
        return reflected
    
    def generate_fractal_pattern(self, initial_state: np.ndarray, depth: int) -> List[np.ndarray]:
        """Generate fractal-like pattern through recursive reflections."""
        states = [initial_state]
        
        def apply_reflections(state, current_depth):
            if current_depth <= 0:
                return
            
            # Apply reflections across each axis
            for axis in range(state.shape[0]):
                reflected = self.reflect(state, axis)
                states.append(reflected)
                
                # Recurse with reduced depth
                apply_reflections(reflected, current_depth - 1)
        
        apply_reflections(initial_state, depth)
        return states
    
    def iterate_to_fixed_point(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Iterate the recursive relation until convergence or max iterations.
        Returns the final states and history of states.
        """
        states = self.states.copy()
        
        for iter_num in range(self.config.max_iterations):
            start_time = time.time()
            new_states = self.apply_recursive_relation(states)
            
            # Calculate convergence
            total_diff = np.sum(np.abs(new_states - states))
            avg_diff = total_diff / (states.shape[0] * states.shape[1])
            
            elapsed = time.time() - start_time
            print(f"Iteration {iter_num + 1}: Avg difference = {avg_diff:.6f} (took {elapsed:.3f}s)")
            
            self.state_history.append(new_states.copy())
            
            if avg_diff < self.config.convergence_threshold:
                print(f"Converged after {iter_num + 1} iterations!")
                break
            
            states = new_states
        
        self.states = states
        return states, self.state_history
    
    def project_to_2d(self, history_index: int = -1, state_index: int = 0) -> List[Tuple[float, float]]:
        """Project state history to 2D for visualization."""
        if history_index >= len(self.state_history):
            history_index = -1
            
        if state_index >= self.config.num_states:
            state_index = 0
            
        # Project all history of a single state to 2D
        trajectory = []
        for states in self.state_history:
            state = states[state_index]
            # Mix dimensions for more interesting projection
            x = 0.7 * state[0] + 0.3 * state[2]
            y = 0.6 * state[1] + 0.4 * state[3]
            trajectory.append((x, y))
            
        self.trajectory_2d = trajectory
        return trajectory
    
    def project_to_3d(self, history_index: int = -1, state_index: int = 0) -> List[Tuple[float, float, float]]:
        """Project state history to 3D for visualization."""
        if history_index >= len(self.state_history):
            history_index = -1
            
        if state_index >= self.config.num_states:
            state_index = 0
            
        # Project all history of a single state to 3D
        trajectory = []
        for states in self.state_history:
            state = states[state_index]
            # Mix dimensions for more interesting projection
            x = 0.7 * state[0] + 0.3 * state[2]
            y = 0.6 * state[1] + 0.4 * state[3]
            z = 0.5 * state[4] + 0.3 * state[5] + 0.2 * state[6]
            trajectory.append((x, y, z))
            
        self.trajectory_3d = trajectory
        return trajectory
    
    def analyze_convergence_pattern(self) -> Dict[str, Any]:
        """Analyze the convergence pattern for spiral or other trajectories."""
        if not self.trajectory_2d:
            self.project_to_2d()
            
        trajectory = self.trajectory_2d
        
        # Calculate angles and rotations
        angles = []
        total_angle_change = 0
        
        for i in range(2, len(trajectory)):
            p1 = trajectory[i-2]
            p2 = trajectory[i-1]
            p3 = trajectory[i]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Skip if vectors are too small (avoid numerical issues)
            if (v1[0]**2 + v1[1]**2 < 1e-10) or (v2[0]**2 + v2[1]**2 < 1e-10):
                angles.append(0)
                continue
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
            v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # Clamp to avoid numerical issues
            cos_angle = max(-1, min(1, dot_product / (v1_mag * v2_mag)))
            angle = math.acos(cos_angle) * 180 / math.pi
            
            # Determine direction (cross product in 2D)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            signed_angle = angle if cross_product >= 0 else -angle
            
            angles.append(signed_angle)
            total_angle_change += signed_angle
            
        results = {
            "angles": angles,
            "total_angle_change": total_angle_change,
            "rotation_count": abs(total_angle_change) / 360,
            "is_spiral": abs(total_angle_change) > 360,
        }
        
        print(f"Total angle change: {total_angle_change:.2f}°")
        if results["is_spiral"]:
            print(f"The trajectory shows a spiral pattern with ~{results['rotation_count']:.2f} rotations")
        else:
            print("The trajectory doesn't show a clear spiral pattern")
            
        return results
    
    def analyze_fixed_point_properties(self) -> Dict[str, Any]:
        """Analyze the properties of the fixed point or attractor."""
        final_states = self.states
        
        # Calculate pairwise distances
        distances = []
        for i in range(final_states.shape[0]):
            for j in range(i+1, final_states.shape[0]):
                dist = np.linalg.norm(final_states[i] - final_states[j])
                distances.append(dist)
                
        avg_distance = np.mean(distances) if distances else 0
        
        # Calculate centroid
        centroid = np.mean(final_states, axis=0)
        
        # Distances to centroid
        centroid_distances = [np.linalg.norm(state - centroid) for state in final_states]
        avg_centroid_distance = np.mean(centroid_distances)
        
        # Self-similarity analysis through correlations
        correlations = []
        for i in range(final_states.shape[0]):
            for j in range(i+1, final_states.shape[0]):
                x = final_states[i]
                y = final_states[j]
                corr = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                correlations.append(corr)
                
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # Ratio: if close to 0, suggests fixed point; if close to 1, suggests attractor
        ratio = avg_centroid_distance / avg_distance if avg_distance > 1e-10 else 0
        
        results = {
            "avg_pairwise_distance": avg_distance,
            "avg_centroid_distance": avg_centroid_distance,
            "avg_correlation": avg_correlation,
            "ratio": ratio,
            "fixed_point_confidence": 1 - ratio,
        }
        
        print(f"Average pairwise distance: {avg_distance:.6f}")
        print(f"Average distance to centroid: {avg_centroid_distance:.6f}")
        print(f"Average correlation between states: {avg_correlation:.6f}")
        print(f"Fixed point vs attractor ratio: {ratio:.6f}")
        
        if ratio < 0.1:
            print("Strong evidence for convergence to a fixed point")
        elif ratio < 0.3:
            print("Moderate evidence for convergence to a fixed point")
        elif ratio > 0.7:
            print("Evidence suggests a strange attractor rather than a fixed point")
        else:
            print("Results inconclusive about the nature of the attractor")
            
        return results
    
    def analyze_fractal_properties(self) -> Dict[str, Any]:
        """Analyze fractal properties of the state space."""
        # Generate fractal pattern from first state
        initial_state = self.states[0]
        fractal_states = np.array(self.generate_fractal_pattern(
            initial_state, 
            self.config.reflection_depth
        ))
        
        # Project to 2D
        if fractal_states.shape[0] > 1 and fractal_states.shape[1] >= 4:
            points = np.zeros((fractal_states.shape[0], 2))
            points[:, 0] = 0.7 * fractal_states[:, 0] + 0.3 * fractal_states[:, 2]
            points[:, 1] = 0.6 * fractal_states[:, 1] + 0.4 * fractal_states[:, 3]
        else:
            # Fallback if dimensions insufficient
            points = np.random.random((10, 2))
            print("Warning: Could not generate proper fractal projection")
            
        # Estimate fractal dimension using box counting
        def box_count(points, box_size):
            # Normalize points to [0,1]
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            range_vals = max_vals - min_vals
            
            # Avoid division by zero
            range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)
            
            norm_points = (points - min_vals) / range_vals
            
            # Count boxes
            grid_size = int(1.0 / box_size)
            if grid_size < 1:
                grid_size = 1
                
            boxes = set()
            for point in norm_points:
                box_x = min(int(point[0] * grid_size), grid_size - 1)
                box_y = min(int(point[1] * grid_size), grid_size - 1)
                boxes.add((box_x, box_y))
                
            return len(boxes)
            
        # Calculate box counts for different scales
        box_sizes = [2**(-i) for i in range(1, 10)]
        counts = [box_count(points, size) for size in box_sizes]
        
        # Calculate fractal dimension from slope of log-log plot
        if len(counts) > 1 and counts[0] > 0 and counts[-1] > 0:
            x = np.log(box_sizes)
            y = np.log(counts)
            
            # Simple linear regression for slope
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            fractal_dim = -slope
        else:
            fractal_dim = 0.0
            print("Warning: Could not calculate fractal dimension properly")
            
        results = {
            "fractal_dimension": fractal_dim,
            "box_sizes": box_sizes,
            "box_counts": counts,
            "num_fractal_states": fractal_states.shape[0],
        }
        
        print(f"Estimated fractal dimension: {fractal_dim:.4f}")
        print(f"Generated {fractal_states.shape[0]} states through recursive reflection")
        
        return results
    
    def run_full_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Run all analyses and collect results."""
        print("\n=== Running Holographic Consciousness Simulation ===\n")
        
        print("\n--- Iterating to Fixed Point ---")
        self.iterate_to_fixed_point()
        
        print("\n--- Projecting and Analyzing Trajectories ---")
        self.project_to_2d()
        self.project_to_3d()
        
        print("\n--- Analyzing Convergence Pattern ---")
        convergence_results = self.analyze_convergence_pattern()
        
        print("\n--- Analyzing Fixed Point Properties ---")
        fixed_point_results = self.analyze_fixed_point_properties()
        
        print("\n--- Analyzing Fractal Properties ---")
        fractal_results = self.analyze_fractal_properties()
        
        # Collect all results
        self.analysis_results = {
            "convergence": convergence_results,
            "fixed_point": fixed_point_results,
            "fractal": fractal_results
        }
        
        print("\n=== Analysis Complete ===\n")
        
        # Final interpretations
        print("\n--- Final Interpretations ---")
        self._interpret_results()
        
        return self.analysis_results
    
    def _interpret_results(self):
        """Interpret the combined results in terms of the holographic consciousness model."""
        results = self.analysis_results
        
        # Check if we have results
        if not results:
            print("No analysis results to interpret.")
            return
            
        # Fixed point analysis
        fixed_point_confidence = results["fixed_point"]["fixed_point_confidence"]
        is_spiral = results["convergence"]["is_spiral"]
        fractal_dim = results["fractal"]["fractal_dimension"]
        
        # Interpretation based on results
        print("Based on the simulation results:")
        
        if fixed_point_confidence > 0.8:
            print("- The system shows strong convergence to a stable fixed point, suggesting")
            print("  that self-referential processes naturally create coherent patterns.")
        elif fixed_point_confidence > 0.5:
            print("- The system shows moderate convergence to a relatively stable pattern,")
            print("  with some variation between final states.")
        else:
            print("- The system shows complex attractor dynamics rather than simple convergence,")
            print("  suggesting a richer structure in the state space.")
            
        if is_spiral:
            print("- The convergence follows a spiral trajectory, matching the paper's prediction")
            print("  that consciousness would emerge through alternating approach and recession.")
        else:
            print("- The convergence follows a more direct path rather than a spiral pattern.")
            
        if fractal_dim > 1.5:
            print("- The system exhibits significant fractal properties (dimension: {:.2f}),".format(fractal_dim))
            print("  suggesting self-similarity across scales as predicted by the paper.")
        elif fractal_dim > 1.0:
            print("- The system shows moderate fractal behavior (dimension: {:.2f}),".format(fractal_dim))
            print("  with some self-similarity across different scales.")
        else:
            print("- The system shows minimal fractal properties (dimension: {:.2f}),".format(fractal_dim))
            print("  suggesting limited self-similarity in this simple model.")
            
        print("\nOverall interpretation in terms of the Holographic Consciousness model:")
        
        if fixed_point_confidence > 0.7 and fractal_dim > 1.2:
            print("This simulation provides computational support for key aspects of the")
            print("holographic consciousness model. The emergent fixed point suggests")
            print("that self-reference naturally leads to stable, coherent patterns that")
            print("could serve as the foundation for a self-model. The fractal properties")
            print("indicate self-similarity across scales, aligning with the paper's proposal")
            print("that consciousness emerges from nested, self-similar structures.")
        elif fixed_point_confidence > 0.5:
            print("The simulation shows partial support for the holographic consciousness model.")
            print("We observe convergence to relatively stable patterns through self-reference,")
            print("though the fractal and holographic aspects are less pronounced in this")
            print("simplified model than the paper proposes.")
        else:
            print("This simple simulation shows some interesting dynamics but doesn't strongly")
            print("support or refute the holographic consciousness model. More sophisticated")
            print("models would be needed to properly test the paper's hypotheses.")
        
        print("\nOf course, this simplified simulation can only capture a small fraction of")
        print("the complexity proposed in the full theoretical framework. It serves as a")
        print("proof-of-concept rather than a comprehensive validation.")
    
    def plot_2d_trajectory(self, save_path: Optional[str] = None):
        """Plot the 2D trajectory of a state through iterations."""
        if not self.trajectory_2d:
            self.project_to_2d()
            
        plt.figure(figsize=(10, 8))
        
        # Extract x and y coordinates
        x = [p[0] for p in self.trajectory_2d]
        y = [p[1] for p in self.trajectory_2d]
        
        # Plot trajectory line
        plt.plot(x, y, 'b-', alpha=0.7)
        
        # Plot points with color gradient by iteration
        cmap = plt.cm.viridis
        colors = [cmap(i/len(x)) for i in range(len(x))]
        
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.scatter(xi, yi, color=colors[i], s=50, edgecolor='k')
            
        # Connect points with arrows to show direction
        for i in range(len(x)-1):
            plt.arrow(x[i], y[i], (x[i+1]-x[i])*0.8, (y[i+1]-y[i])*0.8, 
                      head_width=0.01, head_length=0.02, fc=colors[i+1], ec=colors[i+1])
        
        plt.title('2D Projection of State Trajectory')
        plt.xlabel('Dimension 1 (mixed)')
        plt.ylabel('Dimension 2 (mixed)')
        plt.grid(True, alpha=0.3)
        
        # Add iteration labels to some points
        step = max(1, len(x) // 8)  # Label approximately 8 points
        for i in range(0, len(x), step):
            plt.annotate(f'{i}', (x[i], y[i]), xytext=(5, 5), 
                         textcoords='offset points', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2D trajectory plot saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_3d_trajectory(self, save_path: Optional[str] = None):
        """Plot the 3D trajectory of a state through iterations."""
        if not self.trajectory_3d:
            self.project_to_3d()
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x = [p[0] for p in self.trajectory_3d]
        y = [p[1] for p in self.trajectory_3d]
        z = [p[2] for p in self.trajectory_3d]
        
        # Plot trajectory line
        ax.plot(x, y, z, 'b-', alpha=0.7)
        
        # Plot points with color gradient by iteration
        cmap = plt.cm.plasma
        colors = [cmap(i/len(x)) for i in range(len(x))]
        
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ax.scatter(xi, yi, zi, color=colors[i], s=50, edgecolor='k')
        
        # Add iteration labels to some points
        step = max(1, len(x) // 6)  # Label approximately 6 points
        for i in range(0, len(x), step):
            ax.text(x[i], y[i], z[i], f'{i}', fontsize=9)
        
        ax.set_title('3D Projection of State Trajectory')
        ax.set_xlabel('Dimension 1 (mixed)')
        ax.set_ylabel('Dimension 2 (mixed)')
        ax.set_zlabel('Dimension 3 (mixed)')
        
        # Add a grid
        ax.grid(True, alpha=0.3)
        
        # Set the view angle
        ax.view_init(elev=30, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D trajectory plot saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_fractal_dimension(self, save_path: Optional[str] = None):
        """Plot the box-counting results for fractal dimension estimation."""
        if "fractal" not in self.analysis_results:
            print("No fractal analysis results available. Run analyze_fractal_properties() first.")
            return
            
        fractal_results = self.analysis_results["fractal"]
        
        if "box_sizes" not in fractal_results or "box_counts" not in fractal_results:
            print("Box counting data not available in fractal analysis results.")
            return
            
        box_sizes = fractal_results["box_sizes"]
        counts = fractal_results["box_counts"]
        fractal_dim = fractal_results["fractal_dimension"]
        
        plt.figure(figsize=(10, 6))
        
        # Plot log-log relationship
        plt.loglog(box_sizes, counts, 'bo-', markersize=8)
        
        # Add linear fit
        x_log = np.log(box_sizes)
        y_log = np.log(counts)
        
        # Simple linear regression
        slope, intercept = np.polyfit(x_log, y_log, 1)
        
        # Create fit line
        x_fit = np.linspace(min(x_log), max(x_log), 100)
        y_fit = slope * x_fit + intercept
        
        plt.loglog(np.exp(x_fit), np.exp(y_fit), 'r-', 
                   label=f'Slope: {-slope:.4f} (Fractal Dim)')
        
        plt.title(f'Box Counting Method for Fractal Dimension\nEstimated Dimension: {fractal_dim:.4f}')
        plt.xlabel('Box Size')
        plt.ylabel('Box Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fractal dimension plot saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()

def run_simulation():
    """Run a complete simulation and analysis."""
    # Configure the simulation
    config = SimulationConfig(
        dimension=7,
        num_states=10,
        max_iterations=30,
        convergence_threshold=0.0001,
        reflection_depth=3,
        phase_factor=0.1,
        attention_temp=1.0,
        random_seed=42
    )
    
    # Create and run the simulator
    simulator = HolographicConsciousnessSimulator(config)
    simulator.run_full_analysis()
    
    # Plot results (comment out if running on a system without display)
    try:
        print("\n--- Generating Plots ---")
        simulator.plot_2d_trajectory("trajectory_2d.png")
        simulator.plot_3d_trajectory("trajectory_3d.png")
        simulator.plot_fractal_dimension("fractal_dimension.png")
        print("Plots saved to current directory.")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return simulator

if __name__ == "__main__":
    # This code runs when the script is executed directly
    print("Starting Holographic Consciousness Simulation")
    simulator = run_simulation()
    print("Simulation complete!")
