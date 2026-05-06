import sys
from pathlib import Path
import urllib.request
import pandas
import numpy as np
import matplotlib.pyplot as plt

from data import generate_data, prepare_data
from model import gradient_descent

from sklearn.preprocessing import StandardScaler

INTERACTIVE_URL = (
    "https://edunet.kea.su/repo/EduNet_NLP-web_dependencies/L02/"
    "interactive_visualization.py"
)

def plot_mse(mse_train, mse_test):
    """Simple MSE plot - kept for backward compatibility"""
    plt.figure(figsize=(10, 4))
    plt.title("Learning curve")
    plt.plot(mse_train, label="train")
    plt.plot(mse_test, label="test")
    plt.legend()

    plt.xlabel("iterations", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)

    plt.grid(True)
    plt.show()

def plot_mse_enhanced(mse_train, mse_test, log_scale=False):
    """Enhanced MSE visualization with beautiful styling"""
    # Use a modern style
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Standard view with annotations
    iterations = range(1, len(mse_train) + 1)

    # Plot with gradient colors
    ax1.plot(iterations, mse_train, linewidth=2.5, label='Train MSE',
             color='#2E86AB', marker='o', markevery=len(mse_train)//10, markersize=4)
    ax1.plot(iterations, mse_test, linewidth=2.5, label='Test MSE',
             color='#A23B72', marker='s', markevery=len(mse_test)//10, markersize=4)

    # Fill area between curves to show overfitting
    ax1.fill_between(iterations, mse_train, mse_test,
                      where=(np.array(mse_test) > np.array(mse_train)),
                      alpha=0.3, color='red', label='Overfitting region')

    # Annotate minimum values
    min_train_idx = np.argmin(mse_train)
    min_test_idx = np.argmin(mse_test)

    ax1.annotate(f'Min Train: {mse_train[min_train_idx]:.4f}',
                 xy=(min_train_idx + 1, mse_train[min_train_idx]),
                 xytext=(10, 20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.annotate(f'Min Test: {mse_test[min_test_idx]:.4f}',
                 xy=(min_test_idx + 1, mse_test[min_test_idx]),
                 xytext=(10, -30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='cyan', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax1.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Learning Curve - MSE over Iterations', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right plot: Log scale or comparison view
    if log_scale:
        ax2.semilogy(iterations, mse_train, linewidth=2.5, label='Train MSE (log)',
                     color='#2E86AB', marker='o', markevery=len(mse_train)//10, markersize=4)
        ax2.semilogy(iterations, mse_test, linewidth=2.5, label='Test MSE (log)',
                     color='#A23B72', marker='s', markevery=len(mse_test)//10, markersize=4)
        ax2.set_ylabel('MSE Loss (log scale)', fontsize=14, fontweight='bold')
        ax2.set_title('Learning Curve - Log Scale', fontsize=16, fontweight='bold', pad=20)
    else:
        # Show difference between train and test
        gap = np.array(mse_test) - np.array(mse_train)
        ax2.plot(iterations, gap, linewidth=2.5, color='#F18F01', marker='D',
                 markevery=len(gap)//10, markersize=4)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(iterations, 0, gap, where=(gap > 0),
                         alpha=0.3, color='red', label='Overfitting')
        ax2.fill_between(iterations, 0, gap, where=(gap <= 0),
                         alpha=0.3, color='green', label='Underfitting')
        ax2.set_ylabel('Test MSE - Train MSE', fontsize=14, fontweight='bold')
        ax2.set_title('Generalization Gap', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)

    ax2.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(x, y_true, y_pred, title="Predictions vs Actual"):
    """Scatter plot comparing predictions with actual values"""
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Regression line with data points
    ax1.scatter(x, y_true, alpha=0.6, s=50, c='#2E86AB', label='Actual data', edgecolors='black', linewidth=0.5)
    ax1.plot(x, y_pred, color='#A23B72', linewidth=3, label='Predicted line', linestyle='--')
    ax1.set_xlabel('X', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Regression Fit', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Predicted vs Actual scatter
    ax2.scatter(y_true, y_pred, alpha=0.6, s=50, c='#F18F01', edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax2.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred):
    """Plot residuals to analyze error distribution"""
    plt.style.use('seaborn-v0_8-whitegrid')

    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Residual plot
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, c='#2E86AB', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Right: Residual distribution
    ax2.hist(residuals, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax2.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

def ensure_interactive_module(target_dir: Path) -> bool:
    module_path = target_dir / "interactive_visualization.py"

    if module_path.exists():
        print(f" Module alrady exist: {module_path}")
        return
    
    print(f"Module download from {INTERACTIVE_URL}...")
    try:
        urllib.request.urlretrieve(INTERACTIVE_URL, module_path.as_posix())
        if module_path.exists() and module_path.stat().st_size > 0:
            print(f"The module has already been downloaded: ({module_path.stat().st_size} bytes)")
            return True
        else:
            print(f"File is empty or not create")
            return False
    except Exception as exc:
        print(f"Warning: could not download interactive module: {exc}")
        return False

def load_interactive_plot():
    script_dir = Path(__file__).resolve().parent

    if not ensure_interactive_module(script_dir):
        print("Could not download interactive module. Please check your internet connection and try again.")
        return None

    module_path = script_dir / "interactive_visualization.py"
   
    if not module_path.exists():
        print(f"The file not found: {module_path}")
        return None

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
        print(f"Add in sys.path: {script_dir}")

    try:
        from interactive_visualization import plot_grid_search_2d
    except Exception as exc:
        print(f"Warning: could not import interactive module: {exc}")
        return None

    return plot_grid_search_2d

def main():
    # Generate and prepare data
    x, y = generate_data(n=1000, noise=0.8)
    x_train, x_test, y_train, y_test = prepare_data(x, y)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Scale features for better convergence
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(np.expand_dims(x_train[:, 1], axis=1)).flatten()
    x_test_scaled = scaler.transform(np.expand_dims(x_test[:, 1], axis=1)).flatten()

    # Add bias term to scaled features
    x_train_with_bias = np.column_stack([np.ones(len(x_train_scaled)), x_train_scaled])
    x_test_with_bias = np.column_stack([np.ones(len(x_test_scaled)), x_test_scaled])

    # Load interactive visualization module
    plot_grid_search_2d = load_interactive_plot()

    if plot_grid_search_2d is not None:
        print("\n🎨 Visualization of the search space (before training)...")
        intercepts = np.arange(-15, 25, 0.2)
        slopes = np.arange(-20, 20, 0.2)
        plot_grid_search_2d(x_train_scaled, y_train, slopes, intercepts)
    else:
        print("⚠️  Interactive visualization is not available")

    # Train model with gradient descent
    w_init = np.array([[0.0], [0.0]])
    w, mse_train, mse_test, ws = gradient_descent(
        x_train_with_bias,  # Fixed: use scaled data with bias
        y_train,
        x_test_with_bias,   # Fixed: use scaled data with bias
        y_test,
        w_init,
        alpha=0.05,
        iterations=800,
    )

    print(f"\nFinal weights (scaled): bias = {w[0, 0]:.4f}, slope = {w[1, 0]:.4f}")

    # Enhanced MSE visualization
    print("\n📊 Enhanced MSE Visualization...")
    plot_mse_enhanced(mse_train, mse_test, log_scale=False)

    # Make predictions
    y_train_pred = (x_train_with_bias @ w).flatten()
    y_test_pred = (x_test_with_bias @ w).flatten()

    # Plot predictions vs actual
    print("\n📈 Predictions vs Actual Values...")
    plot_predictions_vs_actual(x_train_scaled, y_train, y_train_pred, title="Training Data")

    # Plot residuals
    print("\n📉 Residual Analysis...")
    plot_residuals(y_train, y_train_pred)

    # Interactive visualization with gradient descent path
    if plot_grid_search_2d is not None:
        print("\n🎯 Visualization of gradient descent path...")

        # Import plot_gradient_descent_2d if available
        try:
            from interactive_visualization import plot_gradient_descent_2d

            # Define tighter range around solution
            intercepts = np.arange(-7.5, 12.5, 0.1)
            slopes = np.arange(-5, 5, 0.1)

            # Transform weights back to original scale for visualization
            ws_array = np.array([w_item.flatten() for w_item in ws])

            plot_gradient_descent_2d(
                x_train_scaled,
                y_train,
                ws_array,
                slopes,
                intercepts,
            )

            # Calculate weights in original scale
            b_original = ws[-1][0, 0] - ws[-1][1, 0] * scaler.mean_[0] / scaler.scale_[0]
            w_original = ws[-1][1, 0] / scaler.scale_[0]

            print(f"\nFinal equation (original scale): y = {w_original:.2f}x + {b_original:.2f}")

        except ImportError:
            print("⚠️  plot_gradient_descent_2d not available in interactive module")
    else:
        print("⚠️  Interactive visualization is not available")

if __name__ == "__main__":
    main()