import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import beaupy
from rich.console import Console
from scipy.stats import multivariate_normal

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    load_data,
    load_study,
    load_best_model,
)

# Initialize console at module level
console = Console()


def main():
    # Test run
    console.print("[bold green]Analyzing the model...[/bold green]")
    console.print("Select a project to analyze:")
    project = select_project()
    console.print("Select a group to analyze:")
    group_name = select_group(project)
    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)
    console.print("Select a device:")
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    # Load the best model
    # study_name = "Optimize_Template"
    # model, config = load_best_model(project, study_name)
    # device = select_device()
    # model = model.to(device)

    ds_train, ds_val = load_data()  # Assuming this is implemented in util.py

    # Get data samples
    console.print("Loading data samples...")
    data_samples = ds_train[:][0].numpy()

    # Transform data to latent space
    console.print("Transforming data to latent space...")
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data_samples, dtype=torch.float32).to(device)

        # Transform to latent space using inverse
        z = data_tensor
        for layer in reversed(model.layers):
            z, _ = layer.inverse(z)

        latent_samples = z.cpu().numpy()

    # Create comprehensive visualization
    console.print("Creating comprehensive visualization...")
    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Original Data Distribution
        h1 = axes[0].hist2d(
            data_samples[:, 0],
            data_samples[:, 1],
            bins=100,
            density=True,
            cmap="viridis",
        )
        axes[0].set_title("Original Data Distribution")
        axes[0].set_xlabel("X1")
        axes[0].set_ylabel("X2")
        plt.colorbar(h1[3], ax=axes[0], label="Density")

        # Plot 2: Latent Space (should be ~N(0,1))
        h2 = axes[1].hist2d(
            latent_samples[:, 0],
            latent_samples[:, 1],
            bins=100,
            density=True,
            cmap="viridis",
        )
        axes[1].set_title("Latent Space (Transformed to N(0,1))")
        axes[1].set_xlabel("Z1")
        axes[1].set_ylabel("Z2")
        plt.colorbar(h2[3], ax=axes[1], label="Density")

        # Add standard normal contours for reference
        x_grid = np.linspace(-4, 4, 100)
        y_grid = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        pos = np.dstack((X, Y))
        rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
        axes[1].contour(X, Y, rv.pdf(pos), colors="red", alpha=0.5, linewidths=2)

        # Plot 3: Model Generated Samples
        num_samples = 8000
        with torch.no_grad():
            generated_samples = model.sample(num_samples).cpu().numpy()

        h3 = axes[2].hist2d(
            generated_samples[:, 0],
            generated_samples[:, 1],
            bins=100,
            density=True,
            cmap="viridis",
        )
        axes[2].set_title("Model Generated Samples")
        axes[2].set_xlabel("X1")
        axes[2].set_ylabel("X2")
        plt.colorbar(h3[3], ax=axes[2], label="Density")

        plt.tight_layout()
        plt.savefig("comprehensive_flow_analysis.png", dpi=300)
        plt.close()

    console.print("[bold green]Saved comprehensive_flow_analysis.png[/bold green]")

    # Additional analysis: Check if latent space is actually N(0,1)
    console.print("\n[bold cyan]Latent Space Statistics:[/bold cyan]")
    console.print(
        f"Mean: [{latent_samples[:, 0].mean():.4f}, {latent_samples[:, 1].mean():.4f}] (should be ~[0, 0])"
    )
    console.print(
        f"Std: [{latent_samples[:, 0].std():.4f}, {latent_samples[:, 1].std():.4f}] (should be ~[1, 1])"
    )

    # Plot detailed latent space transformation
    console.print("\nPlotting detailed latent space transformation...")
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter plot of latent samples
        ax.scatter(
            latent_samples[:, 0],
            latent_samples[:, 1],
            alpha=0.3,
            s=1,
            c="blue",
            label="Transformed Data",
        )

        # Add standard normal contours
        x_grid = np.linspace(-4, 4, 100)
        y_grid = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        pos = np.dstack((X, Y))
        rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
        contours = ax.contour(X, Y, rv.pdf(pos), colors="red", alpha=0.7, linewidths=2)
        ax.clabel(contours, inline=True, fontsize=8)

        ax.set_title("Latent Space vs. Standard Normal Distribution")
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        plt.savefig("latent_space_detail.png", dpi=300)
        plt.close()

    console.print("[bold green]Saved latent_space_detail.png[/bold green]")
    console.print("\n[bold green]✓ Analysis complete![/bold green]")
    console.print("Generated files:")
    console.print("  - comprehensive_flow_analysis.png (3 subplots)")
    console.print("  - latent_space_detail.png (detailed latent space)")


if __name__ == "__main__":
    main()
