import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import beaupy
from rich.console import Console

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
    
    # Plot Original Data Distribution
    console.print("Plotting original data distribution...")
    data_samples = ds_train[:][0].numpy()  # Extract data from tuple (data, labels)

    with plt.style.context(['science', 'nature']):
        fig, ax = plt.subplots()
        h = ax.hist2d(
            data_samples[:, 0],
            data_samples[:, 1],
            bins=100,
            density=True,
            cmap='viridis'
        )
        ax.set_title("Original Data Distribution")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        plt.colorbar(h[3], ax=ax, label='Density')
        plt.savefig("original_data_distribution.png", dpi=300)
        plt.close()

    # Plot Model Generated Samples
    console.print("Generating samples from the model...")
    num_samples = 8000
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(num_samples).cpu().numpy()

    with plt.style.context(['science', 'nature']):
        fig, ax = plt.subplots()
        h = ax.hist2d(
            generated_samples[:, 0],
            generated_samples[:, 1],
            bins=100,
            density=True,
            cmap='viridis'
        )
        ax.set_title("Model Generated Samples")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        plt.colorbar(h[3], ax=ax, label='Density')
        plt.savefig("model_generated_samples.png", dpi=300)
        plt.close()


    # Additional custom analysis can be added here
    # ...


if __name__ == "__main__":
    main()
