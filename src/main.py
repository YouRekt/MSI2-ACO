import argparse
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Ant Colony Optimization for CVRP")
    parser.add_argument("instance_path", type=str, help="Path to the .vrp instance file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--smax", type=float, default=float('inf'), help="Maximum route distance (S_max)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone importance (alpha)")
    parser.add_argument("--beta", type=float, default=2.0, help="Heuristic information importance (beta)")
    parser.add_argument("--rho", type=float, default=0.5, help="Pheromone evaporation rate (rho)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("--- Starting MSI2-ACO ---")
    print(f"Instance: {args.instance_path}")
    print(f"Seed: {args.seed}, S_max: {args.smax}")
    print(f"ACO Parameters: alpha={args.alpha}, beta={args.beta}, rho={args.rho}\n")

if __name__ == "__main__":
    main()