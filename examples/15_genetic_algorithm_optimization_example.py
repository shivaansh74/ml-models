"""
Genetic Algorithm Optimization Example

This example demonstrates how to use the Genetic Algorithm model for solving
various optimization problems, including function optimization, the Knapsack problem,
and the Traveling Salesman Problem (TSP).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import time
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.optimization import GeneticAlgorithmModel

def main():
    """Run the Genetic Algorithm examples."""
    print("Genetic Algorithm Optimization Example")
    print("=====================================\n")
    
    # Example 1: Function Optimization
    print("\n--- FUNCTION OPTIMIZATION EXAMPLE ---\n")
    function_optimization_example()
    
    # Example 2: Knapsack Problem
    print("\n--- KNAPSACK PROBLEM EXAMPLE ---\n")
    knapsack_problem_example()
    
    # Example 3: Traveling Salesman Problem
    print("\n--- TRAVELING SALESMAN PROBLEM EXAMPLE ---\n")
    traveling_salesman_example()

def function_optimization_example():
    """Demonstrate genetic algorithm for function optimization."""
    print("Optimizing a complex mathematical function...\n")
    
    # Define the objective function to maximize
    # We'll use a 2D function with multiple local maxima
    def objective_function(x):
        # Convert binary chromosome to real values if needed
        if x.dtype == np.int64 and set(np.unique(x)).issubset({0, 1}):
            # Assuming 10 bits per variable, 20 bits total
            # Convert first 10 bits to x1, second 10 bits to x2
            x1_bits = x[:10]
            x2_bits = x[10:]
            
            # Binary to decimal conversion
            x1_decimal = sum(bit * (2 ** i) for i, bit in enumerate(reversed(x1_bits)))
            x2_decimal = sum(bit * (2 ** i) for i, bit in enumerate(reversed(x2_bits)))
            
            # Scale to desired range [-5, 5]
            x1 = -5.0 + (10.0 * x1_decimal) / (2**10 - 1)
            x2 = -5.0 + (10.0 * x2_decimal) / (2**10 - 1)
        else:
            # If input is already real-valued
            x1, x2 = x[0], x[1]
        
        # Compute the function value - a complex 2D function with multiple peaks
        term1 = np.sin(0.5 * x1**2 - 0.25 * x2**2 + 3) * np.cos(2 * x1 + 1 - np.exp(x2))
        term2 = 2 * np.exp(-((x1 - 2)**2 + (x2 - 2)**2) / 0.5)
        term3 = 0.5 * np.exp(-((x1 + 2)**2 + (x2 + 2)**2) / 1.0)
        result = term1 + term2 + term3
        
        return result
    
    # Create and train the GA model with binary encoding
    print("Running Genetic Algorithm with binary encoding...")
    ga_binary = GeneticAlgorithmModel(
        population_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.01,
        selection_method="tournament",
        tournament_size=3,
        maximize=True,
        early_stopping_generations=20
    )
    
    ga_binary.train(
        fitness_function=objective_function,
        chromosome_length=20,  # 10 bits per variable
        gene_type="binary"
    )
    
    # Get the best solution
    best_binary_chromosome = ga_binary.predict()
    best_binary_fitness = ga_binary.get_best_fitness()
    
    # Convert binary solution to real values
    x1_bits = best_binary_chromosome[:10]
    x2_bits = best_binary_chromosome[10:]
    
    x1_decimal = sum(bit * (2 ** i) for i, bit in enumerate(reversed(x1_bits)))
    x2_decimal = sum(bit * (2 ** i) for i, bit in enumerate(reversed(x2_bits)))
    
    x1_optimal = -5.0 + (10.0 * x1_decimal) / (2**10 - 1)
    x2_optimal = -5.0 + (10.0 * x2_decimal) / (2**10 - 1)
    
    print(f"Best binary solution: {best_binary_chromosome}")
    print(f"Decoded values: x1 = {x1_optimal:.4f}, x2 = {x2_optimal:.4f}")
    print(f"Fitness: {best_binary_fitness:.4f}")
    
    # Create and train the GA model with real-valued encoding
    print("\nRunning Genetic Algorithm with real-valued encoding...")
    ga_real = GeneticAlgorithmModel(
        population_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_method="tournament",
        tournament_size=3,
        maximize=True,
        early_stopping_generations=20
    )
    
    ga_real.train(
        fitness_function=objective_function,
        chromosome_length=2,  # 2 real variables
        gene_type="real",
        gene_range=(-5.0, 5.0)  # Range for x1 and x2
    )
    
    # Get the best solution
    best_real_chromosome = ga_real.predict()
    best_real_fitness = ga_real.get_best_fitness()
    
    print(f"Best real-valued solution: x1 = {best_real_chromosome[0]:.4f}, x2 = {best_real_chromosome[1]:.4f}")
    print(f"Fitness: {best_real_fitness:.4f}")
    
    # Evaluate the GA performance
    evaluation = ga_real.evaluate()
    print("\nPerformance evaluation:")
    for key, value in evaluation.items():
        print(f"{key}: {value}")
    
    # Plot fitness evolution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    ga_binary.plot_fitness_evolution()
    plt.title("Binary GA Fitness Evolution")
    
    plt.subplot(2, 1, 2)
    ga_real.plot_fitness_evolution()
    plt.title("Real-valued GA Fitness Evolution")
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the function and the solution
    visualize_function_and_solution(objective_function, best_real_chromosome)
    
    # Save the trained model
    print("\nSaving the GA model...")
    ga_real.save_model("../data/ga_function_optimization_model")

def knapsack_problem_example():
    """Demonstrate genetic algorithm for the Knapsack problem."""
    print("Solving the Knapsack problem...\n")
    
    # Define the knapsack problem parameters
    # Items: (value, weight)
    items = [
        (60, 10),   # Item 0
        (100, 20),  # Item 1
        (120, 30),  # Item 2
        (80, 15),   # Item 3
        (90, 25),   # Item 4
        (150, 35),  # Item 5
        (50, 10),   # Item 6
        (70, 15),   # Item 7
        (110, 25),  # Item 8
        (95, 20)    # Item 9
    ]
    
    max_weight = 100  # Maximum weight capacity
    
    # Define the fitness function
    def knapsack_fitness(chromosome):
        total_value = 0
        total_weight = 0
        
        for i, gene in enumerate(chromosome):
            if gene == 1:  # If item is selected
                total_value += items[i][0]
                total_weight += items[i][1]
        
        # Apply penalty if weight exceeds capacity
        if total_weight > max_weight:
            return -total_weight  # Negative fitness for invalid solutions
        
        return total_value
    
    # Create and train the GA model
    print("Running Genetic Algorithm for Knapsack problem...")
    ga_knapsack = GeneticAlgorithmModel(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.05,
        selection_method="tournament",
        tournament_size=3,
        maximize=True,
        early_stopping_generations=20
    )
    
    ga_knapsack.train(
        fitness_function=knapsack_fitness,
        chromosome_length=len(items),
        gene_type="binary"
    )
    
    # Get the best solution
    best_chromosome = ga_knapsack.predict()
    best_fitness = ga_knapsack.get_best_fitness()
    
    # Calculate the total value and weight
    total_value = 0
    total_weight = 0
    selected_items = []
    
    for i, gene in enumerate(best_chromosome):
        if gene == 1:
            total_value += items[i][0]
            total_weight += items[i][1]
            selected_items.append(i)
    
    print(f"Best solution (chromosome): {best_chromosome}")
    print(f"Selected items (indices): {selected_items}")
    print(f"Total value: {total_value}")
    print(f"Total weight: {total_weight}/{max_weight}")
    
    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    ga_knapsack.plot_fitness_evolution()
    plt.title("Knapsack Problem - Fitness Evolution")
    plt.tight_layout()
    plt.show()
    
    # Visualize knapsack solution
    visualize_knapsack_solution(items, best_chromosome, max_weight)

def traveling_salesman_example():
    """Demonstrate genetic algorithm for the Traveling Salesman Problem."""
    print("Solving the Traveling Salesman Problem...\n")
    
    # Define the cities and their coordinates
    cities = {
        'A': (0, 0),
        'B': (1, 5),
        'C': (5, 2),
        'D': (7, 7),
        'E': (3, 6),
        'F': (6, 3),
        'G': (2, 4),
        'H': (8, 1),
        'I': (4, 8),
        'J': (9, 5)
    }
    
    # Calculate distances between cities
    city_names = list(cities.keys())
    n_cities = len(cities)
    distances = np.zeros((n_cities, n_cities))
    
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                xi, yi = cities[city_names[i]]
                xj, yj = cities[city_names[j]]
                distances[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
    
    # Define the fitness function
    def tsp_fitness(chromosome):
        # Calculate the total distance of the route
        total_distance = 0
        for i in range(len(chromosome)):
            from_city = chromosome[i]
            to_city = chromosome[(i + 1) % len(chromosome)]  # Wrap around to the first city
            total_distance += distances[from_city, to_city]
        
        # Invert distance for maximization (shorter distance = higher fitness)
        return 1.0 / total_distance
    
    # Create and train the GA model
    print("Running Genetic Algorithm for TSP...")
    ga_tsp = GeneticAlgorithmModel(
        population_size=100,
        generations=200,
        crossover_rate=0.8,
        mutation_rate=0.2,
        selection_method="tournament",
        tournament_size=5,
        maximize=True,
        early_stopping_generations=30
    )
    
    ga_tsp.train(
        fitness_function=tsp_fitness,
        chromosome_length=n_cities,
        gene_type="permutation"
    )
    
    # Get the best solution
    best_chromosome = ga_tsp.predict()
    best_fitness = ga_tsp.get_best_fitness()
    
    # Calculate the total distance
    total_distance = 0
    route = [city_names[city_idx] for city_idx in best_chromosome]
    
    for i in range(len(best_chromosome)):
        from_city = best_chromosome[i]
        to_city = best_chromosome[(i + 1) % len(best_chromosome)]
        total_distance += distances[from_city, to_city]
    
    print(f"Best route (indices): {best_chromosome}")
    print(f"Best route (cities): {route}")
    print(f"Total distance: {total_distance:.2f}")
    
    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    ga_tsp.plot_fitness_evolution()
    plt.title("TSP - Fitness Evolution")
    plt.tight_layout()
    plt.show()
    
    # Visualize TSP solution
    visualize_tsp_solution(cities, best_chromosome)

def visualize_function_and_solution(objective_function, solution):
    """Visualize the 2D function and the optimal solution."""
    # Create a grid of points
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calculate function values
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = objective_function(np.array([X1[i, j], X2[i, j]]))
    
    # Plot 2D contour
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X1, X2, Z, 20, cmap='viridis')
    plt.colorbar(contour)
    plt.plot(solution[0], solution[1], 'ro', markersize=8)
    plt.title('Function Optimization - Contour Plot')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # Plot 3D surface
    plt.subplot(1, 2, 2, projection='3d')
    ax = plt.gca()
    surface = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.scatter(solution[0], solution[1], objective_function(solution), 
               color='red', s=50, label='Optimal Solution')
    ax.set_title('Function Optimization - 3D Surface')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    
    plt.tight_layout()
    plt.show()

def visualize_knapsack_solution(items, solution, max_weight):
    """Visualize the knapsack problem solution."""
    values = [item[0] for item in items]
    weights = [item[1] for item in items]
    
    # Calculate selected items
    selected_values = [values[i] if solution[i] == 1 else 0 for i in range(len(items))]
    selected_weights = [weights[i] if solution[i] == 1 else 0 for i in range(len(items))]
    
    indices = np.arange(len(items))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    
    # Plot values
    plt.subplot(1, 2, 1)
    plt.bar(indices, values, width, label='Total Value', alpha=0.7)
    plt.bar(indices, selected_values, width, label='Selected', alpha=1.0)
    plt.xlabel('Item Index')
    plt.ylabel('Value')
    plt.title('Item Values')
    plt.xticks(indices)
    plt.legend()
    
    # Plot weights
    plt.subplot(1, 2, 2)
    plt.bar(indices, weights, width, label='Total Weight', alpha=0.7)
    plt.bar(indices, selected_weights, width, label='Selected', alpha=1.0)
    plt.axhline(y=max_weight, color='r', linestyle='-', label=f'Weight Limit ({max_weight})')
    plt.xlabel('Item Index')
    plt.ylabel('Weight')
    plt.title('Item Weights')
    plt.xticks(indices)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot value-weight ratio
    value_weight_ratio = [v/w for v, w in zip(values, weights)]
    plt.figure(figsize=(10, 6))
    plt.bar(indices, value_weight_ratio, width, alpha=0.7)
    for i, selected in enumerate(solution):
        if selected == 1:
            plt.bar(i, value_weight_ratio[i], width, color='green', alpha=1.0)
    plt.xlabel('Item Index')
    plt.ylabel('Value/Weight Ratio')
    plt.title('Item Value-Weight Ratio')
    plt.xticks(indices)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_tsp_solution(cities, route):
    """Visualize the TSP solution."""
    city_names = list(cities.keys())
    
    # Extract coordinates
    x_coords = [cities[city_names[city_idx]][0] for city_idx in route]
    y_coords = [cities[city_names[city_idx]][1] for city_idx in route]
    
    # Add the starting city at the end to complete the route
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    plt.figure(figsize=(10, 8))
    
    # Plot cities
    plt.scatter(
        [cities[city][0] for city in city_names],
        [cities[city][1] for city in city_names],
        s=100, color='blue'
    )
    
    # Label cities
    for city, (x, y) in cities.items():
        plt.annotate(city, (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot route
    plt.plot(x_coords, y_coords, 'r-', alpha=0.7)
    
    # Highlight the start
    plt.scatter(x_coords[0], y_coords[0], s=150, color='green', zorder=5)
    
    plt.title('Traveling Salesman Problem - Optimal Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 