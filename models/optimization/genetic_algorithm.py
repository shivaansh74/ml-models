"""
Genetic Algorithm Implementation

This module provides a Genetic Algorithm implementation that follows the standardized
template structure. Genetic algorithms are optimization techniques inspired by natural
selection, using concepts like mutation, crossover, and selection to find optimal solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import random
from tqdm import tqdm
import copy
import time
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class GeneticAlgorithmModel(MLModel):
    """
    Genetic Algorithm model for optimization problems.
    
    Use case: Function optimization, feature selection, scheduling problems,
    game playing, and other optimization tasks where traditional gradient-based
    methods may not work well.
    """
    
    def __init__(self, population_size=100, generations=100, crossover_rate=0.8,
                mutation_rate=0.2, elitism=True, elitism_ratio=0.1,
                selection_method="tournament", tournament_size=3,
                maximize=True, early_stopping_generations=20, **kwargs):
        """
        Initialize the Genetic Algorithm model.
        
        Args:
            population_size: Size of the population in each generation
            generations: Maximum number of generations to evolve
            crossover_rate: Probability of crossover between two individuals
            mutation_rate: Probability of mutation for each gene
            elitism: Whether to keep the best individuals
            elitism_ratio: Ratio of population to keep as elite
            selection_method: Method for parent selection ('tournament', 'roulette', 'rank')
            tournament_size: Size of tournament when using tournament selection
            maximize: Whether to maximize (True) or minimize (False) the fitness function
            early_stopping_generations: Stop if no improvement after this many generations
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_type = "Genetic Algorithm"
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.elitism_ratio = elitism_ratio
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.maximize = maximize
        self.early_stopping_generations = early_stopping_generations
        
        # Attributes that will be set during training
        self.population = None
        self.chromosome_length = None
        self.gene_type = None
        self.gene_range = None
        self.fitness_function = None
        self.custom_crossover = None
        self.custom_mutation = None
        self.custom_initialization = None
        self.history = []
        self.best_individual = None
        self.best_fitness = float('-inf') if maximize else float('inf')
        
        self.model_params = {
            'population_size': population_size,
            'generations': generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elitism': elitism,
            'elitism_ratio': elitism_ratio,
            'selection_method': selection_method,
            'tournament_size': tournament_size,
            'maximize': maximize,
            'early_stopping_generations': early_stopping_generations,
            **kwargs
        }
    
    def _initialize_population(self, chromosome_length, gene_type='binary', gene_range=None):
        """
        Initialize the population with random chromosomes.
        
        Args:
            chromosome_length: Length of each chromosome
            gene_type: Type of genes ('binary', 'integer', 'real', 'permutation')
            gene_range: Range of values for genes (min, max) for integer/real types
            
        Returns:
            List of chromosomes representing the initial population
        """
        population = []
        
        if self.custom_initialization is not None:
            # Use custom initialization function if provided
            return self.custom_initialization(self.population_size, chromosome_length, gene_type, gene_range)
        
        for _ in range(self.population_size):
            if gene_type == 'binary':
                # Binary chromosomes (0 or 1)
                chromosome = np.random.randint(0, 2, chromosome_length)
            
            elif gene_type == 'integer':
                # Integer chromosomes within range
                if gene_range is None:
                    gene_range = (0, 100)
                chromosome = np.random.randint(gene_range[0], gene_range[1] + 1, chromosome_length)
            
            elif gene_type == 'real':
                # Real-valued chromosomes within range
                if gene_range is None:
                    gene_range = (0.0, 1.0)
                chromosome = np.random.uniform(gene_range[0], gene_range[1], chromosome_length)
            
            elif gene_type == 'permutation':
                # Permutation chromosomes (e.g., for TSP)
                chromosome = np.random.permutation(chromosome_length)
            
            else:
                raise ValueError(f"Unsupported gene type: {gene_type}")
            
            population.append(chromosome)
        
        return population
    
    def _calculate_fitness(self, population):
        """
        Calculate fitness for each individual in the population.
        
        Args:
            population: List of chromosomes
            
        Returns:
            List of fitness values
        """
        fitness_values = []
        
        for chromosome in population:
            fitness = self.fitness_function(chromosome)
            fitness_values.append(fitness)
        
        return np.array(fitness_values)
    
    def _select_parents(self, population, fitness_values):
        """
        Select parents for reproduction based on the selection method.
        
        Args:
            population: List of chromosomes
            fitness_values: List of fitness values
            
        Returns:
            Two parent chromosomes
        """
        if self.selection_method == 'tournament':
            # Tournament selection
            selected_parents = []
            
            for _ in range(2):  # Select two parents
                tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
                tournament_fitness = fitness_values[tournament_indices]
                
                if self.maximize:
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                else:
                    winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                
                selected_parents.append(population[winner_idx])
            
            return selected_parents[0], selected_parents[1]
        
        elif self.selection_method == 'roulette':
            # Roulette wheel selection
            if self.maximize:
                # For maximization problems
                fitness_adjusted = fitness_values - min(fitness_values) + 1e-10
            else:
                # For minimization problems, invert fitness
                fitness_adjusted = 1.0 / (fitness_values + 1e-10)
            
            # Calculate selection probabilities
            selection_probs = fitness_adjusted / np.sum(fitness_adjusted)
            
            # Select two parents
            parent_indices = np.random.choice(len(population), 2, p=selection_probs)
            return population[parent_indices[0]], population[parent_indices[1]]
        
        elif self.selection_method == 'rank':
            # Rank-based selection
            # Sort indices by fitness
            if self.maximize:
                rank_indices = np.argsort(fitness_values)[::-1]
            else:
                rank_indices = np.argsort(fitness_values)
            
            # Assign ranks (higher rank = better fitness)
            ranks = np.zeros(len(population))
            for i, idx in enumerate(rank_indices):
                ranks[idx] = len(population) - i
            
            # Calculate selection probabilities based on ranks
            selection_probs = ranks / np.sum(ranks)
            
            # Select two parents
            parent_indices = np.random.choice(len(population), 2, p=selection_probs)
            return population[parent_indices[0]], population[parent_indices[1]]
        
        else:
            raise ValueError(f"Unsupported selection method: {self.selection_method}")
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create offspring.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two offspring chromosomes
        """
        if self.custom_crossover is not None:
            # Use custom crossover function if provided
            return self.custom_crossover(parent1, parent2)
        
        # Check if crossover should occur
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.gene_type == 'permutation':
            # Order crossover (OX) for permutation problems
            offspring1 = np.zeros_like(parent1)
            offspring2 = np.zeros_like(parent2)
            
            # Select crossover points
            cx_points = sorted(np.random.choice(len(parent1) + 1, 2, replace=False))
            
            # Copy segment from first parent
            offspring1[cx_points[0]:cx_points[1]] = parent1[cx_points[0]:cx_points[1]]
            offspring2[cx_points[0]:cx_points[1]] = parent2[cx_points[0]:cx_points[1]]
            
            # Fill remaining positions with values from second parent
            p1_idx = cx_points[1] % len(parent1)
            p2_idx = cx_points[1] % len(parent2)
            o1_idx = cx_points[1] % len(offspring1)
            o2_idx = cx_points[1] % len(offspring2)
            
            for _ in range(len(parent1) - (cx_points[1] - cx_points[0])):
                # Fill offspring1 from parent2
                while parent2[p2_idx] in offspring1[cx_points[0]:cx_points[1]]:
                    p2_idx = (p2_idx + 1) % len(parent2)
                offspring1[o1_idx] = parent2[p2_idx]
                p2_idx = (p2_idx + 1) % len(parent2)
                o1_idx = (o1_idx + 1) % len(offspring1)
                
                # Fill offspring2 from parent1
                while parent1[p1_idx] in offspring2[cx_points[0]:cx_points[1]]:
                    p1_idx = (p1_idx + 1) % len(parent1)
                offspring2[o2_idx] = parent1[p1_idx]
                p1_idx = (p1_idx + 1) % len(parent1)
                o2_idx = (o2_idx + 1) % len(offspring2)
            
            return offspring1, offspring2
        
        else:
            # One-point or uniform crossover for other gene types
            if len(parent1) <= 2:
                # For very short chromosomes, use uniform crossover
                mask = np.random.randint(0, 2, len(parent1))
                offspring1 = np.where(mask, parent1, parent2)
                offspring2 = np.where(mask, parent2, parent1)
            else:
                # One-point crossover
                crossover_point = np.random.randint(1, len(parent1))
                offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
            return offspring1, offspring2
    
    def _mutate(self, chromosome):
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        if self.custom_mutation is not None:
            # Use custom mutation function if provided
            return self.custom_mutation(chromosome, self.mutation_rate)
        
        mutated_chromosome = chromosome.copy()
        
        if self.gene_type == 'binary':
            # Flip bits with probability mutation_rate
            for i in range(len(chromosome)):
                if np.random.random() < self.mutation_rate:
                    mutated_chromosome[i] = 1 - mutated_chromosome[i]
        
        elif self.gene_type in ['integer', 'real']:
            # Add random noise with probability mutation_rate
            for i in range(len(chromosome)):
                if np.random.random() < self.mutation_rate:
                    if self.gene_type == 'integer':
                        # For integer genes, add a random integer within +/- 10% of range
                        range_size = self.gene_range[1] - self.gene_range[0]
                        noise = np.random.randint(-int(0.1 * range_size), int(0.1 * range_size) + 1)
                        mutated_value = chromosome[i] + noise
                        # Ensure the value stays within range
                        mutated_chromosome[i] = max(self.gene_range[0], min(self.gene_range[1], mutated_value))
                    else:
                        # For real genes, add Gaussian noise
                        range_size = self.gene_range[1] - self.gene_range[0]
                        noise = np.random.normal(0, 0.1 * range_size)
                        mutated_value = chromosome[i] + noise
                        # Ensure the value stays within range
                        mutated_chromosome[i] = max(self.gene_range[0], min(self.gene_range[1], mutated_value))
        
        elif self.gene_type == 'permutation':
            # Swap mutation for permutation problems
            if np.random.random() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(chromosome), 2, replace=False)
                mutated_chromosome[idx1], mutated_chromosome[idx2] = mutated_chromosome[idx2], mutated_chromosome[idx1]
        
        return mutated_chromosome
    
    def train(self, fitness_function, chromosome_length, gene_type='binary', gene_range=None, 
             custom_crossover=None, custom_mutation=None, custom_initialization=None, **kwargs):
        """
        Train the genetic algorithm model to find optimal solutions.
        
        Args:
            fitness_function: Function to evaluate fitness of each individual
            chromosome_length: Length of each chromosome
            gene_type: Type of genes ('binary', 'integer', 'real', 'permutation')
            gene_range: Range of values for genes (min, max) for integer/real types
            custom_crossover: Custom crossover function
            custom_mutation: Custom mutation function
            custom_initialization: Custom initialization function
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        # Update parameters if provided
        for key, value in kwargs.items():
            if key in self.model_params:
                self.model_params[key] = value
                setattr(self, key, value)
        
        # Store configuration
        self.fitness_function = fitness_function
        self.chromosome_length = chromosome_length
        self.gene_type = gene_type
        self.gene_range = gene_range
        self.custom_crossover = custom_crossover
        self.custom_mutation = custom_mutation
        self.custom_initialization = custom_initialization
        
        # Reset history
        self.history = []
        
        # Initialize population
        self.population = self._initialize_population(chromosome_length, gene_type, gene_range)
        
        # Reset best individual
        self.best_individual = None
        self.best_fitness = float('-inf') if self.maximize else float('inf')
        
        # Counter for early stopping
        generations_without_improvement = 0
        
        # Main evolution loop
        for generation in tqdm(range(self.generations), desc="Evolving"):
            # Calculate fitness for current population
            fitness_values = self._calculate_fitness(self.population)
            
            # Keep track of best individual
            if self.maximize:
                best_idx = np.argmax(fitness_values)
                current_best_fitness = fitness_values[best_idx]
                
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_individual = self.population[best_idx].copy()
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            else:
                best_idx = np.argmin(fitness_values)
                current_best_fitness = fitness_values[best_idx]
                
                if current_best_fitness < self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_individual = self.population[best_idx].copy()
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            
            # Store history
            self.history.append({
                'generation': generation,
                'best_fitness': current_best_fitness,
                'avg_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values)
            })
            
            # Early stopping check
            if generations_without_improvement >= self.early_stopping_generations:
                print(f"Stopping early after {generation + 1} generations without improvement")
                break
            
            # Create new population
            new_population = []
            
            # Elitism: Keep best individuals
            if self.elitism:
                elite_count = max(1, int(self.elitism_ratio * self.population_size))
                
                if self.maximize:
                    elite_indices = np.argsort(fitness_values)[-elite_count:]
                else:
                    elite_indices = np.argsort(fitness_values)[:elite_count]
                
                for idx in elite_indices:
                    new_population.append(self.population[idx].copy())
            
            # Fill the rest of the population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self._select_parents(self.population, fitness_values)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                # Add to new population
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Replace old population
            self.population = new_population
        
        self.is_trained = True
        return self
    
    def predict(self, X=None):
        """
        Return the best solution found by the genetic algorithm.
        
        Args:
            X: Not used for genetic algorithms
            
        Returns:
            Best individual found
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return self.best_individual
    
    def get_best_fitness(self):
        """
        Return the fitness of the best solution.
        
        Returns:
            Fitness value of the best individual
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return self.best_fitness
    
    def evaluate(self, X=None, y=None):
        """
        Evaluate the performance of the genetic algorithm.
        
        Args:
            X: Not used
            y: Not used
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Calculate final fitness
        fitness = self.fitness_function(self.best_individual)
        
        # Calculate convergence rate
        if len(self.history) >= 2:
            fitness_values = [gen['best_fitness'] for gen in self.history]
            last_quarter_idx = int(len(fitness_values) * 0.75)
            if last_quarter_idx < len(fitness_values):
                last_quarter_improvement = abs(fitness_values[-1] - fitness_values[last_quarter_idx])
            else:
                last_quarter_improvement = 0
        else:
            last_quarter_improvement = 0
        
        # Calculate diversity in final population
        if self.population is not None:
            diversity = np.mean([np.mean(np.abs(ind - self.best_individual)) 
                               for ind in self.population])
        else:
            diversity = 0
        
        return {
            'best_fitness': fitness,
            'generations': len(self.history),
            'convergence_rate': last_quarter_improvement,
            'population_diversity': diversity
        }
    
    def plot_fitness_evolution(self):
        """
        Plot the evolution of fitness values over generations.
        
        Returns:
            Matplotlib figure
        """
        if not self.history:
            raise ValueError("No history available. Train the model first.")
        
        generations = [gen['generation'] for gen in self.history]
        best_fitness = [gen['best_fitness'] for gen in self.history]
        avg_fitness = [gen['avg_fitness'] for gen in self.history]
        std_fitness = [gen['std_fitness'] for gen in self.history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot best and average fitness
        ax.plot(generations, best_fitness, 'r-', label='Best Fitness')
        ax.plot(generations, avg_fitness, 'b-', label='Average Fitness')
        
        # Add error bands for standard deviation
        ax.fill_between(generations, 
                        [avg - std for avg, std in zip(avg_fitness, std_fitness)],
                        [avg + std for avg, std in zip(avg_fitness, std_fitness)],
                        alpha=0.2, color='blue')
        
        ax.set_title('Fitness Evolution')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_population_diversity(self):
        """
        Plot the diversity of the population over generations.
        
        Returns:
            Matplotlib figure
        """
        if not self.is_trained or not self.history:
            raise ValueError("No history available. Train the model first.")
        
        # Calculate diversity at each generation
        diversity = []
        population_snapshots = []
        
        # Recreate population evolution
        temp_population = self._initialize_population(self.chromosome_length, self.gene_type, self.gene_range)
        fitness_values = self._calculate_fitness(temp_population)
        population_snapshots.append(copy.deepcopy(temp_population))
        
        for gen in range(min(10, len(self.history))):
            # Create new population
            new_population = []
            
            # Elitism: Keep best individuals
            if self.elitism:
                elite_count = max(1, int(self.elitism_ratio * self.population_size))
                
                if self.maximize:
                    elite_indices = np.argsort(fitness_values)[-elite_count:]
                else:
                    elite_indices = np.argsort(fitness_values)[:elite_count]
                
                for idx in elite_indices:
                    new_population.append(temp_population[idx].copy())
            
            # Fill the rest of the population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self._select_parents(temp_population, fitness_values)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                # Add to new population
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Replace old population
            temp_population = new_population
            fitness_values = self._calculate_fitness(temp_population)
            population_snapshots.append(copy.deepcopy(temp_population))
        
        # Calculate diversity for each snapshot
        for pop in population_snapshots:
            # Average Euclidean distance between all pairs of individuals
            sum_distance = 0
            count = 0
            for i in range(len(pop)):
                for j in range(i+1, len(pop)):
                    sum_distance += np.mean(np.abs(pop[i] - pop[j]))
                    count += 1
            if count > 0:
                diversity.append(sum_distance / count)
            else:
                diversity.append(0)
        
        # Plot diversity
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(diversity)), diversity, 'g-o')
        ax.set_title('Population Diversity')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Distance Between Individuals')
        ax.grid(True)
        
        return fig
    
    def save_model(self, filepath):
        """
        Save the genetic algorithm model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Can't save the fitness function directly
        print("Warning: Fitness function cannot be saved and must be provided again when loading.")
        
        model_data = {
            'model_params': self.model_params,
            'chromosome_length': self.chromosome_length,
            'gene_type': self.gene_type,
            'gene_range': self.gene_range,
            'is_trained': self.is_trained,
            'history': self.history,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'population': self.population
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath, fitness_function):
        """
        Load a genetic algorithm model from disk.
        
        Args:
            filepath: Path to the saved model
            fitness_function: Function to evaluate fitness (must be provided again)
            
        Returns:
            self: The loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Load model parameters
        self.model_params = model_data['model_params']
        self.chromosome_length = model_data['chromosome_length']
        self.gene_type = model_data['gene_type']
        self.gene_range = model_data['gene_range']
        self.is_trained = model_data['is_trained']
        self.history = model_data['history']
        self.best_individual = model_data['best_individual']
        self.best_fitness = model_data['best_fitness']
        self.population = model_data['population']
        
        # Set the fitness function
        self.fitness_function = fitness_function
        
        # Set attributes from model_params
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        return self 