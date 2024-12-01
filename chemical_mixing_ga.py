import random  # To generate random numbers for initialization and mutation
import numpy as np  # For numerical computations, especially with arrays


# Chromosome class represents a solution in the population
class Chromosome:
    def __init__(self, proportions, cost):
        """
        Initialize a chromosome with proportions and corresponding cost.
        :param proportions: Array of chemical proportions (floats).
        :param cost: Total cost of the solution (float).
        """
        self.proportions = proportions  # The proportions of chemicals in this solution
        self.cost = cost  # The total cost of the proportions


def main():
    """
    Main function to read input, process datasets, and run the genetic algorithm.
    """
    # Open the input file in read mode
    with open("input.txt", "r") as file:
        # Read the number of datasets to process
        datasets = int(file.readline().strip())
        output = []  # List to store the output for all datasets

        # Loop through each dataset
        for d in range(1, datasets + 1):
            # Read the number of chemicals and the total desired proportion
            params = file.readline().split()
            num_chemicals = int(params[0])
            total_proportion = float(params[1])

            # Read the bounds for each chemical as tuples of (lower_bound, upper_bound)
            bounds_line = file.readline().split()
            bounds = [
                (float(bounds_line[2 * i]), float(bounds_line[2 * i + 1]))
                for i in range(num_chemicals)
            ]

            # Calculate the sum of lower and upper bounds
            lower_bound_sum = sum(b[0] for b in bounds)
            upper_bound_sum = sum(b[1] for b in bounds)

            # Start building the output for the current dataset
            output.append(f"Dataset {d}")

            # If the sum of lower bounds exceeds the total proportion, it's invalid
            if lower_bound_sum > total_proportion:
                output.append(
                    f"Lower Bound Sum that is lower than the total proportion: {lower_bound_sum}"
                )

            # If the sum of upper bounds is less than the total proportion, it's invalid
            if upper_bound_sum < total_proportion:
                output.append(
                    f"Upper Bound Sum that is lower than the total proportion: {upper_bound_sum}"
                )

            # Read the cost per unit of each chemical
            costs = list(map(float, file.readline().split()))

            # Solve the optimization problem using the genetic algorithm
            best_solution = genetic_algorithm(
                num_chemicals, total_proportion, bounds, costs
            )

            # Add the best solution's proportions and total cost to the output
            output.append(f"Chemical Proportions: {best_solution.proportions}")
            output.append(f"Total Cost: {best_solution.cost:.2f}")

    # Write the output to an output file
    with open("output.txt", "w") as file:
        for line in output:
            file.write(line + "\n")



def genetic_algorithm(num_chemicals, total_proportion, bounds, costs):
    population_size = 100
    generations = 500  # Total number of generations
    crossover_rate = 0.9
    mutation_rate = 0.1

    # Initialize the population
    population = initialize_population(
        population_size, num_chemicals, total_proportion, bounds, costs
    )

    for generation in range(generations):  # Track the current generation
        # Selection phase: Create a mating pool using tournament selection
        mating_pool = [tournament_selection(population) for _ in range(population_size)]

        # Crossover phase: Generate offspring
        offspring = []
        for i in range(0, len(mating_pool), 2):
            if i + 1 < len(mating_pool) and random.random() < crossover_rate:
                offspring.extend(
                    two_point_crossover(
                        mating_pool[i],
                        mating_pool[i + 1],
                        total_proportion,
                        bounds,
                        costs,
                    )
                )
            else:
                offspring.append(mating_pool[i])

        # Mutation phase: Mutate offspring
        for child in offspring:
            if random.random() < mutation_rate:
                mutate(child, bounds, total_proportion, costs, generation, generations)

        # Replacement phase: Elitist replacement
        population = elitist_replacement(population, offspring)

    # Return the best solution in the final population
    return min(population, key=lambda c: c.cost)
    
def initialize_population(size, num_chemicals, total_proportion, bounds, costs):
    """
    Initialize a population of Chromosomes with random proportions.
    """
    population = []
    for _ in range(size):
        # Generate random proportions within bounds
        proportions = np.array(
            [random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_chemicals)]
        )
        # Scale proportions to meet the total_proportion constraint
        proportions *= total_proportion / proportions.sum()
        # Calculate the cost of the current solution
        cost = calculate_cost(proportions, costs)
        # Create a new Chromosome object and add to the population
        population.append(Chromosome(proportions, cost))
    return population


def tournament_selection(population):
    """
    Select the better of two random Chromosomes (tournament selection).
    """
    c1, c2 = random.sample(population, 2)
    return c1 if c1.cost < c2.cost else c2


def two_point_crossover(p1, p2, total_proportion, bounds, costs):
    """
    Perform two-point crossover between two parents to produce two offspring.
    """
    n = len(p1.proportions)
    # Select two random crossover points
    point1, point2 = sorted(random.sample(range(n), 2))
    # Create offspring by swapping segments between the parents
    child1 = np.copy(p1.proportions)
    child2 = np.copy(p2.proportions)
    child1[point1:point2], child2[point1:point2] = (
        child2[point1:point2],
        child1[point1:point2],
    )
    # Normalize the offspring to meet the constraints
    normalize_proportions(child1, total_proportion, bounds)
    normalize_proportions(child2, total_proportion, bounds)
    # Return the two offspring as Chromosome objects
    return [
        Chromosome(child1, calculate_cost(child1, costs)),
        Chromosome(child2, calculate_cost(child2, costs)),
    ]


def mutate(chromosome, bounds, total_proportion, costs, generation, max_generations):
    """
    Non-uniformly mutates a single gene of a Chromosome.
    The mutation magnitude decreases as generations progress.
    
    :param chromosome: The Chromosome to mutate.
    :param bounds: Bounds for each gene as a list of (lower_bound, upper_bound).
    :param total_proportion: Desired total proportion constraint.
    :param costs: Cost per unit for each gene.
    :param generation: Current generation number.
    :param max_generations: Total number of generations.
    """
    index = random.randint(0, len(chromosome.proportions) - 1)  # Randomly select a gene to mutate

    # Mutation range is bounded by the gene's lower and upper bounds
    lower, upper = bounds[index]

    # Calculate the mutation magnitude as a fraction of the current generation
    t = generation / max_generations  # Normalized progress (0 to 1)
    delta = (upper - lower) * (1 - t ** 2)  # Non-uniform decay: larger early, smaller later

    # Randomly decide to increase or decrease the selected proportion
    if random.random() < 0.5:
        new_value = chromosome.proportions[index] + delta * random.random()
    else:
        new_value = chromosome.proportions[index] - delta * random.random()

    # Ensure the new value is within bounds
    new_value = max(lower, min(new_value, upper))

    # Update the proportion and normalize
    chromosome.proportions[index] = new_value
    normalize_proportions(chromosome.proportions, total_proportion, bounds)

    # Recalculate the cost for the mutated chromosome
    chromosome.cost = calculate_cost(chromosome.proportions, costs)
def elitist_replacement(old_pop, new_pop):
    """
    Replace the old population with the best solutions from both old and new populations.
    """
    combined = old_pop + new_pop
    combined.sort(key=lambda c: c.cost)  # Sort by cost (lower is better)
    return combined[: len(old_pop)]  # Keep the same population size


def normalize_proportions(proportions, total_proportion, bounds):
    """
    Normalize proportions to meet total_proportion constraint and ensure bounds are respected.
    """
    proportions *= total_proportion / proportions.sum()
    for i in range(len(proportions)):
        proportions[i] = max(bounds[i][0], min(proportions[i], bounds[i][1]))
    for _ in range(100):  # Ensure proportions sum to total_proportion
        excess = total_proportion - proportions.sum()
        if abs(excess) < 1e-6:  # Stop if adjustment is very small
            break
        for i in range(len(proportions)):
            if excess > 0 and proportions[i] < bounds[i][1]:
                adjustment = min(excess, bounds[i][1] - proportions[i])
                proportions[i] += adjustment
                excess -= adjustment
            elif excess < 0 and proportions[i] > bounds[i][0]:
                adjustment = max(excess, bounds[i][0] - proportions[i])
                proportions[i] += adjustment
                excess -= adjustment
            if abs(excess) < 1e-6:
                break
    return proportions


def calculate_cost(proportions, costs):
    """
    Calculate the total cost of the proportions based on unit costs.
    """
    return np.dot(proportions, costs)


if __name__ == "__main__":
    main()
