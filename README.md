About the problem:
The chemical mixing optimization problem involves finding the best proportions of different
chemicals to achieve a desired property (e.g., strength, color intensity, or heat resistance) at the
minimum cost. Each chemical has upper and lower bounds for its proportion, and the overall
mixture must satisfy specific constraints on total proportions while minimizing the cost.
What you are required to do:
Write a genetic algorithm to determine the optimal proportions of chemicals in a mixture to
minimize the total cost while meeting all the constraints.

What the input looks like:
You will be given an input file with the following format:
1. First line: Number of datasets (at least 1).
2. For each dataset:
o First line: Number of chemicals and total proportion constraint (e.g., 100%).
o Next line: Lower and upper bounds for each chemical (two space-separated
numbers per chemical).
o Next line: Cost coefficients for each chemical (a space-separated list, where each
coefficient represents the cost per unit proportion of that chemical).

Example Input:
1
3 100.0
5.0 25.0 10.0 40.0 15.0 35.0
8.5 6.2 7.8
Explanation:
• The input specifies 1 dataset.
• There are 3 chemicals, and the total mixture proportion must sum to 100%.
• Chemical 1 must be between 5% and 25%, Chemical 2 between 10% and 40%, Chemical
3 between 15% and 35%.
• The cost coefficients are 8.5, 6.2, 7.8, and 5, meaning 1% of Chemical 1 costs 8.5 units,
Chemical 2 costs 6.2 units, and so on.

What the output should look like:
The output should specify the dataset index, the optimal proportions for each chemical, and the
total cost of the mixture.
Example Output:
Dataset 1
Chemical Proportions: 20.5 40.0 39.5
Total Cost: 850.75
