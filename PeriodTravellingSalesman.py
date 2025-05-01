import numpy as np
import random
import matplotlib.pyplot as plt

# Params
NUM_DAYS = 50
POP_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.9
DATA_FILE = "vrp8.txt"
mutation_heatmap = np.zeros((NUM_DAYS, NUM_DAYS), dtype=int)


# Loading Data
def load_data(file_path):
    coords = {}
    freq = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(int, line.strip().split()))
            node_id, x, y, f = parts
            coords[node_id] = (x, y)
            freq[node_id] = f
    return coords, freq

# Helper Functions
def euclidean(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def route_distance(route, coords):
    if len(route) <= 1:
        return 0
    dist = 0
    for i in range(len(route) - 1):
        dist += euclidean(coords[route[i]], coords[route[i+1]])
    dist += euclidean(coords[route[-1]], coords[route[0]]) 
    return dist

# Population
def generate_individual(cities, freq):
    schedule = [[] for _ in range(NUM_DAYS)]
    for city in cities:
        days = random.sample(range(NUM_DAYS), freq[city])
        for d in days:
            schedule[d].append(city)
    for day in schedule:
        random.shuffle(day)
    return schedule

def generate_population(pop_size, cities, freq):
    return [generate_individual(cities, freq) for _ in range(pop_size)]

# Objectives
def evaluate(individual, coords):
    day_distances = [route_distance(day, coords) for day in individual]
    total_distance = sum(day_distances)
    variance = np.var(day_distances)
    return total_distance, variance

# Genetic Operators
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    point = random.randint(1, NUM_DAYS - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, freq):
    global mutation_heatmap
    new_ind = [day[:] for day in individual]

    if random.random() < MUTATION_RATE:
        d1, d2 = random.sample(range(NUM_DAYS), 2)
        if new_ind[d1] and new_ind[d2]:
            c1 = random.choice(new_ind[d1])
            c2 = random.choice(new_ind[d2])
            i1, i2 = new_ind[d1].index(c1), new_ind[d2].index(c2)
            new_ind[d1][i1], new_ind[d2][i2] = c2, c1

            mutation_heatmap[d1][d2] += 1
            mutation_heatmap[d2][d1] += 1  

    return repair_individual(new_ind, freq)

def repair_individual(individual, freq):
    counts = {}
    for d in individual:
        for c in d:
            counts[c] = counts.get(c, 0) + 1

    all_cities = set(freq.keys())
    to_add = []
    to_remove = []

    for city in all_cities:
        diff = freq[city] - counts.get(city, 0)
        if diff > 0:
            to_add.extend([city] * diff)
        elif diff < 0:
            to_remove.extend([city] * (-diff))

    for city in to_remove:
        for d in individual:
            if city in d:
                d.remove(city)
                break

    for city in to_add:
        while True:
            d = random.randint(0, NUM_DAYS - 1)
            if city not in individual[d]:
                individual[d].append(city)
                break

    return individual


def plot_city_visits_comparison(individual, freq):
    actual_visits = {}
    for day in individual:
        for city in day:
            actual_visits[city] = actual_visits.get(city, 0) + 1

    cities = sorted(freq.keys())
    required = [freq[city] for city in cities]
    actual = [actual_visits.get(city, 0) for city in cities]

    x = np.arange(len(cities))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, required, width, label='Required Visits')
    plt.bar(x + width/2, actual, width, label='Actual Visits')
    plt.xlabel("City ID")
    plt.ylabel("Number of Visits")
    plt.title("Required vs. Actual City Visits")
    plt.xticks(x, cities, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()


# Pareto Front
def get_pareto_front(population, scores):
    pareto = []
    for i, a in enumerate(scores):
        dominated = False
        for j, b in enumerate(scores):
            if i != j and (b[0] <= a[0] and b[1] <= a[1]) and (b[0] < a[0] or b[1] < a[1]):
                dominated = True
                break
        if not dominated:
            pareto.append((population[i], a))
    return pareto

# GA Loop
def run_ga(file_path):
    coords, freq = load_data(file_path)
    cities = list(coords.keys())
    population = generate_population(POP_SIZE, cities, freq)

    pareto_archive = []
    best_distances = []

    for gen in range(GENERATIONS):
        print(f"Generation {gen + 1}/{GENERATIONS}")  

        scores = [evaluate(ind, coords) for ind in population]
        current_pareto = get_pareto_front(population, scores)

        # Best total distance
        best = min(scores, key=lambda x: x[0])
        best_distances.append(best[0])

        # Merge with archive
        combined = pareto_archive + current_pareto
        combined_solutions = [p[1] for p in combined]
        combined_population = [p[0] for p in combined]
        pareto_archive = get_pareto_front(combined_population, combined_solutions)

        # Tournament selection 
        selected = []
        for _ in range(POP_SIZE):
            i1, i2 = random.sample(range(POP_SIZE), 2)
            if random.random() < 0.5:
                selected.append(population[i1] if evaluate(population[i1], coords)[0] < evaluate(population[i2], coords)[0] else population[i2])
            else:
                selected.append(population[i1] if evaluate(population[i1], coords)[1] < evaluate(population[i2], coords)[1] else population[i2])

        # Reproduction
        next_population = []
        while len(next_population) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, freq)
            c2 = mutate(c2, freq)
            next_population.extend([c1, c2])

        population = next_population[:POP_SIZE]

    return pareto_archive, best_distances

# Visualization
def plot_pareto(pareto):
    x = [p[1][0] for p in pareto]
    y = [p[1][1] for p in pareto]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, c='blue')
    plt.xlabel("Total Distance")
    plt.ylabel("Variance in Daily Distance")
    plt.title("Pareto Front (Trade-Off Curve)")
    plt.grid(True)
    plt.show()

def plot_best_distance_over_time(best_distances):
    plt.figure()
    plt.plot(best_distances)
    plt.xlabel("Generation")
    plt.ylabel("Best Total Distance")
    plt.title("Convergence of Best Total Distance")
    plt.grid(True)
    plt.show()

def plot_mutation_heatmap():
    plt.figure(figsize=(10, 8))
    plt.imshow(mutation_heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Swap Frequency")
    plt.title("Mutation Activity Heatmap (Day-to-Day Swaps)")
    plt.xlabel("Day")
    plt.ylabel("Day")
    plt.xticks(ticks=np.arange(NUM_DAYS))
    plt.yticks(ticks=np.arange(NUM_DAYS))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pareto, best_distances = run_ga(DATA_FILE)
    plot_pareto(pareto)
    plot_best_distance_over_time(best_distances)

    best_individual = pareto[0][0]  
    coords, freq = load_data(DATA_FILE)  
    plot_city_visits_comparison(best_individual, freq)
    plot_mutation_heatmap()

