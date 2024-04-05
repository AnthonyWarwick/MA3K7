#!/usr/bin/env python
# coding: utf-8

# In[324]:


#Simple Case
#CODE 1 : Collision Model where probability of each collision is equal at every stage.
import random

#The collision model below chooses each collision randomly at each stage
def simulate_collisions_corrected(n):
    # Initialize particle counts
    A, B, C = n, n, n
    collisions = 0

    while A > 0 and B > 0 or B > 0 and C > 0 or C > 0 and A > 0:
        types = random.choices(["AB", "BC", "CA"], k=1)[0]
        if types == "AB" and A > 0 and B > 0:
            A -= 1
            B -= 1
            C += 1
        elif types == "BC" and B > 0 and C > 0:
            B -= 1
            C -= 1
            A += 1
        elif types == "CA" and C > 0 and A > 0:
            C -= 1
            A -= 1
            B += 1
        collisions += 1

    # Return the number of remaining particles, regardless of type
    remaining_particles = max(A, B, C)
    return remaining_particles, collisions


# Groups by number of particles in the remaining state
def run_simulations_and_count_duplicates(n, iterations):
    final_states = {}
    for _ in range(iterations):
        remaining_particles, _ = simulate_collisions_corrected(n)
        if remaining_particles in final_states:
            final_states[remaining_particles] += 1
        else:
            final_states[remaining_particles] = 1
    
    # Order the final_states dictionary by the size of the remaining particles 
    ordered_final_states = dict(sorted(final_states.items(), key=lambda item: item[1], reverse=True))
    return ordered_final_states

# Example usage, where n is the number of initial particles of each type.
n = 2
iterations = 10000
final_state_occurrences = run_simulations_and_count_duplicates(n, iterations)

final_state_occurrences


# ### 

# In[325]:


#CODE 2 : Collision Model where probability of each collision is equal at every stage, plotting multiple n's.
import random
import matplotlib.pyplot as plt
import numpy as np

#Identical Collision model as code 1
def simulate_collisions_corrected(n):
    A, B, C = n, n, n
    while A > 0 and B > 0 or B > 0 and C > 0 or C > 0 and A > 0:
        types = random.choices(["AB", "BC", "CA"], k=1)[0]
        if types == "AB" and A > 0 and B > 0:
            A -= 1
            B -= 1
            C += 1
        elif types == "BC" and B > 0 and C > 0:
            B -= 1
            C -= 1
            A += 1
        elif types == "CA" and C > 0 and A > 0:
            C -= 1
            A -= 1
            B += 1
    return max(A, B, C), None

#Groups by number of particles in the remaining state
def run_simulations_and_count_duplicates(n, iterations):
    final_states = {}
    for _ in range(iterations):
        remaining_particles, _ = simulate_collisions_corrected(n)
        final_states[remaining_particles] = final_states.get(remaining_particles, 0) + 1
    return final_states


def simulate_for_multiple_n(n_values, iterations):
    results = {}
    for n in n_values:
        final_states = run_simulations_and_count_duplicates(n, iterations)
        probabilities = {state: count / iterations for state, count in final_states.items()}
        results[n] = probabilities
    return results

# Specify the range of 'n' values here
n_values = range(1, 8)  # Feel free to change this
iterations = 1000  # Adjust as needed for more or fewer simulations

simulation_results = simulate_for_multiple_n(n_values, iterations)

# Plotting
plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'l', 'a']  # Extend or modify as needed for distinct colors

max_n = max(n_values)
plt.xticks(np.arange(2, 2*max_n+1, 2))  # Dynamically alters the x axis corresponding to which n values you choose. 

for i, (n, probabilities) in enumerate(simulation_results.items()):
    color = colors[i % len(colors)]  # Cycle through colors if there are more 'n' values than colors
    states = sorted(probabilities.keys())
    prob_values = [probabilities[state] for state in states]
    plt.plot(states, prob_values, 'x-', color=color, label=f'n={n}')

plt.xlabel('Final States')
plt.ylabel('Probability')
plt.title('Probability of Final States for Different Initial States')
plt.legend()
plt.grid(True)
plt.show()


# In[321]:


#Complex Case
#Code 3, updating our collision model to directly align with the conditions of the question. 
import numpy as np

#Defining the probability model as outlines in the attack phase. k is the number of a particles, l the number of b and m the number of c particles at any given change. 
def calculate_corrected_collision_probability(k, l, m):
    total_particles = k + l + m
    prob_bc_corrected = (l / total_particles) * (m / (k + m)) if (k + m) != 0 else 0
    prob_bc_corrected += (m / total_particles) * (l / (k + l)) if (k + l) != 0 else 0
    return prob_bc_corrected

#What the below does is the following: uses the probability equation to calcualte probabilities, randomly choose 
# a collision based on these probabilties, perform the selected collision and repeat the process until no further collisions can occur. 
def simulate_particle_collisions_corrected(a_count, b_count, c_count, iterations):
    final_states_counts = {}  # Store counts of the final state observed in each iteration

    for _ in range(iterations):
        counts = {'a': a_count, 'b': b_count, 'c': c_count}

        while True:
            # Calculate corrected probabilities for each collision type
            prob_ab = calculate_corrected_collision_probability(counts['c'], counts['a'], counts['b'])
            prob_ac = calculate_corrected_collision_probability(counts['b'], counts['c'], counts['a'])
            prob_bc = calculate_corrected_collision_probability(counts['a'], counts['b'], counts['c'])
            
            # Directly choose a collision type based on calculated probabilities
            collision_probabilities = [prob_ab, prob_ac, prob_bc]
            collision_types = ['ab', 'ac', 'bc']
            collision_type = np.random.choice(collision_types, p=collision_probabilities / np.sum(collision_probabilities))

            # Perform the collision
            if collision_type == 'ab' and counts['a'] > 0 and counts['b'] > 0:
                counts['a'] -= 1
                counts['b'] -= 1
                counts['c'] += 1
            elif collision_type == 'ac' and counts['a'] > 0 and counts['c'] > 0:
                counts['a'] -= 1
                counts['c'] -= 1
                counts['b'] += 1
            elif collision_type == 'bc' and counts['b'] > 0 and counts['c'] > 0:
                counts['b'] -= 1
                counts['c'] -= 1
                counts['a'] += 1

            # If only one type of particle remains, stop the simulation
            if sum(val > 0 for val in counts.values()) <= 1:
                break

        final_particle_count = sum(counts.values())
        final_states_counts[final_particle_count] = final_states_counts.get(final_particle_count, 0) + 1

    return final_states_counts

# Example run with 2 particles of each type and 100 iterations
results = simulate_particle_collisions_corrected(5, 5, 5, 1000)
print(results)


# In[308]:


#Code 4, plots for multiple n 
import numpy as np
import matplotlib.pyplot as plt

#Same as code 3 
def calculate_corrected_collision_probability(k, l, m):
    total_particles = k + l + m
    prob_bc_corrected = (l / total_particles) * (m / (k + m)) if (k + m) != 0 else 0
    prob_bc_corrected += (m / total_particles) * (l / (k + l)) if (k + l) != 0 else 0
    return prob_bc_corrected

#Same as code 3 
def simulate_particle_collisions_corrected(a_count, b_count, c_count, iterations):
    final_states_counts = {}
    for _ in range(iterations):
        counts = {'a': a_count, 'b': b_count, 'c': c_count}
        while True:
            prob_ab = calculate_corrected_collision_probability(counts['c'], counts['a'], counts['b'])
            prob_ac = calculate_corrected_collision_probability(counts['b'], counts['c'], counts['a'])
            prob_bc = calculate_corrected_collision_probability(counts['a'], counts['b'], counts['c'])
            collision_probabilities = [prob_ab, prob_ac, prob_bc]
            collision_types = ['ab', 'ac', 'bc']
            collision_type = np.random.choice(collision_types, p=collision_probabilities / np.sum(collision_probabilities))
            if collision_type == 'ab' and counts['a'] > 0 and counts['b'] > 0:
                counts['a'] -= 1; counts['b'] -= 1; counts['c'] += 1
            elif collision_type == 'ac' and counts['a'] > 0 and counts['c'] > 0:
                counts['a'] -= 1; counts['c'] -= 1; counts['b'] += 1
            elif collision_type == 'bc' and counts['b'] > 0 and counts['c'] > 0:
                counts['b'] -= 1; counts['c'] -= 1; counts['a'] += 1
            if sum(val > 0 for val in counts.values()) <= 1:
                break
        final_particle_count = sum(counts.values())
        final_states_counts[final_particle_count] = final_states_counts.get(final_particle_count, 0) + 1
    return final_states_counts

# Adjustable range of n values and automatic adjustment of the x-axis
n_range = range(1, 8)  # Extend this range as needed
final_states = range(2, 2*max(n_range)+2, 2)

# Store results
results = {n: [] for n in n_range}
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Added 'k' for black if n=7 is included

# Run simulations
for n in n_range:
    state_counts = simulate_particle_collisions_corrected(n, n, n, 1000)
    total_counts = sum(state_counts.values())
    for state in final_states:
        probability = state_counts.get(state, 0) / total_counts
        results[n].append(probability)

# Plotting
plt.figure(figsize=(10, 6))
for n, color in zip(n_range, colors):
    plt.plot(final_states, results[n], marker='o', color=color, label=f'n={n}')

plt.xlabel('Final State (Number of Particles)')
plt.ylabel('Probability')
plt.title('Probability of Final States for different n')
plt.xticks(list(final_states))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid(True)
plt.show()


# In[322]:


#Code 5, gives the specific particles observed to be in the remaning states.
import numpy as np

# Same as code 3 
def calculate_corrected_collision_probability(k, l, m):
    total_particles = k + l + m
    prob_bc_corrected = (l / total_particles) * (m / (k + m)) if (k + m) != 0 else 0
    prob_bc_corrected += (m / total_particles) * (l / (k + l)) if (k + l) != 0 else 0
    return prob_bc_corrected

#Same as code 3 but now we distinguish between the different particles
def simulate_particle_collisions_detailed(a_count, b_count, c_count, iterations):
    final_states_details = {}
    for _ in range(iterations):
        counts = {'a': a_count, 'b': b_count, 'c': c_count}
        while True:
            prob_ab = calculate_corrected_collision_probability(counts['c'], counts['a'], counts['b'])
            prob_ac = calculate_corrected_collision_probability(counts['b'], counts['c'], counts['a'])
            prob_bc = calculate_corrected_collision_probability(counts['a'], counts['b'], counts['c'])
            collision_probabilities = [prob_ab, prob_ac, prob_bc]
            collision_types = ['ab', 'ac', 'bc']
            collision_type = np.random.choice(collision_types, p=collision_probabilities / np.sum(collision_probabilities))
            if collision_type == 'ab' and counts['a'] > 0 and counts['b'] > 0:
                counts['a'] -= 1; counts['b'] -= 1; counts['c'] += 1
            elif collision_type == 'ac' and counts['a'] > 0 and counts['c'] > 0:
                counts['a'] -= 1; counts['c'] -= 1; counts['b'] += 1
            elif collision_type == 'bc' and counts['b'] > 0 and counts['c'] > 0:
                counts['b'] -= 1; counts['c'] -= 1; counts['a'] += 1
            if sum(val > 0 for val in counts.values()) <= 1:
                break
        final_particle_count = sum(counts.values())
        final_state_key = f'{final_particle_count} particles remaining'
        if final_state_key not in final_states_details:
            final_states_details[final_state_key] = {'a': 0, 'b': 0, 'c': 0}
        for particle, count in counts.items():
            if count > 0:
                final_states_details[final_state_key][particle] += 1
    return final_states_details

# Running the simulation 
detailed_results = simulate_particle_collisions_detailed(5, 5,5, 10000)
detailed_results


# In[323]:


#Code 6, plots for multiple intial states 
def simulate_particle_collisions_corrected(a_count, b_count, c_count, iterations):
    final_states_counts = {}
    for _ in range(iterations):
        counts = {'a': a_count, 'b': b_count, 'c': c_count}
        while True:
            prob_ab = calculate_corrected_collision_probability(counts['c'], counts['a'], counts['b'])
            prob_ac = calculate_corrected_collision_probability(counts['b'], counts['c'], counts['a'])
            prob_bc = calculate_corrected_collision_probability(counts['a'], counts['b'], counts['c'])
            collision_probabilities = [prob_ab, prob_ac, prob_bc]
            collision_types = ['ab', 'ac', 'bc']
            collision_type = np.random.choice(collision_types, p=collision_probabilities / np.sum(collision_probabilities))
            if collision_type == 'ab' and counts['a'] > 0 and counts['b'] > 0:
                counts['a'] -= 1; counts['b'] -= 1; counts['c'] += 1
            elif collision_type == 'ac' and counts['a'] > 0 and counts['c'] > 0:
                counts['a'] -= 1; counts['c'] -= 1; counts['b'] += 1
            elif collision_type == 'bc' and counts['b'] > 0 and counts['c'] > 0:
                counts['b'] -= 1; counts['c'] -= 1; counts['a'] += 1
            if sum(val > 0 for val in counts.values()) <= 1:
                break
        final_particle_count = sum(counts.values())
        final_states_counts[final_particle_count] = final_states_counts.get(final_particle_count, 0) + 1
    return final_states_counts

# Specify the initial configurations for the graph
initial_configs = [(5, 5, 5), (5, 5, 4), (5, 4, 4), (5, 4, 3)]
colors = ['blue', 'green', 'red', 'cyan']  # Colors for each line

# Initialize matplotlib plot
plt.figure(figsize=(10, 6))

# Run the simulation and plot for each initial configuration
for config, color in zip(initial_configs, colors):
    results = simulate_particle_collisions_corrected(*config, 1000)
    # Preparing data for plotting, e.g. calculating approximate probabilities 
    final_states = sorted(results.keys())
    probabilities = [results[state] / 1000 for state in final_states]
    plt.plot(final_states, probabilities, marker='o', color=color, label=f'Initial: {config}')

plt.xlabel('Final State (Number of Particles)')
plt.ylabel('Probability')
plt.title('Probability of Final States for Different Initial Configurations')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




