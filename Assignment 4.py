#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt

# We will extend the calculation of the recursive sequence up to k=30
def compute_extended_probability(k, memo={}):
    # Check if the result is already computed
    if k in memo:
        return memo[k]
    
    # Base cases
    if k == 1:
        return 1
    if k == 2:
        return 0.5
    
    # Recursive formula
    if k-1 not in memo:
        memo[k-1] = compute_extended_probability(k-1, memo)
    if k-2 not in memo:
        memo[k-2] = compute_extended_probability(k-2, memo)
    
    memo[k] = 0.5 * memo[k-1] + 0.5 * memo[k-2]
    return memo[k]

# Calculate the probabilities for k from 1 to 30
extended_probabilities = [compute_extended_probability(k) for k in range(1, 31)]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(range(1, 31), extended_probabilities, marker='o')
plt.title('Probability of Landing on Square N')
plt.xlabel('N')
plt.ylabel('Probability')
plt.grid(True)
plt.xticks(range(1, 31))
plt.show()


# In[5]:


# Recursive function to compute the probability of landing on square k
def compute_probability(k, memo={}):
    # Check if the result is already computed
    if k in memo:
        return memo[k]
    
    # Base cases
    if k == 1:
        return 1
    if k == 2:
        return 0.5
    
    # Recursive formula
    memo[k] = 0.5 * compute_probability(k-1, memo) + 0.5 * compute_probability(k-2, memo)
    return memo[k]

# Compute the probabilities for squares 1 through 25
probabilities = {k: compute_probability(k) for k in range(1, 26)}
probabilities


# In[6]:


import matplotlib.pyplot as plt

# Define a function that computes the recursive sequence for different values of p
def compute_probability_for_p(k, p, memo={}):
    # Use a unique dictionary for each value of p
    key = (k, p)
    if key in memo:
        return memo[key]
    
    # Base cases
    if k == 1:
        return 1
    if k == 2:
        return p
    
    # Recursive formula for different p
    if (k-1, p) not in memo:
        memo[(k-1, p)] = compute_probability_for_p(k-1, p, memo)
    if (k-2, p) not in memo:
        memo[(k-2, p)] = compute_probability_for_p(k-2, p, memo)
    
    memo[key] = p * memo[(k-1, p)] + (1-p) * memo[(k-2, p)]
    return memo[key]

# Values of p to plot
p_values = [0.2, 0.4, 0.5, 0.6, 0.8]
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Distinct colors for each plot

# Calculate the probabilities for k from 1 to 30 for each value of p
probabilities_for_p = {p: [compute_probability_for_p(k, p) for k in range(1, 31)] for p in p_values}

# Plot the results for each value of p with distinct colors
plt.figure(figsize=(12, 6))
for i, p in enumerate(p_values):
    plt.plot(range(1, 31), probabilities_for_p[p], marker='o', color=colors[i], label=f'p={p}')

plt.title('Probability of Landing on Square N')
plt.xlabel('N')
plt.ylabel('Probability')
plt.grid(True)
plt.xticks(range(1, 31))
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




