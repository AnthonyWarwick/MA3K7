#!/usr/bin/env python
# coding: utf-8

# In[5]:


def hat_game_simulation(n):
    """
    Simulates the hat game where n numbers (1 through n) are initially in the hat.
    Two numbers are drawn at random, the smaller is subtracted from the larger,
    and the difference is put back in the hat until one number remains.
    
    Parameters:
    n (int): The number of initial papers in the hat, numbered 1 through n.
    
    Returns:
    int: The final number remaining in the hat.
    """
    import random
    
    # Create a list representing numbers in the hat
    numbers_in_hat = list(range(1, n + 1))
    
    # Continue the process until only one number remains
    while len(numbers_in_hat) > 1:
        # Randomly select two different numbers
        a, b = random.sample(numbers_in_hat, 2)
        
        # Remove the selected numbers from the hat
        numbers_in_hat.remove(a)
        numbers_in_hat.remove(b)
        
        # Calculate the difference and add it back to the hat
        difference = abs(a - b)
        if difference != 0:  # Only add non-zero differences
            numbers_in_hat.append(difference)
    
    # Return the last number remaining
    return numbers_in_hat[0] if numbers_in_hat else 0

# Example: Simulate the game with n numbers
n = 100  # You can change this value to simulate with different numbers of papers in the hat 
final_number = hat_game_simulation(n)
final_number


# In[ ]:


def run_hat_game_iterations(n, iterations=10000):
    """
    Runs the hat game simulation for a specified number of iterations
    and returns the frequency of the final numbers.
    
    Parameters:
    n (int): The number of initial papers in the hat, numbered 1 through n.
    iterations (int): The number of times to run the simulation.
    
    Returns:
    dict: A dictionary with final numbers as keys and their frequencies as values.
    """
    final_numbers = [hat_game_simulation(n) for _ in range(iterations)]
    final_numbers_frequency = {num: final_numbers.count(num) for num in set(final_numbers)}
    return final_numbers_frequency

# Specify the value of n for the simulations
n = 2024 # This is the number of papers you start with in the hat

# Run 100 iterations of the hat game simulation
final_numbers_frequency = run_hat_game_iterations(n, 10000)
final_numbers_frequency


# In[ ]:


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Define the exponential function for curve fitting
def simplified_exp_func(x, a, b):
    return a * np.exp(b * x)

# Preparing the data
x_data = np.array(list(final_numbers_frequency.keys()))
y_data = np.array(list(final_numbers_frequency.values()))

# Curve fitting with initial guess
params_simplified, _ = curve_fit(simplified_exp_func, x_data, y_data, p0=(1, 0.01))

# Generating points for the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = simplified_exp_func(x_fit, *params_simplified)

# Plotting with adjusted x-axis labels
x_labels = [0, 250, 500, 1000, 1250, 1500, 1750, 2000, 2024]

plt.figure(figsize=(12, 7))
plt.bar(x_data, y_data, color='skyblue', label='Original Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
plt.xlabel('Number')
plt.ylabel('Frequency')
plt.title('Frequency of Final Numbers with Fitted Exponential Curve')
plt.legend()
plt.xticks(x_labels)  # Use specified x-ticks for clarity
plt.xlim([min(x_data)-10, max(x_data)+10])  # Adjust x-limits to include all data and specified labels
plt.show()


# In[ ]:




