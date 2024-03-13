#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# Define the function
def E(n):
    return n**2 - n

# Create an array of n values from 1 to 20
n = np.arange(1, 21)

# Calculate E[n] values
E_n = E(n)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(n, E_n, marker='o')
plt.title('E[1] vs n')
plt.xlabel('n')
plt.ylabel('E[1]')
plt.xticks(np.arange(1, 21, step=1))  # to show each integer tick on x-axis
plt.grid(True)
plt.show()


# In[4]:


# Full code for plotting the graph

import matplotlib.pyplot as plt
import numpy as np

# Define the functions
def E(n):
    return n**2 - n

def E_k(k, n):
    return -k**2 + n**2 - n + k

# Create an array of n values from 1 to 20
n = np.arange(1, 21)

# Calculate E[n] values
E_n = E(n)

# Plot E[n], E[k] for k=1,2,3,4,5, and E[19] on the same axis with correct domain for each E[k]
plt.figure(figsize=(10, 6))

# Plot E[n] as before
plt.plot(n, E_n, marker='o', label='E[n]', color='black')

# Plot E[k] for k=1,2,3,4,5 in different colors with correct domain
colors = ['blue', 'green', 'red', 'purple', 'orange']
for k in range(1, 6):
    E_k_values = E_k(k, n[k-1:])  # start from n=k
    plt.plot(n[k-1:], E_k_values, marker='o', label=f'E[{k}]', color=colors[k-1])

# Plot E[19] with correct domain (from n=19 to n=20)
E_19_values = E_k(19, n[18:])
plt.plot(n[18:], E_19_values, marker='o', label=f'E[19]', color='cyan')

# Adding labels and title
plt.title('E[k] v n for k=1,2,3,4,5,19')
plt.xlabel('n')
plt.ylabel('E[k]')
plt.xticks(np.arange(1, 21, step=1))  # to show each integer tick on x-axis
plt.grid(True)

# Adding legend
plt.legend()

# Show the plot
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

def e_k(k, n, p):
    if n < k: return 0
    q = 1 - p
    return (k-n)/(q-p) + (p*q)/((p*q - q**2)*(q - p))*((q/p)**k - (q/p)**n)

# Plot for p = 1/4 and n=1,...,10
n_values_p1 = np.arange(1, 11)
plt.figure(figsize=(12, 6))
plt.plot(n_values_p1, [e_k(1, n, 1/4) for n in n_values_p1], 'b-', label='p=0.25')
plt.title('Graph of $e_1$ against $n$ for $p=0.25$ ($n=1,...,10$)')
plt.xlabel('$n$')
plt.ylabel('$e_1$ values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for p = 3/4 and n=1,...,20
n_values_p2 = np.arange(1, 21)
plt.figure(figsize=(12, 6))
plt.plot(n_values_p2, [e_k(1, n, 3/4) for n in n_values_p2], 'g-', label='p=0.75')
plt.title('Graph of $e_1$ against $n$ for $p=0.75$ ($n=1,...,20$)')
plt.xlabel('$n$')
plt.ylabel('$e_1$ values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def e_k(k, n, p):
    if n < k: return 0
    q = 1 - p
    return (k-n)/(q-p) + (p*q)/((p*q - q**2)*(q - p))*((q/p)**k - (q/p)**n)

plt.figure(figsize=(12, 8))
n_values = np.arange(1, 11)
p = 0.25

# Plotting e_1 to e_5
for k in range(1, 6):
    plt.plot(n_values, [e_k(k, n, p) for n in n_values], label=f'e_{k}')

# Adding e_9 with a different line style
plt.plot(n_values, [e_k(9, n, p) for n in n_values], 'k--', label='e_9', linewidth=2)

plt.title('Graph of $e_k$ against $n$ for $p=0.25$')
plt.xlabel('$n$')
plt.ylabel('$e_k$ values')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.show()


# In[ ]:




