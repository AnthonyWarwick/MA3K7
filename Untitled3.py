#!/usr/bin/env python
# coding: utf-8

# In[1]:


def generate_sequence(first_num1, first_num2, n):
    sequence = [first_num1, first_num2]

    for _ in range(n - 2):
        next_num = (sequence[-1] + sequence[-2]) % 10
        sequence.append(next_num)

    return sequence

# Example usage:
first_num1 = int(input("Enter the first number (less than 10): "))
first_num2 = int(input("Enter the second number (less than 10): "))
n = int(input("Enter the number of values to generate: "))

result_sequence = generate_sequence(first_num1, first_num2, n)
print("Generated sequence:", result_sequence)


# In[ ]:





# In[ ]:




