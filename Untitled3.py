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


# In[7]:


def find_repetition_period(first_num1, first_num2):
    sequence = [first_num1, first_num2]
    seen_patterns = {(first_num1, first_num2): 0}

    for i in range(1, 100):  # Limit to prevent infinite loops
        next_num = (sequence[-1] + sequence[-2]) % 10
        sequence.append(next_num)

        current_pattern = (sequence[-2], sequence[-1])

        if current_pattern in seen_patterns:
            return i

        seen_patterns[current_pattern] = i

    return None  # No repetition found within a reasonable limit

# Example usage:
input_first_num1 = int(input("Enter the first number (less than 10): "))
input_first_num2 = int(input("Enter the second number (less than 10): "))

result_period = find_repetition_period(input_first_num1, input_first_num2)
if result_period is not None:
    print(f"The period of the sequence is: {result_period}")
else:
    print("No repetition found within the limit.")


# In[ ]:





# In[ ]:




