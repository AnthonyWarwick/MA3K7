#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(str(cell) for cell in row))

def is_full(matrix):
    return all(all(cell != 'X' for cell in row) for row in matrix)

def script_turn(matrix):
    empty_cells = [(i, j) for i in range(len(matrix)) for j in range(len(matrix)) if matrix[i][j] == 'X']
    if empty_cells:
        i, j = random.choice(empty_cells)
        matrix[i][j] = 1
        print(f"I filled cell ({i+1}, {j+1}) with 1")
    else:
        print("Matrix is already full!")

def user_turn(matrix):
    while True:
        try:
            row, col = map(int, input("Enter row and column numbers to fill with 0, separated by space: ").split())
            if matrix[row-1][col-1] == 'X':
                matrix[row-1][col-1] = 0
                break
            else:
                print("This cell is already filled. Please choose another cell.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid row and column numbers.")

def main():
    n = int(input("How big a matrix? "))
    matrix = [['X' for _ in range(n)] for _ in range(n)]
    
    print_matrix(matrix)
    turn = "you"  # Script starts
    
    while not is_full(matrix):
        if turn == "me":
            user_turn(matrix)
            turn = "you"
        else:
            script_turn(matrix)
            turn = "me"
        print_matrix(matrix)
    
    num_matrix = np.array(matrix, dtype=float)
    determinant = np.linalg.det(num_matrix)
    print(f"The determinant of the filled matrix is: {determinant}")

if __name__ == "__main__":
    main()


# In[ ]:


import numpy as np
import random

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(str(cell) for cell in row))

def is_full(matrix):
    return all(all(cell != 'X' for cell in row) for row in matrix)

def script_turn(matrix):
    empty_cells = [(i, j) for i in range(len(matrix)) for j in range(len(matrix)) if matrix[i][j] == 'X']
    if empty_cells:
        i, j = random.choice(empty_cells)
        matrix[i][j] = 1
        print(f"I filled cell ({i+1}, {j+1}) with 1")
    else:
        print("Matrix is already full!")

def user_turn(matrix):
    while True:
        try:
            row, col = map(int, input("Enter row and column numbers to fill with 0, separated by space: ").split())
            if matrix[row-1][col-1] == 'X':
                matrix[row-1][col-1] = 0
                break
            else:
                print("This cell is already filled. Please choose another cell.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid row and column numbers.")

def main():
    n = int(input("How big a matrix? "))
    matrix = [['X' for _ in range(n)] for _ in range(n)]
    
    print_matrix(matrix)
    turn = "me"  # User starts
    
    while not is_full(matrix):
        if turn == "me":
            user_turn(matrix)
            turn = "you"
        else:
            script_turn(matrix)
            turn = "me"
        print_matrix(matrix)
    
    num_matrix = np.array(matrix, dtype=float)
    determinant = np.linalg.det(num_matrix)
    print(f"The determinant of the filled matrix is: {determinant}")

if __name__ == "__main__":
    main()


# In[ ]:




