import numpy as np

# Завдання 1
# # *
# array1 = np.random.randint(0, 50, 20)
#
# # *
# print(array1)
#
# # *
# threshold_number = int(input("Enter threshold number: "))
# smaller_numbers_than_threshold = array1[array1 > threshold_number]
#
# print(len(smaller_numbers_than_threshold))
#
# # *
# max_number_in_array = array1.max()
# index_max_number = array1.argmax()
#
# print(max_number_in_array, index_max_number)
#
# # *
# descending_array = np.sort(array1)[::-1]
# print(descending_array)

# Завдання 2
# # *
# lower_limit, upper_limit = map(int, input("Enter 'lower limit' and 'upper limit' seperated by a space: ").split(" "))
#
# # *
# rng = np.random.default_rng()
# matrix_5x5 = rng.integers(lower_limit, upper_limit, size=(5, 5))
#
# # *
# print(matrix_5x5)
#
# # *
# main_diagonal = matrix_5x5.diagonal()
#
# print(main_diagonal)
#
# # *
# sum_elements_in_main_diagonal = main_diagonal.sum()
#
# print(sum_elements_in_main_diagonal)
#
# # *
# matrix_5x5 = np.tril(matrix_5x5)
#
# print(matrix_5x5)

# Завдання 3
# # *
# start, end = map(int, input("Enter 'start' and 'end' seperated by a space: ").split(" "))
#
# # *
# sequence = np.arange(start, end)
#
# # *
# matrix_6x5 = sequence.reshape((6, 5))
#
# print(matrix_6x5)
#
# # *
# sums_rows = matrix_6x5.sum(axis=1)
#
# print(sums_rows)
#
# # *
# sums_columns = matrix_6x5.max(axis=0)
#
# print(sums_columns)

# Завдання 4
# # *
# lower_limit1, upper_limit1 = map(int, input("Enter 'lower limit' and 'upper limit' seperated by a space, you can also include negative numbers: ").split(" "))
#
# # *
# array2 = np.random.randint(lower_limit1, upper_limit1, 15)
#
# # *
# print(array2)
#
# # *
# negative_numbers = array2[array2 < 0]
#
# print(negative_numbers)
#
# # *
# array_without_negatives = np.where(array2 < 0, 0, array2)
#
# print(array_without_negatives)
#
# # *
# print(np.count_nonzero(array_without_negatives == 0))

# Завдання 5
# # *
# length = int(input("Enter the length of the arrays: "))
#
# # *
# array3 = np.random.randint(0, 10, length)
# array4 = np.random.randint(10, 20, length)
#
# # *
# print(array3, array4)
#
# # *
# concatenated_array = np.concatenate((array3, array4))
#
# print(concatenated_array)
#
# # *
# print(array3 + array4, array3 - array4)

# Завдання 6
# *
matrix_dimension = input("Enter 'matrix dimension' like 'rows columns': ").split(" ")

number_of_rows = int(matrix_dimension[0])
number_of_columns = int(matrix_dimension[1])

# *
matrix = np.arange(number_of_rows * number_of_columns).reshape(number_of_rows, number_of_columns)

# *
print(matrix)

# *
new_matrix_dimension = input("Enter new 'matrix dimension' like 'rows columns': ").split(" ")

new_number_of_rows = int(new_matrix_dimension[0])
new_number_of_columns = int(new_matrix_dimension[1])

matrix = matrix.reshape(new_number_of_rows, new_number_of_columns)

# *
print(matrix)

# *
minimums = matrix.min(axis=1)
maximums = matrix.max(axis=1)

print(minimums, maximums)

# *
print(matrix.sum())