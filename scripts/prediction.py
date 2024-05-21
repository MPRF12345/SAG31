import numpy as np

def sample_motion_model(n, alpha, yaw, d, x_variables, y_variables):
    def sum_matrices(n, alpha, yaw):
        # Create a vertical matrix of n variables alpha
        matrix_alpha = np.array([alpha] * n).reshape(n, 1)

        # Create a vertical matrix of n constants yaw
        matrix_yaw = np.array([yaw] * n).reshape(n, 1)

        # Calculate the sum of the two matrices
        result_matrix = matrix_alpha + matrix_yaw

        return result_matrix

    def calculate_trig_functions(matrix):
        # Calculate cosine of the matrix
        cos_matrix = np.cos(matrix)

        # Calculate sine of the matrix
        sin_matrix = np.sin(matrix)

        return cos_matrix, sin_matrix

    def multiply_with_d(cos_matrix, sin_matrix, d):
        # Create a 2x𝑛 matrix with left axis as cos_matrix and right axis as sin_matrix
        matrix = np.vstack((cos_matrix, sin_matrix))

        # Multiply with variable 'd'
        result_matrix = d * matrix

        return result_matrix

    def sum_with_variables(x_variables, y_variables, final_result):
        # Create a 2x𝑛 matrix with left axis as x_variables and right axis as y_variables
        variables_matrix = np.vstack((x_variables, y_variables))

        # Calculate the sum with the final result
        result_matrix = final_result + variables_matrix

        return result_matrix

    # Step 1: Sum of matrices alpha and yaw
    result = sum_matrices(n, alpha, yaw)

    # Step 2: Calculate cosine and sine of the result
    cos_result, sin_result = calculate_trig_functions(result)

    # Step 3: Multiply with variable 'd'
    final_result = multiply_with_d(cos_result, sin_result, d)

    # Step 4: Sum with x and y variables
    final_sum = sum_with_variables(x_variables, y_variables, final_result)

    return final_sum
