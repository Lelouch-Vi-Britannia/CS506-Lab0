import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    # Define the vectors inside the test function
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    # Calculate cosine similarity using the function from utils
    result = cosine_similarity(vector1, vector2)
    
    # Manually compute the expected result for verification
    expected_result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    # Compare result with the expected value using np.isclose for floating point comparison
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    # Define a set of points and a query point for the nearest neighbor test
    points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    query_point = np.array([5.1, 6.1])
    
    # Calculate the nearest neighbor using the function from utils
    result = nearest_neighbor(query_point, points)
    
    # Manually determine the index of the nearest neighbor
    distances = np.linalg.norm(points - query_point, axis=1)
    expected_index = np.argmin(distances)
    
    # Assert that the function returns the correct index
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
