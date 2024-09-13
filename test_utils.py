## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    ### YOUR CODE HERE
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    ### YOUR CODE HERE
    
    result = nearest_neighbor(query_point, points)
    
    expected_index = np.argmin(np.linalg.norm(points - query_point, axis=1))
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
