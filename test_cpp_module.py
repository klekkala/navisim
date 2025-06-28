import os
import sys

#!/usr/bin/env python3
import navisim.math_operations as math_operations
# or alternatively:
# from navisim import math_operations

# Test simple function
result = math_operations.add(5, 3)
print(f"5 + 3 = {result}")

# Test function with STL containers
numbers = [1, 2, 3, 4, 5]
multiplied = math_operations.multiply_vector(numbers, 3)
print(f"Original: {numbers}")
print(f"Multiplied by 3: {multiplied}")

# Test class
calc = math_operations.Calculator(10.0)
print(f"Initial: {calc}")

calc.add(5)
print(f"After adding 5: {calc}")

calc.multiply(2)
print(f"After multiplying by 2: {calc}")

print(f"Final value: {calc.get_value()}")

calc.reset()
print(f"After reset: {calc}")