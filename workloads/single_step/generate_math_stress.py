import random
import sys

def generate_expression(depth=0, max_depth=6):
    """
    Recursively generates a random, deeply nested arithmetic expression.
    Forces the CPU to branch heavily and manage a deep call stack.
    """
    # Base case: return a random float between 1.0 and 100.0
    if depth >= max_depth or random.random() < 0.15:
        return str(round(random.uniform(1.0, 100.0), 2))
    
    op = random.choice(['+', '-', '*', '/'])
    left = generate_expression(depth + 1, max_depth)
    right = generate_expression(depth + 1, max_depth)
    
    # Safely prevent direct division by zero
    if op == '/':
        try:
            if float(right) == 0.0:
                right = "1.0"
        except ValueError:
            # 'right' is a nested expression (e.g., "(1.5 + 2.0)").
            # float() fails here, which is expected. We can safely ignore this 
            # because our base numbers are >= 1.0, making an exact 0.0 evaluation highly unlikely.
            pass
        
    return f"({left} {op} {right})"

if __name__ == "__main__":
    # Generate 5,000 distinct nested expressions and chain them together
    num_expressions = 5000
    expressions = [generate_expression() for _ in range(num_expressions)]
    
    massive_string = " + ".join(expressions)
    
    # Save to a text file that the C++ tool can read from
    with open("workloads/single_step/math_stress_payload.txt", "w") as f:
        f.write(massive_string)
        
    print(f"Generated a stress payload with {len(massive_string):,} characters.")