import os
import subprocess
import re

def parse_evaluation_output(output):
    """Parse the output of the evaluation script and extract AP values for easy, medium, and hard subsets."""
    patterns = {
        'easy': r"Easy\s+Val AP: ([\d\.]+)",
        'medium': r"Medium\s+Val AP: ([\d\.]+)",
        'hard': r"Hard\s+Val AP: ([\d\.]+)"
    }

    results = {}
    for subset, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[subset] = float(match.group(1))
        else:
            results[subset] = None

    return results

models_dir = "./weights/test"  # test is the file containing the predictions
subsets = ['easy', 'medium', 'hard']

# Sort model files by their file names to ensure processing in order
sorted_model_files = sorted(os.listdir(models_dir))

for model_file in sorted_model_files:
    model_path = os.path.join(models_dir, model_file)
    
    # Generate predictions using wider_test.py
    subprocess.run(["python", "wider_test.py", model_path])
    
    # Evaluate predictions using eval_tools/evaluation.py
    result = subprocess.run(["python", "eval_tools/evaluation_old.py"], capture_output=True, text=True)
    
    # Parse the results
    results = parse_evaluation_output(result.stdout)
    
    # Append results to the respective files
    for subset in subsets:
        result_file_name = f"results_{subset}.txt"
        with open(result_file_name, 'a') as f:
            f.write(f"Results for model: {model_file}\n")
            f.write(f"{subset.capitalize()} Val AP: {results[subset]}\n\n")

print("Evaluation completed!")
