import subprocess
import json
import os
import argparse

def run_point_generator(exe_path, output_dir, test_case):
    """
    Run PointGenerator executable with given arguments and save output in structured folder.
    """
    name = test_case['name']
    args = test_case['args']

    # Ensure folder exists
    folder_path = os.path.join(output_dir, name)
    os.makedirs(folder_path, exist_ok=True)

    # Construct full command
    cmd = [exe_path] + list(map(str, args))

    # Execute command
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Move generated file into folder
    N = args[0]
    dist_type = args[1]
    generated_file = f"points_{dist_type}_{N}.txt"
    if os.path.exists(generated_file):
        dest_file = os.path.join(folder_path, generated_file)
        os.replace(generated_file, dest_file)
        print(f"Saved to {dest_file}")
    else:
        print(f"Warning: Expected output file {generated_file} not found.")

def main():
    parser = argparse.ArgumentParser(description="Generate point test cases from JSON.")
    parser.add_argument("--json_file",default="testcases.json", help="Path to JSON file with test case definitions")
    parser.add_argument("--exe", default="./gen_points", help="Path to PointGenerator executable")
    parser.add_argument("--out", default="Testcases", help="Output directory for test cases")
    args = parser.parse_args()

    # Load JSON
    with open(args.json_file, 'r') as f:
        testcases = json.load(f)

    # Process each test case
    for test_case in testcases['Testcases']:
        run_point_generator(args.exe, args.out, test_case)

if __name__ == "__main__":
    main()
