import os
import subprocess
import argparse
import csv
from glob import glob
from tqdm import tqdm

def count_lines(file_path):
    """Count number of lines in a text file."""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def run_executable(exe_path, points_file, N, K=None):
    """
    Run the C++ executable.
    - exe_path: path to serial or parallel executable
    - points_file: input file
    - N: number of points
    - K: only for parallel executable
    Returns stdout as string
    """
    cmd = [exe_path, points_file, str(N)]
    if K is not None:
        cmd.append(str(K))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {exe_path} on {points_file}")
        print(result.stderr)
    return result.stdout

def parse_output(output):
    """
    Parse stdout to extract time and hull size.
    Only considers the last three lines:
        Number of input points: X
        Hull vertex count: Y
        CGAL convex hull time: Z ms
    """
    lines = output.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("Output has fewer than 3 lines, cannot parse.")

    # Only look at the last 3 lines
    last_three = lines[-3:]

    hull_size = None
    time_ms = None

    for line in last_three:
        if "Hull vertex count" in line:
            hull_size = int(line.split(":")[1].strip())
        elif "convex hull time" in line:
            time_ms = float(line.split(":")[1].strip().split()[0])

    if hull_size is None or time_ms is None:
        raise ValueError("Failed to parse hull size or time from output.")

    return time_ms, hull_size


def main():
    parser = argparse.ArgumentParser(description="Run serial and parallel convex hull experiments.")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Folder containing dataset folders")
    parser.add_argument("--K_values", type=int, nargs="+", required=True, help="List of K values for parallel runs")
    parser.add_argument("--count", type=int, default=1, help="Number of repetitions for averaging")
    parser.add_argument("--serial_exe", type=str, default="./cgal_convex", help="Path to serial executable")
    parser.add_argument("--parallel_exe", type=str, default="./Extream", help="Path to parallel executable")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of hulls")
    args = parser.parse_args()

    results = []
    dataset_folders = [f for f in glob(os.path.join(args.dataset_folder, "*")) if os.path.isdir(f)]
    csv_file = "results.csv"
    # Check if file already exists to determine whether to write header
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "File Path", "N", "K", "Iteration", "CPU Time", 
            "Parallel Time", "CPU Hull Size", "Parallel Hull Size"
        ])
        if write_header:
            writer.writeheader()

        for iteration in range(1, args.count + 1):
            print(f"\n=== Iteration {iteration}/{args.count} ===")
            for folder in dataset_folders:
                txt_files = glob(os.path.join(folder, "*.txt"))
                for txt_file in txt_files:

                    N = count_lines(txt_file)

                    # --- Run serial code once ---
                    serial_out = run_executable(args.serial_exe, txt_file, N)
                    serial_time, serial_hull = parse_output(serial_out)
                    K_values = args.K_values    
                    
                    if folder=="Gaussian":
                        K_values = [64, 128, 256]
                    
                    for K in K_values:
                        # --- Run parallel code ---
                        parallel_out = run_executable(args.parallel_exe, txt_file, N, K)
                        parallel_time, parallel_hull = parse_output(parallel_out)

                        # --- Hull and visualization paths ---
                        base_name = os.path.basename(txt_file).replace('.txt', '')
                        hull_file_name = f"Hulls/{base_name}_polygon_{K}.txt"
                        vis_png_path = f"Visualise/{base_name}_{K}.png"

                        os.makedirs(os.path.dirname(hull_file_name), exist_ok=True)
                        os.makedirs(os.path.dirname(vis_png_path), exist_ok=True)

                        # --- Optional visualization ---
                        if args.visualize and iteration == 1:
                            visualize_cmd = [
                                "python3", "./visualize.py",
                                txt_file,
                                hull_file_name,
                                str(N),
                                str(K)
                            ]
                            subprocess.run(visualize_cmd)

                        # --- Record row and write immediately ---
                        row = {
                            "File Path": txt_file,
                            "N": N,
                            "K": K,
                            "Iteration": iteration,
                            "CPU Time": serial_time,
                            "Parallel Time": parallel_time,
                            "CPU Hull Size": serial_hull,
                            "Parallel Hull Size": parallel_hull
                        }
                        writer.writerow(row)
                        f.flush()  # ensure it is written to disk immediately

                        print(
                            f"[Iteration {iteration} of {args.count}] "
                            f"File Path: {txt_file}, "
                            f"N: {N}, K: {K}, "
                            f"CPU Time: {serial_time} ms, "
                            f"Parallel Time: {parallel_time} ms, "
                            f"CPU Hull Size: {serial_hull}, "
                            f"Parallel Hull Size: {parallel_hull}"
                        )

if __name__ == "__main__":
    main()
