import sys
import matplotlib.pyplot as plt
import os

def load_points(points_file):
    with open(points_file, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split()
    N, K = int(header[0]), int(header[1])
    points = [tuple(map(float, line.split())) for line in lines[1:]]
    return points, N, K

def load_polygon(polygon_file):
    with open(polygon_file, "r") as f:
        poly_points = [tuple(map(float, line.split())) for line in f]
    return poly_points

def visualize(points, polygon, save_file="output.png"):
    xs, ys = zip(*points)
    plt.figure(figsize=(15, 15))

    # Plot points without legend (no label → won't appear in legend)
    plt.scatter(xs, ys, s=0.1, color="skyblue", alpha=0.8)

    # Draw polygon with border + vertices
    if len(polygon) > 0:
        gx, gy = zip(*polygon)
        plt.plot(list(gx) + [gx[0]], list(gy) + [gy[0]],
                 color="red", alpha=0.7, linewidth=1.5, label="Polygon Border")
        plt.scatter(gx, gy, color="black", alpha=0.8, s=5, zorder=3)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc="upper right")   # legend only for polygon
    plt.title("Extreme Points Polygon")
    
    plt.savefig(save_file, dpi=500, bbox_inches="tight")
    print(f"Saved plot to {save_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize.py points.txt polygon.txt")
        sys.exit(1)

    points_file = sys.argv[1]
    polygon_file = sys.argv[2]

    points, N, K = load_points(points_file)
    polygon = load_polygon(polygon_file)

    print(f"Loaded {N} points and {len(polygon)} polygon vertices")
    base_name = os.path.splitext(os.path.basename(points_file))[0]
    file_name = f"plot_K{K}_{base_name}.png"

    visualize(points, polygon, save_file=file_name)
