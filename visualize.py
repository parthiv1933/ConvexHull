import sys
import matplotlib.pyplot as plt
import os
import math

def load_points(points_file):
    with open(points_file, "r") as f:
        points = [tuple(map(float, line.split())) for line in f]
    return points


def load_polygon(polygon_file):
    with open(polygon_file, "r") as f:
        poly_points = [tuple(map(float, line.split())) for line in f]

    cx = sum(x for x, y in poly_points) / len(poly_points)
    cy = sum(y for x, y in poly_points) / len(poly_points)

    def angle_from_centroid(point):
        x, y = point
        return math.atan2(y - cy, x - cx)

    poly_points.sort(key=angle_from_centroid)
    return poly_points


def visualize(points, polygon, N, K, base_name, save_file="output.png"):
    xs, ys = zip(*points)
    plt.figure(figsize=(15, 15))

    # Plot all input points
    plt.scatter(xs, ys, s=0.2, color="skyblue", alpha=0.8, label="All Points")

    # Draw polygon edges + vertices
    if len(polygon) > 0:
        gx, gy = zip(*polygon)
        
        # Draw closed polygon border
        plt.plot(
            list(gx) + [gx[0]], 
            list(gy) + [gy[0]],
            color="red", 
            alpha=0.8, 
            linewidth=1.2, 
            label="Convex Hull Edges"
        )

        # Highlight polygon vertices
        plt.scatter(gx, gy, color="black", alpha=0.9, s=10, zorder=3, label="Hull Vertices")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc="upper right")
    plt.title("Convex Hull Visualization")

    # Add text info on figure
    text_str = f"Points: {N}\nHull Size: {len(polygon)}\nK: {K}\nFilename: {base_name}"
    plt.gcf().text(0.02, 0.02, text_str, fontsize=12, color="black", bbox=dict(facecolor="white", alpha=0.5))

    plt.savefig(save_file, dpi=250, bbox_inches="tight")
    print(f"Saved plot to {save_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python visualize.py points.txt polygon.txt N K")
        sys.exit(1)

    points_file = sys.argv[1]
    polygon_file = sys.argv[2]
    N = int(sys.argv[3])
    K = int(sys.argv[4])

    points = load_points(points_file)
    polygon = load_polygon(polygon_file)

    print(f"Loaded {len(points)} points (expected {N}) and {len(polygon)} polygon vertices")
    base_name = os.path.splitext(os.path.basename(points_file))[0]
    file_name = f"Visualise/{base_name}_{K}.png"

    visualize(points, polygon, N, K, base_name, save_file=file_name)
