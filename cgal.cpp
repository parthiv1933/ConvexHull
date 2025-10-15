#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;

using namespace std;
using namespace std::chrono;

vector<Point> load_points(const string& filename, int N) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    vector<Point> pts;
    pts.reserve(N);

    double x, y;
    while (fin >> x >> y)
        pts.emplace_back(x, y);

    if ((int)pts.size() != N)
        cerr << "Warning: expected " << N << " points, but read " << pts.size() << endl;

    return pts;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <points_file> <N>\n";
        return 1;
    }

    string points_file = argv[1];
    int N = stoi(argv[2]);

    // --- Load points ---
    auto points = load_points(points_file, N);

    // --- Compute convex hull ---
    vector<Point> hull;
    hull.reserve(N);

    auto start = high_resolution_clock::now();
    CGAL::convex_hull_2(points.begin(), points.end(), back_inserter(hull));
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration_cast<milliseconds>(end - start).count();

    // --- Output results ---
    cout << "Number of input points: " << points.size() << endl;
    cout << "Hull vertex count: " << hull.size() << endl;
    cout << "CGAL convex hull time: " << elapsed_ms << " ms" << endl;

    return 0;
}
