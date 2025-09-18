#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <cmath>

using namespace std;

// Typedef for coordinate type (easy to switch)
typedef float DataType;   // change to double or int if needed

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Point {
    DataType x;
    DataType y;
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <N_points> <distribution>\n";
        cerr << "Distributions: uniform, gaussian, circle\n";
        return 1;
    }

    size_t N = stoull(argv[1]);
    string dist_type = argv[2];

    // Construct filename automatically
    string filename = "points_" + dist_type + "_" + to_string(N) + ".txt";

    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: could not open file " << filename << " for writing\n";
        return 1;
    }

    // Random generator seeded with system clock
    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Distributions
    uniform_real_distribution<DataType> dist_uniform(-1000.0, 1000.0);
    normal_distribution<DataType> dist_gauss(0.0, 300.0); // mean=0, stddev=300
    uniform_real_distribution<DataType> dist_angle(0.0, static_cast<DataType>(2.0 * M_PI));
    uniform_real_distribution<DataType> dist_radius(0.0, 1000.0);

    size_t progress_step = N / 100;  // 1% step
    if (progress_step == 0) progress_step = 1;

    for (size_t i = 0; i < N; i++) {
        DataType x, y;
        if (dist_type == "uniform") {
            x = dist_uniform(rng);
            y = dist_uniform(rng);
        } else if (dist_type == "gaussian") {
            x = dist_gauss(rng);
            y = dist_gauss(rng);
        } else if (dist_type == "circle") {
            DataType angle = dist_angle(rng);
            DataType r = dist_radius(rng);
            x = r * cos(angle);
            y = r * sin(angle);
        } else {
            cerr << "Unknown distribution type: " << dist_type << "\n";
            return 1;
        }

        out << x << " " << y << "\n";

        // Progress indicator
        if (i % progress_step == 0) {
            int percent = static_cast<int>((100.0 * i) / N);
            cout << "\rProgress: " << percent << "%" << flush;
        }
    }

    cout << "\rProgress: 100%" << endl;

    out.close();
    cout << "Generated " << N << " " << dist_type 
         << " points into file " << filename << "\n";

    return 0;
}
