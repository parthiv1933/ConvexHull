/* usage examples:
# Uniform in [-100, 100]
./gen_points 1000 uniform -100 100

# Gaussian with mean=0, stddev=50 will mostly be in [-3*stddev, 3*stddev]
./gen_points 1000 gaussian 0 50

# Inside circle of radius 200
./gen_points 1000 circle 200

# Only boundary of circle radius 200
./gen_points 1000 circle 200 boundary

*/


#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <cmath>

using namespace std;

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
        cerr << "Usage: " << argv[0] << " <N_points> <distribution> [params...]\n";
        cerr << "Distributions:\n";
        cerr << "  uniform <min> <max>\n";
        cerr << "  gaussian <mean> <stddev>\n";
        cerr << "  circle <radius> [boundary]\n";
        return 1;
    }

    size_t N = stoull(argv[1]);
    string dist_type = argv[2];

    // Construct filename 
    string filename = "points_" + dist_type + "_" + to_string(N) + ".txt";

    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: could not open file " << filename << " for writing\n";
        return 1;
    }

    // Random generator seeded with system clock
    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Declare distributions
    uniform_real_distribution<DataType> dist_uniform(0.0, 1.0); // dummy init
    normal_distribution<DataType> dist_gauss(0.0, 1.0);
    uniform_real_distribution<DataType> dist_angle(0.0, static_cast<DataType>(2.0 * M_PI));
    uniform_real_distribution<DataType> dist_radius(0.0, 1.0);

    // Circle config
    bool circle_boundary_only = false;
    DataType circle_radius = 1.0;

    // Configure distributions based on type
    if (dist_type == "uniform") {
        if (argc < 5) {
            cerr << "Usage: " << argv[0] << " N uniform <min> <max>\n";
            return 1;
        }
        DataType min_val = stof(argv[3]);
        DataType max_val = stof(argv[4]);
        dist_uniform = uniform_real_distribution<DataType>(min_val, max_val);
    } 
    else if (dist_type == "gaussian") {
        if (argc < 5) {
            cerr << "Usage: " << argv[0] << " N gaussian <mean> <stddev>\n";
            return 1;
        }
        DataType mean = stof(argv[3]);
        DataType stddev = stof(argv[4]);
        dist_gauss = normal_distribution<DataType>(mean, stddev);
    } 
    else if (dist_type == "circle") {
        if (argc < 4) {
            cerr << "Usage: " << argv[0] << " N circle <radius> [boundary]\n";
            return 1;
        }
        circle_radius = stof(argv[3]);
        dist_radius = uniform_real_distribution<DataType>(0.0, circle_radius);

        if (argc >= 5 && string(argv[4]) == "boundary") {
            circle_boundary_only = true;
        }
    } 
    else {
        cerr << "Unknown distribution type: " << dist_type << "\n";
        return 1;
    }

    size_t progress_step = N / 100;  // 1% step
    if (progress_step == 0) progress_step = 1;

    for (size_t i = 0; i < N; i++) {
        DataType x, y;
        if (dist_type == "uniform") {
            x = dist_uniform(rng);
            y = dist_uniform(rng);
        } 
        else if (dist_type == "gaussian") {
            x = dist_gauss(rng);
            y = dist_gauss(rng);
        } 
        else if (dist_type == "circle") {
            DataType angle = dist_angle(rng);
            DataType r = circle_boundary_only ? circle_radius : dist_radius(rng);
            x = r * cos(angle);
            y = r * sin(angle);
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
         << " points into file " << filename 
         << (circle_boundary_only ? " (boundary only)" : "") 
         << "\n";

    return 0;
}
