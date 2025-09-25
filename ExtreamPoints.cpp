#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cfloat>
#include <chrono>

typedef float DataType;
using namespace std;

#define TILE_SIZE 256

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ======================= Direction Generator =======================
void generate_directions(int K, vector<DataType>& dir_x, vector<DataType>& dir_y) {
    int K2 = K / 2;
    dir_x.resize(K2);
    dir_y.resize(K2);
    for (int i = 0; i < K2; i++) {
        double angle = (2.0 * M_PI * i) / K;
        dir_x[i] = cos(angle);
        dir_y[i] = sin(angle);
    }
}

// ======================= Kernel 1: Block-level reduction =======================
__global__ void block_min_max_kernel(
    const DataType* __restrict__ x,
    const DataType* __restrict__ y,
    int N,
    int K2,
    int blocks_per_dir,
    DataType* block_min_proj,
    DataType* block_max_proj
) {
    extern __shared__ DataType shared[];
    DataType* s_min = shared;           // TILE_SIZE
    DataType* s_max = &shared[blockDim.x]; // TILE_SIZE

    int tid = threadIdx.x;
    int globalBlock = blockIdx.x;
    int dir = globalBlock / blocks_per_dir;
    int dBlock= globalBlock % blocks_per_dir;
    if (dir >= K2) return;

    // DataType dx = dir_x[dir];
    // DataType dy = dir_y[dir];
    //calculate direction on the fly to save memory bandwidth
    double angle = (2.0 * M_PI * dir) / (K2*2);
    DataType dx = cos(angle);
    DataType dy = sin(angle);

    

    DataType local_min = FLT_MAX;
    DataType local_max = -FLT_MAX;

    for (int idx = tid + dBlock * blockDim.x; idx < N; idx += blocks_per_dir * blockDim.x) 
    {
        DataType proj = x[idx] * dx + y[idx] * dy;
        if (proj < local_min) local_min = proj;
        if (proj > local_max) local_max = proj;
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    // intra-block reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + offset]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_idx = globalBlock;
        block_min_proj[out_idx] = s_min[0];
        block_max_proj[out_idx] = s_max[0];
    }
}

// ======================= Kernel 2: Global reduction =======================
__global__ void global_min_max_reduce(
    const DataType* block_min_proj,
    const DataType* block_max_proj,
    int K2,
    int blocks_per_dir,
    DataType* min_proj_half,
    DataType* max_proj_half
) {
    
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= K2) return;

    DataType min_val = FLT_MAX, max_val = -FLT_MAX;
    for (int b = 0; b < blocks_per_dir; b++) {
        int idx = d * blocks_per_dir + b;
        min_val = fminf(min_val, block_min_proj[idx]);
        max_val = fmaxf(max_val, block_max_proj[idx]);
    }

    min_proj_half[d] = min_val;
    max_proj_half[d] = max_val;
}

// ======================= Kernel R: Recover Indices =======================
__global__ void recover_indices(
    const DataType* __restrict__ x,
    const DataType* __restrict__ y,
    const DataType* __restrict__ min_proj_half,
    const DataType* __restrict__ max_proj_half,
    int N,
    int K2,
    int blocks_per_dir,
    int* min_idx_half,
    int* max_idx_half
) {
    // int d = blockIdx.x * blockDim.x + threadIdx.x;
    // if (d >= K2) return;

    int globalBlock = blockIdx.x;
    int dir = globalBlock / blocks_per_dir;
    if (dir >= K2) return;
    int tid = threadIdx.x;
    int dBlock= globalBlock % blocks_per_dir;
    int start = tid + dBlock * blockDim.x;
    int stride = blockDim.x * blocks_per_dir;


    // DataType dx = dir_x[d];
    // DataType dy = dir_y[d];
    //calculate direction on the fly to save memory bandwidth
    double angle = (2.0 * M_PI * dir) / (K2*2);
    DataType dx = cos(angle);
    DataType dy = sin(angle);

    DataType target_min = min_proj_half[dir];
    DataType target_max = max_proj_half[dir];

    for (int i = start; i < N; i += stride) {
        DataType proj = x[i] * dx + y[i] * dy;
        if (fabs(proj - target_min) < 1e-6f) {
            min_idx_half[dir] = i;
            // as int is primitive type and any index will satisfy target_min will work
            // so we can directly assign without atomic operation
        }
        if (fabs(proj - target_max) < 1e-6f) {
            max_idx_half[dir] = i;
        }
    }

    // int found_min = -1;
    // int found_max = -1;

    // for (int i = 0; i < N; i++) {
    //     DataType proj = x[i] * dx + y[i] * dy;
    //     if (found_min == -1 && fabs(proj - target_min) < 1e-6f) found_min = i;
    //     if (found_max == -1 && fabs(proj - target_max) < 1e-6f) found_max = i;
    //     if (found_min != -1 && found_max != -1) break;
    // }

    // min_idx_half[d] = found_min;
    // max_idx_half[d] = found_max;
}

// ======================= Kernel 3: Reconstruct Full K =======================
__global__ void reconstruct_full_K(
    const DataType* min_proj_half,
    const DataType* max_proj_half,
    const int* min_idx_half,
    const int* max_idx_half,
    int K2,
    DataType* full_proj,
    int* full_idx
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= K2) return;

    full_proj[d] = max_proj_half[d];
    full_idx[d]  = max_idx_half[d];

    int j = d + K2;
    full_proj[j] = min_proj_half[d];
    full_idx[j]  = min_idx_half[d];
}


// ======================= Kernel 4: check Point inside Polygon =======================
__global__ void point_in_polygon_kernel(
    const DataType* __restrict__ x,
    const DataType* __restrict__ y,
    int N,
    const DataType* __restrict__ poly_x,
    const DataType* __restrict__ poly_y,
    int K,
    int blocks_per_edge,
    char* inside
) {
    int globalBlock = blockIdx.x;
    int edge = globalBlock / blocks_per_edge;   // which edge this block belongs to
    if (edge >= K) return;
    int eBlock = globalBlock % blocks_per_edge; // which block within edge

    int tid = threadIdx.x;
    int start = eBlock * blockDim.x + tid;
    int stride = blockDim.x * blocks_per_edge;

    // Edge endpoints
    DataType Ax = poly_x[edge];
    DataType Ay = poly_y[edge];
    DataType Bx = poly_x[(edge + 1) % K];
    DataType By = poly_y[(edge + 1) % K];

    // Edge direction
    DataType ex = Bx - Ax;
    DataType ey = By - Ay;

    for (int idx = start; idx < N; idx += stride) {
        if (inside[idx]==0) continue; // already marked false by some other edge

        // Vector from edge start to point
        DataType px = x[idx] - Ax;
        DataType py = y[idx] - Ay;

        // Cross product (CCW polygon: inside if cross >= 0)
        DataType cross = ex * py - ey * px;

        if (cross < (1e-6f)) { // allow small tolerance
            inside[idx] = 0; // mark outside
        }
    }
}


// ======================= MAIN =======================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return -1;
    }

    string input_file = argv[1];
    ifstream fin(input_file);
    if (!fin) {
        cerr << "Error: cannot open input file " << input_file << "\n";
        return -1;
    }

    int N, K;
    fin >> N >> K;
    vector<DataType> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) fin >> h_x[i] >> h_y[i];
    fin.close();

    hipEvent_t start, stop;
    hipEventCreate(&start) ;
     hipEventCreate(&stop) ;

   hipEventRecord(start, 0) ;
    int K2 = K / 2;
    // vector<DataType> h_dir_x, h_dir_y;
    // generate_directions(K, h_dir_x, h_dir_y);

    // GPU buffers
    DataType *d_x, *d_y;
    //  *d_dir_x, *d_dir_y;
    hipMalloc(&d_x, N * sizeof(DataType));
    hipMalloc(&d_y, N * sizeof(DataType));
    // hipMalloc(&d_dir_x, K2 * sizeof(DataType));
    // hipMalloc(&d_dir_y, K2 * sizeof(DataType));
    hipMemcpy(d_x, h_x.data(), N*sizeof(DataType), hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y.data(), N*sizeof(DataType), hipMemcpyHostToDevice);
    // hipMemcpy(d_dir_x, h_dir_x.data(), K2*sizeof(DataType), hipMemcpyHostToDevice);
    // hipMemcpy(d_dir_y, h_dir_y.data(), K2*sizeof(DataType), hipMemcpyHostToDevice);

    // block config
    int blocks_per_dir = (104*(2048/TILE_SIZE))/K2; // tuneable
    cout<< "block_per_direction: "<<blocks_per_dir<<endl;
    int numBlocks = K2 * blocks_per_dir;

    DataType *d_block_min_proj, *d_block_max_proj;
    hipMalloc(&d_block_min_proj, numBlocks*sizeof(DataType));
    hipMalloc(&d_block_max_proj, numBlocks*sizeof(DataType));

    DataType *d_min_proj_half, *d_max_proj_half;
    int *d_min_idx_half, *d_max_idx_half;
    hipMalloc(&d_min_proj_half, K2*sizeof(DataType));
    hipMalloc(&d_max_proj_half, K2*sizeof(DataType));
    hipMalloc(&d_min_idx_half, K2*sizeof(int));
    hipMalloc(&d_max_idx_half, K2*sizeof(int));

    DataType *d_full_proj;
    int *d_full_idx;
    hipMalloc(&d_full_proj, K*sizeof(DataType));
    hipMalloc(&d_full_idx, K*sizeof(int));

    // ===== Launch kernels =====
    size_t shared_mem_bytes = 2 * TILE_SIZE * sizeof(DataType);
    hipLaunchKernelGGL(block_min_max_kernel, dim3(numBlocks), dim3(TILE_SIZE), shared_mem_bytes, 0,
                       d_x, d_y, N, K2, blocks_per_dir,
                       d_block_min_proj, d_block_max_proj);

    int threads2 = 128;
    int blocks2 = (K2 + threads2 -1)/threads2;
    hipLaunchKernelGGL(global_min_max_reduce, dim3(blocks2), dim3(threads2), 0, 0,
                       d_block_min_proj, d_block_max_proj, K2, blocks_per_dir,
                       d_min_proj_half, d_max_proj_half);

    hipLaunchKernelGGL(recover_indices, dim3(numBlocks), dim3(TILE_SIZE), 0, 0,
                       d_x, d_y, d_min_proj_half, d_max_proj_half, N, K2, blocks_per_dir,
                       d_min_idx_half, d_max_idx_half);

    hipLaunchKernelGGL(reconstruct_full_K, dim3(blocks2), dim3(threads2), 0, 0,
                       d_min_proj_half, d_max_proj_half, d_min_idx_half, d_max_idx_half, K2,
                       d_full_proj, d_full_idx);

    // Copy back
    vector<int> h_full_idx(K);
    hipMemcpy(h_full_idx.data(), d_full_idx, K*sizeof(int), hipMemcpyDeviceToHost);

    // Free intermediate buffers
    // hipFree(d_dir_x); hipFree(d_dir_y);
    hipFree(d_block_min_proj); hipFree(d_block_max_proj);
    hipFree(d_min_proj_half); hipFree(d_max_proj_half);
    hipFree(d_min_idx_half); hipFree(d_max_idx_half);
    hipFree(d_full_proj); hipFree(d_full_idx);

    vector<int> unique_indices;
    for (int i = 0; i < K; i++) {
        if(i==0){
            if(h_full_idx[i] != h_full_idx[K-1])
                unique_indices.push_back(h_full_idx[i]);
        }
        else if(h_full_idx[i] != h_full_idx[i-1]) {
            
            unique_indices.push_back(h_full_idx[i]);
        }
    }
    cout << "Number of unique extreme points found: " << unique_indices.size() << "\n";

    K=unique_indices.size();
    // K unique Extream points are stored in unique_indices
    vector<DataType> h_polygon_x(K), h_polygon_y(K);
    for (int i = 0; i < K; i++) {
        h_polygon_x[i] = h_x[unique_indices[i]];
        h_polygon_y[i] = h_y[unique_indices[i]];
    }

    // Allocate device memory to mark points inside the polygon
    char *d_inside;
    hipMalloc(&d_inside, N * sizeof(char));
    vector<char> h_inside(N, 1);
    hipMemcpy(d_inside, h_inside.data(), N*sizeof(char), hipMemcpyHostToDevice);

    // allocate and copy polygon points to device  for further processing
    DataType *d_polygon_x, *d_polygon_y;

    hipMalloc(&d_polygon_x, K * sizeof(DataType));
    hipMalloc(&d_polygon_y, K * sizeof(DataType));
    hipMemcpy(d_polygon_x, h_polygon_x.data(), K*sizeof(DataType), hipMemcpyHostToDevice);
    hipMemcpy(d_polygon_y, h_polygon_y.data(), K*sizeof(DataType), hipMemcpyHostToDevice);

    

    //now using the polygon points (d_polygon_x, d_polygon_y) for further processing
    // kernel which checks if a point is inside the polygon or not and marks it in d_point_inside
    
    int blocks_per_edge = (104*2048/TILE_SIZE)/K; // tuneable
    int edgeNumBlocks = K * blocks_per_edge;

    hipLaunchKernelGGL(point_in_polygon_kernel, dim3(edgeNumBlocks), dim3(TILE_SIZE), 0, 0,
                    d_x, d_y, N,
                    d_polygon_x, d_polygon_y, K,
                    blocks_per_edge,
                    d_inside);

    // Copy result back
    hipMemcpy(h_inside.data(), d_inside, N*sizeof(char), hipMemcpyDeviceToHost);
    int outside_count = 0;
    
    hipEventRecord(stop, 0) ;
    hipEventSynchronize(stop) ;
    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time for GPU computation: " << elapsedTime << " ms\n";
    hipEventDestroy(start) ;
    hipEventDestroy(stop) ;

    // Save points outside the polygon
    ofstream out("points_outside_polygon.txt");
    if (!out) { cerr << "Cannot open points_outside_polygon.txt\n"; return -1; }
    for (int i = 0; i < N; i++) {       
        if (h_inside[i]==0) {
            out << h_x[i] << " " << h_y[i] << "\n";
            outside_count++;
        }
    }
    out.close();
    cout << outside_count<<": Points outside the polygon saved to points_outside_polygon.txt\n";









    
    // Save polygon


    ofstream fout("polygon.txt");
    if (!fout) { cerr << "Cannot open polygon.txt\n"; return -1; }
    for (int idx : unique_indices) {
        fout << h_x[idx] << " " << h_y[idx] << "\n";
    }
    fout.close();
    cout << "Polygon of K extreme points saved to polygon.txt\n";


    // Cleanup
    hipFree(d_polygon_x); hipFree(d_polygon_y);
    hipFree(d_inside);

    hipFree(d_x); hipFree(d_y);
    

    return 0;
}
