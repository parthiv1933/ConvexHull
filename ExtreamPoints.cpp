#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cfloat>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

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

// ======================= Kernel 5: Count Points outside Polygon =======================
__global__ void points_outside_polygon(
    const char* __restrict__ inside,
    int N,
    int* outside_indices,
    int* outside_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (inside[idx] == 0) {
        int pos = atomicAdd(outside_count, 1);
        outside_indices[pos] = idx;
    }
}

// ======================= Kernel 6: Get Coordinates of Points outside Polygon =======================
__global__ void coordinates_of_points_outside_polygon(
    const DataType* __restrict__ x,
    const DataType* __restrict__ y,
    const int* __restrict__ outside_indices,
    int outside_count,
    DataType* outside_x,
    DataType* outside_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outside_count) return;
    int point_idx = outside_indices[idx];
    outside_x[idx] = x[point_idx];
    outside_y[idx] = y[point_idx];
}

// comparator for sorting indices based on x-values, then y-values if tie
struct CompareByXY {
    const DataType* x;
    const DataType* y;
    CompareByXY(const DataType* x_ptr, const DataType* y_ptr) : x(x_ptr), y(y_ptr) {}

    __host__ __device__
    bool operator()(const int& a, const int& b) const {
        if (x[a] < x[b]) return true;
        if (x[a] > x[b]) return false;
        // tie-breaker on y
        return y[a] < y[b];
    }
};

// ======================= 2D Cross Product =======================
__device__ inline DataType cross2d(DataType ax, DataType ay, DataType bx, DataType by) {
    return ax * by - ay * bx;
}

// small epsilon for robust orientation tests
__device__ static inline float ORIENT_EPS() { return 1e-6f; }

// monotone chain that reads source px/py (length n) and writes hull into hx/hy.
// Returns hull size (m). src arrays are NOT modified.
__device__ int monotone_chain_full_src_to_dst(const DataType* __restrict__ srcx,
                                              const DataType* __restrict__ srcy,
                                              int n,
                                              DataType* __restrict__ hx,
                                              DataType* __restrict__ hy)
{
    if (n <= 1) {
        if (n == 1) { hx[0] = srcx[0]; hy[0] = srcy[0]; }
        return n;
    }

    int k = 0;
    float eps = ORIENT_EPS();

    // lower hull
    for (int i = 0; i < n; ++i) {
        DataType xi = srcx[i], yi = srcy[i];
        while (k >= 2) {
            DataType x1 = hx[k-2], y1 = hy[k-2];
            DataType x2 = hx[k-1], y2 = hy[k-1];
            DataType cr = (x2 - x1)*(yi - y1) - (y2 - y1)*(xi - x1);
            if (cr <= eps) --k; else break;
        }
        hx[k] = xi; hy[k] = yi; ++k;
    }

    // upper hull
    int t = k;
    for (int i = n - 2; i >= 0; --i) {
        DataType xi = srcx[i], yi = srcy[i];
        while (k >= t + 1) {
            DataType x1 = hx[k-2], y1 = hy[k-2];
            DataType x2 = hx[k-1], y2 = hy[k-1];
            DataType cr = (x2 - x1)*(yi - y1) - (y2 - y1)*(xi - x1);
            if (cr <= eps) --k; else break;
        }
        hx[k] = xi; hy[k] = yi; ++k;
    }

    if (k > 1) --k; // remove duplicated first
    return k;
}



// ======================= Kernel 7: Build Local Hulls (not used in main) =======================
#define GROUP_SIZE 1024

__global__ void build_local_hulls(const DataType* __restrict__ d_x,
                                  const DataType* __restrict__ d_y,
                                  int N,
                                  DataType* __restrict__ per_hull_x, // size groups*GROUP_SIZE (slot)
                                  DataType* __restrict__ per_hull_y,
                                  int* __restrict__ per_hull_size)
{
    int group_id = blockIdx.x;
    int start = group_id * GROUP_SIZE;
    if (start >= N) return;
    int end = start + GROUP_SIZE;
    if (end > N) end = N;
    int count = end - start;

    int tid = threadIdx.x;

    // allocate shared memory: srcx/srcy and hx/hy each length GROUP_SIZE
    extern __shared__ DataType s_mem[]; // size must be 4 * GROUP_SIZE * sizeof(DataType) when launching
    DataType* srcx = s_mem;                        // GROUP_SIZE
    DataType* srcy = &s_mem[GROUP_SIZE];           // GROUP_SIZE
    DataType* hx   = &s_mem[2*GROUP_SIZE];         // GROUP_SIZE
    DataType* hy   = &s_mem[3*GROUP_SIZE];         // GROUP_SIZE

    
    for (int i = tid; i < count; i += blockDim.x) {
        srcx[i] = d_x[start + i];
        srcy[i] = d_y[start + i];
    }
    __syncthreads();

    if (tid == 0) {
        if (count <= 1) {
            per_hull_size[group_id] = count;
            if (count == 1) {
                // write single point into per_hull slot
                per_hull_x[(size_t)group_id * GROUP_SIZE + 0] = srcx[0];
                per_hull_y[(size_t)group_id * GROUP_SIZE + 0] = srcy[0];
            }
        } else {
            int hull_sz = monotone_chain_full_src_to_dst(srcx, srcy, count, hx, hy);
            per_hull_size[group_id] = hull_sz;
            // write hull (compact) into global per_hull slot
            size_t base = (size_t)group_id * GROUP_SIZE;
            for (int i = 0; i < hull_sz; ++i) {
                per_hull_x[base + i] = hx[i];
                per_hull_y[base + i] = hy[i];
            }
           
        }
    }
}


// ======================= Kernel 8: Merge Hull Pairs  =======================
#define SHARED_LIMIT 2048  // max points we will stage in shared memory (tuneable)
#define BUILD_THREADS  1024// threads for building local hulls per block (<= GROUP_SIZE)
#define MERGE_THREADS  1024    // threads for merge blocks
//
__global__ void merge_pairs_kernel_opt(const DataType* __restrict__ in_x,
                                       const DataType* __restrict__ in_y,
                                       const int* __restrict__ in_offsets,
                                       const int* __restrict__ in_sizes,
                                       int num_in_hulls,
                                       DataType* __restrict__ out_buf_x,
                                       DataType* __restrict__ out_buf_y,
                                       int* __restrict__ out_offsets,
                                       int* __restrict__ out_sizes,
                                       int* __restrict__ d_out_alloc_ptr)
{
    int pair_id = blockIdx.x;
    int left_idx = pair_id * 2;
    int right_idx = left_idx + 1;
    if (left_idx >= num_in_hulls) return;

    int left_off = in_offsets[left_idx];
    int left_sz  = in_sizes[left_idx];

    int right_off = 0;
    int right_sz = 0;
    if (right_idx < num_in_hulls) {
        right_off = in_offsets[right_idx];
        right_sz  = in_sizes[right_idx];
    }

    int total = left_sz + right_sz;
    if (total == 0) {
        out_offsets[pair_id] = 0;
        out_sizes[pair_id] = 0;
        return;
    }

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Fast path: if total <= SHARED_LIMIT, stage concat into shared memory and process there
    if (total <= SHARED_LIMIT) {
    extern __shared__ DataType shared_arr[]; // size at host launch: 4*SHARED_LIMIT*sizeof(DataType)
    DataType* s_srcx = shared_arr;                        // total
    DataType* s_srcy = &shared_arr[SHARED_LIMIT];         // total
    DataType* s_hx   = &shared_arr[2*SHARED_LIMIT];       // total
    DataType* s_hy   = &shared_arr[3*SHARED_LIMIT];       // total

    // copy left
    for (int i = tid; i < left_sz; i += bdim) {
        s_srcx[i] = in_x[left_off + i];
        s_srcy[i] = in_y[left_off + i];
    }
    // copy right
    for (int i = tid; i < right_sz; i += bdim) {
        s_srcx[left_sz + i] = in_x[right_off + i];
        s_srcy[left_sz + i] = in_y[right_off + i];
    }
    __syncthreads();

    if (tid == 0) {
        // build hull from src -> hx
        int k = monotone_chain_full_src_to_dst(s_srcx, s_srcy, total, s_hx, s_hy);

        // reserve global region and copy hull out
        int out_base = atomicAdd(d_out_alloc_ptr, k);
        for (int i = 0; i < k; ++i) {
            out_buf_x[out_base + i] = s_hx[i];
            out_buf_y[out_base + i] = s_hy[i];
        }
        out_offsets[pair_id] = out_base;
        out_sizes[pair_id] = k;
    }
    return;
}
    // Slow path: total > SHARED_LIMIT -> write concatenation to global out buffer then run monotone chain  there
    // Reserve region in out buffer for 'total' entries
    int out_base_total = atomicAdd(d_out_alloc_ptr, total);

    // cooperative copy concatenated sequence into reserved out region
    for (int i = tid; i < left_sz; i += bdim) {
        out_buf_x[out_base_total + i] = in_x[left_off + i];
        out_buf_y[out_base_total + i] = in_y[left_off + i];
    }
    for (int i = tid; i < right_sz; i += bdim) {
        out_buf_x[out_base_total + left_sz + i] = in_x[right_off + i];
        out_buf_y[out_base_total + left_sz + i] = in_y[right_off + i];
    }
    __syncthreads();

    // leader computes monotone chain in-place in global memory (on out_buf arrays)
    if (tid == 0) {
    DataType* srcx = &out_buf_x[out_base_total];
    DataType* srcy = &out_buf_y[out_base_total];
    DataType* hx = &out_buf_x[out_base_total]; // destination can be same region start
    DataType* hy = &out_buf_y[out_base_total];

    int k = monotone_chain_full_src_to_dst(srcx, srcy, total, hx, hy);
    out_offsets[pair_id] = out_base_total;
    out_sizes[pair_id] = k;
}
}


// ======================= MAIN =======================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return -1;
    }

    string input_file = argv[1];
    ifstream fin(input_file);
    if (!fin) {
        cerr << "Error: cannot open input file " << input_file << "\n";
        return -1;
    }

    int N=stoi(argv[2]);
    int K=stoi(argv[3]);

    // fin >> N >> K;
    // // fin >>N;
    // N=100000000;
    // K=128;
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
    int blocks_per_dir = (104*(2048/TILE_SIZE))/K2; // tuneable here 104 is #of Cus and 2048 is max threads per Cu
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
    
    // !!! remove duplicate extream points to be done on gpu !!!
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
    // hipMemcpy(h_inside.data(), d_inside, N*sizeof(char), hipMemcpyDeviceToHost);
    // int outside_count = 0;
    
    //now find outside points on gpu
    int *id_inside;
    hipMalloc(&id_inside, N * sizeof(int));
    int *outside_count;
    hipMalloc(&outside_count, sizeof(int));
    hipMemset(outside_count, 0, sizeof(int));

    hipLaunchKernelGGL( points_outside_polygon, dim3((N+TILE_SIZE-1)/TILE_SIZE), dim3(TILE_SIZE), 0, 0,
                        d_inside, N, id_inside, outside_count);

    int h_outside_count;
    hipMemcpy(&h_outside_count, outside_count, sizeof(int), hipMemcpyDeviceToHost);
    cout << h_outside_count << ": Points outside the polygon\n";
    hipFree(d_polygon_x); hipFree(d_polygon_y);
    hipFree(d_inside);

    // Get coordinates of points outside the polygon
    DataType *d_point_outside_x, *d_point_outside_y;
    hipMalloc(&d_point_outside_x, h_outside_count * sizeof(DataType));
    hipMalloc(&d_point_outside_y, h_outside_count * sizeof(DataType));
    hipLaunchKernelGGL(coordinates_of_points_outside_polygon, dim3((h_outside_count+TILE_SIZE-1)/TILE_SIZE), dim3(TILE_SIZE), 0, 0,
                        d_x, d_y, id_inside, h_outside_count,
                        d_point_outside_x, d_point_outside_y);


    //now thrust based sorting of points outside the polygon based on x and then y coordinates
     // Allocate index array
    int* d_indices;
    hipMalloc(&d_indices, h_outside_count * sizeof(int));

    // Wrap in thrust::device_ptr
    thrust::device_ptr<int> t_indices(d_indices);
    thrust::sequence(t_indices, t_indices + h_outside_count);
    
    // Sort by x and then by y
    thrust::sort(t_indices, t_indices + h_outside_count, CompareByXY(d_point_outside_x, d_point_outside_y));

    // Allocate sorted outputs
    DataType* d_x_sorted;
    DataType* d_y_sorted;
    hipMalloc(&d_x_sorted, h_outside_count * sizeof(DataType));
    hipMalloc(&d_y_sorted, h_outside_count * sizeof(DataType));

    // Gather sorted points
    thrust::device_ptr<DataType> t_point_outside_x(d_point_outside_x);
    thrust::device_ptr<DataType> t_point_outside_y(d_point_outside_y);
    thrust::device_ptr<DataType> t_x_sorted(d_x_sorted);
    thrust::device_ptr<DataType> t_y_sorted(d_y_sorted);
    thrust::gather(t_indices, t_indices + h_outside_count, t_point_outside_x, t_x_sorted);
    thrust::gather(t_indices, t_indices + h_outside_count, t_point_outside_y, t_y_sorted);

    // Copy sorted results back to host
    // vector<DataType> h_x_outside_sorted(h_outside_count);
    // vector<DataType> h_y_outside_sorted(h_outside_count);
    // hipMemcpy(h_x_outside_sorted.data(), d_x_sorted, h_outside_count * sizeof(DataType), hipMemcpyDeviceToHost);
    // hipMemcpy(h_y_outside_sorted.data(), d_y_sorted, h_outside_count * sizeof(DataType), hipMemcpyDeviceToHost);
       hipFree(d_x); hipFree(d_y);
    int group_size = GROUP_SIZE;
    int num_groups = (h_outside_count + group_size - 1) / group_size;
    cout << "GROUP_SIZE=" << GROUP_SIZE << ", num_groups=" << num_groups << "\n";
    
    DataType *d_per_hull_x = nullptr, *d_per_hull_y = nullptr;
    int *d_per_hull_size = nullptr;
    hipMalloc(&d_per_hull_x, (size_t)num_groups * (size_t)group_size * sizeof(DataType)); 
    hipMalloc(&d_per_hull_y, (size_t)num_groups * (size_t)group_size * sizeof(DataType));
    hipMalloc(&d_per_hull_size, (size_t)num_groups * sizeof(int));

        dim3 blocks_build((size_t)num_groups);
        dim3 threads_build((size_t)std::min(BUILD_THREADS, group_size));
        cout << "Launching build_local_hulls: blocks=" << blocks_build.x << " threads=" << threads_build.x << "\n";

        size_t shared_bytes = 4 * GROUP_SIZE * sizeof(DataType);
        hipLaunchKernelGGL(build_local_hulls, blocks_build, threads_build, shared_bytes, 0,
                           d_x_sorted, d_y_sorted, h_outside_count,
                           d_per_hull_x, d_per_hull_y, d_per_hull_size);
        hipDeviceSynchronize();


     // initial in offsets for level 0: each group's slot begins at g*GROUP_SIZE
    int *d_in_offsets = nullptr;
    int *d_in_sizes = nullptr;
    hipMalloc(&d_in_offsets, (size_t)num_groups * sizeof(int)) ;
    hipMalloc(&d_in_sizes, (size_t)num_groups * sizeof(int)) ;

    // prepare offsets on host and copy
    std::vector<int> h_in_offsets(num_groups);
    for (int g = 0; g < num_groups; ++g) h_in_offsets[g] = g * group_size;
    hipMemcpy(d_in_offsets, h_in_offsets.data(), (size_t)num_groups * sizeof(int), hipMemcpyHostToDevice) ;

    // copy sizes (device->device)
    hipMemcpy(d_in_sizes, d_per_hull_size, (size_t)num_groups * sizeof(int), hipMemcpyDeviceToDevice) ;

    // allocate a big buffer for merged hull vertices (size N)
    DataType *d_buf_x = nullptr, *d_buf_y = nullptr;
    hipMalloc(&d_buf_x, (size_t)h_outside_count * sizeof(DataType)) ;
    hipMalloc(&d_buf_y, (size_t)h_outside_count * sizeof(DataType)) ;


    // prepare out offsets/sizes buffers (max needed initially ceil(num_groups/2))
    int max_hulls = num_groups;
    int max_pairs = (max_hulls + 1) / 2;
    int *d_out_offsets = nullptr, *d_out_sizes = nullptr;
    hipMalloc(&d_out_offsets, (size_t)max_pairs * sizeof(int));
    hipMalloc(&d_out_sizes, (size_t)max_pairs * sizeof(int));

    // allocate allocator counter
    int *d_out_alloc_ptr = nullptr;
    hipMalloc(&d_out_alloc_ptr, sizeof(int));

    // set "in" pointers to level-0 storage
    DataType* in_x = d_per_hull_x;
    DataType* in_y = d_per_hull_y;
    int* in_offsets = d_in_offsets;
    int* in_sizes = d_in_sizes;
    int in_num_hulls = num_groups;

    int round = 0;
     while (in_num_hulls > 1) {
        int num_pairs = (in_num_hulls + 1) / 2;
        // reset allocator to 0
        hipMemset(d_out_alloc_ptr, 0, sizeof(int));

        // ensure out_offsets/out_sizes have enough space for num_pairs
        hipFree(d_out_offsets);
        hipFree(d_out_sizes);
        hipMalloc(&d_out_offsets, (size_t)num_pairs * sizeof(int));
        hipMalloc(&d_out_sizes, (size_t)num_pairs * sizeof(int));

        std::cout << "Round " << round << ": in_num_hulls=" << in_num_hulls << " -> num_pairs=" << num_pairs << "\n";

        
       
        size_t shared_mem_bytes = 4 * SHARED_LIMIT * sizeof(DataType); // srcx,srcy,hx,hy

        // Launch (one block per pair)
        hipLaunchKernelGGL(merge_pairs_kernel_opt, dim3(num_pairs), dim3(MERGE_THREADS),
                        shared_mem_bytes, 0,
                        in_x, in_y, in_offsets, in_sizes, in_num_hulls,
                        d_buf_x, d_buf_y, d_out_offsets, d_out_sizes, d_out_alloc_ptr);
 
        hipDeviceSynchronize();

        // after merge: the merged hulls are contiguous in d_buf_x/d_buf_y at offsets given in d_out_offsets,
        // and their sizes are in d_out_sizes. The allocator value contains total points used.

        // prepare next round
        // swap: in_* := out_* ; reuse buffers
        in_x = d_buf_x;
        in_y = d_buf_y;
        in_offsets = d_out_offsets;
        in_sizes = d_out_sizes;
        in_num_hulls = num_pairs;

        // allocate fresh buffers for the next iteration's out 
        round++;
    }

    int h_final_size = 0;
    int h_final_offset = 0;
    hipMemcpy(&h_final_size, in_sizes, sizeof(int), hipMemcpyDeviceToHost) ;
    hipMemcpy(&h_final_offset, in_offsets, sizeof(int), hipMemcpyDeviceToHost);
    cout << "Final hull size: " << h_final_size << "\n";
    std::vector<DataType> h_hull_x(h_final_size), h_hull_y(h_final_size);
    if (h_final_size > 0) {
        hipMemcpy(h_hull_x.data(), &in_x[h_final_offset], (size_t)h_final_size * sizeof(DataType), hipMemcpyDeviceToHost);
        hipMemcpy(h_hull_y.data(), &in_y[h_final_offset], (size_t)h_final_size * sizeof(DataType), hipMemcpyDeviceToHost);
    }

    
    hipFree(outside_count);
    hipFree(id_inside);
    
    
 
    
    
    hipEventRecord(stop, 0) ;
    hipEventSynchronize(stop) ;
    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time for GPU computation: " << elapsedTime << " ms\n";
    hipEventDestroy(start) ;
    hipEventDestroy(stop) ;

    // write to file
    std::ofstream fout("polygon_final.txt");
    if (!fout) std::cerr << "Cannot open polygon_final.txt\n";
    else {
        // fout << h_final_size << "\n";
        for (int i = 0; i < h_final_size; ++i) fout << h_hull_x[i] << " " << h_hull_y[i] << "\n";
        fout.close();
        std::cout << "Wrote polygon_final.txt\n";
    }
   



    // //print all sorted points outside the polygon
    // ofstream out("points_outside_polygon.txt");
    // if (!out) { cerr << "Cannot open points_outside_polygon.txt\n"; return -1; }
    // for (int i = 0; i < h_outside_count; i++) {       
    //         out << h_x_outside_sorted[i] << " " << h_y_outside_sorted[i] << "\n";
    // }
    // out.close();
    // cout << h_outside_count<<": Points outside the polygon saved to points_outside_polygon.txt\n";
    // Save points outside the polygon
    // ofstream out("points_outside_polygon.txt");
    // if (!out) { cerr << "Cannot open points_outside_polygon.txt\n"; return -1; }
    // for (int i = 0; i < N; i++) {       
    //     if (h_inside[i]==0) {
    //         out << h_x[i] << " " << h_y[i] << "\n";
    //         outside_count++;
    //     }
    // }
    // out.close();
    // cout << outside_count<<": Points outside the polygon saved to points_outside_polygon.txt\n";









    
    // Save polygon


    ofstream xout("polygon.txt");
    if (!xout) { cerr << "Cannot open polygon.txt\n"; return -1; }
    for (int idx : unique_indices) {
        xout << h_x[idx] << " " << h_y[idx] << "\n";
    }
    xout.close();
    cout << "Polygon of" << K <<" uniuque extreme points saved to polygon.txt\n";


    // Clean up
    hipFree(d_point_outside_x); hipFree(d_point_outside_y);
    hipFree(d_indices);
    hipFree(d_x_sorted); hipFree(d_y_sorted);
    hipFree(d_per_hull_x); hipFree(d_per_hull_y); hipFree(d_per_hull_size);
    hipFree(d_in_offsets); hipFree(d_in_sizes);
    hipFree(d_buf_x); hipFree(d_buf_y);
    hipFree(d_out_offsets); hipFree(d_out_sizes);
    hipFree(d_out_alloc_ptr);


    
    

    return 0;
}
