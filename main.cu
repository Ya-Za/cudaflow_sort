#include <thrust/host_vector.h>
#include <algorithm>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <taskflow/cudaflow.hpp>
#include <chrono>

// timing
using time_point_t = std::chrono::high_resolution_clock::time_point;

time_point_t tic() { return std::chrono::high_resolution_clock::now(); }

void toc(const time_point_t& start) {
  std::cout << "Elapsed time: ";
  std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(tic()- start).count();
  std::cout << "s\n";
}

int main() {
  constexpr int M = 5; // number of executions
  constexpr int N = 256 << 20; // number of elements
  thrust::host_vector<int> h_v(N); // host vector
  std::generate(h_v.begin(), h_v.end(), std::rand);
  thrust::device_vector<int> d_v{h_v}; // device vector
  time_point_t t;
  tf::cudaFlow cf;

  // thrust::sort
  std::cout << "--- thrust::sort ---\n";
  for (int i = 0; i < M; ++i) {
    t = tic();
    thrust::sort(d_v.begin(), d_v.end());
    toc(t);
  }

  // cudaFlow::sort
  std::cout << "--- cudaFlow::sort ---\n";
  int* p = thrust::raw_pointer_cast(d_v.data());
  for (int i = 0; i < M; ++i) {
    t = tic();
    cf.sort(p, p + N, [] __device__ (int a, int b) { return a < b; });
    cf.offload();
    toc(t);
  }

  // // assert
  // h_v = d_v;
  // if (std::is_sorted(h_v.begin(), h_v.end()))
  //   std::cout << "PASSED.\n";
  // else
  //   std::cerr << "FAILED.\n";

  return 0;
}