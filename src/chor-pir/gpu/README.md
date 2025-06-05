
# chor-pir GPU Backend

This document provides instructions for compiling and running the GPU backend of the `chor-pir` project.

## Compilation

1. **Clone the repository** (if you haven't already):

  ```bash
  git clone <repository-url>
  cd proj/cuPIR/src/chor-pir/gpu
  ```
2. **Build the project:**

  ```bash
  make
  ```

  The compiled binaries will be located in the `build` directory.

## Running

- After compilation, run the executable as follows:

  ```bash
  ./main [r=17 b=12 q=1]
  ```

- `r`, `b`, and `q` denote the number of rows, block size per block, and number of queries (in logarithmic scale). The GPU backend leverages CUDA for parallel execution across queries. 

- The optimal kernels and single and multiple queries are gf2_vector_matrix_mult_optimized_1_uint4 and gf2_vector_matrix_mult_batch_uint4, respectively. uint4 denotes the vectorized structure of 4 uint32_t integers. 

## License

See the main project repository for license information.

