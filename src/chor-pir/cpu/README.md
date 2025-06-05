
# chor-pir CPU Backend

This document provides instructions for compiling and running the CPU backend of the `chor-pir` project.

## Compilation

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone <repository-url>
    cd proj/cuPIR/src/chor-pir/cpu
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

- `r`, `b`, and `q` denote the number of rows, block size per block, and number of queries, (in logarithmic). For more than one query (q>0), a set of kernels are used different from single query. 

## License

See the main project repository for license information.
