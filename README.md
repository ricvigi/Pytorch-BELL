# Blocked Ellpack Sparse Matrix Support in PyTorch

This repository contains the code, experiments, and documentation for my thesis project at **ACSAI, Sapienza University of Rome**, focused on extending PyTorch's sparse matrix capabilities.

## Thesis Overview

Modern machine learning workloads often deal with large sparse matrices, but efficient support for custom sparse formats in PyTorch is still limited. This project introduces and integrates a new sparse format — **Blocked ELLPACK (BELL)** — into the PyTorch ecosystem.

### Goal

To implement and evaluate **Blocked ELLPACK (BELL)** support in PyTorch for improved performance in sparse matrix operations, especially when the sparsity pattern exhibits regular block structure.

---

##  Key Features

- **BELL Format Support**: Blocked ELLPACK implementation in C++ using LibTorch (PyTorch C++ frontend).
- **Automatic Block Size Selection**: Heuristic that selects the optimal block size based on matrix structure.
- **Performance Analysis**: Benchmarks against existing sparse formats (COO, CSR, etc.).
- **Multithreading with OpenMP**: Optimized for parallel computation.
- **Debugging and Visualization Tools**: Tools to inspect and print tensor blocks and layouts.
