#include "myHeaders.cuh"


template <typename T>
__host__ torch::Tensor to_sparse_blockedell(const torch::Tensor& dense);

TORCH_LIBRARY(my_sparse, m)
{
    m.def("to_sparse_blockedell(Tensor self) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_sparse, CPU, m)
{
    m.impl("to_sparse_blockedell", to_sparse_blockedell);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_sparse_blockedell", &to_sparse_blockedell, "Convert to BlockedELL format");
}

