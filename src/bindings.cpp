#include "myHeaders.cuh"




std::tuple<torch::Tensor, BellMetadata> to_sparse_blockedell(torch::Tensor &dense)
{
  return to_sparse_blockedell_impl<float>(dense);
}

torch::Tensor to_sparse_blockedell_temp(torch::Tensor& dense)
{
  auto [tensor, meta] = to_sparse_blockedell_impl<float>(dense);
  return tensor;
}


TORCH_LIBRARY(my_sparse, m)
{
  m.def("to_sparse_blockedell(Tensor self) -> (Tensor, __torch__.my_sparse.BellMetadata)");
}

// // Register the implementation on CPU
// TORCH_LIBRARY_IMPL(my_sparse, CPU, m)
// {
//     m.impl("to_sparse_blockedell", to_sparse_blockedell);
// }

// Python binding
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     pybind11::class_<BellMetadata>(m, "BellMetadata")
//         .def_readonly("ellBlockSize", &BellMetadata::ellBlockSize)
//         .def_readonly("ellCols", &BellMetadata::ellCols)
//         .def_readonly("nnz", &BellMetadata::nnz)
//         .def_readonly("size", &BellMetadata::size)
//         .def_readonly("ellColInd", &BellMetadata::ellColInd)
//         .def_readonly("ellValue", &BellMetadata::ellValue);
//
//     m.def("to_sparse_blockedell", &to_sparse_blockedell, "Convert to BlockedELL format");
// }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("to_sparse_blockedell", &to_sparse_blockedell_temp, "Convert to BlockedELL format");
}




