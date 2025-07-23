#include "myHeaders.cuh"


int
main(int argc, char** argv)
{
  std::string dtype = argv[5];
  if (dtype == "float")
  {
    return run<float>(argc, argv);
  } else if (dtype == "double")
  {
    return run<double>(argc, argv);
  } else if (dtype == "int8_t")
  {
    return run_int<int8_t>(argc, argv);
  } else if (dtype == "int")
  {
    return run_int<int>(argc, argv);
  } else
  {
    std::cerr << "Unsupported dtype: " << dtype << std::endl;
    return EXIT_UNSUPPORTED;
  }
}





