#include "myHeaders.cuh"


int
main(int argc, char **argv)
{
  std::string dtype = argv[5];
  std::cout << dtype << std::endl;
  if (dtype == "float")
  {
    return run<float>(argc, argv);
  } else if (dtype == "double")
  {
    return run<double>(argc, argv);
  } else if (dtype == "int")
  {
    std::cout << "running run_int<int>" << std::endl;
    return run_int<int>(argc, argv);
  } else
  {
    std::cerr << "Unsupported dtype: " << dtype << std::endl;
    return EXIT_UNSUPPORTED;
  }
}





