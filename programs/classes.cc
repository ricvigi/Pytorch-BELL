#include <iostream>
#include <functional>


int main(int argc, char** argv)
{
  int a = 1;
  std::function<int()> f = [a]() mutable -> int { int res = a*9; ++a; printf("res is: %d, a is : %d\n", res, a); return a; };
  auto ff = [&f]() -> void { printf("calling f: %d\n", f()); };
  ff();
  int b[10];
  for (int i = 0; i < 10; i++)
  {
    b[i] = i;
  }
  for (int i = 0; i < 10; i++)
  {
    printf("b[%d] is: %d\n", i, b[i]);
  }
  return 0;
}
