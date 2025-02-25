#include <iostream>

struct myClass
{
public:
  void set_x(int const val) noexcept { x = val; }
  void set_y(int const val) noexcept { y = val; }
  int x = 42;
  int y;
};

struct myOtherClass: myClass
{
  int z = 111;
};
int main(int argc, char** argv)
{
  myOtherClass x;
  x.y = 11;
  printf("x.x is %d, x.y is %d, z is %d\n", x.x, x.y, x.z);
  x.set_x(12);
  x.set_y(13);
  printf("x.x is %d, x.y is %d, z is %d \n", x.x, x.y, x.z);
  return 0;
}
