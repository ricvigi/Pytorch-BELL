/*
 * ---------------------------------------------------------------------------------------
 * File        : findBellParams.cu
 * License     : MIT License (see LICENSE file)
 *
 * License Summary : THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 *                   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *                   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *                   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *                   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *                   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *                   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Description:
 * ----------------------------------------------------------------------------------------
 */

#include <torch/torch.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include "myHeaders.hpp"
#include "cudaHeaders.hpp"

int main(int argc, char** argv)
{
  cGetDeviceProp<<<1,1>>>(points_d, centroids_d, classMap_d, changes_d);
  return EXIT_SUCCESS;
}
