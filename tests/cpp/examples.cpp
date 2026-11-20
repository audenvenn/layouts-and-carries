// Host-only check of the examples quoted in the paper against the C++ CuTe headers.
// Build (no CUDA toolchain needed; headers only):
//   clang++ -std=c++17 -isystem <cuda>/cuda_cudart/include -isystem <cuda>/cuda_crt/include \
//           -isystem <cuda>/cccl/include/cccl -I <cutlass>/include tests/cpp/examples.cpp
#include <iostream>
#include <cute/layout.hpp>
using namespace cute;

template <class LA, class LB>
void check(const char* what, LA const& A, LB const& B) {
  auto R = composition(A, B);
  std::cout << what << "\n  A = " << A << ",  B = " << B << "\n  composition(A,B) = " << R << "\n  its values :";
  for (int i = 0; i < int(size(R)); ++i) std::cout << " " << R(i);
  std::cout << "\n  A(B(i))    :";
  bool same = int(size(R)) == int(size(B));
  for (int i = 0; i < int(size(B)); ++i) { std::cout << " " << A(B(i)); same = same && (i < int(size(R))) && R(i) == A(B(i)); }
  std::cout << "\n  " << (same ? "AGREE" : "DIFFER") << "\n";
}

int main() {
  // 1. reshape a row-major 4x3 matrix as a column-major 3x4 view (single-mode stride check)
  check("row-major 4x3 viewed as 3x4",
        make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<3>{}, Int<1>{})),
        make_layout(make_shape(Int<3>{}, Int<4>{})));
  // 2. every 4th element of a padded 5x2 column-major matrix
  check("padded 5x2, stride-4 tile",
        make_layout(make_shape(Int<5>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{})),
        make_layout(Int<4>{}, Int<4>{}));
  // 3. first four elements of a padded 3x4 matrix viewed as 2x2 (interference between modes)
  check("padded 3x4, first 4 elements as 2x2",
        make_layout(make_shape(Int<3>{}, Int<4>{}), make_stride(Int<1>{}, Int<5>{})),
        make_layout(make_shape(Int<2>{}, Int<2>{})));
  // 4. Shah (2024)
  check("Shah 2024",
        make_layout(make_shape(Int<6>{}, Int<2>{}), make_stride(Int<1>{}, Int<7>{})),
        make_layout(make_shape(Int<3>{}, Int<2>{}), make_stride(Int<2>{}, Int<3>{})));
  // 5. three modes: pairwise fine, jointly carrying
  check("three modes",
        make_layout(make_shape(Int<12>{}, Int<2>{}), make_stride(Int<1>{}, Int<13>{})),
        make_layout(make_shape(Int<2>{}, Int<2>{}, Int<2>{}), make_stride(Int<3>{}, Int<4>{}, Int<6>{})));
  // 6. the documentation example (should agree)
  check("documentation example",
        make_layout(make_shape(Int<6>{}, Int<2>{}), make_stride(Int<8>{}, Int<2>{})),
        make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<3>{}, Int<1>{})));
  return 0;
}
