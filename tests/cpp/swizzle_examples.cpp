// The swizzle examples quoted in the paper (Section 8 and Table 5).  Host-only build:
//   clang++ -std=c++17 -w -isystem refs/cuda-headers/cuda_cudart/include \
//     -isystem refs/cuda-headers/cuda_crt/include -isystem refs/cuda-headers/cccl/include/cccl \
//     -I refs/cutlass/include tests/cpp/swizzle_examples.cpp
#include <cstdint>
#include <iostream>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/layout_composed.hpp>
using namespace cute;

template <class T> long val(T const& t) { return long(int(t)); }

// slice_and_offset(coord, L): print the returned layout + offset, its values, and the truth
template <class L, class Coord, class Truth>
void slice_case(const char* name, L const& layout, Coord const& coord, Truth truth) {
  auto so = slice_and_offset(coord, layout);
  auto sl = get<0>(so);
  long off = val(get<1>(so));
  std::cout << name << ": slice = " << sl << " + " << off << "\n  CuTe : ";
  for (int k = 0; k < int(size(sl)); ++k) std::cout << val(sl(k)) + off << " ";
  std::cout << "\n  truth: ";
  for (int k = 0; k < int(size(sl)); ++k) std::cout << truth(k) << " ";
  std::cout << "\n";
}

int main() {
  // Example 8.x: a slice that decays to a layout with a negative stride (correct)
  {
    auto f = make_layout(make_shape(Int<4>{}, Int<2>{}), make_stride(Int<1>{}, Int<4>{}));
    auto L = composition(Swizzle<1,1,1>{}, f);
    std::cout << "L = " << L << " : ";
    for (int k = 0; k < 8; ++k) std::cout << val(L(k)) << " ";
    std::cout << "\n";
    slice_case("row 1 of L", L, make_coord(_, Int<1>{}), [&](int k) { return val(L(make_coord(k, 1))); });
  }
  // slicing with an inner layout that is not bit-disjoint: modes share a bit
  {
    auto f = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<1>{}, Int<4>{}));
    auto L = composition(Swizzle<1,1,1>{}, f);
    slice_case("Sw<1,1,1> o (2,2,2):(1,1,4) at (1,_,_)", L, make_coord(Int<1>{}, _, _),
               [&](int k) { return val(L(make_coord(1, k % 2, k / 2))); });
  }
  // ... and an injective one with a stride that is not a power of two
  {
    auto f = make_layout(make_shape(Int<2>{}, Int<4>{}), make_stride(Int<1>{}, Int<6>{}));
    auto L = composition(Swizzle<2,1,2>{}, f);
    slice_case("Sw<2,1,2> o (2,4):(1,6) at (1,_), dynamic", L, make_coord(1, _),
               [&](int k) { return val(L(make_coord(1, k))); });
    slice_case("Sw<2,1,2> o (2,4):(1,6) at (1,_), static ", L, make_coord(Int<1>{}, _),
               [&](int k) { return val(L(make_coord(1, k))); });
  }
  // composition(Layout, Swizzle): the layout reverses the bits of one field only
  {
    auto a = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<4>{}), make_stride(Int<8>{}, Int<4>{}, Int<1>{}));
    Swizzle<2,0,2> sw;
    auto r = composition(a, sw);
    std::cout << "(2,2,4):(8,4,1) o Sw<2,0,2> = " << r << "\n  CuTe : ";
    for (int i = 0; i < 16; ++i) std::cout << val(r(i)) << " ";
    std::cout << "\n  truth: ";
    for (int i = 0; i < 16; ++i) std::cout << val(a(sw(i))) << " ";
    std::cout << "\n";
  }
  // right_inverse of a swizzled compact layout
  {
    auto f = make_layout(make_shape(Int<4>{}, Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<8>{}, Int<4>{}));
    auto L = composition(Swizzle<2,0,2>{}, f);
    auto R = right_inverse(L);
    std::cout << "right_inverse(Sw<2,0,2> o (4,2,2):(1,8,4)) = " << R << "\n  L(R(k)): ";
    for (int k = 0; k < 16; ++k) std::cout << val(L(val(R(k)))) << " ";
    std::cout << "\n";
  }
  // cosize
  {
    auto L = composition(Swizzle<1,0,1>{}, make_layout(Int<2>{}, Int<2>{}));
    std::cout << "Sw<1,0,1> o 2:2 takes the values " << val(L(0)) << ", " << val(L(1))
              << "; cosize = " << val(cosize(L)) << "\n";
  }
  // a swizzled layout constructed with an offset that overlaps the fixed coordinate (bit-disjoint inner layout)
  {
    auto f = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));
    auto L = composition(Swizzle<1,0,1>{}, Int<1>{}, f);
    slice_case("Sw<1,0,1> o 1 o (2,2):(1,2) at (1,_)", L, make_coord(Int<1>{}, _),
               [&](int k) { return val(L(make_coord(1, k))); });
  }
  // cosize of a compact layout whose size is not a power of two
  {
    auto L = composition(Swizzle<1,0,1>{}, make_layout(Int<3>{}, Int<1>{}));
    std::cout << "Sw<1,0,1> o 3:1 takes the values " << val(L(0)) << ", " << val(L(1)) << ", " << val(L(2))
              << "; cosize = " << val(cosize(L)) << "\n";
  }
  // upcast of a swizzle with negative shift drops a pair whose source bit lies below the cast
  {
    Swizzle<1,0,-1> s;
    auto u = upcast<2>(s);
    std::cout << "upcast<2>(Sw<1,0,-1>) = " << u << ": x = 1 gives floor(s(x)/2) = " << (s(1) >> 1)
              << " but u(floor(x/2)) = " << u(0) << "\n";
  }
  // the masks are 32-bit signed integers: a field at bit 31 is sign-extended against a 64-bit offset
  {
    Swizzle<1,30,1> s;
    uint64_t x = uint64_t{1} << 32;
    std::cout << std::hex << "Sw<1,30,1> on x = 0x" << x << ": s(x) = 0x" << uint64_t(s(x)) << ", s(s(x)) = 0x"
              << uint64_t(s(s(x))) << ", composition(s,s)(x) = 0x" << uint64_t(composition(s, s)(x)) << std::dec << "\n";
  }
  // the 128-byte swizzle is an involution of a 1024-byte block and a Latin square on chunks
  {
    Swizzle<3,4,3> sw;
    bool ok = true;
    for (int x = 0; x < 1024; ++x) ok = ok && (sw(sw(x)) == x) && (sw(x) / 128 == x / 128);
    for (int z = 0; z < 8; ++z) { int seen = 0; for (int y = 0; y < 8; ++y) seen |= 1 << ((sw(y * 128 + z * 16) / 16) % 8); ok = ok && seen == 255; }
    std::cout << "Swizzle<3,4,3>: involution, row-preserving, Latin square on chunks: " << (ok ? "yes" : "NO") << "\n";
  }
  return 0;
}
