// Host-only checks of real CuTe layout algebra, compiled using CUDA headers.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;
int main() {
    auto nested = make_layout(make_shape(make_shape(Int<3>{}, Int<4>{}), Int<2>{}),
                              make_stride(make_stride(Int<1>{}, Int<3>{}), Int<40>{}));
    int errors = 0;
    for (int c = 0; c < 2; ++c)
        for (int b = 0; b < 4; ++b)
            for (int a = 0; a < 3; ++a)
                errors += int(nested(make_coord(make_coord(a,b),c))) != a + 3*b + 40*c;
    errors += int(size(nested)) != 24 || int(cosize(nested)) != 52;
    auto row = make_layout(make_shape(Int<3>{},Int<4>{}),make_stride(Int<4>{},Int<1>{}));
    auto flat = coalesce(row);
    for (int m=0;m<3;++m)
        for (int n=0;n<4;++n) {
            // CuTe's scalar logical-coordinate enumeration is first-mode-fastest.
            int logical = m + 3*n;
            errors += int(row(make_coord(m,n))) != int(flat(logical));
        }
    auto twice = make_layout(Int<16>{},Int<2>{});
    auto every_third = make_layout(Int<4>{},Int<3>{});
    auto composed = composition(twice,every_third);
    for (int i=0;i<4;++i) errors += int(composed(i)) != 6*i;
    std::printf("nested_size=%d nested_span=%d nested((2,3),1)=%d\n",
                int(size(nested)),int(cosize(nested)),int(nested(make_coord(make_coord(2,3),1))));
    std::printf("%s: 24 nested coordinates, 12 coalesced coordinates, 4 composition coordinates; errors=%d\n",
                errors ? "ALGEBRA FAIL" : "ALGEBRA PASS", errors);
    return errors ? 1 : 0;
}
