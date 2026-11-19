// CuTe coordinate checks. No GPU kernel is launched by this program.
#include <cute/tensor.hpp>
#include <cstdio>
#include <stdexcept>
using namespace cute;
int main() {
    auto row = make_layout(make_shape(Int<3>{}, Int<4>{}),
                           make_stride(Int<4>{}, Int<1>{}));
    auto padded = make_layout(make_shape(Int<3>{}, Int<4>{}),
                              make_stride(Int<8>{}, Int<1>{}));
    int errors = 0;
    for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 4; ++n) {
            const int offset = row(make_coord(m, n));
            errors += offset != 4 * m + n;
            errors += padded(make_coord(m, n)) != 8 * m + n;
            std::printf("%2d%s", offset, n == 3 ? "\n" : " ");
        }
    }
    errors += int(size(row)) != 12 || int(cosize(row)) != 12;
    errors += int(size(padded)) != 12 || int(cosize(padded)) != 20;
    auto contiguous = make_layout(Int<12>{});
    auto divided = logical_divide(contiguous, Int<4>{});
    for (int tile = 0; tile < 3; ++tile)
        for (int within = 0; within < 4; ++within)
            errors += divided(make_coord(within, tile)) != within + 4 * tile;
    std::printf("size(padded)=%d cosize(padded)=%d\n",
                int(size(padded)), int(cosize(padded)));
    std::printf("%s: 24 matrix coordinates, 12 tile coordinates; errors=%d\n",
                errors ? "LAYOUT FAIL" : "LAYOUT PASS", errors);
    return errors ? 1 : 0;
}
