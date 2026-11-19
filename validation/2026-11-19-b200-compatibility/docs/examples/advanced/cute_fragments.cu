// Host-only ownership checks against CUTLASS v4.3.6 MMA traits.
// This program never calls an MMA instruction and launches no GPU kernel.
#include <cute/atom/mma_traits_sm75.hpp>
#include <array>
#include <cstdio>

int main() {
    using Operation = cute::SM75_16x8x8_F32F16F16F32_TN;
    using Traits = cute::MMA_Traits<Operation>;
    typename Traits::ALayout a_layout;
    typename Traits::BLayout b_layout;
    typename Traits::CLayout c_layout;
    std::array<int, 128> a_coverage{}, c_coverage{};
    std::array<int, 64> b_coverage{};
    int errors = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const int group = lane / 4, within = lane % 4;
        for (int value = 0; value < 4; ++value) {
            const int row = group + 8 * (value / 2);
            const int column = 2 * within + value % 2;
            const int expected = row + 16 * column;
            const int a = a_layout(cute::make_coord(lane, value));
            const int c = c_layout(cute::make_coord(lane, value));
            errors += a != expected;
            errors += c != expected;
            if (a < 0 || a >= int(a_coverage.size())) ++errors;
            else ++a_coverage[a];
            if (c < 0 || c >= int(c_coverage.size())) ++errors;
            else ++c_coverage[c];
            if (lane == 10)
                std::printf("lane=10 value=%d C=(%d,%d) trait_offset=%d\n",
                            value, row, column, c);
        }
        for (int value = 0; value < 2; ++value) {
            // The B trait encodes (N,K), while mathematical B is B[k,n].
            const int n = group, k = 2 * within + value;
            const int expected = n + 8 * k;
            const int b = b_layout(cute::make_coord(lane, value));
            errors += b != expected;
            if (b < 0 || b >= int(b_coverage.size())) ++errors;
            else ++b_coverage[b];
        }
    }
    for (int count : a_coverage) errors += count != 1;
    for (int count : b_coverage) errors += count != 1;
    for (int count : c_coverage) errors += count != 1;
    std::printf("%s: C_pairs=128 A_pairs=128 B_pairs=64 errors=%d\n",
                errors ? "FRAGMENT FAIL" : "FRAGMENT PASS", errors);
    return errors ? 1 : 0;
}
