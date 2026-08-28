#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "tensor/tensor.hpp"

#include <cmath>
#include <limits>

using tensor::Tensor;
using tensor::f16ToF32;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("tensor shape and strides", "[tensor]") {
    Tensor t({2, 3, 4});
    REQUIRE(t.size() == 24);
    REQUIRE(t.ndim() == 3);
    // torch row-major: last dim fastest, strides in elements are {12, 4, 1}.
    auto s = t.strides();
    REQUIRE(s[0] == 12);
    REQUIRE(s[1] == 4);
    REQUIRE(s[2] == 1);
}

TEST_CASE("tensor reshape keeps element count", "[tensor]") {
    Tensor t({2, 6});
    for (size_t i = 0; i < t.size(); ++i) t.data()[i] = static_cast<float>(i);
    Tensor r = t.reshape({3, 4});
    REQUIRE(r.shape()[0] == 3);
    REQUIRE(r.shape()[1] == 4);
    REQUIRE(r.data()[7] == 7.0f);
    REQUIRE_THROWS(t.reshape({5, 5}));
}

TEST_CASE("f16 to f32 conversion", "[tensor]") {
    SECTION("normal values") {
        REQUIRE_THAT(f16ToF32(0x3C00), WithinAbs(1.0f, 1e-6));  // 1.0
        REQUIRE_THAT(f16ToF32(0x4200), WithinAbs(3.0f, 1e-6));  // 3.0
        REQUIRE_THAT(f16ToF32(0xBC00), WithinAbs(-1.0f, 1e-6)); // -1.0
        REQUIRE_THAT(f16ToF32(0x3555), WithinAbs(0.33325195f, 1e-6));
    }
    SECTION("zero and signs") {
        REQUIRE(f16ToF32(0x0000) == 0.0f);
        REQUIRE(f16ToF32(0x8000) == -0.0f);
    }
    SECTION("infinity and nan") {
        REQUIRE(std::isinf(f16ToF32(0x7C00)));
        REQUIRE(std::isinf(f16ToF32(0xFC00)));
        REQUIRE(std::isnan(f16ToF32(0x7E00)));
    }
    SECTION("subnormal") {
        // smallest positive subnormal h = 2^-24
        REQUIRE_THAT(f16ToF32(0x0001), WithinRel(std::ldexp(1.0f, -24), 1e-6f));
        // largest subnormal = (1023/1024) * 2^-14
        REQUIRE_THAT(f16ToF32(0x03FF),
                     WithinRel(0.9990234375f * std::ldexp(1.0f, -14), 1e-6f));
    }
    SECTION("round trip through fromF16") {
        const uint16_t raw[] = {0x3C00, 0xC000, 0x4900, 0x0001};
        Tensor t = Tensor::fromF16(raw, {4});
        REQUIRE(t.size() == 4);
        REQUIRE_THAT(t.data()[0], WithinAbs(1.0f, 1e-6));
        REQUIRE_THAT(t.data()[1], WithinAbs(-2.0f, 1e-6));
        REQUIRE_THAT(t.data()[2], WithinAbs(10.0f, 1e-6));
    }
}
