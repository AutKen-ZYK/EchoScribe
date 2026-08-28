#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ops/ops.hpp"

#include <cmath>
#include <random>
#include <vector>

using Catch::Approx;
using ops::attention;
using ops::conv1d;
using ops::KVCache;
using ops::layerNorm;
using ops::linear;
using ops::matmul;
using ops::multiHeadAttention;
using ops::softmax;

namespace {

std::vector<float> randomVec(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// double-precision reference matmul
std::vector<float> refMatmul(const std::vector<float>& a, const std::vector<float>& b, size_t m,
                             size_t k, size_t n) {
    std::vector<float> c(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p) {
            double av = a[i * k + p];
            for (size_t j = 0; j < n; ++j) c[i * n + j] += static_cast<float>(av * b[p * n + j]);
        }
    return c;
}

} // namespace

TEST_CASE("matmul matches reference", "[ops]") {
    const size_t m = 5, k = 7, n = 6;
    auto a = randomVec(m * k, -1, 1, 1);
    auto b = randomVec(k * n, -1, 1, 2);
    std::vector<float> c(m * n);
    matmul(a.data(), b.data(), c.data(), m, k, n);
    auto ref = refMatmul(a, b, m, k, n);
    for (size_t i = 0; i < c.size(); ++i) REQUIRE(c[i] == Approx(ref[i]).margin(1e-4));

    SECTION("identity") {
        std::vector<float> eye(k * k, 0.0f);
        for (size_t i = 0; i < k; ++i) eye[i * k + i] = 1.0f;
        std::vector<float> out(m * k);
        matmul(a.data(), eye.data(), out.data(), m, k, k);
        for (size_t i = 0; i < a.size(); ++i) REQUIRE(out[i] == Approx(a[i]).margin(1e-6));
    }
}

TEST_CASE("linear matches torch semantics y = xW^T + b", "[ops]") {
    const size_t m = 4, k = 3, n = 5;
    auto x = randomVec(m * k, -1, 1, 3);
    auto w = randomVec(n * k, -1, 1, 4); // [n, k]
    auto bias = randomVec(n, -1, 1, 5);

    std::vector<float> y(m * n);
    linear(x.data(), w.data(), bias.data(), y.data(), m, k, n);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double acc = bias[j];
            for (size_t p = 0; p < k; ++p) acc += x[i * k + p] * w[j * k + p];
            REQUIRE(y[i * n + j] == Approx(static_cast<float>(acc)).margin(1e-4));
        }

    SECTION("null bias means zero") {
        std::vector<float> y2(m * n);
        linear(x.data(), w.data(), nullptr, y2.data(), m, k, n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                double acc = 0;
                for (size_t p = 0; p < k; ++p) acc += x[i * k + p] * w[j * k + p];
                REQUIRE(y2[i * n + j] == Approx(static_cast<float>(acc)).margin(1e-4));
            }
    }
}

TEST_CASE("layerNorm normalizes rows to zero mean unit variance", "[ops]") {
    const size_t rows = 6, n = 16;
    auto x = randomVec(rows * n, -3, 3, 6);
    auto gamma = randomVec(n, 0.5f, 1.5f, 7);
    auto beta = randomVec(n, -0.5f, 0.5f, 8);

    std::vector<float> out(rows * n);
    layerNorm(x.data(), gamma.data(), beta.data(), out.data(), rows, n);

    for (size_t i = 0; i < rows; ++i) {
        // mean of gamma*xhat + beta is not exactly 0, but check per-row normalization:
        // (y - beta)/gamma must have mean 0, var 1 (up to eps)
        double m2 = 0, v2 = 0;
        for (size_t j = 0; j < n; ++j) {
            const double xhat = (out[i * n + j] - beta[j]) / gamma[j];
            m2 += xhat;
        }
        m2 /= n;
        for (size_t j = 0; j < n; ++j) {
            const double xhat = (out[i * n + j] - beta[j]) / gamma[j];
            v2 += (xhat - m2) * (xhat - m2);
        }
        v2 /= n;
        REQUIRE(m2 == Approx(0.0).margin(1e-4));
        REQUIRE(v2 == Approx(n / (n + 1e-5)).margin(1e-3));
    }

    SECTION("null gamma/beta") {
        std::vector<float> out2(rows * n);
        layerNorm(x.data(), nullptr, nullptr, out2.data(), rows, n);
        for (size_t i = 0; i < rows; ++i) {
            double mean = 0;
            for (size_t j = 0; j < n; ++j) mean += out2[i * n + j];
            mean /= n;
            REQUIRE(mean == Approx(0.0).margin(1e-4));
            double var = 0;
            for (size_t j = 0; j < n; ++j) var += (out2[i * n + j] - mean) * (out2[i * n + j] - mean);
            var /= n;
            REQUIRE(var == Approx(1.0).margin(1e-3));
        }
    }
}

TEST_CASE("gelu matches tanh approximation reference points", "[ops]") {
    std::vector<float> x = {0.0f, 1.0f, -1.0f, 2.0f, -3.5f, 10.0f};
    std::vector<float> y(x.size());
    ops::gelu(x.data(), y.data(), x.size());

    auto ref = [](double v) {
        return 0.5 * v * (1 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
    };
    REQUIRE(y[0] == Approx(0.0).margin(1e-7));
    REQUIRE(y[1] == Approx(ref(1.0)).margin(1e-6));
    REQUIRE(y[2] == Approx(ref(-1.0)).margin(1e-6));
    REQUIRE(y[3] == Approx(ref(2.0)).margin(1e-6));
    REQUIRE(y[4] == Approx(ref(-3.5)).margin(1e-6));
    // saturates to identity for large positive, 0 for large negative
    REQUIRE(y[5] == Approx(10.0).margin(1e-4));
}

TEST_CASE("softmax rows sum to one and are shift invariant", "[ops]") {
    const size_t rows = 4, n = 9;
    auto x = randomVec(rows * n, -10, 10, 9);
    softmax(x.data(), rows, n);
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0;
        for (size_t j = 0; j < n; ++j) {
            REQUIRE(x[i * n + j] >= 0.0f);
            sum += x[i * n + j];
        }
        REQUIRE(sum == Approx(1.0).margin(1e-5));
    }

    // shift invariance: adding a constant to all elements of a row changes nothing
    auto x2 = randomVec(rows * n, -10, 10, 10);
    auto x3 = x2;
    for (size_t j = 0; j < n; ++j) x3[j] += 100.0f;
    softmax(x2.data(), 1, n);
    softmax(x3.data(), 1, n);
    for (size_t j = 0; j < n; ++j) REQUIRE(x2[j] == Approx(x3[j]).margin(1e-6));
}

TEST_CASE("conv1d matches direct convolution reference", "[ops]") {
    const size_t cIn = 3, cOut = 4, k = 3, t = 11, pad = 1;
    auto x = randomVec(cIn * t, -1, 1, 11);
    auto w = randomVec(cOut * cIn * k, -1, 1, 12);
    auto b = randomVec(cOut, -1, 1, 13);

    std::vector<float> y(cOut * t);
    conv1d(x.data(), w.data(), b.data(), y.data(), cIn, cOut, k, t, pad);

    for (size_t o = 0; o < cOut; ++o)
        for (size_t ti = 0; ti < t; ++ti) {
            double acc = b[o];
            for (size_t c = 0; c < cIn; ++c)
                for (size_t kk = 0; kk < k; ++kk) {
                    const long long src = static_cast<long long>(ti) + static_cast<long long>(kk) -
                                          static_cast<long long>(pad);
                    if (src < 0 || src >= static_cast<long long>(t)) continue;
                    acc += x[c * t + src] * w[o * (cIn * k) + c * k + kk];
                }
            REQUIRE(y[o * t + ti] == Approx(static_cast<float>(acc)).margin(1e-4));
        }
}

TEST_CASE("attention: hand-computed single head", "[ops]") {
    // Q=[1,2] K=[1,1] V=[1,2], dk=1: scores=[1,1;2,2], softmax rows uniform,
    // out = [1.5, 1.5]. causal: row0 -> V0 = 1, row1 -> [1.5, 1.5].
    const float q[2] = {1.0f, 2.0f};
    const float k[2] = {1.0f, 1.0f};
    const float v[2] = {1.0f, 2.0f};
    float out[2] = {0, 0};
    attention(q, k, v, out, 2, 2, 1, 1, false);
    REQUIRE(out[0] == Approx(1.5).margin(1e-5));
    REQUIRE(out[1] == Approx(1.5).margin(1e-5));

    float outC[2] = {0, 0};
    attention(q, k, v, outC, 2, 2, 1, 1, true);
    REQUIRE(outC[0] == Approx(1.0).margin(1e-5));
    REQUIRE(outC[1] == Approx(1.5).margin(1e-5));
}

TEST_CASE("multiHeadAttention equals per-head attention", "[ops]") {
    const size_t tq = 3, tk = 4, nHead = 2, hd = 3;
    auto q = randomVec(tq * nHead * hd, -1, 1, 14);
    auto k = randomVec(tk * nHead * hd, -1, 1, 15);
    auto v = randomVec(tk * nHead * hd, -1, 1, 16);

    std::vector<float> out(tq * nHead * hd, 0.0f);
    multiHeadAttention(q.data(), k.data(), v.data(), out.data(), tq, tk, nHead, hd, false);

    for (size_t h = 0; h < nHead; ++h) {
        std::vector<float> qh(tq * hd), kh(tk * hd), vh(tk * hd), oh(tq * hd);
        for (size_t i = 0; i < tq; ++i)
            for (size_t d = 0; d < hd; ++d) qh[i * hd + d] = q[i * nHead * hd + h * hd + d];
        for (size_t i = 0; i < tk; ++i)
            for (size_t d = 0; d < hd; ++d) {
                kh[i * hd + d] = k[i * nHead * hd + h * hd + d];
                vh[i * hd + d] = v[i * nHead * hd + h * hd + d];
            }
        attention(qh.data(), kh.data(), vh.data(), oh.data(), tq, tk, hd, hd, false);
        for (size_t i = 0; i < tq; ++i)
            for (size_t d = 0; d < hd; ++d)
                REQUIRE(out[i * nHead * hd + h * hd + d] == Approx(oh[i * hd + d]).margin(1e-5));
    }
}

TEST_CASE("kvcache incremental decoding equals batch computation", "[ops]") {
    const size_t nHead = 2, hd = 4, nTokens = 6;
    const size_t dModel = nHead * hd;
    auto q = randomVec(nTokens * dModel, -1, 1, 17);
    auto k = randomVec(nTokens * dModel, -1, 1, 18);
    auto v = randomVec(nTokens * dModel, -1, 1, 19);

    // reference: full causal attention over all tokens at once
    std::vector<float> refOut(nTokens * dModel);
    multiHeadAttention(q.data(), k.data(), v.data(), refOut.data(), nTokens, nTokens, nHead, hd,
                       true);

    // incremental: feed one token at a time
    KVCache cache(nHead, hd, 64);
    for (size_t t = 0; t < nTokens; ++t) {
        std::vector<float> out(dModel);
        cache.forward(q.data() + t * dModel, k.data() + t * dModel, v.data() + t * dModel,
                      out.data(), 1, 1);
        for (size_t d = 0; d < dModel; ++d) {
            REQUIRE(out[d] == Approx(refOut[t * dModel + d]).margin(1e-4));
        }
        REQUIRE(cache.length() == t + 1);
    }
}
