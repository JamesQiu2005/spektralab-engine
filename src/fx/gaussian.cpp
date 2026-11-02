// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* The separable Gaussian machinery and the camera lens blur; rules in gaussian.hpp. */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "fx/fx_internal.hpp"
#include "fx/gaussian.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::fx {

using detail::Fuse;

Branch branch(double sigma) {
    if (!(sigma > 0.0)) return Branch::Identity;
    return sigma < kIirCrossover ? Branch::Fir : Branch::Iir;
}
int fir_radius(double sigma) { return num::gaussian_radius(sigma, kFirTruncate); }
std::vector<float> fir_taps(double sigma) { return num::gaussian_kernel_f32(sigma, kFirTruncate); }

/* iir_coefficients(sigma), float64
 *   q := sigma >= 2.5 ? 0.98711 sigma - 0.96330 : 3.97156 - 4.14554 sqrt(1 - 0.26891 sigma)
 *   b0 := 1.57825 + 2.44413 q + 1.4281 q^2 + 0.422205 q^3
 *   b1 := 2.44413 q + 2.85619 q^2 + 1.26661 q^3
 *   b2 := -(1.4281 q^2 + 1.26661 q^3)
 *   b3 := 0.422205 q^3
 *   B_i := b_i / b0; B := 1 - (b1 + b2 + b3) / b0
 */
IirCoefficients iir_coefficients(double sigma) {
    double q;
    if (sigma >= 2.5) {
        q = 0.98711 * sigma - 0.96330;
    } else {
        q = 3.97156 - 4.14554 * std::sqrt(1.0 - 0.26891 * sigma);
    }
    const double q2 = q * q;
    const double q3 = q2 * q;
    const double b0 = ((1.57825 + 2.44413 * q) + 1.4281 * q2) + 0.422205 * q3;
    const double b1 = (2.44413 * q + 2.85619 * q2) + 1.26661 * q3;
    const double b2 = -(1.4281 * q2 + 1.26661 * q3);
    const double b3 = 0.422205 * q3;
    IirCoefficients c;
    c.B1 = b1 / b0;
    c.B2 = b2 / b0;
    c.B3 = b3 / b0;
    c.B = 1.0 - ((b1 + b2) + b3) / b0;
    return c;
}

std::array<MixtureComponent, 3> exponential_surrogate(const Vec3& weight, const Vec3& lambda_px) {
    static constexpr double kA[3] = {0.1633, 0.6496, 0.1870};
    static constexpr double kR[3] = {0.5360, 1.5236, 2.7684};
    std::array<MixtureComponent, 3> out{};
    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            out[size_t(i)].weight[size_t(c)] = weight[size_t(c)] * kA[i];
            out[size_t(i)].sigma[size_t(c)] = kR[i] * lambda_px[size_t(c)];
        }
    }
    return out;
}

namespace detail {

namespace {

SpkFxPointArgs point_args(const gpu::Plane& p) {
    SpkFxPointArgs a{};
    a.n = uint32_t(p.pixels());
    return a;
}

void check_same_size(const gpu::Plane& a, const gpu::Plane& b) {
    if (a.w != b.w || a.h != b.h) fail_internal("fx: plane sizes differ");
}

SpkFxFirArgs fir_args(const gpu::Plane& p, const Vec3& sigma, const bool iir[3], uint32_t& rmax) {
    SpkFxFirArgs a{};
    a.w = p.w;
    a.h = p.h;
    int r = 0;
    for (int c = 0; c < 3; ++c)
        if (!iir[c] && sigma[size_t(c)] > 0.0) r = std::max(r, fir_radius(sigma[size_t(c)]));
    a.r = uint32_t(r);
    for (int c = 0; c < 3; ++c) {
        float* t = a.taps + c * int(SPK_FX_FIR_STRIDE);
        if (iir[c] || !(sigma[size_t(c)] > 0.0)) {
            t[r] = 1.0f;  // delta
            continue;
        }
        std::vector<float> k = fir_taps(sigma[size_t(c)]);
        const int rc = int(k.size() / 2);
        for (int i = -rc; i <= rc; ++i) t[r + i] = k[size_t(i + rc)];
    }
    rmax = a.r;
    return a;
}

SpkFxIirArgs iir_args(const gpu::Plane& p, const Vec3& sigma, const bool iir[3]) {
    SpkFxIirArgs a{};
    a.w = p.w;
    a.h = p.h;
    for (int c = 0; c < 3; ++c) {
        if (!iir[c]) continue;
        a.mask |= 1u << c;
        const IirCoefficients k = iir_coefficients(sigma[size_t(c)]);
        const double v[4] = {k.B, k.B1, k.B2, k.B3};
        for (int j = 0; j < 4; ++j) {
            const float hi = num::f32(v[j]);
            a.hi[4 * c + j] = hi;
            a.lo[4 * c + j] = num::f32(v[j] - double(hi));
        }
    }
    return a;
}

void set_weight(float* dst, const Vec3& w) {
    for (int c = 0; c < 3; ++c) dst[c] = num::f32(w[size_t(c)]);
}

}  // namespace

void encode_copy(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& x, const gpu::Plane& out) {
    check_same_size(x, out);
    if (same(x, out)) return;
    SpkFxPointArgs a = point_args(x);
    bt.dispatch_1d(ctx.kernel("spk_fx_blur_copy"), a.n, {gpu::Arg::plane(x), gpu::Arg::plane(out), gpu::Arg::value(a)});
}

void encode_lincomb(const core::RenderContext& ctx, gpu::Batch& bt, const Vec3& a, const gpu::Plane& x,
                    const Vec3& b, const gpu::Plane* y, const gpu::Plane& out) {
    check_same_size(x, out);
    if (y) check_same_size(*y, out);
    SpkFxPointArgs q = point_args(x);
    q.y_zero = y ? 0u : 1u;
    set_weight(q.a, a);
    set_weight(q.b, b);
    bt.dispatch_1d(ctx.kernel("spk_fx_blur_lincomb"), q.n,
                   {gpu::Arg::plane(x), gpu::Arg::plane(y ? *y : x), gpu::Arg::plane(out), gpu::Arg::value(q)});
}

void encode_affine(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& x, const Vec3& s,
                   const Vec3& t, const gpu::Plane& out) {
    check_same_size(x, out);
    SpkFxPointArgs q = point_args(x);
    set_weight(q.a, s);
    set_weight(q.b, t);
    bt.dispatch_1d(ctx.kernel("spk_fx_blur_affine"), q.n, {gpu::Arg::plane(x), gpu::Arg::plane(out), gpu::Arg::value(q)});
}

void encode_mul(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& x, const gpu::Plane& y,
                const gpu::Plane& out) {
    check_same_size(x, out);
    check_same_size(y, out);
    SpkFxPointArgs q = point_args(x);
    bt.dispatch_1d(ctx.kernel("spk_fx_blur_mul"), q.n,
                   {gpu::Arg::plane(x), gpu::Arg::plane(y), gpu::Arg::plane(out), gpu::Arg::value(q)});
}

namespace {

void encode_fir(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& in, const gpu::Plane& dst,
                SpkFxFirArgs a, const Fuse& fuse) {
    gpu::Plane tmp = ctx.scratch(in.w, in.h);
    bt.dispatch_2d(ctx.kernel("spk_fx_blur_fir_v"), in.w, in.h,
                   {gpu::Arg::plane(in), gpu::Arg::plane(tmp), gpu::Arg::value(a)});
    a.acc_mode = fuse.mode;
    set_weight(a.weight, fuse.weight);
    const gpu::Plane& acc = (fuse.mode == SPK_FX_ACC_PLANE && fuse.acc) ? *fuse.acc : dst;
    bt.dispatch_2d(ctx.kernel("spk_fx_blur_fir_h"), in.w, in.h,
                   {gpu::Arg::plane(tmp), gpu::Arg::plane(dst), gpu::Arg::plane(acc), gpu::Arg::value(a)});
}

// IIR pair: horizontal into a scratch plane, then vertical into dst (with the optional fuse).
void encode_iir(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& in, const gpu::Plane& dst,
                SpkFxIirArgs a, const Fuse& fuse) {
    gpu::Plane tmp = ctx.scratch(in.w, in.h);
    a.acc_mode = SPK_FX_ACC_NONE;
    bt.dispatch_1d(ctx.kernel("spk_fx_blur_iir_h"), 3u * in.h, {gpu::Arg::plane(in), gpu::Arg::plane(tmp), gpu::Arg::value(a)});
    a.acc_mode = fuse.mode;
    set_weight(a.weight, fuse.weight);
    const gpu::Plane& acc = (fuse.mode == SPK_FX_ACC_PLANE && fuse.acc) ? *fuse.acc : dst;
    bt.dispatch_1d(ctx.kernel("spk_fx_blur_iir_v"), 3u * in.w,
                   {gpu::Arg::plane(tmp), gpu::Arg::plane(dst), gpu::Arg::plane(acc), gpu::Arg::value(a)});
}

}  // namespace

/* encode_blur(ctx, bt, in, dst, sigma, fuse)
 *   channels with sigma >= 3 are recursive; the rest FIR or identity
 *   all FIR:        identity with no fuse: copy; else FIR, fuse on the horizontal pass
 *   all recursive:  recursive, fuse on the vertical pass
 *   mixed + fuse:   blur into a scratch plane, then one linear combination
 *   mixed:          FIR (delta on the recursive channels), then recursive
 */
void encode_blur(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& in, const gpu::Plane& dst,
                 const Vec3& sigma, const Fuse& fuse) {
    check_same_size(in, dst);
    if (fuse.mode == SPK_FX_ACC_PLANE && !fuse.acc) fail_internal("fx: accumulation without a plane");
    bool iir[3];
    int n_iir = 0;
    for (int c = 0; c < 3; ++c) {
        iir[c] = branch(sigma[size_t(c)]) == Branch::Iir;
        n_iir += iir[c];
    }
    uint32_t rmax = 0;
    SpkFxFirArgs fa = fir_args(in, sigma, iir, rmax);
    if (n_iir == 0) {
        if (fuse.mode == SPK_FX_ACC_NONE && rmax == 0) {  // identity, no arithmetic
            encode_copy(ctx, bt, in, dst);
            return;
        }
        encode_fir(ctx, bt, in, dst, fa, fuse);
        return;
    }
    SpkFxIirArgs ia = iir_args(in, sigma, iir);
    if (n_iir == 3) {
        encode_iir(ctx, bt, in, dst, ia, fuse);
        return;
    }
    // Straddling the crossover: FIR channels first (delta on the others), then IIR channels.
    if (fuse.mode != SPK_FX_ACC_NONE) {
        // Blurred completely first, then combined by the pointwise linear combination.
        gpu::Plane t = ctx.scratch(in.w, in.h);
        encode_blur(ctx, bt, in, t, sigma, Fuse{});
        if (fuse.mode == SPK_FX_ACC_ZERO) {
            encode_lincomb(ctx, bt, fuse.weight, t, Vec3{1.0, 1.0, 1.0}, nullptr, dst);
        } else {
            encode_lincomb(ctx, bt, Vec3{1.0, 1.0, 1.0}, *fuse.acc, fuse.weight, &t, dst);
        }
        return;
    }
    if (rmax > 0) {
        gpu::Plane t = ctx.scratch(in.w, in.h);
        encode_fir(ctx, bt, in, t, fa, Fuse{});
        encode_iir(ctx, bt, t, dst, ia, Fuse{});
    } else {
        encode_iir(ctx, bt, in, dst, ia, Fuse{});
    }
}

/* encode_mixture(ctx, bt, in, out, comps)
 *   acc := out, or a scratch plane when in == out
 *   component 0: acc := 0 + w G(in)
 *   component k: acc := acc + w G(in)
 *   component 0 with all sigma 0: acc := 0 in + w in
 *   copy acc to out if they differ
 */
void encode_mixture(const core::RenderContext& ctx, gpu::Batch& bt, const gpu::Plane& in, const gpu::Plane& out,
                    std::span<const MixtureComponent> comps) {
    check_same_size(in, out);
    if (comps.empty()) fail_internal("fx: empty mixture");
    gpu::Plane acc = same(in, out) ? ctx.scratch(in.w, in.h) : out;
    for (size_t k = 0; k < comps.size(); ++k) {
        const MixtureComponent& m = comps[k];
        const bool all_identity = !(m.sigma[0] > 0.0) && !(m.sigma[1] > 0.0) && !(m.sigma[2] > 0.0);
        if (k == 0 && all_identity) {
            encode_lincomb(ctx, bt, Vec3{0.0, 0.0, 0.0}, in, m.weight, &in, acc);  // 0 x + w x
            continue;
        }
        Fuse f;
        f.mode = k == 0 ? SPK_FX_ACC_ZERO : SPK_FX_ACC_PLANE;
        f.acc = &acc;
        f.weight = m.weight;
        encode_blur(ctx, bt, in, acc, m.sigma, f);
    }
    if (!same(acc, out)) encode_copy(ctx, bt, acc, out);
}

}  // namespace detail

void gaussian(const core::RenderContext& ctx, const gpu::Plane& in, gpu::Plane& out, const Vec3& sigma) {
    if (!out) out = ctx.scratch(in.w, in.h);
    gpu::Batch bt = ctx.gpu.batch();
    detail::encode_blur(ctx, bt, in, out, sigma);
    bt.commit_and_wait();
}

void gaussian_mixture(const core::RenderContext& ctx, const gpu::Plane& in, gpu::Plane& out,
                      std::span<const MixtureComponent> components) {
    if (!out) out = ctx.scratch(in.w, in.h);
    gpu::Batch bt = ctx.gpu.batch();
    detail::encode_mixture(ctx, bt, in, out, components);
    bt.commit_and_wait();
}

void linear_combination(const core::RenderContext& ctx, const Vec3& a, const gpu::Plane& x, const Vec3& b,
                        const gpu::Plane& y, gpu::Plane& out) {
    if (!out) out = ctx.scratch(x.w, x.h);
    gpu::Batch bt = ctx.gpu.batch();
    detail::encode_lincomb(ctx, bt, a, x, b, &y, out);
    bt.commit_and_wait();
}

void affine(const core::RenderContext& ctx, const gpu::Plane& x, const Vec3& s, const Vec3& t, gpu::Plane& out) {
    if (!out) out = ctx.scratch(x.w, x.h);
    gpu::Batch bt = ctx.gpu.batch();
    detail::encode_affine(ctx, bt, x, s, t, out);
    bt.commit_and_wait();
}

void lens_blur(const core::RenderContext& ctx, gpu::Plane& exposure, double lens_blur_um) {
    if (!(ctx.pitch_um > 0.0)) fail_internal("fx: lens_blur without a pixel pitch");
    const double s = lens_blur_um / ctx.pitch_um;
    if (!(s > 0.0)) return;
    gaussian(ctx, exposure, exposure, Vec3{s, s, s});
}

/* ---- CPU copy of gaussian(): same operation order; float storage, double recursive state (tests) ---- */

namespace cpu {

namespace {

void fir_pass(const float* in, float* out, uint32_t w, uint32_t h, const Vec3& sigma, const bool iir[3],
              bool vertical) {
    int r = 0;
    for (int c = 0; c < 3; ++c)
        if (!iir[c] && sigma[size_t(c)] > 0.0) r = std::max(r, fir_radius(sigma[size_t(c)]));
    std::vector<float> taps(size_t(3 * (2 * r + 1)), 0.0f);
    for (int c = 0; c < 3; ++c) {
        float* t = taps.data() + c * (2 * r + 1);
        if (iir[c] || !(sigma[size_t(c)] > 0.0)) {
            t[r] = 1.0f;
            continue;
        }
        std::vector<float> k = fir_taps(sigma[size_t(c)]);
        const int rc = int(k.size() / 2);
        for (int i = -rc; i <= rc; ++i) t[r + i] = k[size_t(i + rc)];
    }
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                for (int i = -r; i <= r; ++i) {
                    size_t idx;
                    if (vertical) {
                        idx = size_t(num::reflect_index(int64_t(y) + i, h)) * w + x;
                    } else {
                        idx = size_t(y) * w + size_t(num::reflect_index(int64_t(x) + i, w));
                    }
                    const float p = taps[size_t(c * (2 * r + 1) + i + r)] * in[3 * idx + size_t(c)];
                    acc = acc + p;
                }
                out[3 * (size_t(y) * w + x) + size_t(c)] = acc;
            }
}

/* iir_line(in, out, n, stride, k): one line of the recursive filter
 *   forward:  w[-1] = w[-2] = w[-3] = x[0]; w[i] := B x[i] + B1 w[i-1] + B2 w[i-2] + B3 w[i-3]
 *             u[i] := float(w[i])
 *   backward: y[n] = y[n+1] = y[n+2] = u[n-1]; y[i] := B u[i] + B1 y[i+1] + B2 y[i+2] + B3 y[i+3]
 *             out[i] := float(y[i])
 */
void iir_line(const float* in, float* out, size_t n, size_t stride, const IirCoefficients& k) {
    std::vector<float> u(n);
    double w1 = in[0], w2 = w1, w3 = w1;
    for (size_t i = 0; i < n; ++i) {
        const double t = ((k.B * double(in[i * stride]) + k.B1 * w1) + k.B2 * w2) + k.B3 * w3;
        w3 = w2;
        w2 = w1;
        w1 = t;
        u[i] = float(t);
    }
    double y1 = u[n - 1], y2 = y1, y3 = y1;
    for (size_t i = n; i-- > 0;) {
        const double t = ((k.B * double(u[i]) + k.B1 * y1) + k.B2 * y2) + k.B3 * y3;
        y3 = y2;
        y2 = y1;
        y1 = t;
        out[i * stride] = float(t);
    }
}

}  // namespace

void gaussian(const float* in, float* out, uint32_t w, uint32_t h, const Vec3& sigma) {
    const size_t n = size_t(w) * h * 3;
    bool iir[3];
    int n_iir = 0;
    int rmax = 0;
    for (int c = 0; c < 3; ++c) {
        iir[c] = branch(sigma[size_t(c)]) == Branch::Iir;
        n_iir += iir[c];
        if (!iir[c] && sigma[size_t(c)] > 0.0) rmax = std::max(rmax, fir_radius(sigma[size_t(c)]));
    }
    std::vector<float> cur(in, in + n);
    if (rmax > 0) {
        std::vector<float> t(n);
        fir_pass(cur.data(), t.data(), w, h, sigma, iir, true);
        fir_pass(t.data(), cur.data(), w, h, sigma, iir, false);
    }
    for (int c = 0; c < 3 && n_iir; ++c) {
        if (!iir[c]) continue;
        const IirCoefficients k = iir_coefficients(sigma[size_t(c)]);
        for (uint32_t y = 0; y < h; ++y) iir_line(cur.data() + 3 * size_t(y) * w + c, cur.data() + 3 * size_t(y) * w + c, w, 3, k);
        for (uint32_t x = 0; x < w; ++x) iir_line(cur.data() + 3 * size_t(x) + c, cur.data() + 3 * size_t(x) + c, h, 3 * size_t(w), k);
    }
    std::memcpy(out, cur.data(), n * sizeof(float));
}

}  // namespace cpu

}  // namespace spk::fx
