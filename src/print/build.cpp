// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.

/* build_print_setup(): the non-live print set-up, once per pipeline, float64, narrowed once for the kernels. */
#include <cmath>
#include <cstring>

#include "film/film.hpp"
#include "print/internal.hpp"
#include "res/resources.hpp"
#include "util/error.hpp"
#include "util/numeric.hpp"

namespace spk::print {

using params::P;

// ---- the set-up --------------------------------------------------------------------------------------

namespace {

res::Spectrum need_illuminant(const res::Resources& res, const std::string& name, const char* what) {
    auto s = res::illuminant(*res.blob, name);
    if (!s) fail_user(std::string(what) + " '" + name + "' is not a known illuminant");
    return *s;
}

void narrow(float* dst, const double* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = num::f32(src[i]);
}

}  // namespace


/* build_print_setup(in)
 *   output space: baked, with a transfer function; else SPK_ERR_USER
 *   lamp := illuminant(enlarger_illuminant); else SPK_ERR_USER
 *   neutral := (c, m, y)_filter_neutral; filters := dichroic_filters()
 *   extended_dynamic_range without edr_tone_map: SPK_ERR_USER, also on the scanned path
 *   printed path:
 *     S := paper_sensitivity; curves := paper_model_curves; upload x[K], r[K], y[K][3]
 *     probe_mid := probe_density(0.184); compensation: probe_comp := probe_density(0.184 2^ev)
 *     mean paper curve and mean base, for the levels
 *   medium := scanned ? film : paper
 *   scan := scan_tables(medium, its viewing illuminant, cmfs)
 *   glare tint := the paper's viewing illuminant XYZ, Y = 1
 *   levels: B := 0.01 / 12.92; W := ((0.98 + 0.055) / 1.055)^2.4
 *   to_rgb := XYZ->RGB(out) . CAT02(viewing white -> white of out); no product when allclose
 *   gamut: tables from the engine cache; knee and lightness at their defaults
 *   EDR: domain, toe, shoulder, target table
 *   encode: transfer mode; m_nid := XYZ->RGB . RGB->XYZ of the output space
 */
std::shared_ptr<const PrintSetup> build_print_setup(const PrintSetupInputs& in) {
    const params::ParamSet& p = in.params;
    const res::Resources& res = in.res;
    if (!in.print) fail_internal("build_print_setup: no print profile");
    const res::Profile& film = in.film.profile();

    auto impl = std::make_unique<PrintSetup::Impl>();
    PrintSetup::Impl& s = *impl;
    s.params = std::make_shared<const params::ParamSet>(p);
    s.paper = in.print;
    s.res = &res;
    s.device = in.device;
    s.scanned = p.b(P::scan_film);
    s.film_positive = film.info.is_positive();
    s.paper_positive = s.paper->info.is_positive();
    s.compensation = in.print_exposure_compensation;
    const res::Profile& paper = *s.paper;

    const std::string& out_name = p.s(P::output_color_space);
    if (!res::cctf_mode_of(out_name)) fail_user("unknown colour space '" + out_name + "' (not a baked space with a transfer function)");
    {
        auto cs = res::colour_space(*res.blob, out_name);
        if (!cs) fail_user("colour space '" + out_name + "' has no baked matrices");
        s.out_cs = *cs;
    }

    s.lamp = need_illuminant(res, p.s(P::enlarger_illuminant), "enlarger_illuminant");
    s.filters = dichroic_filters();
    s.neutral = {p.f(P::c_filter_neutral), p.f(P::m_filter_neutral), p.f(P::y_filter_neutral)};

    s.edr = p.b(P::extended_dynamic_range);
    if (s.edr && !paper.edr_tone_map)
        fail_user("extended_dynamic_range: print profile '" + paper.id + "' has no valid edr_tone_map");

    for (size_t l = 0; l < 81; ++l) {
        s.film_beta[l] = film.base_density[l];
        for (size_t c = 0; c < 3; ++c) s.film_eps[3 * l + c] = film.channel_density(l, c);
    }
    for (size_t c = 0; c < 3; ++c) s.film_white_density[c] = num::nanmax_strided(film.density_curves.v.data() + c, film.density_curves.rows, 3);

    const std::vector<double> cmfs = res.blob->f64("colour/cmfs_1931_2deg");

    if (!s.scanned) {
        s.paper_sens = paper_sensitivity(paper);
        const std::vector<double> y = paper_model_curves(paper);
        const size_t K = paper.log_exposure.size();
        s.curve_k = uint32_t(K);
        std::vector<float> tab(K * 5, 0.0f);
        for (size_t i = 0; i < K; ++i) {
            tab[i] = num::f32(paper.log_exposure[i]);
            if (i + 1 < K) {
                const double d = paper.log_exposure[i + 1] - paper.log_exposure[i];
                tab[K + i] = d == 0.0 ? 0.0f : num::f32(1.0 / d);
            }
            for (size_t c = 0; c < 3; ++c) tab[2 * K + 3 * i + c] = num::f32(y[3 * i + c]);
        }
        s.curve_gpu = in.gpu.persistent(tab.size() * sizeof(float), gpu::MemTag::Table);
        std::memcpy(s.curve_gpu.contents(), tab.data(), tab.size() * sizeof(float));

        auto srgb = res::colour_space(*res.blob, "sRGB");
        auto cat16 = res::cat_matrix(*res.blob, "CAT16");
        if (!srgb || !cat16) fail_user("the constants blob lacks sRGB or CAT16 (needed by the mid-grey probe)");
        const res::Spectrum ref = need_illuminant(res, film.info.reference_illuminant, "the film's reference_illuminant");
        const film::SpectralTable& table = in.film.spectral_table();
        ProbeInputs pin;
        pin.film = &film;
        pin.table = &table.f64;
        pin.table_size = table.size;
        pin.srgb_to_xyz = probe_matrix(*srgb, res::chromaticity(*res.blob, ref), *cat16);
        pin.gamma = p.f(P::density_curve_gamma);
        s.probe_mid = probe_density(pin, kMidGrey);
        if (s.compensation) s.probe_comp = probe_density(pin, kMidGrey * std::exp2(p.f(P::exposure_compensation_ev)));

        s.paper_mean_curve.resize(K);
        for (size_t i = 0; i < K; ++i) {
            const double row[3] = {paper.density_curves(i, 0), paper.density_curves(i, 1), paper.density_curves(i, 2)};
            s.paper_mean_curve[i] = num::nanmean(row);
        }
        s.paper_mean_base = num::nanmean(paper.base_density);
    }

    const res::Profile& medium = s.scanned ? film : paper;
    const res::Spectrum view = need_illuminant(res, medium.info.viewing_illuminant, "viewing_illuminant");
    s.scan = scan_tables(medium, view, cmfs);
    s.scan_args.mode = SPK_PRINT_MODE_SCAN;
    narrow(s.scan_args.eps, s.scan.eps.data(), 243);
    narrow(s.scan_args.beta, s.scan.beta.data(), 81);
    narrow(s.scan_args.wt, s.scan.wt.data(), 243);

    s.glare.active = p.b(P::glare_active);
    s.glare.amount = p.f(P::glare_amount);
    {
        const res::Spectrum pv =
            s.scanned ? need_illuminant(res, paper.info.viewing_illuminant, "viewing_illuminant") : view;
        const ScanTables pt = s.scanned ? scan_tables(paper, pv, cmfs) : s.scan;
        s.glare.tint_xyz = pt.xyz_illum;
    }
    s.scanner.lens_blur_px = p.f(P::scanner_lens_blur);

    s.levels.black = p.b(P::scanner_black_correction);
    s.levels.white = p.b(P::scanner_white_correction);
    s.levels.B = 0.01 / 12.92;
    s.levels.W = std::pow((0.98 + 0.055) / 1.055, 2.4);

    {
        auto cat02 = res::cat_matrix(*res.blob, "CAT02");
        if (!cat02) fail_user("the constants blob lacks colour/cat/CAT02");
        const Vec3& w = s.scan.xyz_illum;
        const double sum = (w[0] + w[1]) + w[2];
        const std::array<double, 2> xy_view = {w[0] / sum, w[1] / sum};
        Mat3 m = s.out_cs.xyz_to_rgb;
        if (auto a = res::von_kries(xy_view, s.out_cs.whitepoint, *cat02, res::GuardRef::Target)) m = res::mat3_mul(m, *a);
        narrow(s.to_rgb.m, m.data(), 9);
    }

    if (in.cam16) {
        s.cam = &in.cam16->get(res, out_name);
    } else {
        s.own_cam = build_cam16_tables(res, out_name);
        s.cam = s.own_cam.get();
    }
    {
        const Cam16Tables& t = *s.cam;
        const std::vector<double>& k = t.k;  // F_L, N_bb, N_cb, n, z, A_w, c, N_c, ...
        SpkPrintCam16Args& q = s.cam_args;
        q.rows = t.cmax_rows;
        q.cols = t.cmax_cols;
        narrow(q.m_to_xyz, t.m_to_xyz.data(), 9);
        narrow(q.m_to_rgb, t.m_to_rgb.data(), 9);
        narrow(q.m16, cam16::kM16.data(), 9);
        narrow(q.m16_inv, cam16::kM16InvPublished.data(), 9);
        q.d_rgb[0] = num::f32(k[19]);
        q.d_rgb[1] = num::f32(k[20]);
        q.d_rgb[2] = num::f32(k[21]);
        q.f_l = num::f32(k[0]);
        q.n_bb = num::f32(k[1]);
        q.n_cb = num::f32(k[2]);
        q.n = num::f32(k[3]);
        q.z = num::f32(k[4]);
        q.a_w = num::f32(k[5]);
        q.c = num::f32(k[6]);
        q.n_c = num::f32(k[7]);
        q.j_first = num::f32(k[8]);
        q.j_last = num::f32(k[9]);
        q.h_first = num::f32(k[10]);
        q.h_step = num::f32(k[11]);
        const GamutOptions g;  // the session default: cam16ucs, knee, lightness
        for (int i = 0; i < 3; ++i) {
            q.knee[i] = num::f32(g.knee[i]);
            q.lc[i] = num::f32(g.lightness_compression[i]);
        }
        q.lc_active = g.lightness_compression_active ? 1u : 0u;
        q.j_white = num::f32(k[18]);
        s.cmax_gpu = in.gpu.persistent(t.cmax_f32.size() * sizeof(float), gpu::MemTag::Table);
        std::memcpy(s.cmax_gpu.contents(), t.cmax_f32.data(), t.cmax_f32.size() * sizeof(float));
    }

    if (s.edr) {
        const res::EdrToneMap& e = *paper.edr_tone_map;
        SpkPrintEdrArgs& a = s.edr_args;
        a.n = uint32_t(e.target_log2_y.size());
        for (int c = 0; c < 3; ++c) a.r_y[c] = num::f32(s.out_cs.rgb_to_xyz[3 + c]);
        a.domain_lo = num::f32(e.domain_lo);
        a.domain_span = num::f32(e.domain_hi - e.domain_lo);
        a.toe_y = num::f32(e.toe_y);
        a.shoulder_y = num::f32(e.shoulder_y);
        std::vector<float> t(e.target_log2_y.size());
        for (size_t i = 0; i < t.size(); ++i) t[i] = num::f32(e.target_log2_y[i]);
        s.edr_gpu = in.gpu.persistent(t.size() * sizeof(float), gpu::MemTag::Table);
        std::memcpy(s.edr_gpu.contents(), t.data(), t.size() * sizeof(float));
    }

    s.cctf = p.b(P::output_cctf_encoding);
    s.enc_args.cctf_active = s.cctf ? 1u : 0u;
    s.enc_args.cctf_mode = uint32_t(s.out_cs.cctf);
    {
        const Mat3 nid = res::mat3_mul(s.out_cs.xyz_to_rgb, s.out_cs.rgb_to_xyz);
        narrow(s.enc_args.m_nid, nid.data(), 9);
    }

    auto out = std::shared_ptr<PrintSetup>(new PrintSetup());
    out->p_ = std::move(impl);
    return out;
}

}  // namespace spk::print
