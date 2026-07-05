#include "subsampling.hpp"
#include "backend.hpp"
#include "ggml.h"
#include "ggml_graph.hpp"
#include <cassert>
#include <cstring>
#include <vector>

namespace pk {

// Weights from the GGUF (loader context) are referenced DIRECTLY as graph
// leaves via the shared pk::clone_weight (backend.cpp) — they live in a CPU
// backend buffer (zero-copy). The conv kernels stay F32 (the converter never
// quantizes them); only out.weight is allowlisted and may be f16/q8_0, fed into
// ggml_mul_mat which dequantizes src0. GGUF ne is reverse of the torch shape ==
// ggml's [KW,KH,IC,OC] layout.

Subsampling::Subsampling(const ModelLoader &ml) : ml_(ml) {
  conv_channels_ = (int)ml.config().subsampling_conv_channels;
  d_model_ = (int)ml.config().d_model;
  causal_ = ml.config().causal_downsampling;
}

int Subsampling::valid_out_len(int T, int in_valid_frames) const {
  const int all_paddings = causal_ ? 3 : 2;
  int valid = (in_valid_frames >= 0) ? in_valid_frames : (T - 1);
  int stages = (ml_.config().subsampling_factor == 4) ? 2 : 3;
  for (int st = 0; st < stages; ++st)
    valid = (valid + all_paddings - 3) / 2 + 1;
  return valid;
}

ggml_tensor *
Subsampling::build_graph_batched(ggml_context *ctx, const float *mel,
                                 int n_mels, int T, int B, GraphInputPool &pool,
                                 int &out_Tp, std::vector<int> &out_valid,
                                 const std::vector<int> &valid_in) const {
  const int C = conv_channels_;
  const int F = n_mels; // feature dim (80)
  const ModelLoader &ml = ml_;
  const bool causal = causal_;
  const int factor = ml.config().subsampling_factor;

  std::vector<float> &x_host = pool.alloc_f32((size_t)B * T * F);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t)
      for (int f = 0; f < F; ++f)
        x_host[((size_t)b * T + t) * F + f] =
            mel[((size_t)b * n_mels + f) * T + t];

  int64_t x_ne[4] = {F, T, 1, B};
  ggml_tensor *x =
      pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, x_ne, x_host.data(),
                             x_host.size() * sizeof(float));

  auto pad_causal = [&](ggml_tensor *t) -> ggml_tensor * {
    return ggml_pad_ext(ctx, t, /*lp0*/ 2, /*rp0*/ 1, /*lp1*/ 2, /*rp1*/ 1, 0,
                        0, 0, 0);
  };

  auto mask_time = [&](ggml_tensor *t,
                       const std::vector<int> &vt) -> ggml_tensor * {
    const int H = (int)t->ne[1];
    const int Bx = (int)t->ne[3];
    bool any = false;
    for (int b = 0; b < Bx; ++b) {
      int v = (b < (int)vt.size()) ? vt[b] : H;
      if (v < H) {
        any = true;
        break;
      }
    }
    if (!any)
      return t;
    std::vector<float> &md = pool.alloc_f32((size_t)Bx * H);
    for (int b = 0; b < Bx; ++b) {
      int v = (b < (int)vt.size()) ? vt[b] : H;
      for (int h = 0; h < H; ++h)
        md[(size_t)b * H + h] = (h < v) ? 1.0f : 0.0f;
    }
    int64_t m_ne[4] = {1, H, 1, Bx};
    ggml_tensor *tm = pk::graph_input_tensor(
        ctx, GGML_TYPE_F32, 4, m_ne, md.data(), md.size() * sizeof(float));
    return ggml_mul(ctx, t, tm);
  };

  const int all_paddings = causal_ ? 3 : 2;

  if (factor == 4) {
    std::vector<int> vt_stage0(B), vt_stage1(B);
    for (int b = 0; b < B; ++b) {
      int vi = (b < (int)valid_in.size()) ? valid_in[b] : -1;
      int v0 = (vi >= 0) ? vi : (T - 1);
      int v1 = (v0 + all_paddings - 3) / 2 + 1;
      vt_stage0[b] = v0;
      vt_stage1[b] = v1;
    }

    // Conv 0 (Standard Conv2d)
    ggml_tensor *w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor *b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    x = mask_time(x, vt_stage0);
    if (causal) {
      x = pad_causal(x);
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
    } else {
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1, /*p1*/ 1,
                       /*d0*/ 1, /*d1*/ 1);
    }
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // Conv 2 (Standard Conv2d)
    ggml_tensor *w2 = clone_weight(ctx, ml, "encoder.pre_encode.conv.2.weight");
    ggml_tensor *b2 = clone_weight(ctx, ml, "encoder.pre_encode.conv.2.bias");
    x = mask_time(x, vt_stage1);
    if (causal) {
      x = pad_causal(x);
      x = ggml_conv_2d(ctx, w2, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
    } else {
      x = ggml_conv_2d(ctx, w2, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1, /*p1*/ 1,
                       /*d0*/ 1, /*d1*/ 1);
    }
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b2, 1, 1, C, 1));
    x = ggml_relu(ctx, x);
  } else {
    std::vector<int> vt_stage0(B), vt_stage1(B), vt_stage2(B);
    for (int b = 0; b < B; ++b) {
      int vi = (b < (int)valid_in.size()) ? valid_in[b] : -1;
      int v0 = (vi >= 0) ? vi : (T - 1);
      int v1 = (v0 + all_paddings - 3) / 2 + 1;
      int v2 = (v1 + all_paddings - 3) / 2 + 1;
      vt_stage0[b] = v0;
      vt_stage1[b] = v1;
      vt_stage2[b] = v2;
    }

    // Conv 0 (Standard Conv2d)
    ggml_tensor *w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor *b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    x = mask_time(x, vt_stage0);
    if (causal) {
      x = pad_causal(x);
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
    } else {
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1, /*p1*/ 1,
                       /*d0*/ 1, /*d1*/ 1);
    }
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // Stages 2 & 3 (Depthwise + Pointwise)
    struct StageW {
      const char *dw_w;
      const char *dw_b;
      const char *pw_w;
      const char *pw_b;
    };
    const StageW stages[2] = {
        {"encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
         "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias"},
        {"encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
         "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias"},
    };
    const std::vector<int> *stage_valid_t[2] = {&vt_stage1, &vt_stage2};
    for (int si = 0; si < 2; ++si) {
      const StageW &s = stages[si];
      ggml_tensor *dww = clone_weight(ctx, ml, s.dw_w);
      // Fix for custom exported models where depthwise weights are [KW, KH, C, 1] instead of [KW, KH, 1, C]
      if (dww->ne[2] > 1 && dww->ne[3] == 1) {
          dww = ggml_reshape_4d(ctx, dww, dww->ne[0], dww->ne[1], 1, dww->ne[2]);
      }
      ggml_tensor *dwb = clone_weight(ctx, ml, s.dw_b);
      x = mask_time(x, *stage_valid_t[si]);
      if (causal) {
        x = pad_causal(x);
        x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0,
                                   /*p1*/ 0, /*d0*/ 1, /*d1*/ 1);
      } else {
        x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1,
                                   /*p1*/ 1, /*d0*/ 1, /*d1*/ 1);
      }
      x = ggml_cont(ctx, x);
      x = ggml_add(ctx, x, ggml_reshape_4d(ctx, dwb, 1, 1, C, 1));

      ggml_tensor *pww = clone_weight(ctx, ml, s.pw_w);
      ggml_tensor *pwb = clone_weight(ctx, ml, s.pw_b);
      x = ggml_conv_2d(ctx, pww, x, /*s0*/ 1, /*s1*/ 1, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
      x = ggml_add(ctx, x, ggml_reshape_4d(ctx, pwb, 1, 1, C, 1));
      x = ggml_relu(ctx, x);
    }
  }

  const int Fp = (int)x->ne[0]; // F'
  const int Tp = (int)x->ne[1]; // T'
  ggml_tensor *xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
  ggml_tensor *flat = ggml_reshape_3d(ctx, xp, (int64_t)C * Fp, Tp, B);

  out_valid.assign(B, 0);
  bool any_masked = false;
  for (int b = 0; b < B; ++b) {
    int vi = (b < (int)valid_in.size()) ? valid_in[b] : -1;
    int vo = valid_out_len(T, vi);
    out_valid[b] = (vo > Tp) ? Tp : vo;
    if (vo < Tp)
      any_masked = true;
  }
  if (any_masked) {
    std::vector<float> &outmask = pool.alloc_f32((size_t)B * Tp);
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < Tp; ++t)
        outmask[(size_t)b * Tp + t] = (t < out_valid[b]) ? 1.0f : 0.0f;
    }
    int64_t mk_ne[3] = {1, Tp, B};
    ggml_tensor *mask =
        pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, mk_ne, outmask.data(),
                               outmask.size() * sizeof(float));
    flat = ggml_mul(ctx, flat, mask);
  }

  ggml_tensor *ow = clone_weight(ctx, ml, "encoder.pre_encode.out.weight");
  ggml_tensor *ob = clone_weight(ctx, ml, "encoder.pre_encode.out.bias");
  ggml_tensor *y = ggml_mul_mat(ctx, ow, flat);
  y = ggml_add(ctx, y, ob);

  out_Tp = Tp;
  return y;
}

ggml_tensor *Subsampling::build_graph(ggml_context *ctx,
                                      const std::vector<float> &mel, int n_mels,
                                      int T, GraphInputPool &pool, int &out_Tp,
                                      int &out_valid,
                                      int in_valid_frames) const {
  const int C = conv_channels_;
  const int F = n_mels; // feature dim (80)
  const ModelLoader &ml = ml_;
  const bool causal = causal_;
  const int factor = ml.config().subsampling_factor;

  std::vector<float> &x_host = pool.alloc_f32((size_t)F * T);
  for (int t = 0; t < T; ++t)
    for (int f = 0; f < F; ++f)
      x_host[(size_t)t * F + f] = mel[(size_t)f * T + t];

  int64_t x_ne[4] = {F, T, 1, 1};
  ggml_tensor *x =
      pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, x_ne, x_host.data(),
                             x_host.size() * sizeof(float));

  auto pad_causal = [&](ggml_tensor *t) -> ggml_tensor * {
    return ggml_pad_ext(ctx, t, /*lp0*/ 2, /*rp0*/ 1, /*lp1*/ 2, /*rp1*/ 1, 0,
                        0, 0, 0);
  };

  auto mask_time = [&](ggml_tensor *t, int valid_t) -> ggml_tensor * {
    const int H = (int)t->ne[1];
    if (valid_t >= H)
      return t;
    std::vector<float> &md = pool.alloc_f32(H);
    for (int h = 0; h < H; ++h)
      md[h] = (h < valid_t) ? 1.0f : 0.0f;
    int64_t m_ne[4] = {1, H, 1, 1};
    ggml_tensor *tm = pk::graph_input_tensor(
        ctx, GGML_TYPE_F32, 4, m_ne, md.data(), md.size() * sizeof(float));
    return ggml_mul(ctx, t, tm);
  };

  const int all_paddings = causal_ ? 3 : 2;

  if (factor == 4) {
    int valid_t0 = (in_valid_frames >= 0) ? in_valid_frames : (T - 1);
    int valid_t1 = (valid_t0 + all_paddings - 3) / 2 + 1;

    // Conv 0 (Standard Conv2d)
    ggml_tensor *w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor *b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    if (causal) {
      x = mask_time(x, valid_t0);
      x = pad_causal(x);
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
    } else {
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1, /*p1*/ 1,
                       /*d0*/ 1, /*d1*/ 1);
    }
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // Conv 2 (Standard Conv2d)
    ggml_tensor *w2 = clone_weight(ctx, ml, "encoder.pre_encode.conv.2.weight");
    ggml_tensor *b2 = clone_weight(ctx, ml, "encoder.pre_encode.conv.2.bias");
    if (causal) {
      x = mask_time(x, valid_t1);
      x = pad_causal(x);
      x = ggml_conv_2d(ctx, w2, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
    } else {
      x = ggml_conv_2d(ctx, w2, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1, /*p1*/ 1,
                       /*d0*/ 1, /*d1*/ 1);
    }
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b2, 1, 1, C, 1));
    x = ggml_relu(ctx, x);
  } else {
    int valid_t0 = (in_valid_frames >= 0) ? in_valid_frames : (T - 1);
    int valid_t1 = (valid_t0 + all_paddings - 3) / 2 + 1;
    int valid_t2 = (valid_t1 + all_paddings - 3) / 2 + 1;

    // Conv 0 (Standard Conv2d)
    ggml_tensor *w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
    ggml_tensor *b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
    if (causal) {
      x = mask_time(x, valid_t0);
      x = pad_causal(x);
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
    } else {
      x = ggml_conv_2d(ctx, w0, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1, /*p1*/ 1,
                       /*d0*/ 1, /*d1*/ 1);
    }
    x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
    x = ggml_relu(ctx, x);

    // Stages 2 & 3 (Depthwise + Pointwise)
    struct StageW {
      const char *dw_w;
      const char *dw_b;
      const char *pw_w;
      const char *pw_b;
    };
    const StageW stages[2] = {
        {"encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
         "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias"},
        {"encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
         "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias"},
    };
    int stage_valid_t[2] = {valid_t1, valid_t2};
    for (int si = 0; si < 2; ++si) {
      const StageW &s = stages[si];
      ggml_tensor *dww = clone_weight(ctx, ml, s.dw_w);
      // Fix for custom exported models where depthwise weights are [KW, KH, C, 1] instead of [KW, KH, 1, C]
      if (dww->ne[2] > 1 && dww->ne[3] == 1) {
          dww = ggml_reshape_4d(ctx, dww, dww->ne[0], dww->ne[1], 1, dww->ne[2]);
      }
      ggml_tensor *dwb = clone_weight(ctx, ml, s.dw_b);
      if (causal) {
        x = mask_time(x, stage_valid_t[si]);
        x = pad_causal(x);
        x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0,
                                   /*p1*/ 0, /*d0*/ 1, /*d1*/ 1);
      } else {
        x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 1,
                                   /*p1*/ 1, /*d0*/ 1, /*d1*/ 1);
      }
      x = ggml_cont(ctx, x);
      x = ggml_add(ctx, x, ggml_reshape_4d(ctx, dwb, 1, 1, C, 1));

      ggml_tensor *pww = clone_weight(ctx, ml, s.pw_w);
      ggml_tensor *pwb = clone_weight(ctx, ml, s.pw_b);
      x = ggml_conv_2d(ctx, pww, x, /*s0*/ 1, /*s1*/ 1, /*p0*/ 0, /*p1*/ 0,
                       /*d0*/ 1, /*d1*/ 1);
      x = ggml_add(ctx, x, ggml_reshape_4d(ctx, pwb, 1, 1, C, 1));
      x = ggml_relu(ctx, x);
    }
  }

  const int Fp = (int)x->ne[0]; // F'
  const int Tp = (int)x->ne[1]; // T'
  ggml_tensor *xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
  ggml_tensor *flat = ggml_reshape_2d(ctx, xp, (int64_t)C * Fp, Tp);

  const int valid_out = valid_out_len(T, in_valid_frames);
  if (valid_out < Tp) {
    std::vector<float> &outmask = pool.alloc_f32(Tp);
    for (int t = 0; t < Tp; ++t)
      outmask[t] = (t < valid_out) ? 1.0f : 0.0f;
    int64_t mk_ne[2] = {1, Tp};
    ggml_tensor *mask =
        pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mk_ne, outmask.data(),
                               outmask.size() * sizeof(float));
    flat = ggml_mul(ctx, flat, mask);
  }

  ggml_tensor *ow = clone_weight(ctx, ml, "encoder.pre_encode.out.weight");
  ggml_tensor *ob = clone_weight(ctx, ml, "encoder.pre_encode.out.bias");
  ggml_tensor *y = ggml_mul_mat(ctx, ow, flat);
  y = ggml_add(ctx, y, ob);

  out_Tp = Tp;
  out_valid = (valid_out > Tp) ? Tp : valid_out;
  return y;
}

void Subsampling::forward(const std::vector<float> &mel, int n_mels, int T,
                          std::vector<float> &out, int &Tout,
                          int &d_model) const {
  int valid_len_unused = 0;
  forward(mel, n_mels, T, out, Tout, d_model, valid_len_unused, -1);
}

void Subsampling::forward(const std::vector<float> &mel, int n_mels, int T,
                          std::vector<float> &out, int &Tout, int &d_model,
                          int &valid_len) const {
  forward(mel, n_mels, T, out, Tout, d_model, valid_len, -1);
}

void Subsampling::forward(const std::vector<float> &mel, int n_mels, int T,
                          std::vector<float> &out, int &Tout, int &d_model,
                          int &valid_len, int in_valid_frames) const {
  int Tp = 0, valid = 0;
  GraphInputPool pool;
  bool ok = pk::run_graph(/*mem_bytes*/
                          0, /*n_threads*/ 4,
                          [&](ggml_context *ctx) -> ggml_tensor * {
                            return build_graph(ctx, mel, n_mels, T, pool, Tp,
                                               valid, in_valid_frames);
                          },
                          out);
  assert(ok && "subsampling graph failed");
  (void)ok;
  Tout = Tp;
  d_model = d_model_;
  valid_len = valid;
}

} // namespace pk
