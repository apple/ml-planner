#include "utils.h"


std::vector<torch::Tensor> composite_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return composite_train_fw_cu(
                sigmas, rgbs, deltas, ts,
                rays_a, opacity_threshold);
}


std::vector<torch::Tensor> composite_train_bw(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_ddepth,
    const torch::Tensor dL_drgb,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor depth,
    const torch::Tensor rgb,
    const float opacity_threshold
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_ddepth);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);

    return composite_train_bw_cu(
                dL_dopacity, dL_ddepth, dL_drgb,
                sigmas, rgbs, deltas, ts, rays_a,
                opacity, depth, rgb, opacity_threshold);
}


void composite_test_fw(
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor rgb
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(opacity);
    CHECK_INPUT(depth);
    CHECK_INPUT(rgb);

    composite_test_fw_cu(
        sigmas, rgbs, deltas, ts, hits_t, alive_indices,
        T_threshold, N_eff_samples,
        opacity, depth, rgb);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("composite_train_fw", &composite_train_fw);
    m.def("composite_train_bw", &composite_train_bw);
    m.def("composite_test_fw", &composite_test_fw);
}