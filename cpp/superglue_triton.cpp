#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <utility>

#include "io.h"
#include "viz.h"

using namespace torch;
using namespace torch::indexing;
// namespace fs = std::filesystem;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unpack_result(const IValue &result) {
  auto model_outputs_tuple = result.toTuple();
  std::vector<torch::Tensor> output_tensors;
  int cnt = 0;
  for (auto &m_op : model_outputs_tuple->elements()) {
    output_tensors.push_back(m_op.toTensor());
    std::cout << "shape: " << cnt++ << std::endl;
    std::cout << output_tensors.back().sizes() << std::endl;
    std::cout << "==========" << std::endl;
  }
  return {output_tensors[0][0], output_tensors[1][0], output_tensors[2][0]};
}

std::tuple<torch::Tensor, torch::Tensor>
unpack_superglue_result(const IValue &result) {
  auto model_outputs_tuple = result.toTuple();
  std::vector<torch::Tensor> output_tensors;
  int cnt = 0;
  for (auto &m_op : model_outputs_tuple->elements()) {
    output_tensors.push_back(m_op.toTensor());
    std::cout << "shape: " << cnt++ << std::endl;
    std::cout << output_tensors.back().sizes() << std::endl;
    std::cout << "==========" << std::endl;
  }
  return {output_tensors[0][0], output_tensors[1][0]};
}

torch::Dict<std::string, Tensor> toTensorDict(const torch::IValue &value) {
  return c10::impl::toTypedDict<std::string, Tensor>(value.toGenericDict());
}

int main(int argc, const char *argv[]) {
  if (argc <= 3) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << argv[0] << " <image0> <image1> <downscaled_width>"
              << std::endl;
    return 1;
  }

  torch::manual_seed(1);
  torch::autograd::GradMode::set_enabled(false);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA, 1);
  }

  int target_width = std::stoi(argv[3]);
  Tensor image0 = read_image(std::string(argv[1]), target_width).to(device);
  Tensor image1 = read_image(std::string(argv[2]), target_width).to(device);

  // Look for the TorchScript module files in the executable directory
  // auto executable_dir =
  // fs::weakly_canonical(fs::path(argv[0])).parent_path();
  auto module_path = "SuperPoint.zip";
  // if (!fs::exists(module_path)) {
  //   std::cerr << "Could not find the TorchScript module file " << module_path
  //   << std::endl; return 1;
  // }
  std::cout << "Load SuperPoint" << std::endl;
  torch::jit::script::Module superpoint = torch::jit::load(module_path);
  superpoint.eval();
  superpoint.to(device);

  module_path = "SuperGlue.zip";
  // if (!fs::exists(module_path)) {
  //   std::cerr << "Could not find the TorchScript module file " << module_path
  //   << std::endl; return 1;
  // }
  torch::jit::script::Module superglue = torch::jit::load(module_path);
  superglue.eval();
  superglue.to(device);
  std::cout << "Load SuperGlue" << std::endl;

  using namespace std::chrono;
  Tensor keypoints0, scores0, descriptors0;
  Tensor keypoints1, scores1, descriptors1;
  Tensor indices0, mscores0;

  std::tie(keypoints0, scores0, descriptors0) =
      unpack_result(superpoint.forward({image0}));
  std::tie(keypoints1, scores1, descriptors1) =
      unpack_result(superpoint.forward({image1}));

  std::vector<float> img_shape0({image0.size(3), image0.size(2)});
  std::vector<float> img_shape1({image1.size(3), image1.size(2)});
  torch::Tensor shape0 = torch::from_blob(
      img_shape0.data(), {1, 2}, torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor shape1 = torch::from_blob(
      img_shape1.data(), {1, 2}, torch::TensorOptions().dtype(torch::kFloat32));

  for (int i = 0; i < 3; ++i) {
    std::tie(indices0, mscores0) = unpack_superglue_result(superglue.forward(
        {keypoints0.unsqueeze(0), keypoints1.unsqueeze(0),
         descriptors0.unsqueeze(0), descriptors1.unsqueeze(0),
         scores0.unsqueeze(0), scores1.unsqueeze(0), shape0, shape1}));
  }
  std::cout << img_shape0[0] << ", " << img_shape0[1] << std::endl;
  std::cout << img_shape1[0] << ", " << img_shape1[1] << std::endl;
  auto t0 = high_resolution_clock::now();
  int N = 3;
  for (int i = 0; i < N; ++i) {
    std::tie(indices0, mscores0) = unpack_superglue_result(superglue.forward(
        {keypoints0.unsqueeze(0), keypoints1.unsqueeze(0),
         descriptors0.unsqueeze(0), descriptors1.unsqueeze(0),
         scores0.unsqueeze(0), scores1.unsqueeze(0), shape0, shape1}));
  }
  double period =
      duration_cast<duration<double>>(high_resolution_clock::now() - t0)
          .count() /
      N;
  std::cout << period * 1e3 << " ms, FPS: " << 1 / period << std::endl;

  auto matches = indices0;
  auto valid = at::nonzero(matches > -1).squeeze();
  auto mkpts0 = keypoints0.index_select(0, valid);
  auto mkpts1 = keypoints1.index_select(0, matches.index_select(0, valid));
  auto confidence = mscores0.index_select(0, valid);

  std::cout << "Image #0 keypoints: " << keypoints0.size(0) << std::endl;
  std::cout << "Image #1 keypoints: " << keypoints1.size(0) << std::endl;
  std::cout << "Valid match count: " << valid.size(0) << std::endl;

  cv::Mat plot = make_matching_plot_fast(image0, image1, keypoints0, keypoints1,
                                         mkpts0, mkpts1, confidence);
  cv::imwrite("matches.png", plot);
  std::cout << "Done! Created matches.png for visualization." << std::endl;
}
