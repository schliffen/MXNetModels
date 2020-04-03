#include "fpn.h"
#include "debug.h"
#include "nnutils.h"

FPNImpl::FPNImpl() {}

FPNImpl::FPNImpl(torch::nn::Sequential c1,
                 torch::nn::Sequential c2,
                 torch::nn::Sequential c3,
                 torch::nn::Sequential c4,
                 torch::nn::Sequential c5,
                 uint32_t out_channels)
    : c1_(c1),
      c2_(c2),
      c3_(c3),
      c4_(c4),
      c5_(c5),
      p6_(torch::nn::Functional(torch::max_pool2d,
                                /*kernel_size*/ at::IntList({1}),
                                /*stride*/ at::IntList({2}),
                                /*padding*/ at::IntList({0}),
                                /*dilation*/ at::IntList({1}),
                                /*ceil_mode*/ false)),
      p5_conv1_(torch::nn::Conv2dOptions(2048, out_channels, 1).stride(1)),
      p5_conv2_(SamePad2d(/*kernel_size*/ 3, /*stride*/ 1),
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                        .stride(1))),
      p4_conv1_(torch::nn::Conv2dOptions(1024, out_channels, 1).stride(1)),
      p4_conv2_(SamePad2d(/*kernel_size*/ 3, /*stride*/ 1),
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                        .stride(1))),
      p3_conv1_(torch::nn::Conv2dOptions(512, out_channels, 1).stride(1)),
      p3_conv2_(SamePad2d(/*kernel_size*/ 3, /*stride*/ 1),
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                        .stride(1))),
      p2_conv1_(torch::nn::Conv2dOptions(256, out_channels, 1).stride(1)),
      p2_conv2_(SamePad2d(/*kernel_size*/ 3, /*stride*/ 1),
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                        .stride(1))) {
  register_module("C1", c1_);
  register_module("C2", c2_);
  register_module("C3", c3_);
  register_module("C4", c4_);
  register_module("C5", c5_);
  register_module("P6", p6_);
  register_module("P5_conv1", p5_conv1_);
  register_module("P5_conv2", p5_conv2_);
  register_module("P4_conv1", p4_conv1_);
  register_module("P4_conv2", p4_conv2_);
  register_module("P3_conv1", p3_conv1_);
  register_module("P3_conv2", p3_conv2_);
  register_module("P2_conv1", p2_conv1_);
  register_module("P2_conv2", p2_conv2_);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
FPNImpl::forward(at::Tensor x) {
  x = c1_->forward(x);
  x = c2_->forward(x);
  auto c2_out = x;
  x = c3_->forward(x);
  auto c3_out = x;
  x = c4_->forward(x);
  auto c4_out = x;
  x = c5_->forward(x);
  auto p5_out = p5_conv1_->forward(x);
  auto p4_out =
      p4_conv1_->forward(c4_out) + upsample(p5_out, /*scale_factor*/ 2);
  auto p3_out =
      p3_conv1_->forward(c3_out) + upsample(p4_out, /*scale_factor*/ 2);
  auto p2_out =
      p2_conv1_->forward(c2_out) + upsample(p3_out, /*scale_factor*/ 2);

  p5_out = p5_conv2_->forward(p5_out);
  p4_out = p4_conv2_->forward(p4_out);
  p3_out = p3_conv2_->forward(p3_out);
  p2_out = p2_conv2_->forward(p2_out);

  // P6 is used for the 5th anchor scale in RPN. Generated by subsampling from
  // P5 with stride of 2.
  auto p6_out = p6_->forward(p5_out);

  return {p2_out, p3_out, p4_out, p5_out, p6_out};
}
