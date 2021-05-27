#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <omp.h>
#include <torch/extension.h>

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

// Parse OMP_NUM_THREADS environment variable
// From: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/ParallelCommon.cpp#L28
size_t get_env_num_threads(const char* var_name, size_t def_value = 1) {
  try {
    if (auto* value = std::getenv(var_name)) {
      int nthreads = c10::stoi(value);
      TORCH_CHECK(nthreads > 0);
      return nthreads;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Invalid " << var_name << " variable value, " << e.what();
    TORCH_WARN(oss.str());
  }
  return def_value;
}

// TODO: Could switch to if constexpr once Pytorch support c++17.
// TODO: When reach to really big system, each network could be ran in different GPUs.
template <bool use_stream, bool is_bmm>
class MultiNetFunction : public torch::autograd::Function<MultiNetFunction<use_stream, is_bmm>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor aev,
      int64_t num_networks,
      std::vector<int64_t> num_layers_list,
      std::vector<int64_t> start_layers_list,
      tensor_list idx_list,
      std::vector<Tensor> weight_list,
      std::vector<Tensor> bias_list,
      std::vector<at::Stream> stream_list,
      float celu_alpha) {
    tensor_list to_save;
    std::vector<at::Tensor> outputs;
    Tensor energy_list = at::zeros(num_networks, aev.options());
    int64_t total_layers = start_layers_list.back() + num_layers_list.back();
    std::vector<at::Tensor> intm_result_list(total_layers, Tensor());

    // stream events initialization
    std::vector<cudaEvent_t> event_list;
    cudaEvent_t start_event;
    at::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream();
    if (use_stream) {
      cudaEventCreate(&start_event);
      cudaEventRecord(start_event, current_stream);
      for (int i = 0; i < num_networks; i++) {
        cudaEvent_t tmp_evt;
        cudaEventCreate(&tmp_evt);
        event_list.push_back(tmp_evt);
      }
    }

    // set number of threads for OpenMP
    int max_threads = get_env_num_threads("OMP_NUM_THREADS");
    int num_threads = max_threads < num_networks ? max_threads : num_networks;
    omp_set_num_threads(num_threads);
#ifdef TORCHANI_DEBUG
    printf("fwd: number of host CPUs: %d, number of CPUs using: %d\n", max_threads, num_threads);
#endif

    // loop over networks
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_networks; i++) {
      // only run if species idx is not empty
      if (idx_list[i].size(0) > 0) {
        // disable autograd, view tracking and version counter bumps
        at::AutoDispatchBelowADInplaceOrView guard;
        // switch to different streams for each network
        if (use_stream) {
          cudaStreamWaitEvent(c10::cuda::CUDAStream(stream_list[i]), start_event, 0);
          at::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream(stream_list[i]));
        }
        int start_layers = start_layers_list[i];
        int num_layers = num_layers_list[i];
        Tensor input_ = aev.index_select(0, idx_list[i]);
        if (is_bmm) {
          int num_batch = weight_list[0].size(0);
          input_ = input_.expand({num_batch, -1, -1});
        }

        // loop over layers
        for (int j = 0; j < num_layers; j++) {
          // linear layer
          if (is_bmm) {
            input_ = at::baddbmm(bias_list[start_layers + j], input_, weight_list[start_layers + j]);
          } else {
            input_ = at::addmm(bias_list[start_layers + j], input_, weight_list[start_layers + j]);
          }
          // activation layer if it's not the last layer
          if (j < num_layers - 1) {
            input_ = at::celu_(input_, celu_alpha);
          }
          // save intermediate result for backward
          intm_result_list[start_layers + j] = input_;
        }
        // average over different networks
        if (is_bmm) {
          input_ = input_.mean(0);
        }
        // sum out without cudaMemcpyAsync
        auto tmp_energy = energy_list[i];
        at::sum_out(tmp_energy, input_.view(-1), 0, /* keepdim */ false);

        // default stream waits until all stream finish
        if (use_stream) {
          cudaEventRecord(event_list[i], c10::cuda::CUDAStream(stream_list[i]));
          cudaStreamWaitEvent(current_stream, event_list[i], 0);
        }
      }
    }
    // restore to default streams
    if (use_stream)
      at::cuda::setCurrentCUDAStream(current_stream);

    to_save.push_back(aev);
    ctx->save_for_backward(to_save);
    ctx->saved_data["num_layers_list"] = c10::List<int64_t>(num_layers_list);
    ctx->saved_data["start_layers_list"] = c10::List<int64_t>(start_layers_list);
    ctx->saved_data["stream_list"] = c10::List<at::Stream>(stream_list);
    ctx->saved_data["weight_list"] = c10::List<Tensor>(weight_list);
    ctx->saved_data["intm_result_list"] = c10::List<Tensor>(intm_result_list);
    ctx->saved_data["idx_list"] = c10::List<Tensor>(idx_list);
    ctx->saved_data["celu_alpha"] = (double)celu_alpha;

    return at::sum(energy_list, 0, true);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_o) {
    tensor_list saved_tensors = ctx->get_saved_variables();
    c10::List<int64_t> num_layers_list = ctx->saved_data["num_layers_list"].toIntList();
    c10::List<int64_t> start_layers_list = ctx->saved_data["start_layers_list"].toIntList();
    c10::List<at::Stream> stream_list = ctx->saved_data["stream_list"].to<c10::List<at::Stream>>();
    std::vector<Tensor> weight_list = ctx->saved_data["weight_list"].toTensorVector();
    std::vector<Tensor> intm_result_list = ctx->saved_data["intm_result_list"].toTensorVector();
    std::vector<Tensor> idx_list = ctx->saved_data["idx_list"].toTensorVector();
    float celu_alpha = ctx->saved_data["celu_alpha"].toDouble();
    int num_networks = num_layers_list.size();

    Tensor aev = saved_tensors[0];
    Tensor aev_grad = torch::zeros_like(aev);

    // stream events initialization
    std::vector<cudaEvent_t> event_list;
    cudaEvent_t start_event;
    at::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream();
    if (use_stream) {
      cudaEventCreate(&start_event);
      cudaEventRecord(start_event, current_stream);
      for (int i = 0; i < num_networks; i++) {
        cudaEvent_t tmp_evt;
        cudaEventCreate(&tmp_evt);
        event_list.push_back(tmp_evt);
      }
    }

    // set number of threads for OpenMP
    int max_threads = get_env_num_threads("OMP_NUM_THREADS");
    int num_threads = max_threads < num_networks ? max_threads : num_networks;
    omp_set_num_threads(num_threads);
#ifdef TORCHANI_DEBUG
    printf("bwd: number of host CPUs: %d, number of CPUs using: %d\n", max_threads, num_threads);
#endif

    // loop over networks
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_networks; i++) {
      // only run if species idx is not empty
      if (idx_list[i].size(0) > 0) {
        // disable autograd, view tracking and version counter bumps
        at::AutoDispatchBelowADInplaceOrView guard;
        // switch to different streams for each network
        if (use_stream) {
          cudaStreamWaitEvent(c10::cuda::CUDAStream(stream_list[i]), start_event, 0);
          at::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream(stream_list[i]));
        }
        Tensor input_;
        if (is_bmm) {
          int num_batch = weight_list[0].size(0);
          input_ = (grad_o[0] / num_batch).expand({num_batch, idx_list[i].size(0), -1});
        } else {
          input_ = grad_o[0].expand({idx_list[i].size(0), -1});
        }
        int start_layers = start_layers_list[i];
        int num_layers = num_layers_list[i];

        // loop over layers reversely
        for (int j = num_layers - 1; j >= 0; j--) {
          Tensor weight;
          if (is_bmm) {
            weight = weight_list[start_layers + j].transpose(1, 2);
          } else {
            weight = weight_list[start_layers + j].transpose(0, 1);
          }
          Tensor intermediate_result = intm_result_list[start_layers + j];
          // activation layer backward if it's not the last layer
          if (j < num_layers - 1) {
            input_ =
                at::elu_backward(input_, celu_alpha, 1, 1.0f / celu_alpha, /* is_result */ true, intermediate_result);
          }
          // linear layer backward
          if (is_bmm) {
            input_ = at::bmm(input_, weight);
          } else {
            input_ = at::matmul(input_, weight);
          }
        }
        if (is_bmm) {
          input_ = input_.sum(0);
        }
        aev_grad.index_put_({idx_list[i]}, input_);
        // default stream waits until all stream finish
        if (use_stream) {
          cudaEventRecord(event_list[i], c10::cuda::CUDAStream(stream_list[i]));
          cudaStreamWaitEvent(current_stream, event_list[i], 0);
        }
      }
    }
    if (use_stream)
      at::cuda::setCurrentCUDAStream(current_stream);

    return {aev_grad, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
  }
};

#define MNP_INPUT \
  aev, num_networks, num_layers_list, start_layers_list, idx_list, weight_list, bias_list, stream_list, celu_alpha

Tensor run_autograd(
    Tensor aev,
    int64_t num_networks,
    std::vector<int64_t> num_layers_list,
    std::vector<int64_t> start_layers_list,
    tensor_list idx_list,
    std::vector<Tensor> weight_list,
    std::vector<Tensor> bias_list,
    std::vector<at::Stream> stream_list,
    bool is_bmm,
    double celu_alpha = 0.1) {
  bool use_stream = aev.device().type() == torch::kCUDA;

  if (use_stream) {
    if (is_bmm)
      return MultiNetFunction<true, true>::apply(MNP_INPUT);
    else
      return MultiNetFunction<true, false>::apply(MNP_INPUT);
  } else {
    if (is_bmm)
      return MultiNetFunction<false, true>::apply(MNP_INPUT);
    else
      return MultiNetFunction<false, false>::apply(MNP_INPUT);
  }
}

// check whether the current tensor is the same as last tensor (whether they share the same data_ptr)
bool is_same_tensor(Tensor last, Tensor current) {
  return last.data_ptr() == current.data_ptr();
}

// mnp stands for multi network parallel
TORCH_LIBRARY(mnp, m) {
  m.def("run", run_autograd);
  m.def("is_same_tensor", is_same_tensor);
}

TORCH_LIBRARY_IMPL(mnp, Autograd, m) {
  m.impl("run", run_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
