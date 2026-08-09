#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

void check_solver(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuSOLVER status " +
      std::to_string(static_cast<int>(status)));
  }
}

double host_elapsed_ms(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

double median(std::vector<double> values) {
  if (values.empty()) throw std::runtime_error("median input is empty");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  return values.size() % 2U == 0U ?
    0.5 * (values[middle - 1U] + values[middle]) : values[middle];
}

class Stream {
 public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
               "create stream");
  }
  ~Stream() {
    if (value_ != nullptr) cudaStreamDestroy(value_);
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t get() const noexcept { return value_; }

 private:
  cudaStream_t value_ = nullptr;
};

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&value_), "create event"); }
  ~Event() {
    if (value_ != nullptr) cudaEventDestroy(value_);
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const noexcept { return value_; }

 private:
  cudaEvent_t value_ = nullptr;
};

class Solver {
 public:
  explicit Solver(cudaStream_t stream) {
    check_solver(cusolverDnCreate(&value_), "create solver");
    check_solver(cusolverDnSetStream(value_, stream), "set solver stream");
    check_solver(cusolverDnSetDeterministicMode(
      value_, CUSOLVER_DETERMINISTIC_RESULTS),
      "set deterministic solver mode");
  }
  ~Solver() {
    if (value_ != nullptr) cusolverDnDestroy(value_);
  }
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;
  cusolverDnHandle_t get() const noexcept { return value_; }

 private:
  cusolverDnHandle_t value_ = nullptr;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0U) throw std::runtime_error("zero-size device buffer");
    check_cuda(cudaMalloc(&value_, count_ * sizeof(T)),
               "allocate device buffer");
  }
  ~DeviceBuffer() {
    if (value_ != nullptr) cudaFree(value_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const noexcept { return static_cast<T*>(value_); }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  void* value_ = nullptr;
  std::size_t count_ = 0U;
};

struct Lane {
  Lane(int n, double* matrix, double* eigenvalues)
      : stream(), solver(stream.get()), start(), stop() {
    check_solver(cusolverDnDsyevd_bufferSize(
      solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, matrix, n, eigenvalues, &lwork), "query lane eig workspace");
    if (lwork <= 0) throw std::runtime_error("invalid lane eig workspace");
    work = new DeviceBuffer<double>(static_cast<std::size_t>(lwork));
  }
  ~Lane() { delete work; }
  Lane(const Lane&) = delete;
  Lane& operator=(const Lane&) = delete;

  Stream stream;
  Solver solver;
  Event start;
  Event stop;
  int lwork = 0;
  DeviceBuffer<double>* work = nullptr;
};

float event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
  float value = 0.0F;
  check_cuda(cudaEventElapsedTime(&value, start, stop),
             "read CUDA event elapsed time");
  return value;
}

void submit_eig(
    Lane* lane, int n, double* matrix, double* eigenvalues, int* info) {
  if (lane == nullptr) throw std::runtime_error("null eig lane");
  check_solver(cusolverDnDsyevd(
    lane->solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    n, matrix, n, eigenvalues, lane->work->get(), lane->lwork, info),
    "submit deterministic full eig");
}

struct Measurement {
  double host_submit_ms = 0.0;
  double host_wall_ms = 0.0;
  double device_makespan_ms = 0.0;
  double lane_duration_sum_ms = 0.0;
  double two_lane_overlap_ms = 0.0;
  double first_batch_duration_ms = 0.0;
  double second_batch_duration_ms = 0.0;
  double batch_overlap_ms = 0.0;
};

Measurement run_serial(
    int n, int component_count, std::size_t cells,
    const DeviceBuffer<double>& source,
    DeviceBuffer<double>* matrices,
    DeviceBuffer<double>* eigenvalues,
    DeviceBuffer<int>* info,
    Lane* lane) {
  check_cuda(cudaMemcpy(
    matrices->get(), source.get(), source.bytes(), cudaMemcpyDeviceToDevice),
    "restore serial eig inputs");
  check_cuda(cudaEventRecord(lane->start.get(), lane->stream.get()),
             "record serial eig start");
  const auto submit_start = std::chrono::steady_clock::now();
  for (int component = 0; component < component_count; ++component) {
    submit_eig(
      lane, n,
      matrices->get() + static_cast<std::size_t>(component) * cells,
      eigenvalues->get() + static_cast<std::size_t>(component) * n,
      info->get() + component);
  }
  const double submit_ms = host_elapsed_ms(submit_start);
  check_cuda(cudaEventRecord(lane->stop.get(), lane->stream.get()),
             "record serial eig stop");
  const auto wait_start = std::chrono::steady_clock::now();
  check_cuda(cudaEventSynchronize(lane->stop.get()), "wait serial eig stop");
  Measurement value;
  value.host_submit_ms = submit_ms;
  value.host_wall_ms = submit_ms + host_elapsed_ms(wait_start);
  value.device_makespan_ms = event_elapsed(
    lane->start.get(), lane->stop.get());
  value.lane_duration_sum_ms = value.device_makespan_ms;
  return value;
}

Measurement run_lanes(
    int n, int component_offset, int component_count, std::size_t cells,
    const DeviceBuffer<double>& source,
    DeviceBuffer<double>* matrices,
    DeviceBuffer<double>* eigenvalues,
    DeviceBuffer<int>* info,
    const std::vector<Lane*>& lanes,
    Stream* control,
    Event* ready) {
  check_cuda(cudaMemcpy(
    matrices->get(), source.get(), source.bytes(), cudaMemcpyDeviceToDevice),
    "restore concurrent eig inputs");
  check_cuda(cudaEventRecord(ready->get(), control->get()),
             "record concurrent eig ready");
  for (Lane* lane : lanes) {
    check_cuda(cudaStreamWaitEvent(lane->stream.get(), ready->get(), 0),
               "gate concurrent eig lane");
  }
  std::vector<unsigned char> lane_started(lanes.size(), 0U);
  const auto submit_start = std::chrono::steady_clock::now();
  for (int local_component = 0; local_component < component_count;
       ++local_component) {
    const std::size_t lane_index =
      static_cast<std::size_t>(local_component) % lanes.size();
    Lane* lane = lanes[lane_index];
    if (lane_started[lane_index] == 0U) {
      check_cuda(cudaEventRecord(lane->start.get(), lane->stream.get()),
                 "record concurrent lane start");
      lane_started[lane_index] = 1U;
    }
    const int component = component_offset + local_component;
    submit_eig(
      lane, n,
      matrices->get() + static_cast<std::size_t>(component) * cells,
      eigenvalues->get() + static_cast<std::size_t>(component) * n,
      info->get() + component);
  }
  for (Lane* lane : lanes) {
    check_cuda(cudaEventRecord(lane->stop.get(), lane->stream.get()),
               "record concurrent lane stop");
  }
  const double submit_ms = host_elapsed_ms(submit_start);
  const auto wait_start = std::chrono::steady_clock::now();
  for (Lane* lane : lanes) {
    check_cuda(cudaEventSynchronize(lane->stop.get()),
               "wait concurrent lane stop");
  }

  Measurement value;
  value.host_submit_ms = submit_ms;
  value.host_wall_ms = submit_ms + host_elapsed_ms(wait_start);
  double earliest_start = 1.0e300;
  double latest_start = 0.0;
  double earliest_end = 1.0e300;
  double latest_end = 0.0;
  for (Lane* lane : lanes) {
    const double start_ms = event_elapsed(ready->get(), lane->start.get());
    const double end_ms = event_elapsed(ready->get(), lane->stop.get());
    value.lane_duration_sum_ms += end_ms - start_ms;
    earliest_start = std::min(earliest_start, start_ms);
    latest_start = std::max(latest_start, start_ms);
    earliest_end = std::min(earliest_end, end_ms);
    latest_end = std::max(latest_end, end_ms);
  }
  value.device_makespan_ms = latest_end - earliest_start;
  if (lanes.size() == 2U) {
    value.two_lane_overlap_ms = std::max(0.0, earliest_end - latest_start);
  }
  return value;
}

Measurement run_split_batches(
    int n, int first_component_count, int second_component_count,
    int lanes_per_batch, std::size_t cells,
    const DeviceBuffer<double>& source,
    DeviceBuffer<double>* matrices,
    DeviceBuffer<double>* eigenvalues,
    DeviceBuffer<int>* info,
    const std::vector<Lane*>& lanes,
    Stream* control,
    Event* ready) {
  const int component_count = first_component_count + second_component_count;
  if (first_component_count < 1 || second_component_count < 1 ||
      lanes_per_batch < 1 ||
      lanes.size() != static_cast<std::size_t>(2 * lanes_per_batch)) {
    throw std::runtime_error("invalid split eig batch shape");
  }
  check_cuda(cudaMemcpy(
    matrices->get(), source.get(), source.bytes(), cudaMemcpyDeviceToDevice),
    "restore split eig inputs");
  check_cuda(cudaEventRecord(ready->get(), control->get()),
             "record split eig ready");
  for (Lane* lane : lanes) {
    check_cuda(cudaStreamWaitEvent(lane->stream.get(), ready->get(), 0),
               "gate split eig lane");
  }

  std::vector<unsigned char> lane_started(lanes.size(), 0U);
  const auto submit_start = std::chrono::steady_clock::now();
  for (int component = 0; component < first_component_count; ++component) {
    const int lane_index = component % lanes_per_batch;
    Lane* lane = lanes[static_cast<std::size_t>(lane_index)];
    if (lane_started[static_cast<std::size_t>(lane_index)] == 0U) {
      check_cuda(cudaEventRecord(lane->start.get(), lane->stream.get()),
                 "record first split lane start");
      lane_started[static_cast<std::size_t>(lane_index)] = 1U;
    }
    submit_eig(
      lane, n,
      matrices->get() + static_cast<std::size_t>(component) * cells,
      eigenvalues->get() + static_cast<std::size_t>(component) * n,
      info->get() + component);
  }
  for (int local = 0; local < second_component_count; ++local) {
    const int component = first_component_count + local;
    const int lane_index = lanes_per_batch + local % lanes_per_batch;
    Lane* lane = lanes[static_cast<std::size_t>(lane_index)];
    if (lane_started[static_cast<std::size_t>(lane_index)] == 0U) {
      check_cuda(cudaEventRecord(lane->start.get(), lane->stream.get()),
                 "record second split lane start");
      lane_started[static_cast<std::size_t>(lane_index)] = 1U;
    }
    submit_eig(
      lane, n,
      matrices->get() + static_cast<std::size_t>(component) * cells,
      eigenvalues->get() + static_cast<std::size_t>(component) * n,
      info->get() + component);
  }
  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    if (lane_started[lane_index] == 0U) continue;
    check_cuda(cudaEventRecord(
      lanes[lane_index]->stop.get(), lanes[lane_index]->stream.get()),
      "record split lane stop");
  }
  const double submit_ms = host_elapsed_ms(submit_start);
  const auto wait_start = std::chrono::steady_clock::now();
  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    if (lane_started[lane_index] == 0U) continue;
    check_cuda(cudaEventSynchronize(lanes[lane_index]->stop.get()),
               "wait split lane stop");
  }

  Measurement value;
  value.host_submit_ms = submit_ms;
  value.host_wall_ms = submit_ms + host_elapsed_ms(wait_start);
  double earliest_start = 1.0e300;
  double latest_end = 0.0;
  double first_earliest_start = 1.0e300;
  double first_latest_end = 0.0;
  double second_earliest_start = 1.0e300;
  double second_latest_end = 0.0;
  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    if (lane_started[lane_index] == 0U) continue;
    Lane* lane = lanes[lane_index];
    const double start_ms = event_elapsed(ready->get(), lane->start.get());
    const double end_ms = event_elapsed(ready->get(), lane->stop.get());
    value.lane_duration_sum_ms += end_ms - start_ms;
    earliest_start = std::min(earliest_start, start_ms);
    latest_end = std::max(latest_end, end_ms);
    if (lane_index < static_cast<std::size_t>(lanes_per_batch)) {
      first_earliest_start = std::min(first_earliest_start, start_ms);
      first_latest_end = std::max(first_latest_end, end_ms);
    } else {
      second_earliest_start = std::min(second_earliest_start, start_ms);
      second_latest_end = std::max(second_latest_end, end_ms);
    }
  }
  value.device_makespan_ms = latest_end - earliest_start;
  value.first_batch_duration_ms = first_latest_end - first_earliest_start;
  value.second_batch_duration_ms = second_latest_end - second_earliest_start;
  value.batch_overlap_ms = std::max(
    0.0, std::min(first_latest_end, second_latest_end) -
      std::max(first_earliest_start, second_earliest_start));
  (void)component_count;
  return value;
}

Measurement run_threaded_lanes(
    int n, int component_count, std::size_t cells,
    const DeviceBuffer<double>& source,
    DeviceBuffer<double>* matrices,
    DeviceBuffer<double>* eigenvalues,
    DeviceBuffer<int>* info,
    const std::vector<Lane*>& lanes,
    Stream* control,
    Event* ready) {
  check_cuda(cudaMemcpy(
    matrices->get(), source.get(), source.bytes(), cudaMemcpyDeviceToDevice),
    "restore threaded eig inputs");
  check_cuda(cudaEventRecord(ready->get(), control->get()),
             "record threaded eig ready");
  for (Lane* lane : lanes) {
    check_cuda(cudaStreamWaitEvent(lane->stream.get(), ready->get(), 0),
               "gate threaded eig lane");
  }

  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors(lanes.size());
  workers.reserve(lanes.size());
  const auto submit_start = std::chrono::steady_clock::now();
  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    workers.emplace_back([&, lane_index]() {
      try {
        check_cuda(cudaSetDevice(0), "select threaded benchmark device");
        Lane* lane = lanes[lane_index];
        check_cuda(cudaEventRecord(lane->start.get(), lane->stream.get()),
                   "record threaded lane start");
        for (int component = static_cast<int>(lane_index);
             component < component_count;
             component += static_cast<int>(lanes.size())) {
          submit_eig(
            lane, n,
            matrices->get() + static_cast<std::size_t>(component) * cells,
            eigenvalues->get() + static_cast<std::size_t>(component) * n,
            info->get() + component);
        }
        check_cuda(cudaEventRecord(lane->stop.get(), lane->stream.get()),
                   "record threaded lane stop");
      } catch (...) {
        errors[lane_index] = std::current_exception();
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  const double submit_ms = host_elapsed_ms(submit_start);
  for (const std::exception_ptr& error : errors) {
    if (error) std::rethrow_exception(error);
  }

  const auto wait_start = std::chrono::steady_clock::now();
  for (Lane* lane : lanes) {
    check_cuda(cudaEventSynchronize(lane->stop.get()),
               "wait threaded lane stop");
  }

  Measurement value;
  value.host_submit_ms = submit_ms;
  value.host_wall_ms = submit_ms + host_elapsed_ms(wait_start);
  double earliest_start = 1.0e300;
  double latest_end = 0.0;
  for (Lane* lane : lanes) {
    const double start_ms = event_elapsed(ready->get(), lane->start.get());
    const double end_ms = event_elapsed(ready->get(), lane->stop.get());
    value.lane_duration_sum_ms += end_ms - start_ms;
    earliest_start = std::min(earliest_start, start_ms);
    latest_end = std::max(latest_end, end_ms);
  }
  value.device_makespan_ms = latest_end - earliest_start;
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int n = argc > 1 ? std::atoi(argv[1]) : 351;
    const int first_component_count = argc > 2 ? std::atoi(argv[2]) : 8;
    const int lanes_per_batch = argc > 3 ? std::atoi(argv[3]) : 2;
    const int trials = argc > 4 ? std::atoi(argv[4]) : 7;
    const int second_component_count = argc > 5 ? std::atoi(argv[5]) : 0;
    const bool split_mode = second_component_count > 0;
    const int component_count =
      first_component_count + second_component_count;
    const int lane_count = split_mode ?
      2 * lanes_per_batch : lanes_per_batch;
    if (n < 2 || first_component_count < 1 || lanes_per_batch < 2 ||
        trials < 1 ||
        second_component_count < 0) {
      throw std::runtime_error("invalid benchmark arguments");
    }
    if (!split_mode && component_count < 2) {
      throw std::runtime_error("single-batch benchmark requires two components");
    }
    check_cuda(cudaSetDevice(0), "select benchmark device");

    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t cells = n_size * n_size;
    const std::size_t matrix_values =
      static_cast<std::size_t>(component_count) * cells;
    const std::size_t eigenvalue_count =
      static_cast<std::size_t>(component_count) * n_size;
    std::vector<double> host_source(matrix_values);
    for (int component = 0; component < component_count; ++component) {
      std::vector<double> residual(n_size);
      for (int row = 0; row < n; ++row) {
        residual[static_cast<std::size_t>(row)] =
          std::sin(0.017 * (component + 1) * (row + 1)) +
          std::cos(0.0031 * (component + 3) * (row + 1) * (row + 1));
      }
      double* matrix = host_source.data() +
        static_cast<std::size_t>(component) * cells;
      for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
          matrix[static_cast<std::size_t>(column) * n_size + row] =
            std::abs(residual[static_cast<std::size_t>(row)] -
                     residual[static_cast<std::size_t>(column)]);
        }
      }
    }

    DeviceBuffer<double> source(matrix_values);
    DeviceBuffer<double> serial_matrices(matrix_values);
    DeviceBuffer<double> concurrent_matrices(matrix_values);
    DeviceBuffer<double> serial_eigenvalues(eigenvalue_count);
    DeviceBuffer<double> concurrent_eigenvalues(eigenvalue_count);
    DeviceBuffer<int> serial_info(static_cast<std::size_t>(component_count));
    DeviceBuffer<int> concurrent_info(static_cast<std::size_t>(component_count));
    check_cuda(cudaMemcpy(
      source.get(), host_source.data(), source.bytes(), cudaMemcpyHostToDevice),
      "upload benchmark eig inputs");

    std::vector<Lane*> lanes;
    lanes.reserve(static_cast<std::size_t>(lane_count));
    for (int lane = 0; lane < lane_count; ++lane) {
      lanes.push_back(new Lane(
        n, concurrent_matrices.get(), concurrent_eigenvalues.get()));
    }
    const int pooled_lane_count = std::min(component_count, lanes_per_batch);
    std::vector<Lane*> pooled_lanes(
      lanes.begin(), lanes.begin() + pooled_lane_count);
    const int first_pooled_lane_count =
      std::min(first_component_count, lanes_per_batch);
    const int second_pooled_lane_count = split_mode ?
      std::min(second_component_count, lanes_per_batch) : 0;
    std::vector<Lane*> first_pooled_lanes(
      lanes.begin(), lanes.begin() + first_pooled_lane_count);
    std::vector<Lane*> second_pooled_lanes;
    if (split_mode) {
      second_pooled_lanes.assign(
        lanes.begin(), lanes.begin() + second_pooled_lane_count);
    }
    Lane serial_lane(n, serial_matrices.get(), serial_eigenvalues.get());
    Stream control;
    Event ready;

    run_serial(
      n, component_count, cells, source, &serial_matrices,
      &serial_eigenvalues, &serial_info, &serial_lane);
    run_lanes(
      n, 0, component_count, cells, source, &concurrent_matrices,
      &concurrent_eigenvalues, &concurrent_info, pooled_lanes, &control, &ready);
    if (split_mode) {
      run_lanes(
        n, 0, first_component_count, cells, source, &concurrent_matrices,
        &concurrent_eigenvalues, &concurrent_info, first_pooled_lanes,
        &control, &ready);
      run_lanes(
        n, first_component_count, second_component_count, cells, source,
        &concurrent_matrices, &concurrent_eigenvalues, &concurrent_info,
        second_pooled_lanes, &control, &ready);
      run_split_batches(
        n, first_component_count, second_component_count, lanes_per_batch,
        cells, source, &concurrent_matrices, &concurrent_eigenvalues,
        &concurrent_info, lanes, &control, &ready);
    } else {
      run_threaded_lanes(
        n, component_count, cells, source, &concurrent_matrices,
        &concurrent_eigenvalues, &concurrent_info, pooled_lanes,
        &control, &ready);
    }

    std::vector<double> serial_submit;
    std::vector<double> serial_wall;
    std::vector<double> serial_device;
    std::vector<double> concurrent_submit;
    std::vector<double> concurrent_wall;
    std::vector<double> concurrent_device;
    std::vector<double> concurrent_lane_sum;
    std::vector<double> concurrent_overlap;
    std::vector<double> separate_submit;
    std::vector<double> separate_wall;
    std::vector<double> separate_device;
    std::vector<double> separate_lane_sum;
    std::vector<double> threaded_submit;
    std::vector<double> threaded_wall;
    std::vector<double> threaded_device;
    std::vector<double> threaded_lane_sum;
    std::vector<double> alternate_first_batch_duration;
    std::vector<double> alternate_second_batch_duration;
    std::vector<double> alternate_batch_overlap;
    for (int trial = 0; trial < trials; ++trial) {
      const Measurement serial = run_serial(
        n, component_count, cells, source, &serial_matrices,
        &serial_eigenvalues, &serial_info, &serial_lane);
      const Measurement concurrent = run_lanes(
        n, 0, component_count, cells, source, &concurrent_matrices,
        &concurrent_eigenvalues, &concurrent_info, pooled_lanes,
        &control, &ready);
      serial_submit.push_back(serial.host_submit_ms);
      serial_wall.push_back(serial.host_wall_ms);
      serial_device.push_back(serial.device_makespan_ms);
      concurrent_submit.push_back(concurrent.host_submit_ms);
      concurrent_wall.push_back(concurrent.host_wall_ms);
      concurrent_device.push_back(concurrent.device_makespan_ms);
      concurrent_lane_sum.push_back(concurrent.lane_duration_sum_ms);
      concurrent_overlap.push_back(concurrent.two_lane_overlap_ms);
      Measurement separate;
      if (split_mode) {
        const Measurement first = run_lanes(
          n, 0, first_component_count, cells, source, &concurrent_matrices,
          &concurrent_eigenvalues, &concurrent_info, first_pooled_lanes,
          &control, &ready);
        const Measurement second = run_lanes(
          n, first_component_count, second_component_count, cells, source,
          &concurrent_matrices, &concurrent_eigenvalues, &concurrent_info,
          second_pooled_lanes, &control, &ready);
        separate.host_submit_ms = first.host_submit_ms + second.host_submit_ms;
        separate.host_wall_ms = first.host_wall_ms + second.host_wall_ms;
        separate.device_makespan_ms =
          first.device_makespan_ms + second.device_makespan_ms;
        separate.lane_duration_sum_ms =
          first.lane_duration_sum_ms + second.lane_duration_sum_ms;
      } else {
        separate = concurrent;
      }
      separate_submit.push_back(separate.host_submit_ms);
      separate_wall.push_back(separate.host_wall_ms);
      separate_device.push_back(separate.device_makespan_ms);
      separate_lane_sum.push_back(separate.lane_duration_sum_ms);
      const Measurement alternate = split_mode ?
        run_split_batches(
          n, first_component_count, second_component_count, lanes_per_batch,
          cells, source, &concurrent_matrices, &concurrent_eigenvalues,
          &concurrent_info, lanes, &control, &ready) :
        run_threaded_lanes(
          n, component_count, cells, source, &concurrent_matrices,
          &concurrent_eigenvalues, &concurrent_info, pooled_lanes,
          &control, &ready);
      threaded_submit.push_back(alternate.host_submit_ms);
      threaded_wall.push_back(alternate.host_wall_ms);
      threaded_device.push_back(alternate.device_makespan_ms);
      threaded_lane_sum.push_back(alternate.lane_duration_sum_ms);
      alternate_first_batch_duration.push_back(
        alternate.first_batch_duration_ms);
      alternate_second_batch_duration.push_back(
        alternate.second_batch_duration_ms);
      alternate_batch_overlap.push_back(alternate.batch_overlap_ms);
    }

    std::vector<double> host_serial_matrices(matrix_values);
    std::vector<double> host_concurrent_matrices(matrix_values);
    std::vector<double> host_serial_eigenvalues(eigenvalue_count);
    std::vector<double> host_concurrent_eigenvalues(eigenvalue_count);
    std::vector<int> host_serial_info(static_cast<std::size_t>(component_count));
    std::vector<int> host_concurrent_info(
      static_cast<std::size_t>(component_count));
    check_cuda(cudaMemcpy(
      host_serial_matrices.data(), serial_matrices.get(),
      serial_matrices.bytes(), cudaMemcpyDeviceToHost),
      "download serial eigenvectors");
    check_cuda(cudaMemcpy(
      host_concurrent_matrices.data(), concurrent_matrices.get(),
      concurrent_matrices.bytes(), cudaMemcpyDeviceToHost),
      "download concurrent eigenvectors");
    check_cuda(cudaMemcpy(
      host_serial_eigenvalues.data(), serial_eigenvalues.get(),
      serial_eigenvalues.bytes(), cudaMemcpyDeviceToHost),
      "download serial eigenvalues");
    check_cuda(cudaMemcpy(
      host_concurrent_eigenvalues.data(), concurrent_eigenvalues.get(),
      concurrent_eigenvalues.bytes(), cudaMemcpyDeviceToHost),
      "download concurrent eigenvalues");
    check_cuda(cudaMemcpy(
      host_serial_info.data(), serial_info.get(), serial_info.bytes(),
      cudaMemcpyDeviceToHost), "download serial solver info");
    check_cuda(cudaMemcpy(
      host_concurrent_info.data(), concurrent_info.get(),
      concurrent_info.bytes(), cudaMemcpyDeviceToHost),
      "download concurrent solver info");

    std::size_t eigenvalue_bit_mismatch_count = 0U;
    std::size_t eigenvector_bit_mismatch_count = 0U;
    double maximum_eigenvalue_absolute_error = 0.0;
    double maximum_eigenvector_absolute_error = 0.0;
    for (std::size_t index = 0; index < eigenvalue_count; ++index) {
      if (std::memcmp(
            &host_serial_eigenvalues[index],
            &host_concurrent_eigenvalues[index], sizeof(double)) != 0) {
        eigenvalue_bit_mismatch_count += 1U;
      }
      maximum_eigenvalue_absolute_error = std::max(
        maximum_eigenvalue_absolute_error,
        std::abs(host_serial_eigenvalues[index] -
                 host_concurrent_eigenvalues[index]));
    }
    for (std::size_t index = 0; index < matrix_values; ++index) {
      if (std::memcmp(
            &host_serial_matrices[index],
            &host_concurrent_matrices[index], sizeof(double)) != 0) {
        eigenvector_bit_mismatch_count += 1U;
      }
      maximum_eigenvector_absolute_error = std::max(
        maximum_eigenvector_absolute_error,
        std::abs(host_serial_matrices[index] -
                 host_concurrent_matrices[index]));
    }
    const int serial_failure_count = static_cast<int>(std::count_if(
      host_serial_info.begin(), host_serial_info.end(),
      [](int value) { return value != 0; }));
    const int concurrent_failure_count = static_cast<int>(std::count_if(
      host_concurrent_info.begin(), host_concurrent_info.end(),
      [](int value) { return value != 0; }));

    const double serial_device_median = median(serial_device);
    const double concurrent_device_median = median(concurrent_device);
    const double separate_device_median = median(separate_device);
    const double threaded_device_median = median(threaded_device);
    std::printf(
      "n=%d components=%d first_components=%d second_components=%d "
      "lanes_per_batch=%d trials=%d alternate_mode=%s "
      "serial_submit_median_ms=%.9g serial_wall_median_ms=%.9g "
      "serial_device_median_ms=%.9g "
      "concurrent_submit_median_ms=%.9g concurrent_wall_median_ms=%.9g "
      "concurrent_device_median_ms=%.9g "
      "device_speedup=%.9g lane_duration_sum_median_ms=%.9g "
      "two_lane_overlap_median_ms=%.9g "
      "separate_submit_median_ms=%.9g separate_wall_median_ms=%.9g "
      "separate_device_median_ms=%.9g separate_lane_sum_median_ms=%.9g "
      "alternate_submit_median_ms=%.9g alternate_wall_median_ms=%.9g "
      "alternate_device_median_ms=%.9g alternate_device_speedup=%.9g "
      "separate_to_alternate_device_speedup=%.9g "
      "alternate_lane_duration_sum_median_ms=%.9g "
      "alternate_first_batch_duration_median_ms=%.9g "
      "alternate_second_batch_duration_median_ms=%.9g "
      "alternate_batch_overlap_median_ms=%.9g "
      "eigenvalue_bit_mismatch_count=%zu "
      "eigenvector_bit_mismatch_count=%zu "
      "maximum_eigenvalue_absolute_error=%.17g "
      "maximum_eigenvector_absolute_error=%.17g "
      "serial_failure_count=%d concurrent_failure_count=%d\n",
      n, component_count, first_component_count, second_component_count,
      lanes_per_batch, trials, split_mode ? "split" : "threaded",
      median(serial_submit), median(serial_wall), serial_device_median,
      median(concurrent_submit), median(concurrent_wall),
      concurrent_device_median,
      serial_device_median / concurrent_device_median,
      median(concurrent_lane_sum), median(concurrent_overlap),
      median(separate_submit), median(separate_wall), separate_device_median,
      median(separate_lane_sum),
      median(threaded_submit), median(threaded_wall),
      threaded_device_median, serial_device_median / threaded_device_median,
      separate_device_median / threaded_device_median,
      median(threaded_lane_sum),
      median(alternate_first_batch_duration),
      median(alternate_second_batch_duration),
      median(alternate_batch_overlap),
      eigenvalue_bit_mismatch_count, eigenvector_bit_mismatch_count,
      maximum_eigenvalue_absolute_error, maximum_eigenvector_absolute_error,
      serial_failure_count, concurrent_failure_count);

    for (Lane* lane : lanes) delete lane;
    return serial_failure_count == 0 && concurrent_failure_count == 0 &&
      eigenvalue_bit_mismatch_count == 0U &&
      eigenvector_bit_mismatch_count == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
