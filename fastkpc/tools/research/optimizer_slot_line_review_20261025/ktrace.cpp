// Minimal CUPTI injection tracer (CUDA_INJECTION64_PATH): concurrent-kernel and
// runtime-API activity only, written as fixed binary records to $KTRACE_DIR.
// Used instead of nsys, whose bundled CUPTI crashed (null write in libcupti) on the
// full canonical run. Review diagnostic only; elapsed time under tracing is not
// performance evidence.
#include <cupti.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

#pragma pack(push, 1)
struct KernelRecord {
  uint64_t start, end, lmem_total;
  uint32_t stream, correlation, name, lmem_per_thread, channel, grid_x, block_x;
  uint16_t registers;
};
struct RuntimeRecord {
  uint64_t start, end;
  uint32_t correlation, cbid, thread;
};
#pragma pack(pop)

std::mutex g_mutex;
FILE* g_kernels = nullptr;
FILE* g_runtime = nullptr;
FILE* g_names = nullptr;
std::unordered_map<std::string, uint32_t> g_name_ids;
std::atomic<uint64_t> g_dropped{0}, g_kernel_count{0}, g_runtime_count{0};
std::string g_dir;

void check(CUptiResult result, const char* what) {
  if (result != CUPTI_SUCCESS) {
    const char* text = nullptr;
    cuptiGetResultString(result, &text);
    std::fprintf(stderr, "ktrace: %s failed: %s\n", what, text ? text : "?");
    std::abort();
  }
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
  *size = static_cast<size_t>(64) << 20;
  *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, *size));
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext context, uint32_t stream_id, uint8_t* buffer,
                               size_t, size_t valid) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
        const std::string name = k->name ? k->name : "?";
        auto found = g_name_ids.find(name);
        if (found == g_name_ids.end()) {
          const uint32_t id = static_cast<uint32_t>(g_name_ids.size());
          found = g_name_ids.emplace(name, id).first;
          std::fprintf(g_names, "%u\t%s\n", id, name.c_str());
        }
        KernelRecord out{k->start, k->end, k->localMemoryTotal_v2, k->streamId,
                         k->correlationId, found->second, k->localMemoryPerThread,
                         k->channelID, static_cast<uint32_t>(k->gridX),
                         static_cast<uint32_t>(k->blockX), k->registersPerThread};
        std::fwrite(&out, sizeof(out), 1, g_kernels);
        ++g_kernel_count;
      } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(record);
        RuntimeRecord out{a->start, a->end, a->correlationId, a->cbid, a->threadId};
        std::fwrite(&out, sizeof(out), 1, g_runtime);
        ++g_runtime_count;
      }
    }
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) == CUPTI_SUCCESS) {
    g_dropped += dropped;
  }
  std::free(buffer);
}

void at_exit() {
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  std::lock_guard<std::mutex> lock(g_mutex);
  std::fclose(g_kernels);
  std::fclose(g_runtime);
  std::fclose(g_names);
  FILE* summary = std::fopen((g_dir + "/ktrace-summary.txt").c_str(), "w");
  std::fprintf(summary, "kernels %llu\nruntime %llu\ndropped %llu\n",
               static_cast<unsigned long long>(g_kernel_count.load()),
               static_cast<unsigned long long>(g_runtime_count.load()),
               static_cast<unsigned long long>(g_dropped.load()));
  std::fclose(summary);
}

}  // namespace

extern "C" int InitializeInjection(void) {
  const char* dir = std::getenv("KTRACE_DIR");
  if (dir == nullptr) std::abort();
  g_dir = dir;
  g_kernels = std::fopen((g_dir + "/kernels.bin").c_str(), "wbx");
  g_runtime = std::fopen((g_dir + "/runtime.bin").c_str(), "wbx");
  g_names = std::fopen((g_dir + "/kernel-names.tsv").c_str(), "wx");
  if (!g_kernels || !g_runtime || !g_names) std::abort();
  check(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed), "register");
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable kernel");
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME), "enable runtime");
  check(cuptiActivityFlushPeriod(2000), "flush period");
  std::atexit(at_exit);
  return 1;
}
