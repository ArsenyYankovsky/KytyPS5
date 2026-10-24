// Replays a memory tracker call trace against a fresh MemoryTracker over the
// same guest addresses, with real page protection and upload copies, and prints
// the time per frame by call. Usage:
// memory_trace_replay <memory-trace.bin> [passes]
//
// The trace format ("KYTYMTR1") comes from a recorder in a fork whose tracker
// also arms readbacks and finalizes them later. Those calls map onto this
// tracker's API: an arm does nothing, a finalize downloads and clears the
// range, and the fork-only queries become the nearest query here.
//
// The first frame warms the tracker up (its pages start untracked) and is left
// out of the totals. Upload callbacks only copy the uploaded bytes; the
// emulator's also record Vulkan copies. The checksums cover every range handed
// to an upload callback and every protection change, so two builds that replay
// the same trace with equal checksums made the same uploads and protections.

#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max

namespace {

using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;

enum class Op : uint8_t {
  Frame,
  CpuModified,
  GpuModified,
  GpuUnmarked,
  Untrack,
  Invalidate,
  Upload,
  UploadWritten,
  Download,
  DownloadClear,
  UnarmedDownload,
  QueryReadback,
  Arm,
  Finalize,
  QueryCpu,
  QueryGpu,
  QueryOnlyCpu,
  Count,
};

struct Event {
  uint64_t vaddr = 0;
  uint64_t size = 0;
  uint32_t extra = 0;
  Op op = Op::Frame;
  uint8_t gpu_thread = 0;
  uint16_t reserved = 0;
};
static_assert(sizeof(Event) == 24);

uint64_t g_protect_calls = 0;
uint64_t g_protect_failures = 0;
uint64_t g_protect_hash = 0;

constexpr std::array<const char *, static_cast<size_t>(Op::Count)> OpNames{
    "frame",         "cpu-modified",   "gpu-modified",     "gpu-unmarked",
    "untrack",       "invalidate",     "upload",           "upload-written",
    "download",      "download-clear", "unarmed-download", "query-readback",
    "arm",           "finalize",       "query-cpu",        "query-gpu",
    "query-only-cpu"};

uint64_t Mix(uint64_t hash, uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  return hash;
}

[[noreturn]] void Fail(const char *text) {
  std::fprintf(stderr, "memory_trace_replay: %s\n", text);
  std::exit(1);
}

std::vector<Event> ReadTrace(const char *path, uint32_t &frames,
                             uint64_t &gpu_ns) {
  std::FILE *file = std::fopen(path, "rb");
  if (file == nullptr) {
    Fail("cannot open the trace");
  }
  char magic[8]{};
  uint32_t event_size = 0;
  uint64_t count = 0;
  if (std::fread(magic, 1, 8, file) != 8 ||
      std::memcmp(magic, "KYTYMTR1", 8) != 0 ||
      std::fread(&event_size, sizeof(event_size), 1, file) != 1 ||
      std::fread(&frames, sizeof(frames), 1, file) != 1 ||
      std::fread(&count, sizeof(count), 1, file) != 1 ||
      std::fread(&gpu_ns, sizeof(gpu_ns), 1, file) != 1 ||
      event_size != sizeof(Event)) {
    Fail("not a memory trace of this version");
  }
  std::vector<Event> events(count);
  if (std::fread(events.data(), sizeof(Event), count, file) != count) {
    Fail("truncated trace");
  }
  std::fclose(file);
  return events;
}

bool Touches(Op op) { return op != Op::Frame && op != Op::Untrack; }

// Commits every 64 KiB granule the trace's calls touch, at its guest address.
std::vector<std::pair<uint64_t, uint64_t>>
MapGuestMemory(const std::vector<Event> &events) {
  constexpr uint64_t Granule = 64 * 1024;
  std::map<uint64_t, uint64_t> runs; // begin -> end, merged
  for (const auto &event : events) {
    if (!Touches(event.op) || event.size == 0) {
      continue;
    }
    const auto begin = event.vaddr / Granule * Granule;
    const auto end =
        (event.vaddr + event.size + Granule - 1) / Granule * Granule;
    auto next = runs.upper_bound(begin);
    if (next != runs.begin() && std::prev(next)->second >= begin) {
      --next;
    }
    auto merged_begin = begin;
    auto merged_end = end;
    while (next != runs.end() && next->first <= merged_end) {
      merged_begin = std::min(merged_begin, next->first);
      merged_end = std::max(merged_end, next->second);
      next = runs.erase(next);
    }
    runs[merged_begin] = merged_end;
  }
  std::vector<std::pair<uint64_t, uint64_t>> mapped;
  uint64_t total = 0;
  for (const auto &[begin, end] : runs) {
    void *memory = VirtualAlloc(reinterpret_cast<void *>(begin), end - begin,
                                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (memory != reinterpret_cast<void *>(begin)) {
      std::fprintf(stderr, "cannot map 0x%llx-0x%llx\n",
                   static_cast<unsigned long long>(begin),
                   static_cast<unsigned long long>(end));
      continue;
    }
    mapped.emplace_back(begin, end);
    total += end - begin;
  }
  std::printf("mapped %zu guest ranges, %.1f MiB\n", mapped.size(),
              static_cast<double>(total) / (1024.0 * 1024.0));
  return mapped;
}

uint64_t g_copy_faults = 0;

// The replay's protections follow the tracker's, so an upload should only read
// readable pages.
void CopyGuarded(void *destination, const void *source, size_t size) {
  __try {
    std::memcpy(destination, source, size);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    g_copy_faults++;
  }
}

struct OpTotals {
  uint64_t count = 0;
  uint64_t ns = 0;
  uint64_t bytes = 0;
};

void RunPass(const std::vector<Event> &events, bool print) {
  PageManager page_manager;
  MemoryTracker tracker(page_manager);
  auto staging = std::make_unique<uint8_t[]>(64 * 1024 * 1024);
  size_t cursor = 0;
  uint64_t copied = 0;
  uint64_t upload_hash = 0;

  std::array<OpTotals, static_cast<size_t>(Op::Count)> gpu{};
  std::array<OpTotals, static_cast<size_t>(Op::Count)> other{};
  uint32_t frame = 0;
  std::chrono::steady_clock::time_point measured_start{};
  const auto upload = [&](uint64_t address, uint64_t size) noexcept {
    upload_hash = Mix(Mix(upload_hash, address), size);
    for (uint64_t done = 0; done < size;) {
      const auto bytes =
          std::min<uint64_t>(size - done, 64 * 1024 * 1024 - cursor);
      CopyGuarded(staging.get() + cursor,
                  reinterpret_cast<const void *>(address + done), bytes);
      cursor = (cursor + bytes) % (64 * 1024 * 1024);
      done += bytes;
    }
    copied += size;
  };
  const auto ignore = [](uint64_t, uint64_t) noexcept {};
  for (const auto &event : events) {
    if (event.op == Op::Frame) {
      if (++frame == 1) {
        measured_start = std::chrono::steady_clock::now();
      }
      continue;
    }
    const auto start = std::chrono::steady_clock::now();
    switch (event.op) {
    case Op::CpuModified:
      tracker.MarkRegionAsCpuModified(event.vaddr, event.size);
      break;
    case Op::GpuModified:
      tracker.MarkRegionAsGpuModified(event.vaddr, event.size);
      break;
    case Op::GpuUnmarked:
      tracker.UnmarkRegionAsGpuModified(event.vaddr, event.size);
      break;
    case Op::Untrack:
      // The emulator downloads GPU-written bytes first; calls made before the
      // capture may have left some GPU-dirty in the replay only.
      tracker.UnmarkRegionAsGpuModified(event.vaddr, event.size);
      tracker.UntrackMemory(event.vaddr, event.size);
      break;
    case Op::Invalidate:
      tracker.InvalidateRegion(event.vaddr, event.size, [] {});
      break;
    case Op::Upload:
    case Op::UploadWritten:
      tracker.ForEachUploadRange(event.vaddr, event.size,
                                 event.op == Op::UploadWritten, upload,
                                 []() noexcept {});
      break;
    case Op::Download:
    case Op::UnarmedDownload:
      tracker.ForEachDownloadRange<false>(event.vaddr, event.size, ignore);
      break;
    case Op::DownloadClear:
    case Op::Finalize:
      tracker.ForEachDownloadRange<true>(event.vaddr, event.size, ignore);
      break;
    case Op::QueryCpu:
    case Op::QueryOnlyCpu:
      (void)tracker.IsRegionCpuModified(event.vaddr, event.size);
      break;
    case Op::QueryGpu:
    case Op::QueryReadback:
      (void)tracker.IsRegionGpuModified(event.vaddr, event.size);
      break;
    case Op::Arm:
    case Op::Frame:
    case Op::Count:
      break;
    }
    if (frame == 0) {
      continue; // Warm-up.
    }
    const auto ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    auto &totals = event.gpu_thread != 0 ? gpu : other;
    auto &entry = totals[static_cast<size_t>(event.op)];
    entry.count++;
    entry.ns += ns;
    entry.bytes += event.size;
  }
  const auto wall_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - measured_start)
          .count());
  // PageManager requires every page untracked before it goes.
  uint64_t lowest = UINT64_MAX;
  uint64_t highest = 0;
  for (const auto &event : events) {
    if (event.op != Op::Frame && event.size != 0) {
      lowest = std::min(lowest, event.vaddr);
      highest = std::max(highest, event.vaddr + event.size);
    }
  }
  if (lowest < highest) {
    tracker.UnmarkRegionAsGpuModified(lowest, highest - lowest);
    tracker.UntrackMemory(lowest, highest - lowest);
  }
  const double measured = frame > 1 ? frame - 1 : 1;
  const auto sum = [](const decltype(gpu) &totals) {
    uint64_t all = 0;
    for (const auto &entry : totals) {
      all += entry.ns;
    }
    return all;
  };
  std::printf("pass: Thread_Gpu calls %.3f ms a frame, all calls %.3f ms a "
              "frame, replay wall %.3f ms a frame\n",
              static_cast<double>(sum(gpu)) / 1e6 / measured,
              static_cast<double>(sum(gpu) + sum(other)) / 1e6 / measured,
              static_cast<double>(wall_ns) / 1e6 / measured);
  if (!print) {
    return;
  }
  const auto report = [&](const char *name, const decltype(gpu) &totals) {
    std::printf("%s: %.3f ms a frame\n", name,
                static_cast<double>(sum(totals)) / 1e6 / measured);
    for (size_t op = 0; op < totals.size(); op++) {
      if (totals[op].count == 0) {
        continue;
      }
      std::printf("  %-17s n/frame=%-8.0f %8.3f ms/frame %6.2f us/call %8.1f "
                  "KiB/frame\n",
                  OpNames[op], static_cast<double>(totals[op].count) / measured,
                  static_cast<double>(totals[op].ns) / 1e6 / measured,
                  static_cast<double>(totals[op].ns) / 1e3 /
                      static_cast<double>(totals[op].count),
                  static_cast<double>(totals[op].bytes) / 1024.0 / measured);
    }
  };
  std::printf("frames=%u (first one warms up)\n", frame);
  report("Thread_Gpu calls", gpu);
  report("other threads' calls", other);
  std::printf(
      "uploaded %.1f MiB a frame; protection calls %.0f a frame (%llu failed); "
      "%llu upload copies faulted\n",
      static_cast<double>(copied) / (1024.0 * 1024.0) / measured,
      static_cast<double>(g_protect_calls) / measured,
      static_cast<unsigned long long>(g_protect_failures),
      static_cast<unsigned long long>(g_copy_faults));
  std::printf("checksums: uploads 0x%016llx, protections 0x%016llx\n",
              static_cast<unsigned long long>(upload_hash),
              static_cast<unsigned long long>(g_protect_hash));
}

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  DWORD protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  g_protect_calls++;
  g_protect_hash = Mix(Mix(Mix(g_protect_hash, vaddr), size), protection);
  if (VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                     &old_protection) == 0) {
    // Unmapped by the replay (outside every recorded range): the tracker's
    // state still counts.
    g_protect_failures++;
  }
  return true;
}

} // namespace Libs::LibKernel::Memory

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: memory_trace_replay <memory-trace.bin> [passes]\n");
    return 1;
  }
  uint32_t frames = 0;
  uint64_t gpu_ns = 0;
  const auto events = ReadTrace(argv[1], frames, gpu_ns);
  std::printf("%zu calls over %u frames; in the game Thread_Gpu spent %.3f ms "
              "a frame in them (callbacks included)\n",
              events.size(), frames,
              static_cast<double>(gpu_ns) / 1e6 / std::max(frames, 1u));
  const auto mapped = MapGuestMemory(events);
  const int passes = argc >= 3 ? std::max(1, std::atoi(argv[2])) : 1;
  for (int pass = 0; pass < passes; pass++) {
    g_protect_calls = 0;
    g_protect_failures = 0;
    g_protect_hash = 0;
    g_copy_faults = 0;
    // Each pass starts from writable memory and a fresh tracker.
    for (const auto &[begin, end] : mapped) {
      DWORD old_protection = 0;
      VirtualProtect(reinterpret_cast<void *>(begin), end - begin,
                     PAGE_READWRITE, &old_protection);
    }
    RunPass(events, pass == passes - 1);
  }
  return 0;
}
