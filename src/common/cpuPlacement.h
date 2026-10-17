#ifndef KYTY_COMMON_CPU_PLACEMENT_H_
#define KYTY_COMMON_CPU_PLACEMENT_H_

#include "common/common.h"

#include <cstdint>

// Hybrid CPUs (Intel P-cores and E-cores) on Windows: the scheduler may run the emulator's few
// critical threads on E-cores, or share their P-cores with the game's spinning job workers. Each
// critical thread gets a P-core of its own, both hyperthreads; every other thread of the process
// runs on the remaining cores. Live pinning this way took Demon's Souls on an i7-14700KF from
// 11.6 to 13.2 fps. Hosts with one core type are left alone. KYTY_CPU_PLACEMENT=0 turns it off.
namespace Common::CpuPlacement {

// In order of priority: with fewer spare P-cores, the last roles get none.
enum class Role : uint8_t {
	GpuThread,      // Thread_Gpu: parses guest command buffers and records host commands.
	GpuSubmit,      // CommandScheduler's queue thread: replays recorded commands, submits.
	GuestMain,      // The game's main thread.
	DrawSpeculator, // Prepares draws ahead of Thread_Gpu.
	Count,
};

// Before the threads start (main thread, once).
void Initialize();
// From the role's own thread.
void PlaceCurrentThread(Role role);

} // namespace Common::CpuPlacement

#endif // KYTY_COMMON_CPU_PLACEMENT_H_
