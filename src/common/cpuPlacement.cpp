#include "common/cpuPlacement.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#include <windows.h> // IWYU pragma: keep

#include "common/logging/log.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace Common::CpuPlacement {

namespace {

constexpr auto RoleCount = static_cast<size_t>(Role::Count);

struct RoleCores {
	std::vector<ULONG> cpu_sets; // Both hyperthreads of the role's P-core.
	std::string        logical;  // Their logical processor numbers, for the log.
};

std::array<RoleCores, RoleCount> g_roles;
bool                             g_active = false;

const char* RoleName(Role role) {
	switch (role) {
		case Role::GpuThread: return "Thread_Gpu";
		case Role::GpuSubmit: return "GpuQueueSubmit";
		case Role::GuestMain: return "guest main thread";
		case Role::DrawSpeculator: return "DrawSpeculator";
		case Role::Count: break;
	}
	return "?";
}

} // namespace

void Initialize() {
	if (const char* text = std::getenv("KYTY_CPU_PLACEMENT");
	    text != nullptr && std::strcmp(text, "0") == 0) {
		return;
	}
	ULONG length = 0;
	GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) {
		return;
	}
	std::vector<uint8_t> buffer(length);
	if (GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
	                               length, &length, GetCurrentProcess(), 0) == FALSE) {
		return;
	}
	// By (group, core index), which a core's hyperthreads share.
	std::map<std::pair<WORD, BYTE>, RoleCores> cores;
	std::map<std::pair<WORD, BYTE>, BYTE>      efficiency;
	BYTE                                       fastest = 0;
	BYTE                                       slowest = 0xff;
	for (ULONG offset = 0; offset < length;) {
		const auto* info =
		    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Size == 0) {
			break;
		}
		offset += info->Size;
		if (info->Type != CpuSetInformation) {
			continue;
		}
		const auto& set = info->CpuSet;
		// Allocated to another process exclusively: not ours to use.
		if (set.Allocated != 0 && set.AllocatedToTargetProcess == 0) {
			continue;
		}
		const std::pair<WORD, BYTE> key {set.Group, set.CoreIndex};
		auto&                       core = cores[key];
		core.cpu_sets.push_back(set.Id);
		core.logical +=
		    (core.logical.empty() ? "" : ",") + std::to_string(set.LogicalProcessorIndex);
		efficiency[key] = set.EfficiencyClass;
		fastest         = set.EfficiencyClass > fastest ? set.EfficiencyClass : fastest;
		slowest         = set.EfficiencyClass < slowest ? set.EfficiencyClass : slowest;
	}
	if (fastest == slowest) {
		return;
	}
	size_t performance = 0;
	for (const auto& [key, value]: efficiency) {
		performance += value == fastest ? 1u : 0u;
	}
	// At least half of the P-cores stay shared, for the game's threads.
	const auto reserved = performance / 2 < RoleCount ? performance / 2 : RoleCount;
	if (reserved == 0) {
		return;
	}
	std::vector<ULONG> shared;
	size_t             role = 0;
	for (auto& [key, core]: cores) {
		if (role < reserved && efficiency[key] == fastest) {
			g_roles[role++] = core;
		} else {
			shared.insert(shared.end(), core.cpu_sets.begin(), core.cpu_sets.end());
		}
	}
	if (SetProcessDefaultCpuSets(GetCurrentProcess(), shared.data(),
	                             static_cast<ULONG>(shared.size())) == FALSE) {
		g_roles = {};
		return;
	}
	g_active = true;
}

void PlaceCurrentThread(Role role) {
	const auto index = static_cast<size_t>(role);
	if (!g_active || index >= RoleCount || g_roles[index].cpu_sets.empty()) {
		return;
	}
	const auto& core = g_roles[index];
	if (SetThreadSelectedCpuSets(GetCurrentThread(), core.cpu_sets.data(),
	                             static_cast<ULONG>(core.cpu_sets.size())) == FALSE) {
		return;
	}
	Log::WriteToConsoleAndLog("cpu-placement: " + std::string(RoleName(role)) +
	                          " runs alone on logical processors " + core.logical + "\n");
}

} // namespace Common::CpuPlacement

#else

namespace Common::CpuPlacement {

void Initialize() {}

void PlaceCurrentThread(Role /*role*/) {}

} // namespace Common::CpuPlacement

#endif
