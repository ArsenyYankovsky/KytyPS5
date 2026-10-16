#include "loader/redZonePatcher.h"
#include "loader/x64InstructionEmulator.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#include <xbyak/xbyak.h>

// On a host without SSE4a the loader rewrites immediate EXTRQ/INSERTQ into SSE2 trampolines and
// MONITORX/MWAITX into NOP/PAUSE. This test rewrites every immediate length/index pair and checks
// the result against the exception handler's semantics, that nothing traps any more, and that the
// guest red zone and RFLAGS survive. It forces the rewrite, so it runs on AMD hosts too, where the
// untouched copies execute natively and also check the rewritten low quadword against hardware.

namespace {

using namespace Xbyak::util;

constexpr size_t kModuleSize     = 4 * 1024 * 1024;
constexpr size_t kTrampolineSize = 8 * 1024 * 1024;
constexpr size_t kBlobAlignment  = 16;

std::atomic<int> g_illegal_hits {0};

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "windowsSse4aRewriteTests: failed: %s\n", text);
		std::fflush(stderr);
		std::abort();
	}
}

LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
	if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	g_illegal_hits.fetch_add(1, std::memory_order_relaxed);
	return Loader::X64InstructionEmulator::TryEmulate(info->ContextRecord)
	           ? EXCEPTION_CONTINUE_EXECUTION
	           : EXCEPTION_CONTINUE_SEARCH;
}

int IllegalHits() {
	return g_illegal_hits.load(std::memory_order_relaxed);
}

bool HostHasSse4a() {
	int regs[4] {};
	__cpuid(regs, static_cast<int>(0x80000000u));
	if (static_cast<unsigned>(regs[0]) < 0x80000001u) {
		return false;
	}
	__cpuid(regs, static_cast<int>(0x80000001u));
	return (regs[2] & (1 << 6)) != 0;
}

struct Quadwords {
	uint64_t lo;
	uint64_t hi;
};

// What a blob reports: the destination register, the two red-zone canaries, CF and ZF.
struct BlobOutput {
	Quadwords xmm;
	uint64_t  canary_top;
	uint64_t  canary_bottom;
	uint8_t   carry;
	uint8_t   zero;
	uint8_t   padding[6];
};

using BlobFn = void (*)(BlobOutput* out, const Quadwords* dest, const Quadwords* src,
                        uint64_t canary);

struct Sse4aCase {
	bool      insert;
	uint8_t   dest;
	uint8_t   src;
	uint8_t   length;
	uint8_t   index;
	Quadwords dest_value;
	Quadwords src_value;
	size_t    offset;
};

std::vector<uint8_t> EncodeImmediate(const Sse4aCase& test) {
	// EXTRQ keeps its operand in ModRM.rm with reg 0; INSERTQ puts the destination in reg and the
	// source in rm.
	const uint8_t reg = test.insert ? test.dest : 0;
	const uint8_t rm  = test.insert ? test.src : test.dest;
	const uint8_t rex = 0x40 | ((reg & 8) >> 1) | ((rm & 8) >> 3);

	std::vector<uint8_t> bytes {static_cast<uint8_t>(test.insert ? 0xf2 : 0x66)};
	if (rex != 0x40) {
		bytes.push_back(rex);
	}
	bytes.insert(bytes.end(), {0x0f, 0x78, static_cast<uint8_t>(0xc0 | (reg & 7) << 3 | (rm & 7)),
	                           test.length, test.index});
	return bytes;
}

// extrq xmm(dest), xmm(src): the field comes from src[5:0] (length) and src[13:8] (index).
std::vector<uint8_t> EncodeRegisterExtrq(uint8_t dest, uint8_t src) {
	const uint8_t        rex = 0x40 | ((dest & 8) >> 1) | ((src & 8) >> 3);
	std::vector<uint8_t> bytes {0x66};
	if (rex != 0x40) {
		bytes.push_back(rex);
	}
	bytes.insert(bytes.end(),
	             {0x0f, 0x79, static_cast<uint8_t>(0xc0 | (dest & 7) << 3 | (src & 7))});
	return bytes;
}

// The blob keeps xmm6-xmm15 intact for the caller, puts canaries at both ends of the red zone,
// sets CF and ZF, runs the instruction, and reports everything back.
void EmitBlob(Xbyak::CodeGenerator& gen, uint8_t dest, uint8_t src,
              const std::vector<uint8_t>& instruction) {
	const Xbyak::Xmm dest_xmm(dest);
	const Xbyak::Xmm src_xmm(src);
	gen.sub(rsp, 40);
	gen.movdqu(ptr[rsp], dest_xmm);
	gen.movdqu(ptr[rsp + 16], src_xmm);
	gen.mov(ptr[rsp - 8], r9);
	gen.mov(ptr[rsp - 128], r9);
	gen.movdqu(src_xmm, ptr[r8]);
	gen.movdqu(dest_xmm, ptr[rdx]);
	gen.xor_(eax, eax);
	gen.stc();
	gen.db(instruction.data(), instruction.size());
	gen.setc(al);
	gen.setz(ah);
	gen.movdqu(ptr[rcx], dest_xmm);
	gen.mov(r10, ptr[rsp - 8]);
	gen.mov(ptr[rcx + offsetof(BlobOutput, canary_top)], r10);
	gen.mov(r10, ptr[rsp - 128]);
	gen.mov(ptr[rcx + offsetof(BlobOutput, canary_bottom)], r10);
	gen.mov(ptr[rcx + offsetof(BlobOutput, carry)], ax);
	gen.movdqu(src_xmm, ptr[rsp + 16]);
	gen.movdqu(dest_xmm, ptr[rsp]);
	gen.add(rsp, 40);
	gen.ret();
	gen.align(kBlobAlignment);
}

Quadwords Reference(const Sse4aCase& test) {
	const uint32_t index  = test.index & 0x3fu;
	uint32_t       length = test.length & 0x3fu;
	if (length == 0) {
		length = 64;
	}
	if (length > 64 - index) {
		length = 64 - index;
	}
	const uint64_t mask = length == 64 ? ~uint64_t {0} : (uint64_t {1} << length) - 1;
	if (!test.insert) {
		return {(test.dest_value.lo >> index) & mask, 0};
	}
	const uint64_t src = test.dest == test.src ? test.dest_value.lo : test.src_value.lo;
	return {(test.dest_value.lo & ~(mask << index)) | ((src & mask) << index), test.dest_value.hi};
}

uint64_t NextRandom(uint64_t& state) {
	state ^= state << 13;
	state ^= state >> 7;
	state ^= state << 17;
	return state;
}

struct Region {
	uint8_t* module;
	uint8_t* trampolines;
	uint8_t* untouched;
};

Region AllocateRegion() {
	// One reservation keeps the trampolines within rel32 reach of the code, as in the loader.
	auto* base =
	    static_cast<uint8_t*>(VirtualAlloc(nullptr, kModuleSize * 2 + kTrampolineSize,
	                                       MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
	Check(base != nullptr, "allocate executable test memory");
	return {base, base + kModuleSize, base + kModuleSize + kTrampolineSize};
}

void TestSse4aRewrites() {
	constexpr std::array<uint8_t, 6>                kExtrqRegisters {0, 2, 7, 8, 10, 15};
	constexpr std::array<std::array<uint8_t, 2>, 8> kInsertqRegisters {
	    {{2, 0}, {0, 1}, {1, 0}, {0, 0}, {10, 9}, {15, 15}, {3, 12}, {8, 2}}};

	const Region           region = AllocateRegion();
	Xbyak::CodeGenerator   gen(kModuleSize, region.module);
	std::vector<Sse4aCase> cases;
	uint64_t               random = 0x9e3779b97f4a7c15ull;

	for (int insert = 0; insert < 2; ++insert) {
		for (uint32_t length = 0; length < 64; ++length) {
			for (uint32_t index = 0; index < 64; ++index) {
				const size_t selector = length * 64 + index;
				Sse4aCase    test {
				    .insert     = insert != 0,
				    .length     = static_cast<uint8_t>(length),
				    .index      = static_cast<uint8_t>(index),
				    .dest_value = {NextRandom(random), NextRandom(random)},
				    .src_value  = {NextRandom(random), NextRandom(random)},
				    .offset     = gen.getSize(),
				};
				if (test.insert) {
					const auto& pair = kInsertqRegisters[selector % kInsertqRegisters.size()];
					test.dest        = pair[0];
					test.src         = pair[1];
				} else {
					test.dest = kExtrqRegisters[selector % kExtrqRegisters.size()];
					test.src  = test.dest;
				}
				EmitBlob(gen, test.dest, test.src, EncodeImmediate(test));
				cases.push_back(test);
			}
		}
	}

	// MONITORX on the canary address, then MWAITX without a timer.
	const size_t wait_offset = gen.getSize();
	gen.mov(rax, rcx);
	gen.xor_(ecx, ecx);
	gen.xor_(edx, edx);
	const size_t monitorx_offset = gen.getSize();
	gen.db(0x0f);
	gen.db(0x01);
	gen.db(0xfa);
	gen.xor_(eax, eax);
	gen.xor_(ecx, ecx);
	const size_t mwaitx_offset = gen.getSize();
	gen.db(0x0f);
	gen.db(0x01);
	gen.db(0xfb);
	gen.ret();
	gen.align(kBlobAlignment);

	// The register form is not rewritten; it still goes through the exception handler.
	const size_t register_offset = gen.getSize();
	EmitBlob(gen, 2, 3, EncodeRegisterExtrq(2, 3));

	const size_t code_size = gen.getSize();
	std::memcpy(region.untouched, region.module, code_size);

	std::vector<uintptr_t> function_starts;
	for (const auto& test: cases) {
		function_starts.push_back(reinterpret_cast<uintptr_t>(region.module + test.offset));
	}
	function_starts.push_back(reinterpret_cast<uintptr_t>(region.module + wait_offset));
	function_starts.push_back(reinterpret_cast<uintptr_t>(region.module + register_offset));

	Loader::RegisterRedZonePatchModule(region.module, kModuleSize, region.trampolines,
	                                   kTrampolineSize);
	const auto result = Loader::PatchGuestInstructions(
	    reinterpret_cast<uint64_t>(region.module), code_size, function_starts, false, false, true);
	Check(result.rewritten_amd_instruction_count == cases.size() + 2,
	      "every immediate EXTRQ/INSERTQ and the MONITORX/MWAITX pair are rewritten");
	Check(result.host_emulated_instruction_count == 1,
	      "the register-form EXTRQ keeps red-zone protection for the exception handler");

	constexpr uint8_t kNop[]   = {0x0f, 0x1f, 0x00};
	constexpr uint8_t kPause[] = {0xf3, 0x90, 0x90};
	Check(std::memcmp(region.module + monitorx_offset, kNop, sizeof(kNop)) == 0,
	      "MONITORX becomes a three-byte NOP");
	Check(std::memcmp(region.module + mwaitx_offset, kPause, sizeof(kPause)) == 0,
	      "MWAITX becomes PAUSE");

	const bool     native_sse4a = HostHasSse4a();
	const uint64_t canary       = 0x5ca1ab1e0ddba11ull;
	for (const auto& test: cases) {
		BlobOutput out {};
		const int  before = IllegalHits();
		reinterpret_cast<BlobFn>(region.module + test.offset)(&out, &test.dest_value,
		                                                      &test.src_value, canary);
		Check(IllegalHits() == before, "a rewritten EXTRQ/INSERTQ does not trap");

		const Quadwords expected = Reference(test);
		if (out.xmm.lo != expected.lo || out.xmm.hi != expected.hi) {
			std::fprintf(stderr,
			             "%s dest=xmm%u src=xmm%u length=%u index=%u: got %016llx:%016llx, "
			             "expected %016llx:%016llx\n",
			             test.insert ? "insertq" : "extrq", test.dest, test.src, test.length,
			             test.index, out.xmm.hi, out.xmm.lo, expected.hi, expected.lo);
			Check(false, "rewritten EXTRQ/INSERTQ matches the exception handler");
		}
		Check(out.canary_top == canary && out.canary_bottom == canary,
		      "the rewrite leaves the red zone intact");
		Check(out.carry == 1 && out.zero == 1, "the rewrite leaves RFLAGS intact");

		// Hardware leaves the result undefined when the field runs past bit 63; the exception
		// handler and the rewrite clamp it.
		const uint32_t length = test.length == 0 ? 64 : test.length;
		if (native_sse4a && length + test.index > 64) {
			continue;
		}
		BlobOutput untouched {};
		reinterpret_cast<BlobFn>(region.untouched + test.offset)(&untouched, &test.dest_value,
		                                                         &test.src_value, canary);
		Check(untouched.xmm.lo == out.xmm.lo,
		      native_sse4a ? "rewritten low quadword matches native SSE4a"
		                   : "rewritten low quadword matches the exception handler");
	}

	const int  before_wait = IllegalHits();
	BlobOutput wait_out {};
	reinterpret_cast<void (*)(BlobOutput*)>(region.module + wait_offset)(&wait_out);
	Check(IllegalHits() == before_wait, "rewritten MONITORX/MWAITX do not trap");

	const Quadwords register_dest {0x0123456789abcdefull, 0xfedcba9876543210ull};
	const Quadwords register_field {0x0810, 0}; // length 16, index 8
	BlobOutput      register_out {};
	const int       before_register = IllegalHits();
	reinterpret_cast<BlobFn>(region.module + register_offset)(&register_out, &register_dest,
	                                                          &register_field, canary);
	Check(native_sse4a || IllegalHits() > before_register,
	      "the register-form EXTRQ still reaches the exception handler");
	Check(register_out.xmm.lo == 0xabcd, "extrq xmm2, xmm3 extracts bits [23:8]");
	Check(register_out.canary_top == canary && register_out.canary_bottom == canary,
	      "the emulated register-form EXTRQ leaves the red zone intact");
}

} // namespace

int main() {
	Check(AddVectoredExceptionHandler(1, Handler) != nullptr, "install vectored exception handler");
	TestSse4aRewrites();
	std::printf("windowsSse4aRewriteTests: all passed\n");
	return 0;
}
