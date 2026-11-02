# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 Guangzhou Songzi Intelligent Technology Services Co., Ltd.
#
# spektralab-native build. Plain clang++/metal through xcrun; no CMake.
#
#   make                 build/libspektralab.dylib + build/spektralab.metallib (macOS 15, arm64)
#   make ios             build/ios/libspektralab.a + build/ios/spektralab.metallib (iOS 17, arm64 device)
#                        and the simulator slice (make ios-sim)
#   make ios-sim         build/ios-sim/libspektralab.a + build/ios-sim/spektralab.metallib (iOS 17 simulator, arm64)
#   make xcframework     build/xcframework/spektralab.xcframework (device + simulator libraries, headers)
#   make test            the tests that need no resource data
#   make test RES=<dir>  those, plus the tests that read <dir>, a folder of engine resources
#                        (RES_APP=<dir> adds the checks of a second folder of profiles)
#   make objs-<module>   compile one module's objects only (util res gpu params session film print fx abi)
#   make metal-<file>    compile one kernels/<file>.metal to AIR only
#   make clean
#
# BUILD=<dir> puts every product in another directory, so several builds can run in parallel:
#   make BUILD=build_film objs-film
# `make -k` keeps compiling the other modules' objects when one file is broken.

BUILD      ?= build
BUILD_DATE ?= $(shell date -u +%Y-%m-%d)
RES        ?=
RES_APP    ?=
.DEFAULT_GOAL := all

MODULES := util res gpu params session film print fx abi
KERNELS := $(basename $(notdir $(wildcard kernels/*.metal)))

# ---- toolchains --------------------------------------------------------------------------------
MAC_CXX      := xcrun -sdk macosx clang++
MAC_METAL    := xcrun -sdk macosx metal
MAC_METALLIB := xcrun -sdk macosx metallib
IOS_CXX      := xcrun -sdk iphoneos clang++
IOS_METAL    := xcrun -sdk iphoneos metal
IOS_METALLIB := xcrun -sdk iphoneos metallib
IOS_LIBTOOL  := xcrun -sdk iphoneos libtool
SIM_CXX      := xcrun -sdk iphonesimulator clang++
SIM_METAL    := xcrun -sdk iphonesimulator metal
SIM_METALLIB := xcrun -sdk iphonesimulator metallib
SIM_LIBTOOL  := xcrun -sdk iphonesimulator libtool

# Host code: C++20, -O2, IEEE arithmetic (no fast math, no FP contraction: set-up values feed the
# tables bit for bit), hidden visibility (only spk_* is exported).
CXXFLAGS_COMMON := -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter \
                   -fno-fast-math -ffp-contract=off \
                   -fvisibility=hidden -fvisibility-inlines-hidden \
                   -Iinclude -Isrc -Ikernels \
                   -DSPK_BUILDING_LIBRARY -DSPK_BUILD_DATE=\"$(BUILD_DATE)\" \
                   -MMD -MP
OBJCXXFLAGS := -fobjc-arc
MAC_TARGET  := -arch arm64 -mmacosx-version-min=15.0
IOS_TARGET  := -arch arm64 -miphoneos-version-min=17.0
SIM_TARGET  := -arch arm64 -mios-simulator-version-min=17.0

# Kernels: safe math + precise fp32 functions, stamped for the minimum OS.
# No debug info and no embedded sources, so the metallib is byte-reproducible.
METALFLAGS_COMMON := -std=metal3.1 -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise \
                     -Ikernels -Wall
MAC_METAL_TARGET  := -mmacosx-version-min=15.0
IOS_METAL_TARGET  := -target air64-apple-ios17.0
SIM_METAL_TARGET  := -target air64-apple-ios17.0-simulator

LDFLAGS_MAC := -dynamiclib -install_name @rpath/libspektralab.dylib \
               -Wl,-exported_symbols_list,src/abi/exports.list \
               -framework Metal -framework Foundation

# ---- sources ------------------------------------------------------------------------------------
define module_sources
SRC_$(1) := $$(wildcard src/$(1)/*.cpp) $$(wildcard src/$(1)/*.mm)
MACOBJ_$(1) := $$(patsubst src/%,$(BUILD)/obj/mac/%.o,$$(SRC_$(1)))
IOSOBJ_$(1) := $$(patsubst src/%,$(BUILD)/obj/ios/%.o,$$(SRC_$(1)))
SIMOBJ_$(1) := $$(patsubst src/%,$(BUILD)/obj/ios-sim/%.o,$$(SRC_$(1)))
.PHONY: objs-$(1) ios-objs-$(1) ios-sim-objs-$(1)
objs-$(1): $$(MACOBJ_$(1))
ios-objs-$(1): $$(IOSOBJ_$(1))
ios-sim-objs-$(1): $$(SIMOBJ_$(1))
endef
$(foreach m,$(MODULES),$(eval $(call module_sources,$(m))))

MAC_OBJS := $(foreach m,$(MODULES),$(MACOBJ_$(m)))
IOS_OBJS := $(foreach m,$(MODULES),$(IOSOBJ_$(m)))
MAC_AIR  := $(patsubst %,$(BUILD)/air/mac/%.air,$(KERNELS))
IOS_AIR  := $(patsubst %,$(BUILD)/air/ios/%.air,$(KERNELS))
SIM_OBJS := $(foreach m,$(MODULES),$(SIMOBJ_$(m)))
SIM_AIR  := $(patsubst %,$(BUILD)/air/ios-sim/%.air,$(KERNELS))

DYLIB    := $(BUILD)/libspektralab.dylib
METALLIB := $(BUILD)/spektralab.metallib
IOS_LIB  := $(BUILD)/ios/libspektralab.a
IOS_MLIB := $(BUILD)/ios/spektralab.metallib
SIM_LIB  := $(BUILD)/ios-sim/libspektralab.a
SIM_MLIB := $(BUILD)/ios-sim/spektralab.metallib
XCFW     := $(BUILD)/xcframework/spektralab.xcframework

.PHONY: all ios ios-device ios-sim xcframework test clean
all: $(DYLIB) $(METALLIB)

# ---- macOS objects -------------------------------------------------------------------------------
$(BUILD)/obj/mac/%.cpp.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(MAC_CXX) $(MAC_TARGET) $(CXXFLAGS_COMMON) -c $< -o $@

$(BUILD)/obj/mac/%.mm.o: src/%.mm
	@mkdir -p $(dir $@)
	$(MAC_CXX) $(MAC_TARGET) $(CXXFLAGS_COMMON) $(OBJCXXFLAGS) -c $< -o $@

$(DYLIB): $(MAC_OBJS) src/abi/exports.list
	@mkdir -p $(dir $@)
	$(MAC_CXX) $(MAC_TARGET) $(LDFLAGS_MAC) $(MAC_OBJS) -o $@

# ---- kernels -------------------------------------------------------------------------------------
$(BUILD)/air/mac/%.air: kernels/%.metal $(wildcard kernels/*.h)
	@mkdir -p $(dir $@)
	$(MAC_METAL) $(METALFLAGS_COMMON) $(MAC_METAL_TARGET) -c $< -o $@

$(METALLIB): $(MAC_AIR)
	$(MAC_METALLIB) $(MAC_AIR) -o $@

.PHONY: $(patsubst %,metal-%,$(KERNELS))
$(patsubst %,metal-%,$(KERNELS)): metal-%: $(BUILD)/air/mac/%.air

# ---- iOS -----------------------------------------------------------------------------------------
$(BUILD)/obj/ios/%.cpp.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(IOS_CXX) $(IOS_TARGET) $(CXXFLAGS_COMMON) -c $< -o $@

$(BUILD)/obj/ios/%.mm.o: src/%.mm
	@mkdir -p $(dir $@)
	$(IOS_CXX) $(IOS_TARGET) $(CXXFLAGS_COMMON) $(OBJCXXFLAGS) -c $< -o $@

$(IOS_LIB): $(IOS_OBJS)
	@mkdir -p $(dir $@)
	rm -f $@
	$(IOS_LIBTOOL) -static -o $@ $(IOS_OBJS)

$(BUILD)/air/ios/%.air: kernels/%.metal $(wildcard kernels/*.h)
	@mkdir -p $(dir $@)
	$(IOS_METAL) $(METALFLAGS_COMMON) $(IOS_METAL_TARGET) -c $< -o $@

$(IOS_MLIB): $(IOS_AIR)
	@mkdir -p $(dir $@)
	$(IOS_METALLIB) $(IOS_AIR) -o $@

ios-device: $(IOS_LIB) $(IOS_MLIB)
ios: ios-device ios-sim

# ---- iOS simulator -------------------------------------------------------------------------------
# The simulator build compiles the TARGET_OS_SIMULATOR path of gpu.mm (private texture + blit).
$(BUILD)/obj/ios-sim/%.cpp.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(SIM_CXX) $(SIM_TARGET) $(CXXFLAGS_COMMON) -c $< -o $@

$(BUILD)/obj/ios-sim/%.mm.o: src/%.mm
	@mkdir -p $(dir $@)
	$(SIM_CXX) $(SIM_TARGET) $(CXXFLAGS_COMMON) $(OBJCXXFLAGS) -c $< -o $@

$(SIM_LIB): $(SIM_OBJS)
	@mkdir -p $(dir $@)
	rm -f $@
	$(SIM_LIBTOOL) -static -o $@ $(SIM_OBJS)

$(BUILD)/air/ios-sim/%.air: kernels/%.metal $(wildcard kernels/*.h)
	@mkdir -p $(dir $@)
	$(SIM_METAL) $(METALFLAGS_COMMON) $(SIM_METAL_TARGET) -c $< -o $@

$(SIM_MLIB): $(SIM_AIR)
	@mkdir -p $(dir $@)
	$(SIM_METALLIB) $(SIM_AIR) -o $@

ios-sim: $(SIM_LIB) $(SIM_MLIB)

# Both static libraries paired with the public header. The metallibs ship in the app's resources
# folder, one per slice.
xcframework: $(IOS_LIB) $(SIM_LIB) include/spektralab.h
	rm -rf $(XCFW)
	@mkdir -p $(dir $(XCFW))
	xcodebuild -create-xcframework -library $(IOS_LIB) -headers include -library $(SIM_LIB) -headers include \
	    -output $(XCFW)

# ---- tests ---------------------------------------------------------------------------------------
TEST_SRCS := $(wildcard tests/*.cpp)
TEST_BINS := $(patsubst tests/%.cpp,$(BUILD)/tests/%,$(TEST_SRCS))
# Unit tests link the module objects directly (internal symbols are hidden in the dylib).
TEST_LINK_OBJS := $(MACOBJ_util) $(MACOBJ_res) $(MACOBJ_gpu) $(MACOBJ_fx) $(BUILD)/obj/mac/session/tiers.cpp.o $(BUILD)/obj/mac/session/progress.cpp.o

# Per-test extra objects (target-specific TEST_EXTRA_OBJS plus a prerequisite line).
$(BUILD)/tests/test_film: TEST_EXTRA_OBJS = $(MACOBJ_film) $(MACOBJ_params)
$(BUILD)/tests/test_film: $(MACOBJ_film) $(MACOBJ_params)
$(BUILD)/tests/test_params: TEST_EXTRA_OBJS = $(MACOBJ_params)
$(BUILD)/tests/test_params: $(MACOBJ_params)

$(BUILD)/tests/%: tests/%.cpp $(TEST_LINK_OBJS)
	@mkdir -p $(dir $@) $(BUILD)/tmp
	$(MAC_CXX) $(MAC_TARGET) $(CXXFLAGS_COMMON) $(OBJCXXFLAGS) -DSPK_TEST_TMP=\"$(abspath $(BUILD))/tmp\" $< $(TEST_LINK_OBJS) $(TEST_EXTRA_OBJS) \
	    -framework Metal -framework Foundation -o $@

T := $(BUILD)/tests
PYTEST := PYTHONDONTWRITEBYTECODE=1 python3
TEST_RES := $(BUILD)/tmp/res_test
FAST_RES := $(BUILD)/tmp/res_fast

test: all $(TEST_BINS)
	$(T)/test_util
	$(T)/test_params
	$(T)/test_session $(BUILD)
	$(T)/test_gpu $(BUILD)
ifeq ($(RES),)
	$(T)/test_fx $(BUILD)
	@echo "SKIPPED: test_res, test_film, test_print, abi_smoke, session_check and the grain part of test_fx read resource data. Run: make test RES=<dir>"
else
	rm -rf $(TEST_RES) $(FAST_RES)
	cp -R $(RES) $(TEST_RES)
	cp $(METALLIB) $(TEST_RES)/spektralab.metallib
	# A resources copy whose metallib was built with FAST math: engine creation must refuse it.
	cp -R $(RES) $(FAST_RES)
	$(MAC_METAL) -std=metal3.1 -fmetal-math-mode=fast -Ikernels $(MAC_METAL_TARGET) -c kernels/probe.metal -o $(BUILD)/tmp/probe_fast.air
	$(MAC_METALLIB) $(BUILD)/tmp/probe_fast.air -o $(FAST_RES)/spektralab.metallib
	$(T)/test_fx $(BUILD) $(TEST_RES)
	$(T)/test_res $(TEST_RES) $(RES_APP)
	$(T)/test_film $(TEST_RES)
	$(T)/test_print $(TEST_RES)
	$(PYTEST) tests/abi_smoke.py $(DYLIB) $(TEST_RES) $(FAST_RES)
	$(PYTEST) tests/session_check.py $(abspath $(DYLIB)) $(abspath $(TEST_RES))
endif

clean:
	rm -rf $(BUILD)

-include $(MAC_OBJS:.o=.d) $(IOS_OBJS:.o=.d) $(SIM_OBJS:.o=.d) $(TEST_BINS:=.d)
