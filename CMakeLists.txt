cmake_minimum_required(VERSION 3.25)
project(gsic VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GSIC_ENABLE_GPU "Build the OpenGL compute backend" ON)
option(GSIC_ENABLE_TRACY "Enable Tracy profiler instrumentation (requires 'tracy' vcpkg feature)" OFF)
option(GSIC_BUILD_TESTS "Build unit tests" ON)

# Link-time optimization for release binaries.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
endif()

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(nfd CONFIG REQUIRED)
find_package(Stb REQUIRED)
if(GSIC_ENABLE_TRACY)
  find_package(Tracy CONFIG REQUIRED)
endif()

# ---------------------------------------------------------------- core library
add_library(gsic_core STATIC
  src/core/image.cpp
  src/core/renderer.cpp
  src/core/trainer.cpp
  src/core/codec.cpp
  src/core/metrics.cpp
  src/core/thread_pool.cpp
  src/core/kernels_dispatch.cpp
  src/core/kernels_baseline.cpp
)
target_include_directories(gsic_core PUBLIC src PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(gsic_core PUBLIC zstd::libzstd)

# The hot kernels are compiled once per instruction set and selected at
# runtime, so one binary runs well on both old and new machines. Detection
# asks the compiler rather than trusting CMAKE_SYSTEM_PROCESSOR, whose value
# is inconsistent across 32-bit toolchains (x86, i686, i386, X86).
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
int main() { return 0; }
#else
#error not x86
#endif
" GSIC_TARGET_IS_X86)

if(GSIC_TARGET_IS_X86)
  if(MSVC)
    set(GSIC_AVX2_FLAGS /arch:AVX2)
  else()
    set(GSIC_AVX2_FLAGS -mavx2 -mfma)
  endif()
  target_sources(gsic_core PRIVATE src/core/kernels_avx2.cpp)
  set_source_files_properties(src/core/kernels_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "${GSIC_AVX2_FLAGS}")
  target_compile_definitions(gsic_core PRIVATE GSIC_HAVE_AVX2_TU=1)
endif()

# 32-bit builds: SSE2 is guaranteed on x86-64 but not on 32-bit x86, where the
# default is x87. Asking for SSE2 costs nothing (every CPU since 2001 has it)
# and lets the floating point code vectorize. This is applied per source
# rather than to the target, so it does not collide with the AVX2 kernel's
# own architecture flag.
if(GSIC_TARGET_IS_X86 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
  if(MSVC)
    set(GSIC_SSE2_FLAGS /arch:SSE2)
  else()
    set(GSIC_SSE2_FLAGS -msse2 -mfpmath=sse)
  endif()
  get_target_property(gsic_core_sources gsic_core SOURCES)
  list(REMOVE_ITEM gsic_core_sources src/core/kernels_avx2.cpp)
  set_property(SOURCE ${gsic_core_sources} APPEND
               PROPERTY COMPILE_OPTIONS ${GSIC_SSE2_FLAGS})
endif()

# Fast-math only on the numeric kernels; the rest of the code (codec, I/O)
# keeps strict IEEE behaviour.
if(MSVC)
  set(GSIC_FAST_MATH /fp:fast)
else()
  set(GSIC_FAST_MATH -ffast-math -fno-finite-math-only)
endif()
foreach(f src/core/kernels_baseline.cpp src/core/kernels_avx2.cpp)
  set_property(SOURCE ${f} APPEND PROPERTY COMPILE_OPTIONS ${GSIC_FAST_MATH})
endforeach()

if(GSIC_ENABLE_GPU)
  target_sources(gsic_core PRIVATE src/core/gpu_backend.cpp)
  target_compile_definitions(gsic_core PUBLIC GSIC_ENABLE_GPU=1)
  target_link_libraries(gsic_core PUBLIC glad::glad glfw)
endif()

if(GSIC_ENABLE_TRACY)
  target_compile_definitions(gsic_core PUBLIC GSIC_ENABLE_TRACY=1)
  target_link_libraries(gsic_core PUBLIC Tracy::TracyClient)
endif()

if(MSVC)
  target_compile_options(gsic_core PRIVATE /W4 /permissive-)
  # MSVC needs this for correct feature detection macros.
  target_compile_options(gsic_core PUBLIC /Zc:__cplusplus)
else()
  target_compile_options(gsic_core PRIVATE -Wall -Wextra)
endif()

# ------------------------------------------------------------------------- cli
add_executable(gsic src/cli/main.cpp)
target_link_libraries(gsic PRIVATE gsic_core)

# ------------------------------------------------------------------------- gui
add_executable(gsic-gui
  src/app/main.cpp
  src/app/app.cpp
)
if(WIN32)
  target_sources(gsic-gui PRIVATE src/app/windows.rc)
  # GUI subsystem (no console window) while keeping a standard main().
  set_target_properties(gsic-gui PROPERTIES WIN32_EXECUTABLE ON)
  target_link_options(gsic-gui PRIVATE /ENTRY:mainCRTStartup)
  # A 32-bit process gets 2 GB of address space by default; this raises it to
  # 4 GB when the host is 64-bit, which materially widens the image sizes a
  # 32-bit build can handle.
  if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_link_options(gsic-gui PRIVATE /LARGEADDRESSAWARE)
    target_link_options(gsic PRIVATE /LARGEADDRESSAWARE)
  endif()
endif()
target_link_libraries(gsic-gui PRIVATE gsic_core imgui::imgui glfw glad::glad nfd::nfd)

# ------------------------------------------------------------------ logo tool
# Regenerates the icon and Store assets from fixed constants. Not built by
# default; run `cmake --build <dir> --target logo` after changing the mark.
add_executable(gsic_make_logo EXCLUDE_FROM_ALL tools/make_logo.cpp)
target_link_libraries(gsic_make_logo PRIVATE gsic_core)
target_include_directories(gsic_make_logo PRIVATE ${Stb_INCLUDE_DIR})

add_custom_target(logo
  COMMAND gsic_make_logo "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS gsic_make_logo
  COMMENT "Regenerating icon and Store assets")

# ----------------------------------------------------------------------- tests
if(GSIC_BUILD_TESTS)
  enable_testing()
  add_executable(gsic_tests tests/test_core.cpp)
  target_link_libraries(gsic_tests PRIVATE gsic_core)
  add_test(NAME core COMMAND gsic_tests)
endif()
