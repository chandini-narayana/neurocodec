# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(neurocodec_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── zlib (the .nec v2 payload is DEFLATE-compressed) ──────────────────────
find_package(ZLIB REQUIRED)

# ── ONNX Runtime ──────────────────────────────────────────────────────────
# Either point at an existing install:   cmake .. -DORT_ROOT=/path/to/onnxruntime
# or let CMake download the official prebuilt release for this platform.
set(ORT_VERSION "1.17.3" CACHE STRING "ONNX Runtime version to download")

if(NOT ORT_ROOT)
  set(ORT_BASE "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ORT_URL "${ORT_BASE}/onnxruntime-linux-x64-${ORT_VERSION}.tgz")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_URL "${ORT_BASE}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ORT_URL "${ORT_BASE}/onnxruntime-osx-universal2-${ORT_VERSION}.tgz")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ORT_URL "${ORT_BASE}/onnxruntime-win-x64-${ORT_VERSION}.zip")
  else()
    message(FATAL_ERROR
      "No prebuilt ONNX Runtime for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
      "Pass -DORT_ROOT=<path to an onnxruntime install>.")
  endif()

  include(FetchContent)
  FetchContent_Declare(onnxruntime URL ${ORT_URL})
  FetchContent_MakeAvailable(onnxruntime)
  set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")
endif()

# ── Build the inference CLI ───────────────────────────────────────────────
add_executable(nc_decode infer.cpp)

# Prebuilt archives ship headers directly under include/
target_include_directories(nc_decode PRIVATE "${ORT_ROOT}/include")
target_link_directories(nc_decode PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(nc_decode PRIVATE onnxruntime ZLIB::ZLIB)

if(UNIX)
  target_link_libraries(nc_decode PRIVATE dl pthread)
  # Find libonnxruntime next to / relative to the binary at run time
  set_target_properties(nc_decode PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN")
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS nc_decode DESTINATION bin)

if(WIN32)
  install(FILES "${ORT_ROOT}/lib/onnxruntime.dll" DESTINATION bin)
else()
  file(GLOB ORT_LIBS "${ORT_ROOT}/lib/libonnxruntime*")
  install(FILES ${ORT_LIBS} DESTINATION bin)
endif()
