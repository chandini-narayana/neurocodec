# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(neurocodec_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ONNX Runtime (fetched automatically) ──────────────────────────────────
include(FetchContent)

# Detect platform for the right ONNX Runtime release
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz")
  set(ORT_HASH "SHA256=fc4b18b42f3ad62de0d3a34e67e60cf8")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-osx-universal2-1.17.3.tgz")
  set(ORT_HASH "SHA256=placeholder_mac")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-win-x64-1.17.3.zip")
  set(ORT_HASH "SHA256=placeholder_win")
endif()

FetchContent_Declare(onnxruntime URL ${ORT_URL})
FetchContent_MakeAvailable(onnxruntime)

set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")

# ── Build the inference CLI ───────────────────────────────────────────────
add_executable(nc_decode infer.cpp)

target_include_directories(nc_decode PRIVATE "${ORT_ROOT}/include")
target_link_directories(nc_decode PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(nc_decode PRIVATE onnxruntime)

if(UNIX)
  target_link_libraries(nc_decode PRIVATE dl pthread)
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS nc_decode DESTINATION bin)

# Copy ORT shared library next to the binary
if(UNIX)
  install(FILES "${ORT_ROOT}/lib/libonnxruntime.so.1.17.3" DESTINATION bin)
endif()
