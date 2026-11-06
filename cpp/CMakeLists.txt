cmake_minimum_required(VERSION 3.13)
project(fireredpunc_axera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(BSP_MSP_DIR "${CMAKE_SOURCE_DIR}/toolchains/ax650n_bsp_sdk/msp/out" CACHE PATH
    "AX650 BSP msp/out directory containing include/ and lib/")
if(NOT EXISTS "${BSP_MSP_DIR}/include/ax_engine_api.h")
  message(FATAL_ERROR
          "AX650 BSP not found: ${BSP_MSP_DIR}. Set BSP_MSP_DIR to msp/out.")
endif()

set(TARGET_NAME fireredpunc_ax650)
add_executable(${TARGET_NAME}
  main.cpp
  src/engine_wrapper.cpp
)
target_include_directories(${TARGET_NAME} PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${BSP_MSP_DIR}/include
)
target_link_directories(${TARGET_NAME} PRIVATE ${BSP_MSP_DIR}/lib)
target_compile_definitions(${TARGET_NAME} PRIVATE DEFAULT_MODELS_DIR="models")
target_compile_options(${TARGET_NAME} PRIVATE
  -O3 -Wall -Wextra -Wpedantic
  -fvisibility=hidden -fdata-sections -ffunction-sections
)
target_link_libraries(${TARGET_NAME} PRIVATE ax_sys ax_engine)
target_link_options(${TARGET_NAME} PRIVATE
  -Wl,--enable-new-dtags -Wl,--allow-shlib-undefined
)
set_target_properties(${TARGET_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)
