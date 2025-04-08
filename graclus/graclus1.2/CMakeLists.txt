cmake_minimum_required(VERSION 3.10)
project(GraclusLocalBuild)

include(ExternalProject)

# Path to the existing Graclus source
set(GRACLUS_SOURCE_DIR ${CMAKE_SOURCE_DIR})

# Where to install Graclus after building
set(GRACLUS_INSTALL_DIR ${CMAKE_BINARY_DIR}/graclus_install)

ExternalProject_Add(
    Graclus
    SOURCE_DIR ${GRACLUS_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/graclus_build
    CONFIGURE_COMMAND ""                   # No configure step; it's a plain Makefile
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    INSTALL_DIR ${GRACLUS_INSTALL_DIR}
)