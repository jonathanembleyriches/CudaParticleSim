# cmake for regular project
# cmake_minimum_required(VERSION 3.10)
# project(MyCudaAdventure CUDA)
#
# # Automatically include all .cu files from the src/ directory
# file(GLOB CUDA_SOURCES src/*.cu)
#
# # Include all source files
# set(SOURCES main.cu ${CUDA_SOURCES})
#
# # Specify the executable target
# add_executable(MyCudaProgram ${SOURCES})
#
# # Set CUDA-specific properties (optional, depending on your GPU)
# set_target_properties(MyCudaProgram PROPERTIES
#     CUDA_ARCHITECTURES "75" # Adjust to your GPU's compute capability
# )

#Cmake for my sim
cmake_minimum_required(VERSION 3.10)
project(MyCudaAdventure CUDA)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
file(GLOB CUDA_SOURCES src/*.cu)

add_executable(MyCudaProgram SimMain.cu ${CUDA_SOURCES})
set_target_properties(MyCudaProgram PROPERTIES CUDA_ARCHITECTURES "75")

# --- Find OpenGL ---
find_package(OpenGL REQUIRED)
if (OPENGL_FOUND)
  target_link_libraries(MyCudaProgram PRIVATE ${OPENGL_LIBRARIES})
endif()

# --- Find GLEW in CONFIG mode to ensure we get the vcpkg config ---
find_package(GLEW CONFIG REQUIRED)
target_link_libraries(MyCudaProgram PRIVATE GLEW::GLEW)

# --- Find FreeGLUT (also installed via vcpkg, found similarly) ---
find_package(GLUT REQUIRED)
target_link_libraries(MyCudaProgram PRIVATE GLUT::GLUT)
# target_link_libraries(MyCudaProgram PRIVATE GLUT::GLUT)

