cmake_minimum_required(VERSION 4.0)
project(Comp426A3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable first
add_executable(Comp426A3 main.cpp)

# Include GLEW headers
include_directories(/opt/homebrew/include)

# Link directories (Homebrew)
link_directories(/opt/homebrew/lib)


find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)


target_link_libraries(Comp426A3 PRIVATE
        OpenCL::OpenCL
        OpenGL::GL
        GLUT::GLUT
        /opt/homebrew/lib/libGLEW.dylib
)

