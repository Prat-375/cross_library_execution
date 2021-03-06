# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gurumurt/Desktop/gpu_libraries_prototype

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gurumurt/Desktop/gpu_libraries_prototype

# Include any dependencies generated for this target.
include CMakeFiles/cross_library.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cross_library.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cross_library.dir/flags.make

CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.o: Base/BaseCompute.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.o -c /home/gurumurt/Desktop/gpu_libraries_prototype/Base/BaseCompute.cpp

CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gurumurt/Desktop/gpu_libraries_prototype/Base/BaseCompute.cpp > CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.i

CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gurumurt/Desktop/gpu_libraries_prototype/Base/BaseCompute.cpp -o CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.s

CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.o: ThrustOperations/ThrustComputeOps.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/gurumurt/Desktop/gpu_libraries_prototype/ThrustOperations/ThrustComputeOps.cu -o CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.o

CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.o: BoostOperations/BoostComputeOps.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.o -c /home/gurumurt/Desktop/gpu_libraries_prototype/BoostOperations/BoostComputeOps.cpp

CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gurumurt/Desktop/gpu_libraries_prototype/BoostOperations/BoostComputeOps.cpp > CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.i

CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gurumurt/Desktop/gpu_libraries_prototype/BoostOperations/BoostComputeOps.cpp -o CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.s

CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.o: evaluation/executeQueries.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.o -c /home/gurumurt/Desktop/gpu_libraries_prototype/evaluation/executeQueries.cpp

CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gurumurt/Desktop/gpu_libraries_prototype/evaluation/executeQueries.cpp > CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.i

CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gurumurt/Desktop/gpu_libraries_prototype/evaluation/executeQueries.cpp -o CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.s

CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.o: ThrustOperations/ThrustAdapter.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/gurumurt/Desktop/gpu_libraries_prototype/ThrustOperations/ThrustAdapter.cu -o CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.o

CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.o: BoostOperations/BoostAdapter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.o -c /home/gurumurt/Desktop/gpu_libraries_prototype/BoostOperations/BoostAdapter.cpp

CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gurumurt/Desktop/gpu_libraries_prototype/BoostOperations/BoostAdapter.cpp > CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.i

CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gurumurt/Desktop/gpu_libraries_prototype/BoostOperations/BoostAdapter.cpp -o CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.s

CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.o: ArrayFireOperations/afComputeOps.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/gurumurt/Desktop/gpu_libraries_prototype/ArrayFireOperations/afComputeOps.cu -o CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.o

CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.o: CMakeFiles/cross_library.dir/flags.make
CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.o: ArrayFireOperations/ArrayFireAdapter.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.o"
	/usr/local/cuda-10.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/gurumurt/Desktop/gpu_libraries_prototype/ArrayFireOperations/ArrayFireAdapter.cu -o CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.o

CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cross_library
cross_library_OBJECTS = \
"CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.o" \
"CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.o" \
"CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.o" \
"CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.o" \
"CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.o" \
"CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.o" \
"CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.o" \
"CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.o"

# External object files for target cross_library
cross_library_EXTERNAL_OBJECTS =

libcross_library.a: CMakeFiles/cross_library.dir/Base/BaseCompute.cpp.o
libcross_library.a: CMakeFiles/cross_library.dir/ThrustOperations/ThrustComputeOps.cu.o
libcross_library.a: CMakeFiles/cross_library.dir/BoostOperations/BoostComputeOps.cpp.o
libcross_library.a: CMakeFiles/cross_library.dir/evaluation/executeQueries.cpp.o
libcross_library.a: CMakeFiles/cross_library.dir/ThrustOperations/ThrustAdapter.cu.o
libcross_library.a: CMakeFiles/cross_library.dir/BoostOperations/BoostAdapter.cpp.o
libcross_library.a: CMakeFiles/cross_library.dir/ArrayFireOperations/afComputeOps.cu.o
libcross_library.a: CMakeFiles/cross_library.dir/ArrayFireOperations/ArrayFireAdapter.cu.o
libcross_library.a: CMakeFiles/cross_library.dir/build.make
libcross_library.a: CMakeFiles/cross_library.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX static library libcross_library.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cross_library.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cross_library.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cross_library.dir/build: libcross_library.a

.PHONY : CMakeFiles/cross_library.dir/build

CMakeFiles/cross_library.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cross_library.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cross_library.dir/clean

CMakeFiles/cross_library.dir/depend:
	cd /home/gurumurt/Desktop/gpu_libraries_prototype && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gurumurt/Desktop/gpu_libraries_prototype /home/gurumurt/Desktop/gpu_libraries_prototype /home/gurumurt/Desktop/gpu_libraries_prototype /home/gurumurt/Desktop/gpu_libraries_prototype /home/gurumurt/Desktop/gpu_libraries_prototype/CMakeFiles/cross_library.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cross_library.dir/depend

