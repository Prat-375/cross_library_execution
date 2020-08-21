# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /home/hkumar/clion-2020.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/hkumar/clion-2020.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hkumar/cross_library_execution

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hkumar/cross_library_execution

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/hkumar/clion-2020.2/bin/cmake/linux/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/hkumar/clion-2020.2/bin/cmake/linux/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hkumar/cross_library_execution/CMakeFiles /home/hkumar/cross_library_execution/CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hkumar/cross_library_execution/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named cross_library_execution

# Build rule for target.
cross_library_execution: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 cross_library_execution
.PHONY : cross_library_execution

# fast build rule for target.
cross_library_execution/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/build
.PHONY : cross_library_execution/fast

Plug/Compute.o: Plug/Compute.cpp.o

.PHONY : Plug/Compute.o

# target to build an object file
Plug/Compute.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/Plug/Compute.cpp.o
.PHONY : Plug/Compute.cpp.o

Plug/Compute.i: Plug/Compute.cpp.i

.PHONY : Plug/Compute.i

# target to preprocess a source file
Plug/Compute.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/Plug/Compute.cpp.i
.PHONY : Plug/Compute.cpp.i

Plug/Compute.s: Plug/Compute.cpp.s

.PHONY : Plug/Compute.s

# target to generate assembly for a file
Plug/Compute.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/Plug/Compute.cpp.s
.PHONY : Plug/Compute.cpp.s

Thrust_SwitchBoard/ThrustCompute.o: Thrust_SwitchBoard/ThrustCompute.cu.o

.PHONY : Thrust_SwitchBoard/ThrustCompute.o

# target to build an object file
Thrust_SwitchBoard/ThrustCompute.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/Thrust_SwitchBoard/ThrustCompute.cu.o
.PHONY : Thrust_SwitchBoard/ThrustCompute.cu.o

Thrust_SwitchBoard/ThrustCompute.i: Thrust_SwitchBoard/ThrustCompute.cu.i

.PHONY : Thrust_SwitchBoard/ThrustCompute.i

# target to preprocess a source file
Thrust_SwitchBoard/ThrustCompute.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/Thrust_SwitchBoard/ThrustCompute.cu.i
.PHONY : Thrust_SwitchBoard/ThrustCompute.cu.i

Thrust_SwitchBoard/ThrustCompute.s: Thrust_SwitchBoard/ThrustCompute.cu.s

.PHONY : Thrust_SwitchBoard/ThrustCompute.s

# target to generate assembly for a file
Thrust_SwitchBoard/ThrustCompute.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/Thrust_SwitchBoard/ThrustCompute.cu.s
.PHONY : Thrust_SwitchBoard/ThrustCompute.cu.s

adapter.o: adapter.cu.o

.PHONY : adapter.o

# target to build an object file
adapter.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/adapter.cu.o
.PHONY : adapter.cu.o

adapter.i: adapter.cu.i

.PHONY : adapter.i

# target to preprocess a source file
adapter.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/adapter.cu.i
.PHONY : adapter.cu.i

adapter.s: adapter.cu.s

.PHONY : adapter.s

# target to generate assembly for a file
adapter.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/adapter.cu.s
.PHONY : adapter.cu.s

main.o: main.cu.o

.PHONY : main.o

# target to build an object file
main.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/main.cu.o
.PHONY : main.cu.o

main.i: main.cu.i

.PHONY : main.i

# target to preprocess a source file
main.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/main.cu.i
.PHONY : main.cu.i

main.s: main.cu.s

.PHONY : main.s

# target to generate assembly for a file
main.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cross_library_execution.dir/build.make CMakeFiles/cross_library_execution.dir/main.cu.s
.PHONY : main.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... cross_library_execution"
	@echo "... Plug/Compute.o"
	@echo "... Plug/Compute.i"
	@echo "... Plug/Compute.s"
	@echo "... Thrust_SwitchBoard/ThrustCompute.o"
	@echo "... Thrust_SwitchBoard/ThrustCompute.i"
	@echo "... Thrust_SwitchBoard/ThrustCompute.s"
	@echo "... adapter.o"
	@echo "... adapter.i"
	@echo "... adapter.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

