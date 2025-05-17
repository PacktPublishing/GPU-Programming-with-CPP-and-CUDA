# Introduction

<p>This chapter presents a C++ library that uses the traditional steps for building, but we also have three implementations of wrappers to expose the code to Python. 
We also provide a run_tests.sh inside the `python` folder that runs the corresponding `test.py` file inside each alternative of the wrappers. The specific commands are listed bellow:</p>

## To Build the C++ Library
To build we may simply:
 1. create a `build` folder inside of `ch9` folder
 2. run cmake .. 
 3. run make

 ## For CTypes
 There is no need to build, just run the code with the Python interpreter. There are two files:
 1. test_ctypes.py that uses the library
 2. test.py that will be used on the script that runs all the wrappers

 ## For Wrapper
 It will be built by setup.py wit the following command:
 1. python3 setup.py build_ext --inplace 

 <p>We also have two test files here:</p>
 1. test_vector_add_wrapper.py to test the wrapper individually with the Python interpreter
 2. test.py that will be used on the script that runs all the wrappers

 ## For the NumPy Wrapper
 It will be built by setup.py wit the following command:
 1. python3 setup.py build_ext --inplace 

 <p>We also have two test files here:</p>
 1. test_vector_add_np_wrapper.py to test the wrapper individually with the Python interpreter
 2. test.py that will be used on the script that runs all the wrappers

## To Execute all Tests
First we have to run the setup.py command for the wrappers, after that we can run:
1. run_tests.sh inside the `python` folder, this script will collect the names of the subfolders and will execute 10 times each `test.py` program. After each iteration it increases the inpute size so that we can observe the impact on overall performance.