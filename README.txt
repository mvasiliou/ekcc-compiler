Authors: Michael Vasiliou

External Packages:
PyYAML==3.11
ply==3.11
llvmlite==0.30.0


Usage: bin/ekcc.py [-h] [-?] [-v] [--emit-ast] [--emit-llvm] [-O] [-o OUTPUT_FILE]
                   input_file

Extended Kaleidoscope Compiler

This compiler implements the Extended Kaleidoscope language as defined here: https://bcain-llvm.readthedocs.io/projects/llvm/en/latest/tutorial/LangImpl01/#the-basic-language

Input files are lexed into a stream of tokens, then parsed into an abstract syntax tree and then passed to LLVM
bindings to generate an intermediate representation. The compiler supports Just In Time execution, so 
code can be run immediately after compilation.

Sample .ek files can be found in the test/ directory.


positional arguments:
  input_file      Source file to compile

optional arguments:
  -h, --help, -?      Show this help message and exit
  -v, --verbose       Turn on verbose logging
  --emit-ast          Write out the abstract syntax tree as a file
  --emit-llvm         Write out the LLVM IR as a file
  --jit               Execute code immediately after compilation
  --emit-stats        Log stats for runtime of program
  -o OUTPUT_FILE      Output file path
  -O                  Enable optimizations. Pass in space separated list of any of the following:
    dead_arg
    constant_merge
    gvn
    licm
    inline
    inline_50
    inline_100
    inline_200
    inline_300
    opt_3
    opt_2
    opt_1
    unroll_loops
    loop_vectorize
    global_dce
    dce
    ipsccp
    global_optimizer
    sccp