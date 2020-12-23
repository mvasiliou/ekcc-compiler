Authors: Michael Vasiliou

External Packages:
PyYAML==3.11
ply==3.11
llvmlite==0.30.0

(Coincidental versioning!)


Usage: bin/ekcc.py [-h] [-?] [-v] [--emit-ast] [--emit-llvm] [-O] [-o OUTPUT_FILE]
                   input_file

Extended Kaleidoscope Compiler

positional arguments:
  input_file      Source file to compile

optional arguments:
  -h, --help      show this help message and exit
  -?
  -v, --verbose   Turn on verbose logging
  --emit-ast
  --emit-llvm
  --jit
  --emit-stats
  -o OUTPUT_FILE  Output file path
  -O              Enable optimizations. Pass in space separated list of any of the following:
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

