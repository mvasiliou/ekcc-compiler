for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/inline.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/inline.ek --emit-stats -O inline
    echo ===================================
done