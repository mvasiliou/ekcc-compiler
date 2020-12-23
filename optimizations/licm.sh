for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/loop_invariant.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/loop_invariant.ek --emit-stats -O licm
    echo ===================================
done