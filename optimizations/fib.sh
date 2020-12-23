n=100

for i in {100..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/fib.ek --extra-args 46
    echo ===================================
done

for i in {100..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/fib.ek --extra-args 46 -O
    echo ===================================
done

