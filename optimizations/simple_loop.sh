for i in {100..1}
do
    echo ===================================
     python ../bin/ekcc.py --emit-llvm --jit ../test/simple_loop.ek
     echo ===================================
done

for i in {100..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/simple_loop.ek -O
    echo ===================================
done
