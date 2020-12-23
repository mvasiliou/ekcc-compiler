for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num.ek --emit-stats -O gvn
    echo ===================================
done


for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_3.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_3.ek --emit-stats -O gvn
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_4.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_4.ek --emit-stats -O gvn
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_5.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_5.ek --emit-stats -O gvn
    echo ===================================
done


for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_6.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_6.ek --emit-stats -O gvn
    echo ===================================
done


for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_7.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_7.ek --emit-stats -O gvn
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_8.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_8.ek --emit-stats -O gvn
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_9.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_9.ek --emit-stats -O gvn
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_10.ek --emit-stats
    echo ===================================
done

for i in {10..1}
do
    echo ===================================
    python ../bin/ekcc.py --emit-llvm --jit ../test/global_var_num_10.ek --emit-stats -O gvn
    echo ===================================
done
