#!/bin/bash
# Parameter sweep for vision encoder GEMM kernels

set -e

HIPCC="hipcc -O3 --offload-arch=gfx1100"
KERNEL="kernels/src/gemm_f16_wmma_sweep.hip"
BENCH="benchmarks/vision/bench_sweep.cpp"
OUTPUT_DIR="/tmp/sweep_results"

mkdir -p "$OUTPUT_DIR"

# Vision encoder shapes
declare -a SHAPES=(
    "4608:1536:19520:qkv_proj"
    "1536:1536:19520:proj"
    "4224:1536:19520:fc1"
    "1536:4224:19520:fc2"
)

# Sweep parameters
declare -a NB_VALUES=(4 8 16)
declare -a M_TILE_VALUES=(16 32)
declare -a K_UNROLL_VALUES=(8 16)
declare -a PERSISTENT_VALUES=(384 768 1152)
declare -a WAVES_VALUES=(4 8 12)

compile_and_bench() {
    local nb=$1
    local m_tile=$2
    local k_unroll=$3
    local persistent=$4
    local waves=$5
    local shape_str=$6
    
    IFS=':' read -r m k n name <<< "$shape_str"
    
    local suffix="nb${nb}_mt${m_tile}_ku${k_unroll}_pg${persistent}_wv${waves}"
    local kernel_bin="$OUTPUT_DIR/kernel_${suffix}.hip"
    
    # Compile with these parameters
    cat > "$kernel_bin" << EOF
#define NB $nb
#define M_TILE $m_tile
#define K_UNROLL $k_unroll
#define PERSISTENT_GROUPS $persistent
#define WAVES_PER_SIMD $waves
#include "/home/kread/git/hipfire/$KERNEL"
EOF
    
    # Compile
    if ! $HIPCC -c "$kernel_bin" -o "$OUTPUT_DIR/${suffix}.o" 2>/dev/null; then
        echo "FAIL: $name $suffix - compilation failed"
        return 1
    fi
    
    # Run benchmark (using existing bench binary with kernel path)
    local result
    result=$(./benchmarks/vision/bench_sweep "$OUTPUT_DIR/${suffix}.o" $m $k $n "gemm_f16_wmma_sweep_persistent" 2>&1)
    if [ $? -ne 0 ]; then
        echo "FAIL: $name $suffix - benchmark failed"
        return 1
    fi
    
    # Parse result (format: "time_gflops_utilization")
    local time tflops pct
    time=$(echo "$result" | awk '{print $1}')
    tflops=$(echo "$result" | awk '{print $2}')
    pct=$(echo "$result" | awk '{print $3}')
    
    echo "PASS: $name nb=$nb mt=$m_tile ku=$k_unroll pg=$persistent wv=$waves | ${time}us ${tflops}T ${pct}%"
    
    # Log to file
    echo "$name,$nb,$m_tile,$k_unroll,$persistent,$waves,$time,$tflops,$pct" >> "$OUTPUT_DIR/results.csv"
}

# Initialize CSV
echo "shape,nb,m_tile,k_unroll,persistent_groups,waves_per_simd,time_us,tflops,utilization_pct" > "$OUTPUT_DIR/results.csv"

echo "Starting parameter sweep..."
echo "Total combinations: ${#SHAPES[@]} × ${#NB_VALUES[@]} × ${#M_TILE_VALUES[@]} × ${#K_UNROLL_VALUES[@]} × ${#PERSISTENT_VALUES[@]} × ${#WAVES_VALUES[@]}"

# Stage 1: Fixed baseline sweep
echo "Stage 1: Baseline sweep (NB=8, M_TILE=16, K_UNROLL=16)"
for shape in "${SHAPES[@]}"; do
    compile_and_bench 8 16 16 768 8 "$shape" || true
done

# Stage 2: Sweep M_TILE (fix NB=8, K_UNROLL=16, PERSISTENT=768, WAVES=8)
echo -e "\nStage 2: M_TILE sweep"
for shape in "${SHAPES[@]}"; do
    for m_tile in "${M_TILE_VALUES[@]}"; do
        compile_and_bench 8 $m_tile 16 768 8 "$shape" || true
    done
done

# Stage 3: Sweep K_UNROLL (fix NB=8, M_TILE=16, PERSISTENT=768, WAVES=8)
echo -e "\nStage 3: K_UNROLL sweep"
for shape in "${SHAPES[@]}"; do
    for k_unroll in "${K_UNROLL_VALUES[@]}"; do
        compile_and_bench 8 16 $k_unroll 768 8 "$shape" || true
    done
done

# Stage 4: Sweep persistent groups (fix NB=8, M_TILE=16, K_UNROLL=16, WAVES=8)
echo -e "\nStage 4: Persistent groups sweep"
for shape in "${SHAPES[@]}"; do
    for persistent in "${PERSISTENT_VALUES[@]}"; do
        compile_and_bench 8 16 16 $persistent 8 "$shape" || true
    done
done

# Stage 5: Sweep waves per SIMD (fix NB=8, M_TILE=16, K_UNROLL=16, PERSISTENT=768)
echo -e "\nStage 5: Waves per SIMD sweep"
for shape in "${SHAPES[@]}"; do
    for waves in "${WAVES_VALUES[@]}"; do
        compile_and_bench 8 16 16 768 $waves "$shape" || true
    done
done

echo -e "\nSweep complete! Results in $OUTPUT_DIR/results.csv"
echo -e "\nTop 5 configurations per shape:"
for shape in "${SHAPES[@]}"; do
    IFS=':' read -r m k n name <<< "$shape"
    echo -e "\n$name:"
    grep "^$name," "$OUTPUT_DIR/results.csv" | sort -t',' -k7 -n | head -5
done
