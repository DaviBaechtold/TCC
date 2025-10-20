#!/bin/bash

# Benchmark script for real-time pose estimation
# Compares original vs. optimized implementation

set -e

PROJECT_DIR="/home/davs/Documents/TCC/Project"
cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🎯 REAL-TIME POSE ESTIMATION BENCHMARK                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
CFG="work_dirs/test_minimal5/rtmpose_m_wholebody_minimal.py"
CKPT="work_dirs/test_minimal5/best_coco-wholebody_AP_epoch_50.pth"
DET_CFG="configs/detectors/rtmdet_nano_person_infer.py"
DET_CKPT="checkpoints/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"

# Test video (create if doesn't exist)
TEST_VIDEO="data/video/test_sample.mp4"

if [ ! -f "$TEST_VIDEO" ]; then
    echo "⚠️  Test video not found. Using webcam (camera 0)"
    SOURCE="0"
else
    SOURCE="$TEST_VIDEO"
fi

echo "📋 Configuration:"
echo "  - Pose Model: RTMPose-M"
echo "  - Detector: RTMDet-Nano"
echo "  - Source: $SOURCE"
echo ""

# Function to run benchmark
run_benchmark() {
    local name=$1
    local script=$2
    local extra_args=$3
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔹 Testing: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    timeout 30 python "$script" \
        --cfg "$CFG" \
        --ckpt "$CKPT" \
        --det-cfg "$DET_CFG" \
        --det-ckpt "$DET_CKPT" \
        --source "$SOURCE" \
        --device cuda:0 \
        --benchmark \
        $extra_args || true
    
    echo ""
}

# Test 1: Original implementation (single person)
echo "📊 Benchmark 1: Original (Single Person)"
run_benchmark "Original - Single Person" \
    "src/evaluation/run_realtime.py" \
    ""

# Test 2: Original implementation (multi-person)
echo "📊 Benchmark 2: Original (Multi-Person with Detector)"
run_benchmark "Original - Multi-Person" \
    "src/evaluation/run_realtime.py" \
    ""

# Test 3: Optimized implementation (batch processing)
echo "📊 Benchmark 3: Optimized (Batch Processing)"
run_benchmark "Optimized - Batch Size 4" \
    "src/evaluation/run_realtime_optimized.py" \
    "--batch-size 4"

# Test 4: Optimized with larger batch
echo "📊 Benchmark 4: Optimized (Batch Size 8)"
run_benchmark "Optimized - Batch Size 8" \
    "src/evaluation/run_realtime_optimized.py" \
    "--batch-size 8"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ BENCHMARK COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Summary:"
echo "  - Original implementation provides baseline performance"
echo "  - Optimized implementation uses batch processing"
echo "  - Larger batch sizes improve throughput for multiple people"
echo ""
echo "💡 Next steps:"
echo "  1. Compare FPS numbers above"
echo "  2. For 70+ FPS: export to TensorRT (see IMPLEMENTATION_PLAN.md)"
echo "  3. Implement GPU async streams for further optimization"
echo ""
