#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Navigate into it
cd build

# Generate Makefiles
cmake ..

# Compile the code
make

# Go back to the root directory
cd ..

echo ""
echo "Build complete! Run from the repository root, for example: ./build/v4_0_lpa_ici_2D_sparse_scale_fused_gpu"
