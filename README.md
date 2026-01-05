# StableGS

1. generate pairs_uni.pth by mvs.py
2. cd submodules/diff-gaussian-rasterization_radegs; python setup.py build_ext --inplace -j8; cd submodules/diff-gaussian-rasterization_taming; python setup.py build_ext --inplace -j8
3. training refer to run.sh