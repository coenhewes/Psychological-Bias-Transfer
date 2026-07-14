#!/usr/bin/env bash
set +u
pip install --upgrade pip
pip uninstall -y numpy scipy
pip install "numpy<2" "scipy" "vllm" "transformers>=4.51,<5" "huggingface-hub>=0.34.0,<1.0" 2>&1 | tail -5
echo "vllm stack install done"
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -2
