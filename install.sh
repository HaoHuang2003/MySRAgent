git clone git@github.com:yuzhTHU/MySRAgent.git ./42-MySRAgent
cd 42-MySRAgent

git clone git@github.com:GAIR-NLP/SR-Scientist.git ./src/sr_scientist
git clone git@github.com:deep-symbolic-mathematics/llm-srbench.git src/llm-srbench
git clone git@hf.co:datasets/nnheui/llm-srbench ./data/llm-srbench

conda create -p ./venv python=3.12 -y
conda activate ./venv
pip install -e .