git clone git@github.com:yuzhTHU/MySRAgent.git ./SRAgent && cd SRAgent

# git clone git@github.com:GAIR-NLP/SR-Scientist.git ./src/sr_scientist
# git clone git@hf.co:datasets/nnheui/llm-srbench ./data/llm-srbench-data
git clone git@github.com:deep-symbolic-mathematics/llm-srbench.git ./data/llm-srbench-code

conda create -p ./venv python=3.12 -y && conda activate ./venv && pip install -e .[llm,dev]
