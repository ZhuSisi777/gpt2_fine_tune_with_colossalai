# install dependency in this repo
pip install -r requirements.txt


git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency required for ColossalSAI
pip install -r ColossalAI/requirements/requirements.txt
# install colossalai
pip install ColossalAI/

# run fine tune process
torchrun run_finetune_gpt2.py
