FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN pip install jupyterlab   
RUN pip install packaging
RUN pip install ninja
RUN pip install datasets==2.18.0

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install nano

RUN pip install numpy
RUN pip install pandas
RUN pip install tqdm
RUN pip install evaluate
RUN pip install deepspeed
RUN pip install tensorboard
RUN pip install scikit-learn
RUN pip install sentencepiece
RUN pip install torch==2.3.0
RUN pip install transformers==4.38.2
RUN pip install peft==0.11.0
RUN pip install accelerate==0.30.0
RUN pip install bitsandbytes==0.43.1
RUN pip install wandb
RUN pip install fire
RUN pip install flake8==7.0.0
RUN pip install mmh3==4.1.0
RUN pip install xformers==0.0.26.post1
RUN pip install flash-attn==2.5.9.post1 --no-build-isolation
RUN pip install tiktoken
RUN pip install --no-deps vllm==0.4.3
RUN pip install vllm_flash_attn=2.5.9

RUN pip uninstall --yes transformer_engine

RUN pip install nltk
RUN pip install inflect
RUN pip install pydantic==2.7.4
RUN pip install filelock==3.14.0
RUN pip install rouge-score
RUN pip install pymorphy2

WORKDIR /workdir
