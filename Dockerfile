# set pytorch:2.2.2-cuda12.1-cudnn8-deval as the base
From pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Copy application files
COPY /home/jason/AgenticIR /home/jason/CLIP4Cir ./

# build postgresql environment
RUN apt-get update  \
  && apt-get install -y git vim  \
  && apt-get install -y postgresql postgresql-client  \
  && /etc/init.d/postgresql start  \
  && su postgres && createdb $db_name && exit  \
  && pip install pgvector  && apt install postgresql-server-dev-14  \
  && git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git && cd pgvector && make && make install && cd .. \
  && apt-get install libpq-dev && pip install psycopg2  \
  && sed -i '90s/.*/local   all             postgres                                peer/' /etc/postgresql/14/main/pg_hba.conf \
  && apt-get install systemd  && service postgresql restart

# build environment
RUN pip install numpy==1.24.1 torch==2.1.0 opencv-python==4.8.0.76  \
  && conda create -y -n clip4cir -y python=3.8  \
  && source activate clip4cir  \
  && conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0  \
  && conda install -y -c anaconda pandas=1.4.2 \
  && pip install -y comet-ml==3.21.0, urllib3==1.26.18  \
  && pip install -y git+https://github.com/openai/CLIP.git  \
  && pip install pgvector  \
  && apt-get install libpq-dev && pip install psycopg2  \
\
  && conda create -y -n agenticir python=3.10  \
  && source activate agenticir  \
  && apt-get install -y ffmpeg libsm6 libxext6  \
  && cd /app/AgenticIR && pip install -r installation/requirements.txt  \
  && pip install -y git+https://github.com/openai/CLIP.git \
\
  && conda create -y -n depictqa python=3.10  \
  && source activate depictqa  \
  && pip install -r /app/AgenticIR/DepictQA/requirements.txt  \
  && cd /app/AgenticIR/DepictQA && sh launch_service.sh  \
\
  && conda create -y -n dehazeformer python=3.7  \
  && source activate dehazeformer  \
  && pip install -r /app/AgenticIR/executor/dehazing/tools/DehazeFormer/requirements.txt  \
\
  && conda create -y -n diffbir python=3.10  \
  && source activate diffbir  \
  && pip install -r /app/AgenticIR/executor/super_resolution/tools/DiffBIR/requirements.txt  \
\
  && conda create -y -n drbnet python=3.8  \
  && source activate drbnet  \
  && pip install -r /app/AgenticIR/executor/defocus_deblurring/tools/DRBNet/requirements.txt  \
\
  && conda create -y -n fbcnn python=3.10.18  \
  && source activate fbcnn  \
  && pip install -r /app/AgenticIR/executor/jpeg_compression_artifact_removal/tools/FBCNN/requirements.txt  \
\
  && conda create -y -n hat python=3.10.18  \
  && source activate hat  \
  && pip install -r /app/AgenticIR/executor/super_resolution/tools/HAT/requirements.txt  \
  && python setup.py develop  \
\
  && conda create -y -n ifan python=3.8.20  \
  && source activate ifan  \
  && pip install -r /app/AgenticIR/executor/defocus_deblurring/tools/IFAN/requirements.txt  \
\
  && conda create -y -n maxim python=3.10.18  \
  && source activate maxim  \
  && pip install -r /app/AgenticIR/executor/denoising/tools/maxim/requirements.txt  \
  && pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  \
  && pip install .  \
\
  && conda create -y -n mprnet python=3.7.16  \
  && source activate mprnet  \
  && pip install -r /app/AgenticIR/executor/denoising/tools/MPRNet/requirements.txt  \
\
  && conda create -y -n restormer python=3.7.16  \
  && source activate restormer  \
  && pip install -r /app/AgenticIR/executor/denoising/tools/Restormer/requirements.txt  \
\
  && conda create -y -n ridcp python=3.8.20  \
  && source activate ridcp  \
  && pip install -r /app/AgenticIR/executor/dehazing/tools/RIDCP_dehazing/requirements.txt  \
\
  && conda create -y -n swinir python=3.10.18  \
  && source activate swinir  \
  && pip install -r /app/AgenticIR/executor/denoising/tools/SwinIR/requirements.txt  \
\
  && conda create -y -n xrestormer python=3.10.18  \
  && source activate xrestormer  \
  && pip install -r /app/AgenticIR/executor/denoising/tools/X-Restormer/requirements.txt  \
  && python setup.py develop  \
  && sed -i '8s/.*/from torchvision.transforms.functional import rgb_to_grayscale/' /opt/conda/envs/xrestormer/lib/python3.10/site-packages/basicsr/data/degradations.py  \

# test conda environment
RUN if ["$do_test" == "true"]; then cd /app/AgenticIR && sh test_env.sh; fi













