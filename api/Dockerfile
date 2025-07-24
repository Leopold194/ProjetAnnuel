FROM rustlang/rust:nightly

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# intall cmake
RUN apt-get update && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*

# installle git
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --break-system-packages

RUN pip install maturin --break-system-packages

RUN git clone https://github.com/Leopold194/ProjetAnnuel.git

WORKDIR /ProjetAnnuel

RUN cargo build

RUN set -ex && echo $(maturin build -o dist --release --quiet > /dev/null && find dist -name '*.whl' | sort | tail -n1) > path.txt

RUN pip install $(cat path.txt) --break-system-packages

RUN cd .. && rm -rf ProjetAnnuel/

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt --break-system-packages

RUN pip install numpy --break-system-packages

RUN pip install mysql-connector-python --break-system-packages

RUN pip install gunicorn --break-system-packages

ENV ENVIRONMENT=prod

EXPOSE 5000

# Without gunicorn
CMD ["python3", "api.py"]

# With gunicorn
#CMD ["sh", "-c", "\
#  WORKERS=$(( $(nproc) * 2 + 1 )) && \
#  exec gunicorn -w $WORKERS -b 0.0.0.0:5000 api:app\
#"]
