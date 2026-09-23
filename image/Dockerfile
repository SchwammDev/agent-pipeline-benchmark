FROM debian:bookworm-slim

ARG LIUBAI_REF=6862349a8599702a8a205fb9e6c2724765247118

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/liubai-mind \
    && git -C /opt/liubai-mind remote add origin https://github.com/SchwammDev/liubai-mind \
    && git -C /opt/liubai-mind fetch --depth 1 origin "$LIUBAI_REF" \
    && git -C /opt/liubai-mind checkout --detach FETCH_HEAD

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && PATH="${HOME}/.local/bin:${PATH}" bash /opt/liubai-mind/setup.sh

RUN chmod 711 /root \
    && chmod -R a+rwX /root/.pi \
    && mkdir -p /root/.cache \
    && chmod -R a+rwX /root/.cache \
    && chmod -R a+rX /root/.config /root/.local /opt/liubai-mind

ENV PATH="/root/.local/bin:${PATH}"
