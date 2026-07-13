# Embodied voice-avatar — single image for Coolify.
# Bundles Node + ffmpeg + Piper (TTS, FR voice) + Rhubarb (visemes).
FROM node:20-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg wget unzip ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Piper (TTS) + French voice ----------------------------------------
ARG PIPER_URL=https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
RUN wget -qO /tmp/piper.tgz "$PIPER_URL" \
    && tar -xzf /tmp/piper.tgz -C /opt \
    && rm /tmp/piper.tgz
ENV LD_LIBRARY_PATH=/opt/piper

RUN mkdir -p /opt/voices \
    && wget -qO /opt/voices/fr_FR-siwis-medium.onnx \
       "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx?download=true" \
    && wget -qO /opt/voices/fr_FR-siwis-medium.onnx.json \
       "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json?download=true"

# ---- Rhubarb (visemes) --------------------------------------------------
ARG RHUBARB_URL=https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip
RUN wget -qO /tmp/rhubarb.zip "$RHUBARB_URL" \
    && unzip -q /tmp/rhubarb.zip -d /opt \
    && mv /opt/Rhubarb-Lip-Sync-1.13.0-Linux /opt/rhubarb \
    && chmod +x /opt/rhubarb/rhubarb \
    && rm /tmp/rhubarb.zip

# ---- app ----------------------------------------------------------------
WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev
COPY server.js ./
COPY public ./public

ENV PORT=3000 \
    PIPER_BIN=/opt/piper/piper \
    PIPER_VOICE=/opt/voices/fr_FR-siwis-medium.onnx \
    RHUBARB_BIN=/opt/rhubarb/rhubarb \
    FFMPEG_BIN=ffmpeg
EXPOSE 3000
CMD ["node", "server.js"]
