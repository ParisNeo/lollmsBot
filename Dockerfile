# LollmsBot Sovereign Swarm Node
FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node dependencies for WhatsApp bridge (if needed)
RUN mkdir -p /root/.lollmsbot/whatsapp-bridge
COPY lollmsbot/channels/whatsapp.py /tmp/whatsapp.py
# We create a dummy package.json to pre-install node deps
RUN cd /root/.lollmsbot/whatsapp-bridge && \
    npm install whatsapp-web.js qrcode-terminal

# Copy Python project
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .[all]

# Copy the source code
COPY . .

# Environment Defaults
ENV LOLLMSBOT_HOST=0.0.0.0
ENV LOLLMSBOT_PORT=8800
ENV LOLLMSBOT_DATA_DIR=/app/data

# Create directory for persistent data
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8800

# Start the Daemon
CMD ["lollmsbot", "gateway"]