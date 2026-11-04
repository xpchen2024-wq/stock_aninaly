# ============================================================================
# AI Stock Analysis Platform - Dockerfile
# ============================================================================
FROM python:3.11-slim

# System dependencies for building TA-Lib from source.
# libta-lib0 is no longer available in Debian trixie, so we build TA-Lib from
# source. We also need `automake` because on aarch64 (Apple Silicon / OrbStack)
# the config.guess/config.sub bundled with TA-Lib 0.4.0 (2006) cannot detect
# the architecture and ./configure fails with "cannot guess build type".
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make wget unzip automake \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library from source (works on x86_64 and aarch64)
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && cp /usr/share/automake-*/config.guess . \
    && cp /usr/share/automake-*/config.sub . \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
