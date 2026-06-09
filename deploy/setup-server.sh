#!/usr/bin/env bash
# Executar NO SERVIDOR (Ubuntu 24.04), como root ou com sudo.
set -euo pipefail

APP_DIR="/opt/gerador_sequencial"
REPO_URL="https://github.com/korgloriws/gerador_sequencial.git"
HOST_PORT="${HOST_PORT:-5080}"

echo "==> Criando pasta isolada em ${APP_DIR}"
mkdir -p "${APP_DIR}"
cd "${APP_DIR}"

if [ ! -d .git ]; then
  git clone "${REPO_URL}" .
else
  git pull origin main || git pull origin master
fi

echo "==> Preparando volumes de dados"
mkdir -p data/uploads
if [ ! -f data/model_weights.json ]; then
  cp model_weights.json data/model_weights.json 2>/dev/null || echo '{}' > data/model_weights.json
fi

echo "==> Subindo container (porta ${HOST_PORT})"
docker compose down 2>/dev/null || true
docker compose up -d --build

echo ""
echo "Pronto. Acesse: http://SEU_IP:${HOST_PORT}"
echo "Exemplo: http://31.97.167.75:${HOST_PORT}"
echo ""
echo "Comandos úteis:"
echo "  cd ${APP_DIR} && docker compose logs -f"
echo "  cd ${APP_DIR} && docker compose restart"
