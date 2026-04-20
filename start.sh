#!/bin/bash
cd "$(dirname "$0")"

echo "🚀 启动后端..."
uvicorn server:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
sleep 2

echo "🖥️  启动 Electron..."
cd electron && npx electron .
kill $BACKEND_PID 2>/dev/null
