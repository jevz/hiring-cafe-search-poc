#!/bin/bash
set -e

echo "Starting HiringCafe Search..."

# Start backend
echo "Starting API server on :8000..."
uvicorn api.main:app --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "Backend ready!"
        break
    fi
    sleep 1
done

# Start frontend
echo "Starting frontend on :5173..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "================================"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "================================"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
