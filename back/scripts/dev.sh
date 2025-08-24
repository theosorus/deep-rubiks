#!/bin/bash
echo "📦 Installing dependencies..."

cd "$(dirname "$0")/../app" || exit

poetry install

echo "🚀 Starting development server..."
poetry run python app.py