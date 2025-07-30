#!/bin/bash
echo "📦 Installing dependencies..."

# Aller dans le dossier back (pas app)
cd "$(dirname "$0")/../app" || exit

# Installer les dépendances avec Poetry
poetry install

# Lancer l'application
echo "🚀 Starting development server..."
poetry run python app.py