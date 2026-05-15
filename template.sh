#!/bin/bash

echo "Creating AI Research Assistant structure..."

mkdir -p app/data
mkdir -p chroma_db
mkdir -p uploads

touch app/main.py
touch app/ingest.py
touch app/rag.py
touch app/utils.py

touch requirements.txt
touch .env
touch README.md
touch .gitignore

echo "Done."