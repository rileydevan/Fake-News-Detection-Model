#!/bin/bash
# run.sh - Launch Script for Fake News Detection DSS

echo "======================================"
echo "Fake News Detection DSS"
echo "======================================"
echo ""

# Check if Ollama is running
echo "Checking Ollama status..."
if ! pgrep -x "ollama" > /dev/null
then
    echo "⚠️  Ollama is not running!"
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
    echo "✅ Ollama started"
else
    echo "✅ Ollama is running"
fi

# Check if required models are available
echo ""
echo "Checking for required models..."

models=("gpt-oss:20b" "gemma3:27b" "qwen3:30b" "deepseek-r1:14b" "gemma3:12b")
missing_models=()

for model in "${models[@]}"
do
    if ! ollama list | grep -q "$model"; then
        missing_models+=("$model")
    fi
done

if [ ${#missing_models[@]} -ne 0 ]; then
    echo "⚠️  Missing models detected:"
    for model in "${missing_models[@]}"
    do
        echo "   - $model"
    done
    echo ""
    read -p "Would you like to download missing models? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        for model in "${missing_models[@]}"
        do
            echo "Downloading $model..."
            ollama pull "$model"
        done
    else
        echo "⚠️  Application may not work properly without all models"
    fi
else
    echo "✅ All models available"
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Dependencies not installed"
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "✅ Dependencies installed"
fi

# Launch the application
echo ""
echo "======================================"
echo "Launching Application..."
echo "======================================"
echo ""
echo "🌐 Opening browser at http://localhost:8501"
echo "📝 Press Ctrl+C to stop the server"
echo ""

streamlit run main.py
