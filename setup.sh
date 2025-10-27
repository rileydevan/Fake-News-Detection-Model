#!/bin/bash
# setup.sh - Setup Script for Fake News Detection DSS

echo "======================================"
echo "Fake News Detection DSS - Setup"
echo "======================================"
echo ""

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo "Step 1: Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found!"
    echo "Please install Python 3.9 or higher from https://python.org"
    exit 1
fi

# Check pip
echo ""
echo "Step 2: Checking pip installation..."
if command_exists pip3; then
    echo "✅ pip found"
else
    echo "❌ pip not found!"
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Install Python dependencies
echo ""
echo "Step 3: Installing Python dependencies..."
echo "This may take a few minutes..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Error installing Python dependencies"
    exit 1
fi

# Check Ollama
echo ""
echo "Step 4: Checking Ollama installation..."
if command_exists ollama; then
    echo "✅ Ollama found"
else
    echo "❌ Ollama not found!"
    echo ""
    echo "Please install Ollama:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    read -p "Would you like to install Ollama now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
        echo "✅ Ollama installed"
    else
        echo "⚠️  Please install Ollama manually before running the application"
        exit 1
    fi
fi

# Start Ollama service
echo ""
echo "Step 5: Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
    echo "✅ Ollama service started"
else
    echo "✅ Ollama service already running"
fi

# Download models
echo ""
echo "Step 6: Downloading required LLM models..."
echo "⚠️  This will download ~50GB of models and may take 30-60 minutes"
echo "Models: gpt-oss:20b, gemma3:27b, qwen3:30b, deepseek-r1:14b, gemma3:12b"
echo ""
read -p "Download all models now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    models=("gpt-oss:20b" "gemma3:27b" "qwen3:30b" "deepseek-r1:14b" "gemma3:12b")
    
    for model in "${models[@]}"; do
        echo ""
        echo "Downloading $model..."
        ollama pull "$model"
        
        if [ $? -eq 0 ]; then
            echo "✅ $model downloaded"
        else
            echo "❌ Error downloading $model"
        fi
    done
    
    echo ""
    echo "✅ All models downloaded"
else
    echo "⚠️  Models not downloaded. You can download them later with:"
    echo "     ollama pull <model-name>"
fi

# Create launch script
echo ""
echo "Step 7: Setting up launch script..."
chmod +x run.sh
echo "✅ Launch script configured"

# Final message
echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "To start the application, run:"
echo "  ./run.sh"
echo ""
echo "Or manually with:"
echo "  streamlit run main.py"
echo ""
echo "📝 Don't forget to:"
echo "   1. Get a SerpAPI key from https://serpapi.com"
echo "   2. Enter your API key in the application sidebar"
echo ""
echo "📖 For more information, see README.md"
echo ""
