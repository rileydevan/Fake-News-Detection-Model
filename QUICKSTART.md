# 🚀 Quick Start Guide

Get the Fake News Detection DSS up and running in 5 minutes!

## Prerequisites

- Python 3.9+
- 8GB+ RAM (16GB recommended)
- 50GB+ free disk space (for models)
- SerpAPI account (free tier available)

## Installation

### Option 1: Automated Setup (Recommended)

**Unix/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```batch
setup.bat
```

### Option 2: Manual Setup

1. **Install Ollama:**
   ```bash
   # Linux/Mac
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows: Download from https://ollama.com/download
   ```

2. **Pull at least one model:**
   ```bash
   ollama pull gpt-oss:20b
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

**Unix/Mac:**
```bash
./run.sh
```

**Windows:**
```batch
run.bat
```

**Or manually:**
```bash
streamlit run main.py
```

The application will open at `http://localhost:8501`

## First Steps

1. **Enter SerpAPI Key**
   - Get key from https://serpapi.com
   - Enter in sidebar
   
2. **Select Model**
   - Choose `gpt-oss:20b` for best results
   - Or `gemma3:12b` for faster processing

3. **Test with a Claim**
   ```
   "COVID-19 vaccines contain microchips for tracking"
   ```

4. **View Results**
   - Check verdict, confidence, and evidence
   - Download report if needed

## Common Issues

### "Ollama not found"
```bash
# Start Ollama service
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull gpt-oss:20b
```

### "Memory error"
- Use smaller model (gemma3:12b)
- Close other applications
- Reduce number of search results in settings

## Quick Tips

- 💡 **First time?** Use single query mode with a simple claim
- 💡 **Batch processing?** Prepare CSV with "text" column
- 💡 **Slow?** Use smaller models or reduce search results
- 💡 **Better accuracy?** Increase search results and use LLM-based NLI

## Need Help?

See full documentation in `README.md` or create an issue on GitHub.

---

**Ready to detect fake news!** 🔍
