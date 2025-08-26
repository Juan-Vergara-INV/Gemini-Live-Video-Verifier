#!/bin/bash

# Video Content Analyzer Dev Container Setup Script
echo "🚀 Setting up Video Content Analyzer development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update

# Install system dependencies for OpenCV and Tesseract
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtkglext1-dev \
    libgtkglext1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config

# Install additional Tesseract language packs (optional)
echo "🌐 Installing additional Tesseract language packs..."
sudo apt-get install -y \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu

# Upgrade pip
echo "📈 Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Verify installations
echo "✅ Verifying installations..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
python -c "import pytesseract; print('Tesseract is available')"

# Create output directories
echo "📁 Creating output directories..."
mkdir -p analysis_output
mkdir -p temp

# Set up Git (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚠️  Git user not configured. You may want to run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi

echo ""
echo "🎉 Setup complete! Your development environment is ready."
echo ""
echo "📋 Available commands:"
echo "  • streamlit run video_analyzer.py  - Start the web interface"
echo "  • python video_analyzer.py         - Run in script mode"
echo ""
echo "🌐 Streamlit will be available at http://localhost:8501"
echo ""
