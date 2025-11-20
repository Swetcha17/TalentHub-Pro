#!/bin/bash

echo "🚀 TalentHub Pro - Quick Setup Script"
echo "======================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✅ Node.js version: $(node --version)"
echo "✅ npm version: $(npm --version)"
echo "✅ Python version: $(python3 --version 2>/dev/null || python --version)"
echo ""

# Step 1: Install Node.js dependencies
echo "📦 Step 1: Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi
echo "✅ Node.js dependencies installed"
echo ""

# Step 2: Install Python dependencies
echo "📦 Step 2: Installing Python dependencies..."
echo "   This may take a few minutes..."
pip3 install flask flask-cors PyMuPDF pdfplumber python-docx pytesseract pillow easyocr spacy 2>/dev/null || \
pip install flask flask-cors PyMuPDF pdfplumber python-docx pytesseract pillow easyocr spacy

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Some Python dependencies may have failed to install"
    echo "   You can install them manually later if needed"
fi
echo ""

# Step 3: Download spaCy model
echo "📦 Step 3: Downloading spaCy language model..."
python3 -m spacy download en_core_web_sm 2>/dev/null || python -m spacy download en_core_web_sm
if [ $? -eq 0 ]; then
    echo "✅ spaCy model downloaded"
else
    echo "⚠️  Failed to download spaCy model (optional)"
fi
echo ""

# Step 4: Build React app
echo "🔨 Step 4: Building React application..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Failed to build React app"
    exit 1
fi
echo "✅ React app built successfully"
echo ""

# Step 5: Rename parse file if needed
if [ -f "parse_modified.py" ] && [ ! -f "parse.py" ]; then
    echo "📝 Step 5: Renaming parse_modified.py to parse.py..."
    mv parse_modified.py parse.py
    echo "✅ File renamed"
elif [ -f "parse_modified.py" ]; then
    echo "📝 Step 5: parse.py exists, keeping parse_modified.py as backup..."
else
    echo "✅ Step 5: parse.py is ready"
fi
echo ""

echo "🎉 Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. Start the server:"
echo "      python parse.py --web"
echo ""
echo "   2. Open your browser to:"
echo "      http://localhost:5001"
echo ""
echo "💡 For development mode (hot reload):"
echo "   Terminal 1: npm start"
echo "   Terminal 2: python parse.py --web"
echo ""
echo "📖 See README.md for more details"
