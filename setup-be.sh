#!/bin/bash

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║   GREENLOCK AI CHATBOT - BACKEND SETUP        ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${BLUE}➜${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    echo "Please install Python 3.9 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Check pip
print_status "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed!"
    echo "Please install pip3 first"
    exit 1
fi
print_success "pip3 is available"

# Create virtual environment
print_status "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate || {
    print_error "Failed to activate virtual environment"
    exit 1
}
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_status "Installing dependencies (this may take a few minutes)..."
echo ""

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || {
        print_error "Failed to install dependencies"
        exit 1
    }
    print_success "All dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p uploads
mkdir -p web_content
mkdir -p vectorstore
print_success "Directories created (uploads, web_content, vectorstore)"

# Create .env file if not exists
print_status "Setting up environment file..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created from .env.example"
    else
        cat > .env << EOF
# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=deepseek/deepseek-chat-v3

# Server Config
DEBUG=True
HOST=0.0.0.0
PORT=8000
EOF
        print_success ".env file created"
    fi
    
    echo ""
    print_warning "⚠️  IMPORTANT: Edit .env file and add your OpenRouter API key!"
    echo ""
    echo "   1. Get your API key from: https://openrouter.ai/keys"
    echo "   2. Edit .env file: nano .env"
    echo "   3. Replace 'your_openrouter_key_here' with your actual key"
    echo ""
else
    print_success ".env file already exists"
fi

# Verify installation
print_status "Verifying installation..."
python3 -c "import fastapi" && print_success "FastAPI installed" || print_error "FastAPI installation failed"
python3 -c "import langchain" && print_success "LangChain installed" || print_error "LangChain installation failed"
python3 -c "import sentence_transformers" && print_success "Sentence Transformers installed" || print_error "Sentence Transformers installation failed"

# Create run script
print_status "Creating run script..."
cat > run-backend.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Greenlock Backend..."
echo ""

# Activate venv
source venv/bin/activate

# Check .env
if ! grep -q "your_openrouter_key_here" .env; then
    python main.py
else
    echo "⚠️  WARNING: Please configure your OpenRouter API key in .env file!"
    echo ""
    echo "1. Edit .env: nano .env"
    echo "2. Add your API key from https://openrouter.ai/keys"
    echo ""
    exit 1
fi
EOF

chmod +x run-backend.sh
print_success "Run script created (run-backend.sh)"

# Summary
echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║          SETUP COMPLETED SUCCESSFULLY!         ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "📋 Next Steps:"
echo ""
echo "   1. Configure API Key:"
echo "      nano .env"
echo "      (Get key from: https://openrouter.ai/keys)"
echo ""
echo "   2. Start the server:"
echo "      ./run-backend.sh"
echo "      or manually:"
echo "      source venv/bin/activate && python main.py"
echo ""
echo "   3. Access the server:"
echo "      🌐 API: http://localhost:8000"
echo "      📖 Docs: http://localhost:8000/docs"
echo ""
echo "   4. Upload documents (optional):"
echo "      - Place PDF files in uploads/ folder"
echo "      - Or use: POST /upload-greenlock-doc"
echo ""
echo "🔍 Features:"
echo "   ✅ Hybrid Search (FAISS + BM25)"
echo "   ✅ Cross-Encoder Re-ranking"
echo "   ✅ Query Rewriting (3 variations)"
echo "   ✅ Real-time WebSocket chat"
echo ""
echo "📚 Knowledge Base: 2 PDFs + 15 website pages"
echo "⚡ Response time: ~800-1200ms"
echo ""
print_success "Setup complete! Happy coding! 🚀"
echo ""