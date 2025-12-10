#!/bin/bash

# BTSC-UNet-ViT Setup Script
# This script sets up both backend and frontend

set -e

echo "🚀 Setting up BTSC-UNet-ViT..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}   BTSC-UNet-ViT Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

# Check Python
echo -e "\n${YELLOW}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Node.js
echo -e "\n${YELLOW}[2/5] Checking Node.js...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} Node.js found: $NODE_VERSION"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Setup Backend
echo -e "\n${YELLOW}[3/5] Setting up Backend...${NC}"
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Backend dependencies installed"

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating backend .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Backend .env created (please configure paths)"
else
    echo -e "${GREEN}✓${NC} Backend .env already exists"
fi

cd ..

# Setup Frontend
echo -e "\n${YELLOW}[4/5] Setting up Frontend...${NC}"

echo "Installing Node.js dependencies..."
npm install
echo -e "${GREEN}✓${NC} Frontend dependencies installed"

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating frontend .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Frontend .env created"
else
    echo -e "${GREEN}✓${NC} Frontend .env already exists"
fi

# Final checks
echo -e "\n${YELLOW}[5/5] Running verification...${NC}"

echo "Building frontend..."
npm run build > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Frontend builds successfully"

echo "Checking Python syntax..."
cd backend
source venv/bin/activate
python3 -m py_compile app/main.py app/config.py
echo -e "${GREEN}✓${NC} Backend syntax valid"
cd ..

# Success message
echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"

echo -e "\n${BLUE}Next steps:${NC}"
echo "1. Configure dataset paths in backend/.env"
echo "2. Start backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "3. Start frontend: npm run dev"
echo "4. Open http://localhost:5173"

echo -e "\n${BLUE}Training models:${NC}"
echo "• UNet: python -m app.models.unet.train_unet"
echo "• ViT: python -m app.models.vit.train_vit"

echo -e "\n${BLUE}Documentation:${NC}"
echo "• Main: README.md"
echo "• Backend: backend/README.md"
echo "• Frontend: frontend_README.md"
echo "• Summary: IMPLEMENTATION_SUMMARY.md"

echo -e "\n${GREEN}Happy coding! 🧠🔬${NC}"
