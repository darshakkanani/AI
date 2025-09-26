#!/bin/bash

echo "Setting up Hair Removal with SAM+LaMa Pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install segment-anything
echo "Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install lama-cleaner
echo "Installing LaMa Cleaner..."
pip install lama-cleaner

echo "Setup complete!"
echo ""
echo "To use the hair removal tool:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run with SAM+LaMa: python3 hair_removal.py input.jpg -o output.jpg --method sam-lama"
echo "3. Run with classical method: python3 hair_removal.py input.jpg -o output.jpg --method classical"
echo ""
echo "The SAM model will be automatically downloaded on first use (~2.5GB)"
