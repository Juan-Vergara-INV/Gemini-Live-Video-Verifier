# Video Content Analyzer - Dev Container Setup

This project includes a complete Dev Container configuration for seamless development with all dependencies pre-installed.

### Getting Started

1. **Clone or open this repository in VS Code**

2. **Open in Dev Container**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Select the command and wait for the container to build

3. **Start the application**
   ```bash
   streamlit run video_analyzer.py
   ```

4. **Access the web interface**
   - Open http://localhost:8501 in your browser
   - The port will be automatically forwarded from the container

## 📦 What's Included

### Pre-installed Dependencies
- **Python 3.11** with all required packages
- **OpenCV** for computer vision
- **Tesseract OCR** with multiple language packs
- **Streamlit** for the web interface
- **Pandas** for data export
- **NumPy** for numerical operations

### VS Code Extensions
- Python development tools (Python, Pylint, Black formatter)
- Jupyter notebook support
- JSON editing support
- Additional productivity extensions

### System Dependencies
- OpenCV system libraries
- Tesseract OCR engine with language packs (English, Spanish, French, German)
- All necessary C++ libraries for image processing

## 🛠️ Development Features

### Port Forwarding
- **Port 8501**: Streamlit application (automatically forwarded)

### Environment Variables
- `PYTHONPATH`: Set to workspace root
- `STREAMLIT_SERVER_HEADLESS`: Configured for container use
- `STREAMLIT_SERVER_ENABLE_CORS`: Disabled for security

### File Structure
```
.devcontainer/
├── devcontainer.json    # Dev Container configuration
└── setup.sh           # Post-creation setup script
```

## 🔧 Customization

### Adding Python Packages
1. Update `requirements.txt`
2. Rebuild the container: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container"

### Adding System Packages
1. Update `.devcontainer/setup.sh`
2. Rebuild the container

### VS Code Settings
Customize the VS Code settings in `.devcontainer/devcontainer.json` under `customizations.vscode.settings`.

## 🚨 Troubleshooting

### Container Won't Start
- Ensure Docker Desktop is running
- Check Docker has sufficient resources allocated
- Try rebuilding: "Dev Containers: Rebuild Container"

### Tesseract Not Working
- The setup script installs Tesseract and language packs
- If issues persist, rebuild the container

### Port Conflicts
- If port 8501 is in use, VS Code will automatically use the next available port
- Check the PORTS tab in VS Code's terminal panel

### Performance Issues
- Allocate more memory to Docker Desktop (recommended: 4GB+)
- Close unnecessary applications