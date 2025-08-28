# 🎥 Gemini Live Video Verifier

A comprehensive video analysis system that provides multi-modal content detection, language fluency analysis, speaker diarization, and quality assurance validation through an intuitive three-screen Streamlit interface.

## 🔍 Overview

The Gemini Live Video Verifier is designed for analyzing video content to detect specific text elements, assess language fluency, and verify voice audibility. It's particularly useful for:

- Content moderation and verification
- Language learning assessment
- Multi-speaker audio analysis
- Automated quality assurance testing
- Educational content validation

## ✨ Features

### 🖥️ Three-Screen Interface

1. **Input Parameters Screen**: Configure analysis settings
2. **Video Analysis Screen**: Real-time processing with progress indicators
3. **QA Report Screen**: Comprehensive results and export options

### 🔍 Multi-Modal Analysis

- **Text Detection**: OCR-based content recognition
- **Audio Language Analysis**: Multi-language speech verification
- **Voice Audibility**: Detection of multiple distinct voices
- **Quality Assurance**: Automated validation checks

### 🛡️ Security & Authorization

- Google Sheets integration for user authorization
- Session-based isolation and security
- Input validation and sanitization
- Resource management and monitoring

### 📊 Advanced Analytics

- Real-time progress tracking
- Confidence scoring for all detections
- Detailed screenshot capture
- Comprehensive reporting with export capabilities

### 💻 Software Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **FFmpeg**: Required for audio processing
- **Tesseract OCR**: Required for text recognition

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Juan-Vergara-INV/Gemini-Live-Video-Verifier.git
cd video-analyzer
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Google Sheets Integration Setup

Create a `secrets.toml` file in your `.streamlit` directory:

```toml
[google]
apps_script_url = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
sheets_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID"
sheet_id = "YOUR_SHEET_ID"

[settings]
default_sheet_name = "Submissions"
max_retries = 2
timeout_base = 45

[performance]
enable_gpu_acceleration = false
max_memory_usage_mb = 1800
enable_concurrent_analysis = true
whisper_model_size = "tiny"
audio_cache_size = 2

[cloud]
environment = "streamlit_cloud"
enable_memory_monitoring = true
aggressive_cleanup = true
reduce_logging = true
```

### 2. Language Configuration

Supported languages for analysis:

- es-419
- hi-IN
- ja-JP
- ko-KR
- de-DE
- en-IN
- fr-FR
- ar-EG
- pt-BR
- id-ID
- ko-JA
- zh-CN
- ru-RU
- ml-IN
- sv-SE
- te-IN
- vi-VN
- tr-TR
- bn-IN
- it-IT
- zh-TW
- pl-PL
- nl-NL
- th-TH
- ko-ZH

## 📖 Usage

### 1. Start the Application

```bash
streamlit run video_analyzer.py
```

### 2. Access the Web Interface

Open your browser and navigate to: `http://localhost:8501`

### 3. Input Parameters (Screen 1)

#### Required Fields:

- **Question ID**: Unique identifier for the analysis session
- **Alias Email**: Email address for session tracking
- **QA Email Address**: Quality assurance analyst email
- **Video File**: Upload your video file (MP4, AVI, MOV, MKV)

#### Configuration Options:

- **Target Language**: Select the expected language for fluency analysis
- **Frame Interval**: Time between analyzed frames (1-10 seconds)
  - Lower intervals = more accurate analysis but slower processing
  - Higher intervals = faster processing but potentially missed content

### 4. Video Analysis (Screen 2)

The system will automatically:

1. **Extract audio** from the uploaded video
2. **Process video frames** at the specified interval
3. **Perform OCR** to detect text content
4. **Analyze audio** for language and voice characteristics
5. **Generate screenshots** at detection points
6. **Create analysis report** with all findings

#### Real-time Progress Indicators:

- Overall analysis progress
- Current processing stage
- Memory usage monitoring
- Frame processing status

### 5. QA Report (Screen 3)

Review comprehensive results including:

- **Overall Quality Score**: Pass/Fail status
- **Detailed Detection Results**: Individual findings with confidence scores
- **Screenshots**: Visual evidence of detections
- **Export Options**: Send results to Google Sheets

## 🎯 Analysis Types

### 1. Text Detection

**Purpose**: Identifies specific text content in video frames using OCR

**Target Texts**:

- "2.5 Flash" - Flash configuration detection
- "Eval Mode: Native Audio Output" - System mode verification

**Process**:

1. Frame extraction at specified intervals
2. Image preprocessing (denoising, thresholding)
3. OCR text recognition using Tesseract
4. Fuzzy text matching with error correction
5. Bounding box detection and screenshot capture

### 2. Audio Language Analysis

**Purpose**: Verifies language fluency and accuracy in speech

**Features**:

- Multi-language support (15+ languages)
- Whisper-based transcription
- Fluency scoring algorithms
- Confidence assessment

**Process**:

1. Audio extraction from video
2. Voice activity detection
3. Speech segmentation
4. Language identification using Whisper
5. Fluency assessment based on multiple factors

**Fluency Indicators**:

- Word count and complexity
- Average word length
- Transcription quality
- Language match confidence

### 3. Voice Audibility Detection

**Purpose**: Determines if multiple distinct voices are present and audible

**Methodology**:

- Voice activity detection using energy analysis
- Feature extraction (pitch, spectral centroid, MFCCs)
- Voice diversity analysis
- Multi-speaker detection algorithms

**Key Metrics**:

- Voice segment count
- Total voice duration
- Voice diversity score
- Audibility confidence

### 4. Quality Assurance Checks

**Automated validation** across four key areas:

#### Flash Text Predominance

- Verifies "2.5 Flash" appears in OCR results
- Checks for pattern variations and OCR errors
- **Pass Criteria**: Text found in any frame

#### Eval Mode Detection

- Searches for "Eval Mode: Native Audio Output"
- Validates system configuration display
- **Pass Criteria**: Full or partial text match

#### Language Fluency Assessment

- Analyzes speech quality and accuracy
- Calculates fluency coverage percentage
- **Pass Criteria**: ≥60% fluent segments OR ≥60% fluency score

#### Voice Audibility Verification

- Confirms both user and model voices are present
- Assesses voice separation and clarity
- **Pass Criteria**: Both voices audible with ≥30% confidence

## 📊 Quality Assurance

### Pass/Fail Criteria

**PASS Requirements**:

- All four QA checks must pass
- Overall score ≥ 75%
- No critical errors in analysis

**FAIL Conditions**:

- Any QA check fails
- Overall score < 75%
- Analysis errors or insufficient data

### Quality Indicators

**Confidence Levels**:

- 🟢 **High (90%+)**: Very reliable detection
- 🟡 **Good (70-89%)**: Reliable detection
- 🟠 **Fair (50-69%)**: Moderate confidence
- 🔴 **Low (<50%)**: Uncertain detection

## 🔒 Security Features

### Authentication & Authorization

- **Google Sheets Verification**: User credentials validated against authorized lists
- **Session-based Access**: Each analysis session is isolated and secured
- **Input Validation**: All user inputs are sanitized and validated
- **Path Security**: File operations restricted to safe directories

### Data Privacy

- **Temporary File Cleanup**: Automatic deletion of processing files
- **Session Isolation**: Each user session is completely isolated
- **Secure File Handling**: All file operations use validated paths
- **No Data Persistence**: Videos and audio are not permanently stored

## 📤 Export & Integration

### Google Sheets Export

**Features**:

- Automatic data export to configured Google Sheets
- Screenshot upload to Google Drive
- Structured data format with session metadata
- Real-time export status tracking

## 🛠️ Troubleshooting

### Common Issues

#### 1. Installation Problems

**Tesseract not found**:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows: Add Tesseract to PATH
set PATH=%PATH%;C:\Program Files\Tesseract-OCR
```

**FFmpeg not found**:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

#### 2. Runtime Issues

**Analysis failures**:

- Check video file format (must be MP4)
- Verify file size (must be ≤500MB)
- Ensure good video quality for text detection
- Check audio clarity for language analysis

**Authorization failures**:

- Verify Google Sheets configuration
- Check user credentials in authorization sheet
- Ensure QA email is in approved list

### Getting Help

1. **Check Logs**: Review console output for error messages
2. **Verify Configuration**: Ensure all settings are correct
3. **Test with Sample Data**: Use a small, high-quality video file
4. **Contact Support**: Provide session ID and error details

## 📞 Support

For support, issues, or feature requests:

1. Check the troubleshooting section above
2. Check the Help section within the application
3. Check application logs for error details
4. Contact **Juan Vergara** on Slack for technical support
