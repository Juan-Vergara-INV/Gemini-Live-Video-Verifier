"""
Gemini Live Video Verifier Tool for Multi-Modal Content Detection.

A video analysis tool that provides text recognition, language fluency analysis, and speaker diarization.
The tool includes quality assurance validation and a multi-screen Streamlit interface.
"""

import gc
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, Set
import signal
import sys
import requests

import cv2
import numpy as np
import pytesseract
import streamlit as st

import librosa
import speech_recognition as sr
from pydub import AudioSegment
import whisper
from scipy.ndimage import median_filter

from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials


class Config:
    """Centralized application configuration constants."""
    
    # Frame processing parameters
    DEFAULT_FRAME_INTERVAL: float = 10.0
    
    # OCR configuration
    OCR_CONFIG: str = '--psm 3'
    OCR_DENOISING_ENABLED: bool = True
    OCR_THRESHOLD_ENABLED: bool = True
    
    # Video format support
    SUPPORTED_VIDEO_FORMATS: List[str] = ['.mp4']
    
    # UI annotation settings
    ANNOTATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
    ANNOTATION_FONT_SCALE: float = 0.5
    ANNOTATION_THICKNESS: int = 2
    
    # Resource limits
    MAX_FILE_SIZE: int = 1024 * 1024 * 1024 # 1GB
    MAX_VIDEO_DURATION: int = 600  # 10 minutes
    MIN_VIDEO_DURATION: int = 30   # 30 seconds
    
    # Color definitions for UI annotations and bounding boxes
    ANNOTATION_COLORS: Dict[str, Tuple[int, int, int]] = {
        'GREEN': (0, 255, 0),
        'RED': (0, 0, 255),
        'YELLOW': (0, 255, 255),
        'WHITE': (255, 255, 255),
    }
    
    # Language configuration for display and Whisper mapping
    LANGUAGE_CONFIG: Dict[str, Tuple[str, str]] = {
        'es-419': ('es-419', 'es'), 'hi-IN': ('hi-IN', 'hi'), 'ja-JP': ('ja-JP', 'ja'), 'ko-KR': ('ko-KR', 'ko'),
        'de-DE': ('de-DE', 'de'), 'en-IN': ('en-IN', 'en'), 'fr-FR': ('fr-FR', 'fr'), 'ar-EG': ('ar-EG', 'ar'),
        'pt-BR': ('pt-BR', 'pt'), 'id-ID': ('id-ID', 'id'), 'ko-JA': ('ko-JA', 'ko'), 'zh-CN': ('zh-CN', 'zh'),
        'ru-RU': ('ru-RU', 'ru'), 'ml-IN': ('ml-IN', 'ml'), 'sv-SE': ('sv-SE', 'sv'), 'te-IN': ('te-IN', 'te'),
        'vi-VN': ('vi-VN', 'vi'), 'tr-TR': ('tr-TR', 'tr'), 'bn-IN': ('bn-IN', 'bn'), 'it-IT': ('it-IT', 'it'),
        'zh-TW': ('zh-TW', 'zh'), 'pl-PL': ('pl-PL', 'pl'), 'nl-NL': ('nl-NL', 'nl'), 'th-TH': ('th-TH', 'th'),
        'ko-ZH': ('ko-ZH', 'ko'),
    }

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Get dict of supported language codes to display names."""
        return {locale: display_name for locale, (display_name, _) in cls.LANGUAGE_CONFIG.items()}

    @classmethod
    def get_language_display_name(cls, language_code: str) -> str:
        """Get display name for a language code."""
        if language_code in cls.LANGUAGE_CONFIG:
            return cls.LANGUAGE_CONFIG[language_code][0]
        return language_code if language_code else "No language inferred, please review Question ID"
    
    @classmethod
    def locale_to_whisper_language(cls, locale_code: str) -> str:
        """Convert locale code to Whisper language code."""
        if locale_code in cls.LANGUAGE_CONFIG:
            return cls.LANGUAGE_CONFIG[locale_code][1]
        return locale_code

    @classmethod
    def whisper_language_to_locale(cls, whisper_language: str, target_locale: str = None) -> str:
        """Convert Whisper language code back to locale format."""
        if target_locale and cls.locale_to_whisper_language(target_locale) == whisper_language:
            return target_locale
        return whisper_language if whisper_language else "unknown"
    
    @classmethod
    def is_portrait_mobile_resolution(cls, width: int, height: int) -> bool:
        """Check if video resolution is portrait mobile phone format."""
        if width >= height:
            return False
        
        aspect_ratio = height / width
        
        if 1.6 <= aspect_ratio <= 2.5:
            if 360 <= width <= 1600:
                return True
        
        return False
    
    @classmethod
    def validate_video_properties(cls, video_path: str) -> Dict[str, Any]:
        """Validate and extract basic video properties including duration and resolution."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Cannot open video file'}
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            validation_results = {
                'duration_valid': duration >= cls.MIN_VIDEO_DURATION,
                'resolution_valid': cls.is_portrait_mobile_resolution(width, height),
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'min_duration_required': cls.MIN_VIDEO_DURATION
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return {'error': str(e)}


class TargetTexts:
    """Target text definitions for detection."""
    
    FLASH_TEXT: str = "2.5 Flash"
    PRO_TEXT: str = "2.5 Pro"
    ALIAS_NAME_TEXT: str = "Roaring tiger"
    EVAL_MODE_TEXT: str = "Eval Mode: Native Audio Output"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class DetectionType(Enum):
    """Content detection types enumeration."""
    TEXT = auto()
    LANGUAGE_FLUENCY = auto()
    VOICE_AUDIBILITY = auto()


@dataclass
class DetectionRule:
    """Detection rule configuration."""
    name: str
    detection_type: DetectionType
    parameters: Dict[str, Any]


@dataclass
class DetectionResult:
    """Provides detailed information about detection operations including performance metrics and debugging information."""
    rule_name: str
    timestamp: float
    frame_number: int
    detected: bool
    details: Dict[str, Any]
    screenshot_path: Optional[Union[str, Path]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary representation."""

        return {
            'rule_name': self.rule_name,
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'detected': self.detected,
            'details': self.details,
            'screenshot_path': str(self.screenshot_path) if self.screenshot_path else None,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TaskVerifier:
    """Verifies question_id and alias_email inputs."""
    
    def __init__(self):
        config = ConfigurationManager.get_secure_config()
        self.SHEET_URL = config["verifier_sheet_url"]
        self.SHEET_ID = config["verifier_sheet_id"]
        self.QUESTION_IDS_SHEET = "Question IDs"
        self.ALIAS_EMAILS_SHEET = "Alias Emails"
    
    def verify_inputs(self, question_id: str, alias_email: str) -> Tuple[bool, str]:
        """Verify if question_id and alias_email are both authorized."""
        try:
            if not question_id or not question_id.strip():
                return False, "Question ID cannot be empty"
            
            if not alias_email or not alias_email.strip():
                return False, "Alias email cannot be empty"
            
            return self._check_both_inputs(question_id.strip(), alias_email.strip().lower())

        except Exception as e:
            logger.error(f"Authorization verification error: {e}")
            return False, f"Authorization check failed: {str(e)}"
    
    def _check_both_inputs(self, question_id: str, alias_email: str) -> Tuple[bool, str]:
        """Check if both question_id and alias_email are authorized."""
        try:
            question_ids = self._fetch_sheet_data(self.QUESTION_IDS_SHEET)
            alias_emails = self._fetch_sheet_data(self.ALIAS_EMAILS_SHEET)
            
            question_found = False
            for entry in question_ids:
                if str(entry).strip() == question_id:
                    question_found = True
                    break
            
            if not question_found:
                return False, f"Question ID '{question_id}' not found in authorized list"
            
            email_found = False
            for entry in alias_emails:
                if str(entry).strip().lower() == alias_email.lower():
                    email_found = True
                    break
            
            if not email_found:
                return False, f"Alias email '{alias_email}' not found in authorized list"
            
            return True, f"Both Question ID '{question_id}' and email '{alias_email}' are authorized"
            
        except Exception as e:
            logger.error(f"Input check error: {e}")
            return False, f"Failed to verify inputs: {str(e)}"

    def _fetch_sheet_data(self, sheet_name: str) -> List[str]:
        """Fetch data from a specific sheet."""
        try:
            config = ConfigurationManager.get_secure_config()
            gas_url = config["apps_script_url"]
            
            payload = {
                "action": "getAuthorizedEntries",
                "sheetName": sheet_name,
                "data": {}
            }
            
            response = requests.post(
                gas_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                raw_data = result.get("data", [])
                values = []
                
                for item in raw_data:
                    if isinstance(item, dict):
                        if sheet_name == "Question IDs":
                            question_id = item.get('question_id', item.get('id', ''))
                            if question_id and str(question_id).strip():
                                values.append(str(question_id).strip())
                        elif sheet_name == "Alias Emails":
                            alias_email = item.get('alias_email', item.get('email', ''))
                            if alias_email and str(alias_email).strip():
                                values.append(str(alias_email).strip())
                        else:
                            for value in item.values():
                                if value and str(value).strip():
                                    values.append(str(value).strip())
                                    break
                    elif item and str(item).strip():
                        values.append(str(item).strip())
                
                return values
            else:
                error_msg = result.get('message', result.get('error', 'Unknown error'))
                logger.error(f"Google Sheets API error for {sheet_name}: {error_msg}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Network error fetching {sheet_name} sheet: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching {sheet_name} sheet: {e}")
            return []


class SessionManager:
    """Session and resource management system."""
    
    _instance = None
    _lock = threading.RLock()
    _active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._session_base_dir = Path("sessions")
        self._session_base_dir.mkdir(exist_ok=True)
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())
    
    def create_session(self, session_id: str) -> str:
        """Create a new session directory and return its path."""
        with self._lock:
            session_dir = self._session_base_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            self._active_sessions[session_id] = {
                'directory': str(session_dir),
                'created_at': datetime.now().timestamp(),
                'files': []
            }
            
            logger.info(f"Created session directory: {session_dir}")
            return str(session_dir)
    
    def get_session_directory(self, session_id: str) -> Optional[str]:
        """Get the directory path for a session."""
        with self._lock:
            if session_id in self._active_sessions:
                return self._active_sessions[session_id]['directory']
            
            session_dir = self._session_base_dir / session_id
            if session_dir.exists():
                return str(session_dir)
            
            return None
    
    def add_file_to_session(self, session_id: str, file_path: str):
        """Track a file as part of a session."""
        with self._lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id]['files'].append(file_path)
    
    def save_frame(self, session_id: str, frame: np.ndarray, filename: str) -> Optional[str]:
        """Save a frame to the session directory."""
        try:
            session_dir = self.get_session_directory(session_id)
            if not session_dir:
                logger.error(f"Session directory not found for {session_id}")
                return None
            
            filepath = os.path.join(session_dir, filename)
            
            if frame is None or frame.size == 0:
                logger.error(f"Invalid frame data for {filepath}")
                return None
            
            success = cv2.imwrite(filepath, frame)
            if success and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                self.add_file_to_session(session_id, filepath)
                return filepath
            else:
                logger.error(f"Failed to save frame: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Frame save error: {e}")
            return None
    
    def create_temp_file(self, session_id: str, prefix: str, suffix: str = ".tmp") -> str:
        """Create a temporary file within the session directory."""
        session_dir = self.get_session_directory(session_id)
        if not session_dir:
            raise ValueError(f"Session {session_id} not found")
        
        safe_prefix = re.sub(r'[^\w-]', '_', prefix)[:50]
        timestamp = int(datetime.now().timestamp())
        random_part = str(uuid.uuid4())[:8]
        
        filename = f"{safe_prefix}_{timestamp}_{random_part}{suffix}"
        filepath = os.path.join(session_dir, filename)
        
        self.add_file_to_session(session_id, filepath)
        return filepath
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and all its files."""
        try:
            with self._lock:
                session_dir = self.get_session_directory(session_id)
                if not session_dir or not os.path.exists(session_dir):
                    return
                
                try:
                    shutil.rmtree(session_dir)
                    logger.info(f"Removed session directory: {session_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove session directory {session_dir}: {e}")
                
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
                
        except Exception as e:
            logger.error(f"Session cleanup error for {session_id}: {e}")
    
    def cleanup_old_sessions(self) -> None:
        """Clean up sessions older than 20 minutes."""
        current_time = time.time()
        max_age_seconds = 20 * 60  # 20 minutes
        try:
            for session_dir in self._session_base_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                try:
                    dir_mtime = session_dir.stat().st_mtime
                    if current_time - dir_mtime > max_age_seconds:
                        session_id = session_dir.name
                        try:
                            if (hasattr(st, 'session_state') and 
                                hasattr(st.session_state, 'session_id') and
                                st.session_state.session_id == session_id):
                                continue
                        except:
                            pass
                        self.cleanup_session(session_id)
                except Exception as e:
                    logger.error(f"Error checking session {session_dir.name}: {e}")
        except Exception as e:
            logger.error(f"Old sessions cleanup error: {e}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self._active_sessions.keys())


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return _session_manager


class ScreenManager:
    """Manages the screen interface state."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables for screen management."""
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 'input'
        
        if 'session_id' not in st.session_state:
            session_manager = get_session_manager()
            st.session_state.session_id = session_manager.generate_session_id()
        
        if 'question_id' not in st.session_state:
            st.session_state.question_id = ""
        if 'alias_email' not in st.session_state:
            st.session_state.alias_email = ""
        if 'is_authorized' not in st.session_state:
            st.session_state.is_authorized = False
        if 'video_file' not in st.session_state:
            st.session_state.video_file = None
        if 'selected_language' not in st.session_state:
            st.session_state.selected_language = ""
        if 'task_type' not in st.session_state:
            st.session_state.task_type = ""
        if 'frame_interval' not in st.session_state:
            st.session_state.frame_interval = Config.DEFAULT_FRAME_INTERVAL
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analyzer_instance' not in st.session_state:
            st.session_state.analyzer_instance = None
        if 'qa_checker' not in st.session_state:
            st.session_state.qa_checker = None

        if 'analysis_in_progress' not in st.session_state:
            st.session_state.analysis_in_progress = False
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False
        if 'validation_error_shown' not in st.session_state:
            st.session_state.validation_error_shown = False
    
    @staticmethod
    def navigate_to_screen(screen: str):
        """Navigate to a specific screen."""
        if st.session_state.get('current_screen') == 'input' and screen != 'input':
            for key in list(st.session_state.keys()):
                if key.startswith('input_') or key in ['FormSubmitter:input-', 'FormSubmitter:analysis-']:
                    try:
                        del st.session_state[key]
                    except:
                        pass
        
        st.session_state.current_screen = screen
        st.rerun()
    
    @staticmethod
    def reset_session_for_new_analysis():
        """Completely reset session state for a new analysis."""
        target_screen = 'input'
        
        current_session_id = st.session_state.get('session_id')
        if current_session_id:
            session_manager = get_session_manager()
            session_manager.cleanup_session(current_session_id)
            logger.info(f"Cleaned up session {current_session_id}")
        
        keys_to_clear = list(st.session_state.keys())
        
        for key in keys_to_clear:
            try:
                del st.session_state[key]
            except:
                logger.warning(f"Failed to clear session state key: {key}")
        
        session_manager = get_session_manager()
        st.session_state.current_screen = target_screen
        st.session_state.session_id = session_manager.generate_session_id()
        st.session_state.question_id = ""
        st.session_state.alias_email = ""
        st.session_state.video_file = None
        st.session_state.selected_language = ""
        st.session_state.task_type = ""
        st.session_state.frame_interval = Config.DEFAULT_FRAME_INTERVAL
        st.session_state.analysis_results = None
        st.session_state.analyzer_instance = None
        st.session_state.qa_checker = None
        st.session_state.analysis_in_progress = False
        st.session_state.analysis_started = False
        st.session_state.validation_error_shown = False
        
        try:
            if hasattr(st, 'cache_data') and hasattr(st.cache_data, 'clear'):
                st.cache_data.clear()
        except:
            pass
        
        try:
            if hasattr(st, 'cache_resource') and hasattr(st.cache_resource, 'clear'):
                st.cache_resource.clear()
        except:
            pass
        
        logger.info(f"Reset session for new analysis with new session ID: {st.session_state.session_id}")
        st.rerun()
    
    @staticmethod
    def get_current_screen() -> str:
        """Get the current screen."""
        return st.session_state.get('current_screen', 'input')
    
    @staticmethod
    def _cleanup_previous_session():
        """Clean up screenshots and files from the current session."""
        try:
            current_session_id = st.session_state.get('session_id')
            if current_session_id:
                session_manager = get_session_manager()
                session_manager.cleanup_session(current_session_id)
                logger.debug(f"Cleanup session {current_session_id}")
            else:
                logger.debug("No current session ID to clean up")
                        
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")


class InputScreen:
    """First screen: Input form for analysis parameters."""
    _verifier = None

    @classmethod
    def _get_verifier(cls):
        """Get or create the verifier instance."""
        if cls._verifier is None:
            cls._verifier = TaskVerifier()
        return cls._verifier
    
    @staticmethod
    def _infer_language_from_question_id(question_id: str) -> str:
        """Infer target language from question ID."""
        clean_id = question_id.strip()
        
        primary_match = re.search(r'human_eval_([a-z]{2,3}-[A-Z]{2,4})(?:_[A-Za-z]*)?(?:\+|$)', clean_id)
        if primary_match:
            lang_code = primary_match.group(1)
            if lang_code in Config.LANGUAGE_CONFIG:
                return lang_code
        
        for lang_code in Config.LANGUAGE_CONFIG:
            if re.search(re.escape(lang_code), clean_id, re.IGNORECASE):
                return lang_code

    @staticmethod
    def _infer_task_type_from_question_id(question_id: str) -> str:
        """Infer task type from question ID."""
        clean_id = question_id.strip().lower()
        
        if 'monolingual' in clean_id:
            return 'Monolingual'
        
        if 'code_mixed' in clean_id or 'code-mixed' in clean_id:
            return 'Code-mixed'
        
        if 'language_learning' in clean_id or 'language-learning' in clean_id:
            return 'Language Learning'
        
        return 'Unknown'

    @staticmethod
    def render():
        """Render the input screen UI."""
        main_container = st.container()
        
        if ScreenManager.get_current_screen() != 'input':
            main_container.empty()
            return
            
        with main_container:
            InputScreen._render_title_and_divider()
            InputScreen._render_form_fields()
            st.divider()
            InputScreen._render_validation_and_navigation()

    @staticmethod
    def _render_title_and_divider():
        """Render title and divider for input screen."""
        with st.container():
            st.title("1️⃣ Input Parameters")
            st.divider()

    @staticmethod
    def _render_form_fields():
        """Render form input fields."""
        if ScreenManager.get_current_screen() != 'input':
            return
            
        col1, col2 = st.columns(2)
        with col1:
            question_id = st.text_input(
                "Question ID *",
                value=st.session_state.question_id,
                placeholder="Enter question identifier",
                help="Unique identifier for this analysis session (must be authorized in the Question IDs sheet, target language will be automatically inferred)",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.question_id = question_id
        with col2:
            alias_email = st.text_input(
                "Alias Email Address *",
                value=st.session_state.alias_email,
                placeholder="alias-email@gmail.com",
                help="Email address for this analysis session (must be authorized in the Alias Emails sheet)",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.alias_email = alias_email
        st.divider()
        st.subheader("📁 Video File")
        video_file = st.file_uploader(
            "Upload Video File *",
            type=Config.SUPPORTED_VIDEO_FORMATS,
            help=f"Supported formats: {', '.join(Config.SUPPORTED_VIDEO_FORMATS)} (Max: {Config.MAX_FILE_SIZE // 1024 // 1024}MB)",
            disabled=st.session_state.get('analysis_in_progress', False)
        )
        st.session_state.video_file = video_file

        question_id = st.session_state.get('question_id', '')
        inferred_language = InputScreen._infer_language_from_question_id(question_id)
        st.session_state.selected_language = inferred_language
        
        inferred_task_type = InputScreen._infer_task_type_from_question_id(question_id)
        st.session_state.task_type = inferred_task_type

        st.info(f"⏳ **Minimum Video Duration Required**: {Config.MIN_VIDEO_DURATION} seconds")

        st.info("📱 **Video Resolution**: Video must have standard mobile phone resolution")

        if question_id:
            language_display = Config.get_language_display_name(inferred_language)
            if language_display is None:
                language_display = inferred_language
            st.info(f"🗣️ **Target Language**: {language_display}")
            
            if inferred_task_type != 'Unknown':
                st.info(f"🎯 **Task Type**: {inferred_task_type}")
            else:
                st.info(f"🎯 **Task Type**: Will be inferred from Question ID")
        else:
            st.info(f"🗣️ **Target Language**: Will be inferred from Question ID")
            st.info(f"🎯 **Task Type**: Will be inferred from Question ID")

        if video_file:
            InputScreen._render_video_validation_and_properties(video_file)

    @staticmethod
    def _render_validation_and_navigation():
        """Render validation messages and navigation buttons."""
        if ScreenManager.get_current_screen() != 'input':
            return
        errors = InputScreen._validate_inputs()
        display_errors = [error for error in errors if error != "validation_error"]
        if errors:
            for error in display_errors:
                st.error(f"❌ {error}")
            st.warning("⚠️ Please complete all required fields and ensure authorization before proceeding.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            is_disabled = bool(errors) or st.session_state.get('analysis_in_progress', False)
            button_text = "Validating..." if st.session_state.get('analysis_in_progress', False) else "Start Analysis"
            
            if st.button(button_text, type="primary", use_container_width=True, disabled=is_disabled, key="input_start_analysis_btn"):
                st.session_state.analysis_in_progress = True
                st.rerun()
                
        if st.session_state.get('analysis_in_progress', False) and not st.session_state.get('analysis_started', False):
            st.session_state.analysis_started = True
            InputScreen._handle_start_analysis()

    @staticmethod
    def _handle_start_analysis():
        """Handle the Start Analysis button: authorization, state cleanup and navigation."""
        question_id = st.session_state.question_id.strip()
        alias_email = st.session_state.alias_email.strip()
        
        inferred_language = InputScreen._infer_language_from_question_id(question_id)
        st.session_state.selected_language = inferred_language
        
        inferred_task_type = InputScreen._infer_task_type_from_question_id(question_id)
        st.session_state.task_type = inferred_task_type
        
        try:
            if question_id and alias_email:
                verifier = InputScreen._get_verifier()
                is_authorized, auth_message = verifier.verify_inputs(question_id, alias_email)
                
                if not is_authorized:
                    st.session_state.analysis_in_progress = False
                    st.session_state.analysis_started = False
                    st.session_state.validation_error_shown = True
                    st.error(f"❌ **Authorization Failed**: {auth_message}")
                    time.sleep(3)
                    st.rerun()
                    return
                
                InputScreen._cleanup_previous_analysis_state()
                gc.collect()
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                
                st.session_state.analysis_in_progress = False
                st.session_state.analysis_started = False
                ScreenManager.navigate_to_screen('analysis')
            
        except Exception as e:
            st.session_state.analysis_in_progress = False
            st.session_state.analysis_started = False
            st.session_state.validation_error_shown = True
            logger.error(f"Authorization check error: {e}")
            st.error(f"❌ **Authorization Check Failed**: Unable to verify inputs - {str(e)}")
            st.rerun()
            return

    @staticmethod
    def _cleanup_previous_analysis_state():
        """Clean up previous analysis state from session."""
        if hasattr(st.session_state, 'cached_temp_path'):
            try:
                if os.path.exists(st.session_state.cached_temp_path):
                    os.unlink(st.session_state.cached_temp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up cached temp file: {e}")
            finally:
                del st.session_state.cached_temp_path
        
        if hasattr(st.session_state, 'cached_video_file_name'):
            del st.session_state.cached_video_file_name
            
        st.session_state.analysis_results = None
        st.session_state.analyzer_instance = None
        st.session_state.qa_checker = None
        st.session_state.analysis_in_progress = False
        st.session_state.analysis_started = False
        st.session_state.validation_error_shown = False
        if hasattr(st.session_state, 'analysis_session_id'):
            del st.session_state.analysis_session_id

    @staticmethod
    def _validate_inputs() -> List[str]:
        """Validate all input parameters and return errors for logic but not display."""
        errors = []
        if not st.session_state.question_id.strip():
            errors.append("validation_error")
        if not st.session_state.alias_email.strip():
            errors.append("validation_error")
        elif not InputScreen._is_valid_email(st.session_state.alias_email):
            errors.append("Please enter a valid email address")
        if not st.session_state.video_file:
            errors.append("validation_error")
        else:
            video_validation = st.session_state.get('video_validation', {})
            if video_validation:
                duration_valid = video_validation.get('duration_valid', False)
                resolution_valid = video_validation.get('resolution_valid', False)
                
                if not duration_valid:
                    logger.debug(f"Video duration invalid: {video_validation.get('duration', 0)}s. Video must be at least {video_validation.get('min_duration_required', Config.MIN_VIDEO_DURATION)}s.")
                    errors.append("validation_error")
                if not resolution_valid:
                    logger.debug(f"Video resolution invalid: {video_validation.get('width', 0)}x{video_validation.get('height', 0)}. Video must be in portrait mobile format.")
                    errors.append("validation_error")

        return errors

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Basic email validation."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def _render_video_validation_and_properties(video_file):
        """Render video validation results and properties display."""
        try:
            # Extract video duration and resolution on upload
            session_id = st.session_state.get('session_id', 'temp')
            temp_path, _ = StreamlitInterface.create_temp_video(video_file, session_id)
            
            if not temp_path:
                st.error("❌ Could not process video file")
                return
            
            validation_results = Config.validate_video_properties(temp_path)
            
            if 'error' in validation_results:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
                st.error(f"❌ Video validation failed: {validation_results['error']}")
                return
            
            # Store validation results in session state
            st.session_state.cached_temp_path = temp_path
            st.session_state.cached_video_file_name = video_file.name
            st.session_state.video_validation = validation_results
            
            # Display validation results
            col1, col2 = st.columns(2)
            
            with col1:
                duration = validation_results.get('duration', 0)
                duration_valid = validation_results.get('duration_valid', False)
                min_duration = validation_results.get('min_duration_required', Config.MIN_VIDEO_DURATION)
                
                if duration_valid:
                    st.success(f"✅ **Duration**: {duration:.1f}s (≥ {min_duration}s required)")
                else:
                    st.error(f"❌ **Duration**: {duration:.1f}s (minimum {min_duration}s required)")
                    
            with col2:
                width = validation_results.get('width', 0)
                height = validation_results.get('height', 0)
                resolution_valid = validation_results.get('resolution_valid', False)
                
                if resolution_valid:
                    st.success(f"✅ **Resolution**: {width}x{height} (Portrait mobile format)")
                else:
                    if width >= height:
                        st.error(f"❌ **Resolution**: {width}x{height} (Must be portrait orientation)")
                    else:
                        st.error(f"❌ **Resolution**: {width}x{height} (Not standard mobile portrait format)")
                
        except Exception as e:
            st.error(f"❌ Could not validate video: {str(e)}")
            logger.error(f"Video validation error: {e}")


class AnalysisScreen:
    """Second screen: Analysis progress and results."""
    @staticmethod
    def render():
        """Render the analysis screen."""
        st.title("2️⃣ Video Analysis")
        st.divider()
        
        if st.session_state.analysis_results is not None:
            AnalysisScreen._render_completed_analysis()
        else:
            AnalysisScreen._start_analysis()

    @staticmethod
    def _start_analysis():
        """Start the video analysis process."""
        st.subheader("🔄 Processing Video...")
        overall_progress = st.progress(0, text="Initializing analysis...")
        try:
            session_id = st.session_state.session_id
            
            # Use the already created temp video from validation step
            if (hasattr(st.session_state, 'cached_temp_path') and
                hasattr(st.session_state, 'cached_video_file_name') and
                st.session_state.cached_video_file_name == st.session_state.video_file.name and
                os.path.exists(st.session_state.cached_temp_path)):
                
                video_path = st.session_state.cached_temp_path
                temp_files = [video_path]
                overall_progress.progress(10, text="Using validated video file, initializing analyzer...")
            else:
                # Fallback: create temp video if not already cached
                video_path, temp_files = StreamlitInterface.create_temp_video(
                    st.session_state.video_file, session_id
                )
                overall_progress.progress(10, text="Video processed, initializing analyzer...")

            if not video_path:
                st.error("❌ Failed to prepare video file for analysis")
                return
            with VideoContentAnalyzer(session_id=session_id) as analyzer:
                AnalysisScreen._setup_and_run_analysis(analyzer, video_path, overall_progress)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}")
            AnalysisScreen._render_analysis_error_buttons()

    @staticmethod
    def _setup_and_run_analysis(analyzer, video_path, overall_progress):
        default_rules = create_detection_rules(target_language=st.session_state.selected_language)
        analyzer.rules = default_rules
        rule_types = {}
        for rule in analyzer.rules:
            rule_type = rule.detection_type.value
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        overall_progress.progress(20, text=f"Added analysis rules...")
        
        def progress_callback(percentage, message):
            try:
                overall_progress.progress(percentage, text=message)
            except Exception as e:
                logger.warning(f"Progress update failed: {e}")
        
        results = analyzer.analyze_video(
            video_path,
            frame_interval=st.session_state.frame_interval,
            progress_callback=progress_callback
        )
        overall_progress.progress(90, text="Analysis complete, generating report...")
        analyzer.export_results("results.json")
        
        analyzer.cleanup_temp_files()
        
        video_duration = analyzer._get_video_duration() if hasattr(analyzer, '_get_video_duration') else 0.0
        if video_duration == 0.0:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = total_frames / fps if fps > 0 else 0.0
                    cap.release()
            except Exception as e:
                logger.warning(f"Failed to calculate video duration: {e}")
                video_duration = 0.0
        st.session_state.video_duration = video_duration
        st.session_state.analysis_results = results
        st.session_state.analysis_session_id = analyzer.session_id
        st.session_state.analyzer_instance = analyzer
        st.session_state.qa_checker = QualityAssuranceChecker(results)
        overall_progress.progress(100, text="Analysis complete!")

    @staticmethod
    def _render_analysis_error_buttons():
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Analysis"):
                st.rerun()
        with col2:
            if st.button("⬅️ Back to Input"):
                ScreenManager.navigate_to_screen('input')

    @staticmethod
    def _render_completed_analysis():
        """Render completed analysis results."""
        st.subheader("📊 Analysis Reports")
        results = st.session_state.analysis_results
        
        # Display total analysis time
        total_analysis_time = AnalysisScreen._get_total_analysis_time()
        if total_analysis_time > 0:
            st.info(f"⏱️ **Total Analysis Time:** {total_analysis_time:.2f} seconds")
        
        if st.session_state.qa_checker:
            overall = st.session_state.qa_checker.get_qa_summary()
            
            if overall['passed']:
                st.success(f"✅ **Quality Assurance: PASSED** - All requirements met!")
            else:
                st.error(f"❌ **Quality Assurance: FAILED** - {overall['checks_passed']}/{overall['total_checks']} checks passed")
            
            st.divider()
        
        if results:
            AnalysisScreen._render_individual_detections(results)
        
        AnalysisScreen._render_navigation_buttons()

    @staticmethod
    def _render_individual_detections(results):
        text_results = [r for r in results if 'Text Detection' in r.rule_name]
        language_results = [r for r in results if 'Language Detection' in r.rule_name]
        voice_results = [r for r in results if 'Voice Audibility' in r.rule_name]
        
        if text_results:
            AnalysisScreen._render_text_detections(text_results)
            
        if language_results:
            language_qa_info = AnalysisScreen._get_qa_info_for_result(language_results[0]) if language_results else None
            language_qa_status = ""
            if language_qa_info:
                language_qa_status = " - ✅ PASS" if language_qa_info['passed'] else " - ❌ FAIL"
            
            with st.expander(f"🗣️ Language Fluency Analysis{language_qa_status}", expanded=False):
                for result in language_results[:15]:
                    AnalysisScreen._render_audio_detection_result(result)
                    
        if voice_results:
            voice_qa_info = AnalysisScreen._get_qa_info_for_result(voice_results[0]) if voice_results else None
            voice_qa_status = ""
            if voice_qa_info:
                voice_qa_status = " - ✅ PASS" if voice_qa_info['passed'] else " - ❌ FAIL"
            
            with st.expander(f"👥 Voice Audibility Analysis{voice_qa_status}", expanded=False):
                for result in voice_results[:15]:
                    AnalysisScreen._render_audio_detection_result(result)

    @staticmethod
    def _render_text_detections(text_results):
        """Render text detections."""
        flash_results = [r for r in text_results if '2.5 Flash' in r.rule_name]
        alias_results = [r for r in text_results if 'Alias Name' in r.rule_name]
        eval_results = [r for r in text_results if 'Eval Mode' in r.rule_name]
        
        if flash_results:
            flash_positive = [r for r in flash_results if r.detected]
            
            flash_qa_info = AnalysisScreen._get_qa_info_for_result(flash_results[0]) if flash_results else None
            flash_qa_status = ""
            if flash_qa_info:
                flash_qa_status = " - ✅ PASS" if flash_qa_info['passed'] else " - ❌ FAIL"
            
            if flash_positive:
                with st.expander(f"📝 2.5 Flash Text Detection{flash_qa_status}", expanded=False):
                    AnalysisScreen._render_text_detection_result(flash_positive[0])
                    
            else:
                with st.expander(f"📝 2.5 Flash Text Detection{flash_qa_status}", expanded=False):
                    validation_status = flash_qa_info.get('validation_status', 'not_found') if flash_qa_info else 'not_found'
                    
                    if validation_status == 'incorrect_model':
                        st.error("**Incorrect Model Detected:** 2.5 Pro was found instead of 2.5 Flash")
                        
                        pro_detection_result = None
                        for result in flash_results:
                            if (result.details and 
                                result.details.get('model_validation', {}).get('status') == 'incorrect_model'):
                                pro_detection_result = result
                                break
                        
                        if pro_detection_result:
                            AnalysisScreen._render_incorrect_model_result(pro_detection_result)
                    else:
                        st.error("**2.5 Flash** was not detected in any frame of the video")
                    
                    if flash_qa_info:
                        st.markdown(f"**QA Feedback:** {flash_qa_info['details']}")
        
        if alias_results:
            alias_positive = [r for r in alias_results if r.detected]
            
            alias_qa_info = AnalysisScreen._get_qa_info_for_result(alias_results[0]) if alias_results else None
            alias_qa_status = ""
            if alias_qa_info:
                alias_qa_status = " - ✅ PASS" if alias_qa_info['passed'] else " - ❌ FAIL"
            
            if alias_positive:
                with st.expander(f"📝 Alias Name Text Detection{alias_qa_status}", expanded=False):
                    AnalysisScreen._render_text_detection_result(alias_positive[0])
                    
            else:
                with st.expander(f"📝 Alias Name Text Detection{alias_qa_status}", expanded=False):
                    st.error("**Roaring tiger** was not detected in any frame of the video")
                    
                    if alias_qa_info:
                        st.markdown(f"**QA Feedback:** {alias_qa_info['details']}")
        
        if eval_results:
            eval_positive = [r for r in eval_results if r.detected]
            
            eval_qa_info = AnalysisScreen._get_qa_info_for_result(eval_results[0]) if eval_results else None
            eval_qa_status = ""
            if eval_qa_info:
                eval_qa_status = " - ✅ PASS" if eval_qa_info['passed'] else " - ❌ FAIL"
            
            if eval_positive:
                with st.expander(f"📝 Eval Mode Text Detection{eval_qa_status}", expanded=False):
                    AnalysisScreen._render_text_detection_result(eval_positive[0])
            
            else:
                with st.expander(f"📝 Eval Mode Text Detection{eval_qa_status}", expanded=False):
                    st.error("**Eval Mode: Native Audio Output** was not detected in any frame of the video")
                    if eval_qa_info:
                        st.markdown(f"**QA Feedback:** {eval_qa_info['details']}")

    @staticmethod
    def _render_navigation_buttons():
        submit_enabled = False
        submit_message = ""
        
        if st.session_state.qa_checker:
            overall = st.session_state.qa_checker.get_qa_summary()
            submit_enabled = overall['passed']
            if not submit_enabled:
                failed_checks = overall['total_checks'] - overall['checks_passed']
                submit_message = f"Your video cannot be submitted because it failed {failed_checks} quality check(s). Please review the analysis results and make necessary adjustments."
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                AnalysisScreen._cleanup_and_reset_for_new_analysis()
        with col2:
            if st.button("Submit Video", type="primary", use_container_width=True, disabled=not submit_enabled):
                if submit_enabled:
                    ScreenManager.navigate_to_screen('qa')
                    
        if not submit_enabled and submit_message:
            st.warning(f"⚠️ {submit_message}")

    @staticmethod
    def _cleanup_and_reset_for_new_analysis():
        current_session_id = st.session_state.get('analysis_session_id', st.session_state.get('session_id'))
        if current_session_id:
            session_manager = get_session_manager()
            session_manager.cleanup_session(current_session_id)
        
        try:
            session_manager = get_session_manager()
            session_manager.cleanup_old_sessions()
        except Exception as e:
            logger.debug(f"Error cleaning old sessions: {e}")
        
        ScreenManager._cleanup_previous_session()
        gc.collect()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        ScreenManager.reset_session_for_new_analysis()

    @staticmethod
    def _get_total_analysis_time() -> float:
        """Get the total analysis time from the analyzer instance."""
        try:
            analyzer = st.session_state.get('analyzer_instance')
            if analyzer and hasattr(analyzer, 'performance_metrics'):
                return analyzer.performance_metrics.get('total_analysis_time', 0.0)
            return 0.0
        except Exception as e:
            logger.debug(f"Could not retrieve total analysis time: {e}")
            return 0.0

    @staticmethod
    def _get_qa_info_for_result(result):
        """Get QA information for a specific result."""
        if not st.session_state.qa_checker:
            return None
        
        qa_results = st.session_state.qa_checker.get_detailed_results()
        
        if 'Language Detection' in result.rule_name:
            return qa_results.get('language_fluency')
        elif 'Voice Audibility' in result.rule_name:
            return qa_results.get('voice_audibility')
        elif 'Text Detection' in result.rule_name and '2.5 Flash' in result.rule_name:
            return qa_results.get('flash_presence')
        elif 'Text Detection' in result.rule_name and 'Alias Name' in result.rule_name:
            return qa_results.get('alias_name_presence')
        elif 'Text Detection' in result.rule_name and 'Eval Mode' in result.rule_name:
            return qa_results.get('eval_mode_presence')
        
        return None

    @staticmethod
    def _render_text_detection_result(result):
        """Render text detections."""
        qa_info = AnalysisScreen._get_qa_info_for_result(result)
        header_text = f"### {result.rule_name}"
        with st.container():
            st.markdown(header_text)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**🎯 Detection Details:**")
                st.write(f"• **Timestamp:** {result.timestamp:.2f}s")
                st.write(f"• **Frame:** {result.frame_number}")
                if hasattr(result, 'details') and result.details:
                    if isinstance(result.details, dict):
                        if 'detected_text' in result.details:
                            st.write(f"**📝 Detected Text:**")
                            st.code(result.details['detected_text'], language=None)
            with col2:
                AnalysisScreen._render_screenshot_section(result)

            if qa_info:
                st.markdown(f"**QA Feedback:** {qa_info['details']}")

    @staticmethod
    def _render_incorrect_model_result(result):
        """Render details for incorrect model detection (2.5 Pro found instead of 2.5 Flash)."""
        with st.container():
            st.markdown("### Incorrect Model Detected")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**🎯 Detection Details:**")
                st.write(f"• **Timestamp:** {result.timestamp:.2f}s")
                st.write(f"• **Frame:** {result.frame_number}")
                if hasattr(result, 'details') and result.details:
                    if isinstance(result.details, dict):
                        if 'detected_text' in result.details:
                            st.write(f"**📝 Detected Text:**")
                            st.code(result.details['detected_text'], language=None)
                        model_info = result.details.get('model_validation', {})
            with col2:
                AnalysisScreen._render_screenshot_section(result)

    @staticmethod
    def _render_screenshot_section(result):
        """Render screenshot section."""
        screenshot_path = result.screenshot_path
        if not screenshot_path:
            st.warning("⚠️ No screenshot generated for positive detection" if result.detected else "No screenshot (no detection)")
            return

        if not os.path.isabs(screenshot_path):
            session_id = getattr(st.session_state, 'analysis_session_id', None)
            if session_id:
                session_manager = get_session_manager()
                session_dir = session_manager.get_session_directory(session_id)
                if session_dir:
                    screenshot_path = os.path.join(session_dir, os.path.basename(screenshot_path))

        if not os.path.exists(screenshot_path):
            if result.detected:
                st.error("🚨 Screenshot missing for positive detection!")
                st.caption(f"Expected path: {screenshot_path}")
            else:
                st.caption("Screenshot not available")
            return

        try:
            file_size = os.path.getsize(screenshot_path)
            is_readable = os.access(screenshot_path, os.R_OK)
            if not is_readable or file_size == 0:
                msg = f"Screenshot exists but not readable (size: {file_size})" if result.detected else "Screenshot not readable"
                if result.detected:
                    st.error(msg)
                else:
                    with st.expander("🖼️ Frame Screenshot", expanded=False):
                        st.error(msg)
                return

            is_pro_detection = (
                hasattr(result, 'details') and result.details and
                isinstance(result.details, dict) and
                result.details.get('model_validation', {}).get('status') == 'incorrect_model'
            )
            
            if result.detected or is_pro_detection:
                st.markdown("**Detection Screenshot:**")
                st.image(screenshot_path, width=400)
            else:
                with st.expander("🖼️ Frame Screenshot", expanded=False):
                    st.image(screenshot_path, caption=f"Frame {result.frame_number} at {result.timestamp:.2f}s", width=300)
        except Exception as e:
            st.error(f"Error displaying screenshot: {e}")

    @staticmethod
    def _render_audio_detection_result(result):
        qa_info = AnalysisScreen._get_qa_info_for_result(result)
        
        header_text = f"### {result.rule_name}"
        
        with st.container():
            st.markdown(header_text)
            
            if hasattr(result, 'details') and result.details:
                if isinstance(result.details, dict):
                    if 'Language Detection' in result.rule_name:
                        if 'analysis_failed_reason' in result.details:
                            st.markdown(f"**Analysis Status:** {result.details['analysis_failed_reason']}")
                            st.markdown("**Explanation:** The fluency could not be analyzed because there were no audible voices in the video.")
                        else:
                            if 'detected_language' in result.details:
                                whisper_detected = result.details['detected_language']
                                target_locale = result.details.get('target_language')
                                locale_format = Config.whisper_language_to_locale(whisper_detected, target_locale)
                                display_name = Config.get_language_display_name(locale_format)
                                if display_name is None:
                                    display_name = whisper_detected if whisper_detected else "Unknown"
                                st.markdown(f"**Detected Language:** {display_name}")
                            if 'transcription' in result.details and result.details['transcription']:
                                transcription = result.details['transcription']
                                st.markdown("**Full Audio Transcription:**")
                                st.text_area("Full Audio Transcription", transcription, height=200, disabled=True, key=f"transcript_{result.timestamp}", label_visibility="hidden")
                    elif 'Voice Audibility' in result.rule_name:
                        if 'voice_detected' in result.details:
                            st.markdown(f"**Voice Activity:** {'Yes' if result.details['voice_detected'] else 'No'}")
                        if 'num_audible_voices' in result.details:
                            st.markdown(f"**Number of audible voices:** {result.details['num_audible_voices']}")
                        if 'both_voices_audible' in result.details:
                            st.markdown(f"**Both voices audible:** {result.details['both_voices_audible']}")
            
            if qa_info:
                st.markdown(f"**QA Feedback:** {qa_info['details']}")


class GoogleDriveIntegration:
    """Google Drive integration for creating submission folders."""
    
    def __init__(self):
        self.service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Drive service using service account."""
        try:
            scopes = ['https://www.googleapis.com/auth/drive']
            credentials = None
            
            # Option 1: Use credentials from Streamlit secrets (recommended for production)
            try:
                service_account_info = ConfigurationManager.get_google_service_account_info()
                if service_account_info:
                    credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
                    logger.info("Using Google service account credentials from Streamlit secrets")
            except Exception as e:
                logger.warning(f"Could not load credentials from secrets: {e}")
            
            # Option 2: Fall back to environment variables
            if not credentials:
                credentials_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
                if credentials_json:
                    credentials_info = json.loads(credentials_json)
                    credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
                    logger.info("Using Google service account credentials from environment variable JSON")
                else:
                    # Option 3: Use credentials file path from environment variable
                    credentials_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_PATH')
                    if credentials_path and os.path.exists(credentials_path):
                        credentials = Credentials.from_service_account_file(credentials_path, scopes=scopes)
                        logger.info("Using Google service account credentials from file path")
            
            if not credentials:
                logger.error("No Google credentials found. Please configure credentials in Streamlit secrets or set environment variables")
                self.service = None
                return
            
            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("Google Drive service initialized successfully")
            
        except ImportError:
            logger.error("Google API client libraries not installed")
            self.service = None
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in GOOGLE_SERVICE_ACCOUNT_JSON environment variable")
            self.service = None
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            self.service = None
    
    def create_submission_folder(self, question_id: str, alias_email: str) -> Optional[str]:
        """Create a Google Drive folder for video submission."""
        if not self.service:
            logger.error("Google Drive service not available")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"Video_Submission_{question_id}_{alias_email.split('@')[0]}_{timestamp}"
            
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': []
            }
            
            folder = self.service.files().create(body=folder_metadata, fields='id,name,webViewLink').execute()
            
            permission_metadata = {
                'role': 'writer',
                'type': 'anyone'
            }
            
            self.service.permissions().create(
                fileId=folder.get('id'),
                body=permission_metadata
            ).execute()
            
            folder_link = folder.get('webViewLink')
            logger.info(f"Created Google Drive folder: {folder_name}")
            
            return folder_link
            
        except Exception as e:
            logger.error(f"Failed to create Google Drive folder: {e}")
            return None


class VideoSubmissionScreen:
    """Third screen: Video submission with Google Drive integration."""
    @staticmethod
    def render():
        """Render the video submission screen."""
        st.title("📤 Video Submission")
        st.divider()
        
        VideoSubmissionScreen._initialize_submission_state()
        VideoSubmissionScreen._render_submission_content()

    @staticmethod
    def _initialize_submission_state():
        """Initialize session state for submission screen."""
        if 'drive_folder_link' not in st.session_state:
            st.session_state.drive_folder_link = ""
        if 'folder_generated' not in st.session_state:
            st.session_state.folder_generated = False
        if 'task_submitted' not in st.session_state:
            st.session_state.task_submitted = False
        if 'submission_locked' not in st.session_state:
            st.session_state.submission_locked = False

    @staticmethod
    def _render_submission_content():
        """Render the main submission content."""
        video_file = st.session_state.get('video_file')
        file_name = video_file.name if video_file else "your video file"
        
        st.info(f"""
        **Please follow these steps to submit your video:**
        
        1. Click the "Generate" button below to create a dedicated Google Drive folder for your submission
        2. Once generated, use the provided Google Drive link to upload your video file: **{file_name}**
        3. After uploading your video to the Google Drive folder, click the "Submission Completed" button to complete the process
        
        ⚠️ **Important:** Make sure to upload the exact same video file that was analyzed in the previous steps.
        """)
        
        st.subheader("🔗 Google Drive Folder Generation")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            drive_link = st.text_input(
                "Google Drive Folder Link",
                value=st.session_state.drive_folder_link,
                disabled=True,
                placeholder="Click 'Generate' to create a Google Drive folder link",
                label_visibility="collapsed"
            )
        
        with col2:
            if not st.session_state.folder_generated:
                if st.button("🔄 Generate", use_container_width=True, type="primary"):
                    VideoSubmissionScreen._generate_drive_folder()
            else:
                st.button("✅ Generated", use_container_width=True, type="secondary", disabled=True)
        
        if st.session_state.folder_generated:
            st.markdown(f'🔗 [Open Google Drive Folder]({st.session_state.drive_folder_link})')
        
        if st.session_state.folder_generated:
            st.divider()
            
            if not st.session_state.submission_locked:
                if st.button("✅ Submission Completed", use_container_width=True, type="primary"):
                    VideoSubmissionScreen._handle_task_submission()
            else:
                VideoSubmissionScreen._render_submission_success()

    @staticmethod
    def _generate_drive_folder():
        """Generate Google Drive folder for submission."""
        try:
            with st.spinner("Creating Google Drive folder..."):
                question_id = st.session_state.get('question_id')
                alias_email = st.session_state.get('alias_email')
                
                drive_integration = GoogleDriveIntegration()
                
                if not drive_integration.service:
                    st.error("❌ Google Drive service is not available. Please try again later.")
                    return
                
                folder_link = drive_integration.create_submission_folder(question_id, alias_email)
                
                if folder_link:
                    st.session_state.drive_folder_link = folder_link
                    st.session_state.folder_generated = True
                    st.rerun()
                else:
                    st.error("❌ Failed to create Google Drive folder. Please try again.")
                    
        except Exception as e:
            logger.error(f"Error generating Google Drive folder: {e}")
            st.error(f"❌ An error occurred while creating the folder: {str(e)}")

    @staticmethod
    def _handle_task_submission():
        """Handle the task submission process."""
        try:
            st.session_state.task_submitted = True
            st.session_state.submission_locked = True
            
            question_id = st.session_state.get('question_id')
            alias_email = st.session_state.get('alias_email')
            timestamp = datetime.now().isoformat()
            
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error handling task submission: {e}")
            st.error(f"❌ An error occurred during submission: {str(e)}")

    @staticmethod
    def _render_submission_success():
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️ Back to Results", use_container_width=True):
                ScreenManager.navigate_to_screen('analysis')
        
        with col2:
            if st.button("🔄 Start New Analysis", use_container_width=True, type="primary"):
                VideoSubmissionScreen._start_new_analysis_session()

    @staticmethod
    def _start_new_analysis_session():
        """Start a completely new analysis session."""
        current_session_id = st.session_state.get('analysis_session_id', st.session_state.get('session_id'))
        if current_session_id:
            session_manager = get_session_manager()
            session_manager.cleanup_session(current_session_id)
        
        try:
            session_manager = get_session_manager()
            session_manager.cleanup_old_sessions()
        except Exception as e:
            logger.debug(f"Error cleaning old sessions: {e}")
        
        ScreenManager._cleanup_previous_session()
        
        gc.collect()
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        ScreenManager.reset_session_for_new_analysis()


class ConfigurationManager:
    """Centralized configuration management."""
    
    _config_cache: Optional[Dict[str, str]] = None
    _cache_lock = threading.Lock()
    _cache_timestamp: float = 0
    _cache_ttl: float = 300
    
    @classmethod
    def get_secure_config(cls, force_refresh: bool = False) -> Dict[str, str]:
        """Get secure configuration from Streamlit secrets."""
        current_time = time.time()
        
        if (not force_refresh and cls._config_cache is not None and 
            (current_time - cls._cache_timestamp) < cls._cache_ttl):
            return cls._config_cache.copy()
        
        with cls._cache_lock:
            if (not force_refresh and cls._config_cache is not None and 
                (current_time - cls._cache_timestamp) < cls._cache_ttl):
                return cls._config_cache.copy()
            
            try:
                if "google" not in st.secrets:
                    raise ValueError("Google configuration missing from secrets")
                    
                google_config = st.secrets["google"]
                required_keys = ["apps_script_url", "verifier_sheet_url", "verifier_sheet_id"]
                
                for key in required_keys:
                    if key not in google_config:
                        raise ValueError(f"Required configuration missing: {key}")
                
                config = {
                    "apps_script_url": google_config["apps_script_url"],
                    "verifier_sheet_url": google_config["verifier_sheet_url"],
                    "verifier_sheet_id": google_config["verifier_sheet_id"],
                    "verifier_sheet_name": google_config["verifier_sheet_name"]
                }
                
                cls._config_cache = config
                cls._cache_timestamp = current_time
                
                return config.copy()
                
            except Exception as e:
                logger.error(f"Secure configuration error: {e}")
                raise RuntimeError(f"Cannot load secure configuration: {e}") from e
    
    @classmethod
    def get_google_service_account_info(cls) -> Optional[Dict[str, Any]]:
        """Get Google service account credentials from secrets."""
        try:
            if "google_service_account" in st.secrets:
                return dict(st.secrets["google_service_account"])
            return None
        except Exception as e:
            logger.error(f"Failed to load Google service account info: {e}")
            return None


class TextMatcher:
    """Text matching with OCR error correction and fuzzy matching."""
    SIMILARITY_THRESHOLDS = {
        'word_match': 0.75,
        'partial_match': 0.75,
        'character_match': 0.6,
        'character_strict': 0.8
    }
    
    OCR_CORRECTIONS: Dict[str, str] = {
        '2s flash': '2.5 flash', '2.s flash': '2.5 flash', '2,5 flash': '2.5 flash',
        '25 flash': '2.5 flash',
        '2s pro': '2.5 pro', '2.s pro': '2.5 pro', '2,5 pro': '2.5 pro',
        '25 pro': '2.5 pro',
        'fiash': 'flash', 'flasb': 'flash', 'fash': 'flash',
        'flashy': 'flash', 'flast': 'flash', 'flach': 'flash',
        'pno': 'pro', 'prn': 'pro', 'pro.': 'pro',
        'evai mode': 'eval mode', 'eval rode': 'eval mode',
        'native audio output': 'native audio output',
        'roannng tiger': 'roaring tiger', 'roaring tiqer': 'roaring tiger', 'roaring tigee': 'roaring tiger',
        'roaring tger': 'roaring tiger', 'roaring ticer': 'roaring tiger', 'roanng tiger': 'roaring tiger',
        'roarmg tiger': 'roaring tiger', 'roarrng tiger': 'roaring tiger', 'roarirg tiger': 'roaring tiger',
        'roaring.tiger': 'roaring tiger', 'roaring . tiger': 'roaring tiger', 'roaring. tiger': 'roaring tiger',
        'roaring .tiger': 'roaring tiger', 'roaring..tiger': 'roaring tiger',
        'rn': 'm', 'vv': 'w', '1': 'l'
    }

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate text similarity using normalized Levenshtein distance."""
        if not text1 or not text2:
            return 0.0
        text1, text2 = text1.lower().strip(), text2.lower().strip()
        if text1 == text2:
            return 1.0
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0
        return TextMatcher._compute_levenshtein_similarity(text1, text2, len1, len2)

    @staticmethod
    def _compute_levenshtein_similarity(text1: str, text2: str, len1: int, len2: int) -> float:
        """Compute Levenshtein similarity using space-optimized algorithm."""
        if len(text1) > len(text2):
            text1, text2 = text2, text1
        len1, len2 = len(text1), len(text2)
        prev_row = list(range(len1 + 1))
        curr_row = [0] * (len1 + 1)
        for j in range(1, len2 + 1):
            curr_row[0] = j
            for i in range(1, len1 + 1):
                if text1[i-1] == text2[j-1]:
                    curr_row[i] = prev_row[i-1]
                else:
                    curr_row[i] = 1 + min(
                        prev_row[i],
                        curr_row[i-1],
                        prev_row[i-1]
                    )
            prev_row, curr_row = curr_row, prev_row
        edit_distance = prev_row[len1]
        return max(0.0, 1.0 - (edit_distance / len2))

    @classmethod
    def apply_ocr_corrections(cls, text: str) -> str:
        """Apply common OCR error corrections to improve recognition accuracy."""
        if not text:
            return text
        corrected = text.lower()
        for incorrect, correct in cls.OCR_CORRECTIONS.items():
            corrected = corrected.replace(incorrect, correct)
        return corrected

    @classmethod
    def match_text(cls, detected: str, expected: str, enable_fuzzy: bool = True) -> Tuple[bool, str]:
        """Precise text matching that requires more exact matches to avoid false positives."""
        if not detected or not expected:
            return False, 'empty_input'
        detected_lower = detected.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Special case for "roaring tiger" - alias name
        if expected_lower == "roaring tiger":
            if cls._match_roaring_tiger_variants(detected_lower):
                return True, 'roaring_tiger_variant_match'
        
        # Strategy 1: Exact phrase match (word boundaries respected)
        if cls._exact_phrase_match(detected_lower, expected_lower):
            return True, 'exact_phrase_match'
        # Strategy 2: OCR error correction with word boundaries
        corrected_detected = cls.apply_ocr_corrections(detected_lower)
        if cls._exact_phrase_match(corrected_detected, expected_lower):
            return True, 'ocr_corrected_phrase'
        # Strategy 3: Reverse check for very specific cases
        if (len(detected_lower) >= 6 and
            len(expected_lower) > len(detected_lower) * 1.5 and
            detected_lower in expected_lower and
            ' ' not in detected_lower):
            word_significance = len(detected_lower) / len(expected_lower)
            if word_significance >= 0.25:
                return True, 'reverse_substring'
        if not enable_fuzzy:
            return False, 'no_fuzzy_match'
        expected_words = expected_lower.split()
        detected_words = detected_lower.split()
        overall_similarity = cls.calculate_similarity(detected_lower, expected_lower)
        if overall_similarity >= cls.SIMILARITY_THRESHOLDS['character_strict']:
            return True, f'character_similarity_{overall_similarity:.2f}'
        return False, f'no_match_similarity_{overall_similarity:.2f}'

    @classmethod
    def _match_roaring_tiger_variants(cls, detected_text: str) -> bool:
        """Matching for roaring tiger with various separators."""
        pattern = r'\broaring\s*[.\s_-]*\s*tiger\b'
        return bool(re.search(pattern, detected_text))

    @classmethod
    def _exact_phrase_match(cls, detected: str, expected: str) -> bool:
        expected_words = expected.split()
        if len(expected_words) == 1:
            pattern = r'\b' + re.escape(expected) + r'\b'
            return bool(re.search(pattern, detected))
        else:
            escaped_words = [re.escape(word) for word in expected_words]
            pattern = r'\b' + r'\s+'.join(escaped_words) + r'\b'
            return bool(re.search(pattern, detected))


class AudioAnalyzer:
    """Audio processing for voice and language detection with voice separation capabilities."""
    
    _shared_whisper_model = None
    _whisper_model_lock = threading.RLock()
    _model_loading_condition = threading.Condition(_whisper_model_lock)
    _model_loading = False
    
    def __init__(self):
        self.whisper_model = None
        self.supported_languages = Config.get_supported_languages()
        self.voice_features_cache = {}
    
    def load_whisper_model(self, model_size: str = "base") -> bool:
        """Load Whisper transcription model with thread-safe shared loading."""
        try:
            model_size = "base"
            
            with AudioAnalyzer._model_loading_condition:
                if AudioAnalyzer._shared_whisper_model is not None:
                    self.whisper_model = AudioAnalyzer._shared_whisper_model
                    return True
                
                while AudioAnalyzer._model_loading:
                    AudioAnalyzer._model_loading_condition.wait()
                    
                    if AudioAnalyzer._shared_whisper_model is not None:
                        self.whisper_model = AudioAnalyzer._shared_whisper_model
                        return True
                
                AudioAnalyzer._model_loading = True
                try:
                    AudioAnalyzer._shared_whisper_model = whisper.load_model(model_size)
                    self.whisper_model = AudioAnalyzer._shared_whisper_model
                    logger.info(f"Whisper model '{model_size}' loaded successfully")
                    return True
                finally:
                    AudioAnalyzer._model_loading = False
                    AudioAnalyzer._model_loading_condition.notify_all()
                    
        except Exception as e:
            logger.error(f"Whisper model load failed: {e}")
            with AudioAnalyzer._model_loading_condition:
                AudioAnalyzer._model_loading = False
                AudioAnalyzer._model_loading_condition.notify_all()
            return False
    
    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio track from video."""
        try:
            audio = AudioSegment.from_file(video_path)
            audio.export(output_path, format="wav")
            return True
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    def analyze_full_audio_fluency(self, audio_path: str, target_language: str) -> Optional[Dict[str, Any]]:
        """Analyze language fluency for entire audio file."""
        if self.whisper_model is None:
            return {'detected_language': 'unknown', 'is_fluent': False, 'confidence': 0.0}
        
        try:
            whisper_language = Config.locale_to_whisper_language(target_language)
            
            result = self.whisper_model.transcribe(
                audio_path, 
                task="transcribe", 
                fp16=False,
                condition_on_previous_text=False,
                temperature=0.0
            )
            
            detected_language = result.get('language', 'unknown')
            transcription = result.get('text', '').strip()
            
            words = transcription.split()
            total_words = len(words)
            
            no_space_languages = {'ja', 'zh', 'ko', 'th'}
            if detected_language in no_space_languages and total_words == 1:
                char_per_word = 2.5 if detected_language in {'ja', 'ko'} else 2.0
                total_words = max(1, int(len(transcription.strip()) / char_per_word))
            
            if total_words == 0:
                return None
            
            is_target_language = detected_language == whisper_language
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            if is_target_language and total_words >= 3 and len(transcription) >= 10:
                fluency_score = 1.0
                is_fluent = True
            else:
                fluency_score = 0.0
                is_fluent = False
            
            return {
                'detected_language': detected_language,
                'target_language': target_language,
                'whisper_language': whisper_language,
                'is_fluent': is_fluent,
                'fluency_score': fluency_score,
                'confidence': result.get('avg_logprob', fluency_score),
                'transcription': transcription,
                'total_words': total_words,
                'avg_word_length': avg_word_length,
                'full_audio_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Full audio language analysis failed: {e}")
            return {
                'detected_language': 'unknown',
                'is_fluent': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def detect_voice_activity(self, audio_path: str, frame_length: int = 2048, 
                             hop_length: int = 512) -> Dict[str, Any]:
        """Detect voice activity in audio using multi-feature approach."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate multiple features for robust VAD
            # 1. RMS energy
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 2. Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 3. Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            
            # 4. Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
            
            # Use adaptive threshold based on noise floor estimation
            sorted_rms = np.sort(rms)
            noise_floor = np.mean(sorted_rms[:int(len(sorted_rms) * 0.1)])  # Bottom 10% as noise
            energy_threshold = noise_floor * 3
            
            # Alternative energy threshold if noise floor is too low
            if energy_threshold < np.percentile(rms, 10):
                energy_threshold = np.percentile(rms, 10)
            
            # ZCR: Human speech has moderate ZCR
            zcr_low = np.percentile(zcr, 20)
            zcr_high = np.percentile(zcr, 80)
            
            # Spectral: Human voice typically in 85-3000 Hz range
            voice_freq_mask = (spectral_centroid > 85) & (spectral_centroid < 3000)
            
            # Combined voice activity detection
            # Method 1: Energy + ZCR
            voice_activity_energy = (rms > energy_threshold) & (zcr > zcr_low) & (zcr < zcr_high)
            
            # Method 2: Spectral characteristics
            voice_activity_spectral = voice_freq_mask & (rms > noise_floor * 1.5)
            
            # Combine both methods (OR operation for sensitivity)
            voice_activity = voice_activity_energy | voice_activity_spectral
            
            # Apply median filter to smooth out short gaps
            voice_activity = median_filter(voice_activity.astype(float), size=7) > 0.5
            
            # Post-processing: remove very short segments
            voice_activity = self._remove_short_segments(voice_activity, min_length=5)
            
            # Calculate percentage of frames with voice
            voice_ratio = np.sum(voice_activity) / len(voice_activity) if len(voice_activity) > 0 else 0
            
            # Get time stamps of voice activity
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            voice_segments = []
            
            # Find continuous voice segments
            in_segment = False
            start_time = 0
            
            for i, is_voice in enumerate(voice_activity):
                if is_voice and not in_segment:
                    start_time = times[i]
                    in_segment = True
                elif not is_voice and in_segment:
                    end_time = times[i]
                    if end_time - start_time > 0.2:  # Minimum 200ms segments
                        voice_segments.append((start_time, end_time))
                    in_segment = False
            
            # Handle last segment
            if in_segment and times[-1] - start_time > 0.2:
                voice_segments.append((start_time, times[-1]))

            # If we have good energy throughout but no segments, likely continuous speech
            if len(voice_segments) == 0 and np.mean(rms) > noise_floor * 5:
                voice_segments = [(0, float(len(y) / sr))]
                voice_ratio = 0.8  # Assume 80% voice if continuous energy
            
            return {
                'has_voice': voice_ratio > 0.02 or len(voice_segments) > 0,
                'voice_ratio': float(voice_ratio),
                'num_segments': len(voice_segments),
                'total_voice_duration': sum(end - start for start, end in voice_segments),
                'segments': voice_segments,
                'audio_duration': float(len(y) / sr),
                'mean_energy': float(np.mean(rms)),
                'energy_threshold': float(energy_threshold),
                'noise_floor': float(noise_floor)
            }
            
        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            return {
                'has_voice': False,
                'voice_ratio': 0.0,
                'num_segments': 0,
                'total_voice_duration': 0.0,
                'segments': [],
                'audio_duration': 0.0
            }
    
    def _remove_short_segments(self, activity: np.ndarray, min_length: int) -> np.ndarray:
        """Remove segments shorter than min_length frames."""
        result = activity.copy()
        
        changes = np.diff(np.concatenate(([0], activity.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for start, end in zip(starts, ends):
            if end - start < min_length:
                result[start:end] = False
                
        return result
    
    def _find_histogram_peaks(self, hist: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find peaks in histogram for bimodal distribution detection."""
        peaks = []
        
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.05:
                if not peaks or all(abs(i - p) >= min_distance for p in peaks):
                    peaks.append(i)
        
        return peaks
    
    def analyze_speaker_count(self, audio_path: str, vad_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze audio to estimate the number of distinct speakers.
        Uses improved approach based on spectral features, pitch, and temporal analysis.
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # First check if there's any substantial audio
            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                return {
                    'estimated_speakers': 0,
                    'confidence': 0.0,
                    'feature_variance_ratio': 0.0,
                    'has_multiple_speakers': False,
                    'audio_features': {}
                }
            
            # Extract MFCC features for speaker characteristics
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Calculate spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # Calculate temporal variations in MFCCs (speaker changes)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Combine features for analysis
            # Focus on MFCC variations which are good for speaker discrimination
            mfcc_temporal_var = np.var(mfccs, axis=1)
            mfcc_delta_var = np.var(mfcc_delta, axis=1)
            
            # Advanced speaker separation using temporal dynamics
            # Analyze how MFCCs change over time to detect speaker transitions
            hop_length = 512  # Default hop length
            window_size = int(sr * 0.5 / hop_length)  # 0.5 second windows
            if mfccs.shape[1] > window_size * 2:
                mfcc_windows = []
                # Use smaller step size for better resolution
                step_size = max(1, window_size // 4)  # 125ms steps
                
                for i in range(0, mfccs.shape[1] - window_size, step_size):
                    window = mfccs[:, i:i+window_size]
                    mfcc_windows.append(np.mean(window, axis=1))
                
                # Calculate distances between windows to detect speaker changes
                if len(mfcc_windows) > 2:
                    mfcc_windows = np.array(mfcc_windows)
                    distances = []
                    for i in range(1, len(mfcc_windows)):
                        dist = np.linalg.norm(mfcc_windows[i] - mfcc_windows[i-1])
                        distances.append(dist)
                    
                    # High distance variance indicates speaker changes
                    distance_variance = np.var(distances) if distances else 0
                    max_distance = np.max(distances) if distances else 0
                    mean_distance = np.mean(distances) if distances else 0
                    
                else:
                    distance_variance = 0
                    max_distance = 0
            else:
                distance_variance = 0
                max_distance = 0
            
            # Calculate feature statistics
            spectral_var = np.var(spectral_centroids)
            bandwidth_var = np.var(spectral_bandwidth)
            
            # Normalize variations
            mfcc_norm_var = mfcc_temporal_var / (np.max(mfcc_temporal_var) + 1e-8)
            delta_norm_var = mfcc_delta_var / (np.max(mfcc_delta_var) + 1e-8)
            
            # Multiple speakers cause higher variation in mid-range MFCCs (2-8)
            mid_mfcc_var_ratio = np.mean(mfcc_norm_var[2:8])
            delta_var_ratio = np.mean(delta_norm_var[2:8])
            
            # Pitch analysis for speaker discrimination
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 80 and pitch < 400:  # Human voice range
                    pitch_values.append(pitch)
            
            pitch_variance = np.std(pitch_values) if len(pitch_values) > 10 else 0
            pitch_range = max(pitch_values) - min(pitch_values) if len(pitch_values) > 10 else 0
            
            speaker_count = 1  # Default to 1 speaker if voice is present
            confidence = 0.7   # Base confidence
            
            # Multiple speaker indicators with weighted scoring
            multi_speaker_score = 0.0
            indicators = []
            
            # 1. MFCC variation
            if mid_mfcc_var_ratio > 0.3:
                score_contrib = min(0.25, (mid_mfcc_var_ratio - 0.3) * 2.5)
                multi_speaker_score += score_contrib
                indicators.append(f"mfcc_var({mid_mfcc_var_ratio:.2f})={score_contrib:.2f}")
                
            if delta_var_ratio > 0.35:
                score_contrib = min(0.25, (delta_var_ratio - 0.35) * 2.0)
                multi_speaker_score += score_contrib
                indicators.append(f"delta_var({delta_var_ratio:.2f})={score_contrib:.2f}")
                
            # 2. Temporal dynamics
            # Only contribute if there's significant variance AND distance
            if distance_variance > 1.0:
                score_contrib = min(0.25, (distance_variance - 1.0) / 3)
                multi_speaker_score += score_contrib
                indicators.append(f"temporal_var({distance_variance:.2f})={score_contrib:.2f}")
            
            if max_distance > 7.0:
                score_contrib = min(0.15, (max_distance - 7.0) / 15)
                multi_speaker_score += score_contrib
                indicators.append(f"max_transition({max_distance:.2f})={score_contrib:.2f}")
                
            # 3. Pitch-based analysis 
            if len(pitch_values) > 100:  # Need sufficient pitch data
                # Check for consistent pitch patterns vs varied patterns
                pitch_percentiles = np.percentile(pitch_values, [10, 25, 50, 75, 90])
                pitch_iqr = pitch_percentiles[3] - pitch_percentiles[1]  # Interquartile range
                
                # Only contribute if BOTH IQR and range are high
                if pitch_iqr > 60 and pitch_range > 180:
                    score_contrib = min(0.2, (pitch_iqr - 60) / 100 * 0.2)
                    multi_speaker_score += score_contrib
                    indicators.append(f"pitch_spread({pitch_iqr:.1f})={score_contrib:.2f}")
                
                # Bimodal pitch distribution is strong indicator
                if len(pitch_values) > 500:
                    pitch_hist, pitch_bins = np.histogram(pitch_values, bins=20)
                    pitch_hist_norm = pitch_hist / np.sum(pitch_hist)
                    peaks = self._find_histogram_peaks(pitch_hist_norm)
                    if len(peaks) >= 2:
                        # Check if peaks are far apart (different speakers)
                        peak_distance = abs(peaks[1] - peaks[0])
                        if peak_distance > 3:
                            score_contrib = min(0.25, peak_distance / 10 * 0.25)
                            multi_speaker_score += score_contrib
                            indicators.append(f"bimodal_pitch={len(peaks)}peaks,dist={peak_distance}")
            
            # 4. Audio duration and voice ratio analysis
            if vad_info is None:
                vad_info = self.voice_features_cache.get('vad_results', {})
            voice_ratio = vad_info.get('voice_ratio', 0)
            total_duration = vad_info.get('audio_duration', 0)
            
            # 1 voice has much lower voice ratio
            if voice_ratio > 0.5:  # High voice ratio indicates conversation
                score_contrib = min(0.3, (voice_ratio - 0.5) * 0.6)
                multi_speaker_score += score_contrib
                indicators.append(f"voice_ratio({voice_ratio:.2f})={score_contrib:.2f}")
            elif voice_ratio < 0.35:  # Low voice ratio strongly suggests single speaker
                # Strong negative contribution to multi-speaker score
                multi_speaker_score -= 0.4
                indicators.append(f"low_voice_ratio({voice_ratio:.2f})=-0.4")
                
            # Additional penalty for short duration with low voice ratio
            if total_duration < 15 and voice_ratio < 0.4:
                multi_speaker_score -= 0.3
                indicators.append(f"short_monologue(dur={total_duration:.1f})=-0.3")
            
            # 5. Duration check
            if total_duration > 60 and voice_ratio > 0.6:  # Long conversation
                multi_speaker_score += 0.2
                indicators.append(f"long_conversation(dur={total_duration:.1f},ratio={voice_ratio:.2f})=0.2")
            
            # 6. Special case detection
            if pitch_range > 200 and distance_variance > 1.0 and voice_ratio > 0.4:
                speaker_count = 2
                confidence = 0.9
            # High voice ratio with reasonable duration is strong indicator
            elif voice_ratio > 0.6 and total_duration > 30:
                speaker_count = 2
                confidence = 0.85
            # Clear single speaker pattern
            elif voice_ratio < 0.35 and total_duration < 20:
                speaker_count = 1
                confidence = 0.85
            else:
                # Determine speaker count based on score
                # Require positive score AND minimum indicators for 2 speakers
                if multi_speaker_score >= 0.4 and voice_ratio > 0.35:
                    speaker_count = 2
                    confidence = min(0.9, 0.6 + multi_speaker_score * 0.3)
                else:
                    speaker_count = 1  # Default to 1 speaker
                    # For 1 speaker, boost confidence if indicators are low or negative
                    if multi_speaker_score < 0:
                        confidence = 0.9  # Very confident it's 1 speaker
                    elif multi_speaker_score < 0.2:
                        confidence = 0.8
                    else:
                        confidence = max(0.5, 0.7 - multi_speaker_score * 0.5)
            
            # Store VAD results for use in analysis if not passed directly
            if vad_info and 'voice_ratio' in vad_info:
                self.voice_features_cache['vad_results'] = vad_info
            
            return {
                'estimated_speakers': speaker_count,
                'confidence': float(confidence),
                'feature_variance_ratio': float(mid_mfcc_var_ratio),
                'has_multiple_speakers': speaker_count >= 2,
                'multi_speaker_score': float(multi_speaker_score),
                'distance_variance': float(distance_variance),
                'max_distance': float(max_distance),
                'audio_features': {
                    'mfcc_variance': float(np.mean(mfcc_temporal_var)),
                    'mfcc_delta_variance': float(np.mean(mfcc_delta_var)),
                    'spectral_variance': float(spectral_var),
                    'pitch_variance': float(pitch_variance),
                    'pitch_range': float(pitch_range),
                    'num_pitch_values': len(pitch_values)
                }
            }
            
        except Exception as e:
            logger.error(f"Speaker count analysis failed: {e}")
            return {
                'estimated_speakers': 0,
                'confidence': 0.0,
                'feature_variance_ratio': 0.0,
                'has_multiple_speakers': False,
                'audio_features': {}
            }
    
    def analyze_voice_audibility(self, audio_path: str) -> Dict[str, Any]:
        """
        Comprehensive voice audibility analysis combining VAD and speaker detection.
        Returns whether there are 0, 1, or 2 audible voices.
        """
        try:
            vad_results = self.detect_voice_activity(audio_path)
            
            if not vad_results['has_voice'] or (vad_results['voice_ratio'] < 0.02 and vad_results['num_segments'] == 0):
                return {
                    'num_audible_voices': 0,
                    'passed_qa': False,
                    'confidence': 0.9,
                    'details': 'No audible voices detected',
                    'vad_info': vad_results,
                    'speaker_info': None,
                    'voice_ratio': vad_results['voice_ratio'],
                    'total_voice_duration': vad_results['total_voice_duration'],
                    'has_multiple_speakers': False
                }
            
            speaker_results = self.analyze_speaker_count(audio_path, vad_info=vad_results)
            
            # Get initial speaker count
            num_voices = speaker_results['estimated_speakers']
            
            # If speaker analysis failed but VAD found voice, assume at least 1 speaker
            if num_voices == 0 and vad_results['voice_ratio'] > 0.1:
                num_voices = 1
                speaker_results['confidence'] = 0.5
            
            # Validate speaker count with VAD results
            if vad_results['voice_ratio'] < 0.05 and vad_results['num_segments'] < 2:
                # Very minimal voice activity - cap at 1 speaker
                if num_voices > 1:
                    num_voices = 1
                    speaker_results['confidence'] *= 0.7
            
            # QA passes only if exactly 2 voices are detected
            passed_qa = (num_voices == 2)
            
            # Calculate combined confidence based on both analyses
            # Weight speaker confidence more heavily as it's more specific
            vad_confidence = min(1.0, vad_results['voice_ratio'] * 2)
            speaker_confidence = speaker_results['confidence']
            
            # Combined confidence: 30% VAD, 70% speaker analysis
            confidence = (0.3 * vad_confidence + 0.7 * speaker_confidence)
            
            # Boost confidence if both analyses agree
            if num_voices == 2 and speaker_results['has_multiple_speakers'] and vad_results['voice_ratio'] > 0.3:
                confidence = min(0.95, confidence * 1.1)
            
            # Generate detailed description
            if num_voices == 0:
                details = 'No audible voices detected'
            elif num_voices == 1:
                details = 'Only one audible voice detected'
            elif num_voices == 2:
                details = 'Two audible voices detected'
            else:
                details = f'{num_voices} voices detected (expected 2)'
            
            # Add voice activity info to description
            if vad_results['voice_ratio'] > 0:
                details += f' - {vad_results["voice_ratio"]:.1%} voice activity'
            
            return {
                'num_audible_voices': num_voices,
                'passed_qa': passed_qa,
                'confidence': float(confidence),
                'details': details,
                'vad_info': vad_results,
                'speaker_info': speaker_results,
                'voice_ratio': vad_results['voice_ratio'],
                'total_voice_duration': vad_results['total_voice_duration'],
                'has_multiple_speakers': speaker_results.get('has_multiple_speakers', False)
            }
            
        except Exception as e:
            logger.error(f"Voice audibility analysis failed: {e}")
            return {
                'num_audible_voices': 0,
                'passed_qa': False,
                'confidence': 0.0,
                'details': f'Analysis failed: {str(e)}',
                'vad_info': None,
                'speaker_info': None,
                'voice_ratio': 0.0,
                'total_voice_duration': 0.0,
                'has_multiple_speakers': False
            }


class VideoContentAnalyzer:
    """Video content analysis system."""
    
    def __init__(self, session_id: Optional[str] = None) -> None:
        gc.collect()
        
        self.session_manager = get_session_manager()
        self.session_id = session_id or self.session_manager.generate_session_id()
        
        self.session_dir = self.session_manager.create_session(self.session_id)
        
        self.rules: List[DetectionRule] = []
        self.results: List[DetectionResult] = []
        self.temp_files: List[str] = []
        self.screenshot_files: List[str] = []
        self._lock = threading.RLock()
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self.analysis_start_time: Optional[float] = None
        self.analysis_end_time: Optional[float] = None
        self.total_frames_processed: int = 0
        self.video_duration: float = 0.0
        self.progress_callback: Optional[callable] = None
        
        self.performance_metrics = {
            'flash_detection_time': 0.0,
            'alias_detection_time': 0.0,
            'eval_mode_detection_time': 0.0,
            'audio_analysis_time': 0.0,
            'total_analysis_time': 0.0,
            'frames_analyzed': 0,
            'audio_extraction_time': 0.0,
            'ocr_processing_time': 0.0
        }
        
        try:
            self.audio_analyzer = self._initialize_audio_analyzer()
        except Exception as e:
            self.audio_analyzer = None
    
    def _update_progress(self, percentage: float, message: str):
        """Update progress via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(percentage / 100.0, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    class ProgressTracker:
        """Context manager for tracking progress within a range."""
        
        def __init__(self, analyzer, start_pct: float, end_pct: float, operation_name: str):
            self.analyzer = analyzer
            self.start_pct = start_pct
            self.end_pct = end_pct
            self.operation_name = operation_name
            self.range_size = end_pct - start_pct
            
        def update(self, progress_ratio: float, message: str = None):
            """Update progress within the allocated range."""
            current_pct = self.start_pct + (progress_ratio * self.range_size)
            display_message = message or f"{self.operation_name}: {progress_ratio*100:.1f}%"
            self.analyzer._update_progress(current_pct, display_message)
        
        def __enter__(self):
            self.analyzer._update_progress(self.start_pct, f"Starting {self.operation_name}...")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.analyzer._update_progress(self.end_pct, f"{self.operation_name} completed")
            else:
                self.analyzer._update_progress(self.end_pct, f"{self.operation_name} failed")
    
    def _initialize_audio_analyzer(self) -> Optional[AudioAnalyzer]:
        """Initialize audio analyzer."""
        try:
            analyzer = AudioAnalyzer()
            return analyzer
        except Exception as e:
            return None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            with self._lock:
                self._is_processing = False
        finally:
            pass
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        with self._lock:
            try:
                self._is_processing = False
                
                if self.temp_files:
                    for temp_file in self.temp_files:
                        try:
                            if os.path.exists(temp_file):
                                os.unlink(temp_file)
                                logger.debug(f"Removed temp file: {temp_file}")
                        except Exception as e:
                            logger.debug(f"Could not remove temp file {temp_file}: {e}")
                    self.temp_files.clear()
                    
                logger.debug(f"Session {self.session_id} analysis completed, directory preserved")
                
            except Exception as e:
                logger.error(f"Temp cleanup error for session {self.session_id}: {e}")
    
    def analyze_video(self, video_path: str, 
                     frame_interval: float = Config.DEFAULT_FRAME_INTERVAL,
                     progress_callback: callable = None) -> List[DetectionResult]:
        """Thread-safe video analysis with progress tracking."""
        
        if not self._processing_lock.acquire(blocking=False):
            raise RuntimeError("Analysis already in progress for this session")
        
        try:
            with self._lock:
                self._is_processing = True
                self.progress_callback = progress_callback
            
            self._validate_analysis_inputs(video_path, frame_interval)
            
            active_rules = self.rules
            if not active_rules:
                return []
            
            self._initialize_analysis()
            
            try:
                analysis_context = self._create_analysis_context(video_path)
                self._execute_analysis(analysis_context, frame_interval)
                return self.results
                
            except Exception as e:
                logger.error(f"Analysis failed (session: {self.session_id}): {e}")
                raise RuntimeError(f"Analysis failed: {e}") from e
            
        finally:
            with self._lock:
                self._is_processing = False
                self.progress_callback = None
            self._processing_lock.release()
    
    def _validate_analysis_inputs(self, video_path: str, frame_interval: float) -> None:
        """Input validation."""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
    
    def _initialize_analysis(self) -> None:
        """Initialize analysis state with thread safety."""
        with self._lock:
            self.results = []
            self.analysis_start_time = datetime.now().timestamp()
            self.analysis_end_time = None
            self.total_frames_processed = 0
            self.performance_metrics = {
                'flash_detection_time': 0.0,
                'alias_detection_time': 0.0,
                'eval_mode_detection_time': 0.0,
                'audio_analysis_time': 0.0,
                'total_analysis_time': 0.0,
                'frames_analyzed': 0,
                'audio_extraction_time': 0.0,
                'ocr_processing_time': 0.0
            }
    
    def _create_analysis_context(self, video_path: str) -> Dict[str, Any]:
        """Create analysis context."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if duration > Config.MAX_VIDEO_DURATION:
                cap.release()
                raise ValueError(f"Video too long: {duration:.1f}s (max: {Config.MAX_VIDEO_DURATION}s)")
            
            self.video_duration = duration
            
            audio_path = self._setup_audio_analysis(video_path)
            
            return {
                'cap': cap,
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'audio_path': audio_path,
                'video_path': video_path
            }
            
        except Exception as e:
            cap.release()
            raise
    
    def _execute_analysis(self, context: Dict[str, Any], frame_interval: float) -> None:
        """Execute the video analysis."""
        try:
            self._update_progress(5, "Preparing analysis parameters...")
            frame_params = self._calculate_frame_parameters(
                context, frame_interval
            )
            
            self._update_progress(10, "Starting audio analysis...")

            audio_start_time = time.time()
            if context['audio_path']:
                self._process_audio_analysis(
                    context['audio_path'], 
                    frame_params['start_time'], 
                    frame_params['end_time'], 
                    frame_interval
                )
            else:
                self._update_progress(20, "No audio analysis needed")
            self.performance_metrics['audio_analysis_time'] = time.time() - audio_start_time
            
            self._update_progress(25, "Audio analysis complete, starting video frame processing...")

            video_start_time = time.time()
            self._process_video_frames(context, frame_params, frame_interval)
            video_processing_time = time.time() - video_start_time
            
            self._update_progress(95, "Finalizing analysis results...")

            self.analysis_end_time = datetime.now().timestamp()
            self.performance_metrics['total_analysis_time'] = self.analysis_end_time - (self.analysis_start_time or 0)
            
            logger.info(f"Performance breakdown - Flash: {self.performance_metrics['flash_detection_time']:.2f}s, "
                       f"Alias Name: {self.performance_metrics['alias_detection_time']:.2f}s, "
                       f"Eval Mode: {self.performance_metrics['eval_mode_detection_time']:.2f}s, "
                       f"Audio: {self.performance_metrics['audio_analysis_time']:.2f}s")
            
        finally:
            if context.get('cap'):
                context['cap'].release()
            
            if not self.analysis_end_time:
                self.analysis_end_time = datetime.now().timestamp()
                self.performance_metrics['total_analysis_time'] = self.analysis_end_time - (self.analysis_start_time or 0)
    
    def _calculate_frame_parameters(self, context: Dict[str, Any], 
                                   frame_interval: float) -> Dict[str, Any]:
        """Calculate frame processing parameters."""
        fps = context['fps']
        total_frames = context['total_frames']
        duration = context['duration']
        
        start_frame = 0
        end_frame = total_frames
        frame_step = max(1, int(frame_interval * fps))
        
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_step': frame_step,
            'start_time': 0.0,
            'end_time': duration
        }
    
    def _process_video_frames(self, context: Dict[str, Any], 
                             frame_params: Dict[str, Any], 
                             frame_interval: float) -> None:
        """Process video frames for detection."""
        cap = context['cap']
        fps = context['fps']
        
        visual_rules = [rule for rule in self.rules 
                       if rule.detection_type not in [DetectionType.LANGUAGE_FLUENCY, 
                                     DetectionType.VOICE_AUDIBILITY]]

        if not visual_rules:
            return

        flash_rules = [rule for rule in visual_rules if "2.5 Flash" in rule.name]
        alias_name_rules = [rule for rule in visual_rules if "Alias Name" in rule.name]
        eval_mode_rules = [rule for rule in visual_rules if "Eval Mode" in rule.name]

        if flash_rules or alias_name_rules or eval_mode_rules:
            self._process_text_detection_rules(cap, fps, flash_rules, eval_mode_rules, frame_params)

        self.total_frames_processed = 0
    
    def _process_text_detection_rules(self, cap: cv2.VideoCapture, fps: float,
                                               flash_rules: List[DetectionRule], 
                                               eval_mode_rules: List[DetectionRule],
                                               frame_params: Dict[str, Any]) -> None:
        """Process text detection rules."""        
        flash_detected = False
        flash_detection_timestamp = None
        
        alias_name_rules = [rule for rule in self.rules if "Alias Name" in rule.name]
        
        if flash_rules:
            self._update_progress(30, "Analyzing first 5 seconds for '2.5 Flash' and 'Alias Name' detection...")
            flash_start_time = time.time()
            flash_detection_timestamp = self._process_flash_and_alias_detection_frame_by_frame(cap, fps, flash_rules, alias_name_rules)
            flash_detected = flash_detection_timestamp is not None
            detection_time = time.time() - flash_start_time
            self.performance_metrics['flash_detection_time'] = detection_time
            self.performance_metrics['alias_detection_time'] = detection_time
            
            if flash_detected:
                self._update_progress(50, f"'2.5 Flash' detected at {flash_detection_timestamp:.1f}s! Searching for 'Eval Mode'...")
            else:
                self._update_progress(45, "'2.5 Flash' not found in first 5 seconds. Continuing with 'Eval Mode' search...")
        
        if flash_detected and self._check_if_eval_mode_already_detected():
            logger.info("Both 2.5 Flash and Eval Mode detected - stopping text analysis immediately")
            self._update_progress(90, "Both text elements detected! Analysis nearly complete...")
            return
        
        if eval_mode_rules:
            eval_start_time = time.time()
            if flash_detected:
                logger.info(f"2.5 Flash found at {flash_detection_timestamp:.2f}s - starting Eval Mode search from this point")
                self._update_progress(55, f"Searching for 'Eval Mode' starting from {flash_detection_timestamp:.1f}s...")
                self._process_eval_mode_adaptive_search(cap, fps, eval_mode_rules, flash_detection_timestamp, frame_params)
            else:
                logger.info("2.5 Flash not detected in first 5 seconds - starting Eval Mode search from 5 seconds")
                self._update_progress(50, "Searching for 'Eval Mode' starting from 5 seconds...")
                self._process_eval_mode_adaptive_search(cap, fps, eval_mode_rules, 5.0, frame_params)
            
            self.performance_metrics['eval_mode_detection_time'] = time.time() - eval_start_time
            self._update_progress(85, "Text detection analysis completed.")
    
    def _check_if_eval_mode_already_detected(self) -> bool:
        """Check if Eval Mode has already been detected."""
        with self._lock:
            for result in self.results:
                if "Eval Mode" in result.rule_name and result.detected:
                    return True
        return False
    
    def _process_eval_mode_adaptive_search(self, cap: cv2.VideoCapture, fps: float,
                                         eval_mode_rules: List[DetectionRule],
                                         start_time: float, frame_params: Dict[str, Any]) -> None:
        """Process Eval Mode detection with adaptive interval strategy:
        - 1-second intervals for first 5 seconds of search
        - 10-second intervals after that
        """
        logger.info(f"Starting adaptive Eval Mode search from {start_time:.2f}s")
        
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        try:
            phase1_end_time = start_time + 5.0
            phase1_interval = 1.0
            
            self._update_progress(60, f"Phase 1: Searching 'Eval Mode' with 1s intervals ({start_time:.1f}s-{phase1_end_time:.1f}s)")
            logger.info(f"Phase 1: Searching with {phase1_interval}s intervals from {start_time:.2f}s to {phase1_end_time:.2f}s")
            eval_detected = self._search_eval_mode_in_time_range(
                cap, fps, eval_mode_rules, start_time, phase1_end_time, phase1_interval, 60, 70
            )
            
            if eval_detected:
                logger.info("Eval Mode found in Phase 1 - stopping search")
                self._update_progress(80, "'Eval Mode' detected! Text analysis complete.")
                return
            
            phase2_start_time = phase1_end_time
            phase2_end_time = frame_params['end_frame'] / fps
            phase2_interval = 10.0
            
            if phase2_start_time < phase2_end_time:
                self._update_progress(72, f"Phase 2: Searching 'Eval Mode' with 10s intervals ({phase2_start_time:.1f}s-{phase2_end_time:.1f}s)")
                logger.info(f"Phase 2: Searching with {phase2_interval}s intervals from {phase2_start_time:.2f}s to {phase2_end_time:.2f}s")
                eval_detected = self._search_eval_mode_in_time_range(
                    cap, fps, eval_mode_rules, phase2_start_time, phase2_end_time, phase2_interval, 72, 82
                )
                
                if eval_detected:
                    self._update_progress(85, "'Eval Mode' detected! Text analysis complete.")
            
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    def _search_frames_for_text(self, cap: cv2.VideoCapture, fps: float, 
                                rules: List[DetectionRule], start_time: float, 
                                end_time: float, interval: float,
                                progress_start: float = 60, progress_end: float = 80,
                                search_type: str = "generic") -> Optional[float]:
        """Generic frame search method for text detection rules."""
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)
        frame_step = max(1, int(interval * fps))
        
        frames_processed = 0
        total_frames_in_range = (end_frame - start_frame) // frame_step
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        try:
            for frame_idx in range(start_frame, end_frame, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    continue
                
                timestamp = frame_idx / fps
                frames_processed += 1
                self.performance_metrics['frames_analyzed'] += 1
                
                if total_frames_in_range > 0:
                    phase_progress = frames_processed / total_frames_in_range
                    current_progress = progress_start + (progress_end - progress_start) * phase_progress
                    self._update_progress(current_progress, 
                                        f"Searching '{search_type}' at {timestamp:.1f}s ({interval}s interval)")
                
                for rule in rules:
                    try:
                        result = self._apply_visual_rule(rule, frame, timestamp, frame_idx)
                        
                        with self._lock:
                            self.results.append(result)
                            
                        if result.detected:
                            logger.info(f"✓ {search_type} detected at {timestamp:.2f}s (frame {frame_idx}) with {interval}s interval")
                            return timestamp
                            
                    except Exception as e:
                        logger.error(f"{search_type} rule {rule.name} failed at frame {frame_idx}: {e}")
                
                log_frequency = 10 if interval < 5.0 else 3
                if frames_processed % log_frequency == 0:
                    logger.info(f"{search_type} search progress: {frames_processed} frames processed ({timestamp:.1f}s, {interval}s interval)")
            
            logger.info(f"{search_type} search completed for range {start_time:.1f}s-{end_time:.1f}s: processed {frames_processed} frames with {interval}s interval - NOT DETECTED")
            return None
            
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    def _search_eval_mode_in_time_range(self, cap: cv2.VideoCapture, fps: float,
                                       eval_mode_rules: List[DetectionRule],
                                       start_time: float, end_time: float, 
                                       interval: float, 
                                       progress_start: float = 60, 
                                       progress_end: float = 80) -> bool:
        """Search for Eval Mode in a specific time range with given interval."""
        detection_timestamp = self._search_frames_for_text(
            cap, fps, eval_mode_rules, start_time, end_time, interval,
            progress_start, progress_end, "Eval Mode"
        )
        return detection_timestamp is not None
    
    def _process_flash_and_alias_detection_frame_by_frame(self, cap: cv2.VideoCapture, fps: float, 
                                                         flash_rules: List[DetectionRule], 
                                                         alias_name_rules: List[DetectionRule]) -> Optional[float]:
        """Analyze the first 5 seconds frame by frame for both "2.5 Flash" and "Alias Name" text.
        Returns the timestamp of flash detection if found, None otherwise."""
        if not flash_rules and not alias_name_rules:
            return None
        
        logger.info("Processing 2.5 Flash and Alias Name detection using frame-by-frame analysis in first 5 seconds...")
        
        combined_rules = flash_rules + alias_name_rules
        
        flash_detection_timestamp = self._search_frames_for_combined_text(
            cap, fps, flash_rules, alias_name_rules, 0.0, 5.0, 1.0, 30, 45
        )
        
        if flash_detection_timestamp is None:
            logger.info(f"2.5 Flash analysis completed: analyzed 5 frames (at 0s, 1s, 2s, 3s, 4s) - NOT DETECTED")
            
            with self._lock:
                for flash_rule in flash_rules:
                    summary_result = DetectionResult(
                        rule_name=flash_rule.name,
                        timestamp=5.0,
                        frame_number=int(4.0 * fps),
                        detected=False,
                        details={
                            'frame_by_frame_analysis': True,
                            'analysis_window': '0-5 seconds (5 frames at 0s, 1s, 2s, 3s, 4s)',
                            'frames_analyzed': 5,
                            'search_method': 'selective_frame_sampling'
                        }
                    )
                    self.results.append(summary_result)
                
                for alias_rule in alias_name_rules:
                    summary_result = DetectionResult(
                        rule_name=alias_rule.name,
                        timestamp=5.0,
                        frame_number=int(4.0 * fps),
                        detected=False,
                        details={
                            'frame_by_frame_analysis': True,
                            'analysis_window': '0-5 seconds (5 frames at 0s, 1s, 2s, 3s, 4s)',
                            'frames_analyzed': 5,
                            'search_method': 'selective_frame_sampling'
                        }
                    )
                    self.results.append(summary_result)
        
        return flash_detection_timestamp

    def _search_frames_for_combined_text(self, cap: cv2.VideoCapture, fps: float, 
                                        flash_rules: List[DetectionRule], 
                                        alias_name_rules: List[DetectionRule],
                                        start_time: float, end_time: float, interval: float,
                                        progress_start: float = 30, progress_end: float = 45) -> Optional[float]:
        """Search for both flash and alias name text in the same frames."""
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)
        frame_step = max(1, int(interval * fps))
        
        frames_processed = 0
        total_frames_in_range = (end_frame - start_frame) // frame_step
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        flash_detection_timestamp = None
        alias_detection_logged = False
        alias_screenshot_saved = False
        flash_screenshot_saved = False
        
        try:
            for frame_idx in range(start_frame, end_frame, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    continue
                
                timestamp = frame_idx / fps
                frames_processed += 1
                self.performance_metrics['frames_analyzed'] += 1
                
                if total_frames_in_range > 0:
                    phase_progress = frames_processed / total_frames_in_range
                    current_progress = progress_start + (progress_end - progress_start) * phase_progress
                    self._update_progress(current_progress, 
                                        f"Searching '2.5 Flash' and 'Alias Name' at {timestamp:.1f}s")
                
                flash_detected_this_frame = False
                for rule in flash_rules:
                    try:
                        if flash_screenshot_saved:
                            rule_params = rule.parameters.copy()
                            rule_params['save_screenshot'] = False
                            modified_rule = DetectionRule(
                                name=rule.name,
                                detection_type=rule.detection_type,
                                parameters=rule_params
                            )
                            result = self._apply_visual_rule(modified_rule, frame, timestamp, frame_idx)
                        else:
                            result = self._apply_visual_rule(rule, frame, timestamp, frame_idx)
                        
                        with self._lock:
                            self.results.append(result)
                            
                        if result.detected and flash_detection_timestamp is None:
                            flash_detection_timestamp = timestamp
                            flash_detected_this_frame = True
                            flash_screenshot_saved = True
                            logger.info(f"✓ 2.5 Flash detected at {timestamp:.2f}s (frame {frame_idx})")
                            
                    except Exception as e:
                        logger.error(f"Flash rule {rule.name} failed at frame {frame_idx}: {e}")
                
                for rule in alias_name_rules:
                    try:
                        if alias_screenshot_saved:
                            rule_params = rule.parameters.copy()
                            rule_params['save_screenshot'] = False
                            modified_rule = DetectionRule(
                                name=rule.name,
                                detection_type=rule.detection_type,
                                parameters=rule_params
                            )
                            result = self._apply_visual_rule(modified_rule, frame, timestamp, frame_idx)
                        else:
                            result = self._apply_visual_rule(rule, frame, timestamp, frame_idx)
                        
                        with self._lock:
                            self.results.append(result)
                            
                        if result.detected:
                            if not alias_detection_logged:
                                alias_detection_logged = True
                                logger.info(f"✓ Alias Name detected at {timestamp:.2f}s (frame {frame_idx})")
                            if not alias_screenshot_saved:
                                alias_screenshot_saved = True
                            
                    except Exception as e:
                        logger.error(f"Alias Name rule {rule.name} failed at frame {frame_idx}: {e}")
                
                flash_found = flash_detection_timestamp is not None
                alias_found = alias_detection_logged
                
                if flash_rules and alias_name_rules:
                    if flash_found and alias_found:
                        logger.info(f"✓ Both 2.5 Flash and Alias Name detected - stopping search early at {timestamp:.2f}s")
                        break
                elif flash_rules and not alias_name_rules:
                    if flash_found:
                        logger.info(f"✓ 2.5 Flash detected - stopping search early at {timestamp:.2f}s")
                        break
                elif alias_name_rules and not flash_rules:
                    if alias_found:
                        logger.info(f"✓ Alias Name detected - stopping search early at {timestamp:.2f}s")
                        break
                
                log_frequency = 10 if interval < 5.0 else 3
                if frames_processed % log_frequency == 0:
                    logger.info(f"Combined text search progress: {frames_processed} frames processed ({timestamp:.1f}s)")
            
            logger.info(f"Combined text search completed for range {start_time:.1f}s-{end_time:.1f}s: processed {frames_processed} frames")
            return flash_detection_timestamp
            
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    def _setup_audio_analysis(self, video_path: str) -> Optional[str]:
        """Setup audio extraction for audio rules."""
        audio_rules = [rule for rule in self.rules 
                  if rule.detection_type in [DetectionType.LANGUAGE_FLUENCY, 
                               DetectionType.VOICE_AUDIBILITY]]
        
        if not audio_rules or not self.audio_analyzer:
            return None
        
        try:
            self._update_progress(6, "Extracting audio from video...")
            audio_extraction_start = time.time()
            audio_path = self.session_manager.create_temp_file(self.session_id, "temp_audio", ".wav")
            
            if self.audio_analyzer.extract_audio(video_path, audio_path):
                self.performance_metrics['audio_extraction_time'] = time.time() - audio_extraction_start
                
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    self.temp_files.append(audio_path)
                    
                    self._update_progress(8, "Loading Whisper model for language analysis...")
                    try:
                        whisper_load_start = time.time()
                        self.audio_analyzer.load_whisper_model()
                        self.performance_metrics['whisper_model_load_time'] = time.time() - whisper_load_start
                        self._update_progress(9, "Whisper model loaded successfully")
                    except Exception as e:
                        logger.warning(f"Whisper model load failed: {e}")
                        self.performance_metrics['whisper_model_load_time'] = 0.0
                    
                    return audio_path
                else:
                    return None
            else:
                self.performance_metrics['audio_extraction_time'] = time.time() - audio_extraction_start
                return None
                
        except Exception as e:
            self.performance_metrics['audio_extraction_time'] = time.time() - audio_extraction_start if 'audio_extraction_start' in locals() else 0.0
            return None
    
    def _save_frame(self, frame: np.ndarray, rule_name: str, timestamp: float) -> Optional[str]:
        """Save frame as screenshot in session directory."""
        try:
            safe_rule_name = re.sub(r'[^\w-]', '_', rule_name)[:50]
            filename = f"{safe_rule_name}_{timestamp:.2f}s.png"
            
            filepath = self.session_manager.save_frame(self.session_id, frame, filename)
            
            if filepath:
                with self._lock:
                    self.screenshot_files.append(filepath)
                return filepath
            return None
                
        except Exception as e:
            logger.error(f"Frame save error: {e}")
            return None
    
    def _apply_visual_rule(self, rule: DetectionRule, frame: np.ndarray,
                          timestamp: float, frame_number: int) -> DetectionResult:
        """Apply visual detection rule to frame."""
        start_time = datetime.now().timestamp()
        
        try:
            result = self._execute_detection_rule(rule, frame, timestamp, frame_number)
            result.processing_time = datetime.now().timestamp() - start_time
            return result
        except Exception as e:
            logger.error(f"Rule execution failed for {rule.name}: {e}")
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=False, details={'error': str(e)}
            )
    
    def _execute_detection_rule(self, rule: DetectionRule, frame: np.ndarray,
                               timestamp: float, frame_number: int) -> DetectionResult:
        """Execute specific detection type."""        
        detection_methods = {
            DetectionType.TEXT: self._detect_text
        }
        
        if rule.detection_type in detection_methods:
            return detection_methods[rule.detection_type](rule, frame, timestamp, frame_number)
        else:
            raise NotImplementedError(f"Detection type {rule.detection_type} not implemented")
    
    def _process_audio_analysis(self, audio_path: str, start_time: float, 
                              end_time: float, frame_interval: float) -> None:
        """Audio analysis for language detection and voice audibility."""
        if not self.audio_analyzer:
            return
        
        try:
            audio_rules = [rule for rule in self.rules 
                          if rule.detection_type in [DetectionType.LANGUAGE_FLUENCY, 
                                                   DetectionType.VOICE_AUDIBILITY]]
            
            if not audio_rules:
                return
            
            voice_audibility_rules = [r for r in audio_rules if r.detection_type == DetectionType.VOICE_AUDIBILITY]
            language_rules = [r for r in audio_rules if r.detection_type == DetectionType.LANGUAGE_FLUENCY]
            
            total_audio_tasks = len(voice_audibility_rules) + len(language_rules)
            completed_tasks = 0
            
            if voice_audibility_rules:
                self._update_progress(12, "Analyzing voice audibility...")
                completed_tasks = self._process_voice_audibility_rules(
                    voice_audibility_rules, audio_path, start_time, completed_tasks, total_audio_tasks
                )
            
            if language_rules:
                self._update_progress(18, "Analyzing language fluency...")
                self._process_language_fluency_rules(
                    language_rules, audio_path, start_time, completed_tasks, total_audio_tasks
                )
                        
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
    
    def _process_voice_audibility_rules(self, rules: List[DetectionRule], audio_path: str, 
                                       start_time: float, completed_tasks: int, 
                                       total_audio_tasks: int) -> int:
        """Process voice audibility rules."""
        for rule in rules:
            try:
                result = self.audio_analyzer.analyze_voice_audibility(audio_path)
                
                if result:
                    detection_result = DetectionResult(
                        rule_name=rule.name,
                        detected=result['passed_qa'],
                        timestamp=start_time,
                        frame_number=int(start_time * 30),
                        details={
                            'num_audible_voices': result['num_audible_voices'],
                            'voice_detected': result['num_audible_voices'] > 0,
                            'both_voices_audible': result['num_audible_voices'] == 2,
                            'details': result['details'],
                            'voice_ratio': result.get('voice_ratio', 0.0),
                            'total_voice_duration': result.get('total_voice_duration', 0.0),
                            'has_multiple_speakers': result.get('has_multiple_speakers', False),
                            'audio_features': result.get('speaker_info', {}).get('audio_features', {}) if result.get('speaker_info') else {}
                        }
                    )
                    self.results.append(detection_result)
                
                completed_tasks += 1
                progress = 12 + (completed_tasks / total_audio_tasks) * 8
                self._update_progress(int(progress), f"Voice audibility analysis complete ({completed_tasks}/{total_audio_tasks})")
                
            except Exception as e:
                logger.error(f"Voice audibility analysis failed for rule {rule.name}: {e}")
                completed_tasks += 1
        
        return completed_tasks
    
    def _process_language_fluency_rules(self, rules: List[DetectionRule], audio_path: str, 
                                       start_time: float, completed_tasks: int, 
                                       total_audio_tasks: int) -> None:
        """Process language fluency rules."""
        try:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                audio_duration = len(y) / sr
            except:
                audio_duration = 0
            
            for i, rule in enumerate(rules):
                target_language = rule.parameters.get('target_language')
                
                self._update_progress(18 + i, f"Transcribing audio for {Config.get_language_display_name(target_language)}...")
                
                fluency_result = self.audio_analyzer.analyze_full_audio_fluency(audio_path, target_language)
                
                detection_result = self._create_language_detection_result(
                    rule, fluency_result, target_language, start_time, audio_duration
                )
                
                self.results.append(detection_result)
                
                completed_tasks += 1
                progress = 18 + ((completed_tasks - len([r for r in rules if r != rule])) / total_audio_tasks) * 7
                self._update_progress(int(progress), f"Language analysis complete for {Config.get_language_display_name(target_language)}")
                
        except Exception as e:
            logger.error(f"Language fluency analysis failed: {e}")
    
    def _create_language_detection_result(self, rule: DetectionRule, fluency_result: Optional[Dict[str, Any]], 
                                         target_language: str, start_time: float, 
                                         audio_duration: float) -> DetectionResult:
        """Create detection result for language analysis."""
        if fluency_result is None:
            return DetectionResult(
                rule_name=rule.name,
                timestamp=start_time,
                frame_number=0,
                detected=False,
                details={
                    'target_language': target_language,
                    'detected_language': 'unknown',
                    'is_fluent': False,
                    'fluency_score': 0.0,
                    'transcription': '',
                    'audio_duration': audio_duration,
                    'total_words': 0,
                    'avg_word_length': 0,
                    'full_audio_analysis': True,
                    'analysis_type': 'Full Audio Transcription',
                    'analysis_failed_reason': 'No audible voices detected for transcription.',
                    'fluency_indicators': {
                        'word_count': 0,
                        'avg_word_length': 0,
                        'language_match': False,
                        'has_transcription': False,
                        'audio_duration': audio_duration,
                    }
                }
            )
        else:
            return DetectionResult(
                rule_name=rule.name,
                timestamp=start_time,
                frame_number=0,
                detected=fluency_result.get('is_fluent', False),
                details={
                    'target_language': target_language,
                    'detected_language': fluency_result.get('detected_language', 'unknown'),
                    'is_fluent': fluency_result.get('is_fluent', False),
                    'fluency_score': fluency_result.get('fluency_score', 0.0),
                    'transcription': fluency_result.get('transcription', ''),
                    'audio_duration': audio_duration,
                    'total_words': fluency_result.get('total_words', 0),
                    'avg_word_length': fluency_result.get('avg_word_length', 0),
                    'full_audio_analysis': True,
                    'analysis_type': 'Full Audio Transcription',
                    'fluency_indicators': {
                        'word_count': fluency_result.get('total_words', 0),
                        'avg_word_length': fluency_result.get('avg_word_length', 0),
                        'language_match': fluency_result.get('detected_language') == Config.locale_to_whisper_language(target_language),
                        'has_transcription': len(fluency_result.get('transcription', '')) > 0,
                        'audio_duration': audio_duration,
                    }
                }
            )
    
    def _find_text_bounding_box(self, expected_text: str, boxes: dict) -> Optional[List[int]]:
        """Find bounding box for expected text using fuzzy matching."""
        expected_words = expected_text.lower().split()
        found_boxes = []
        
        for word in expected_words:
            best_match_box = None
            best_similarity = 0
            
            for i, box_word in enumerate(boxes['text']):
                box_word_clean = box_word.strip().lower()
                if not box_word_clean:
                    continue
                
                box = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                
                if box_word_clean == word:
                    best_match_box = box
                    break
                
                similarity = TextMatcher.calculate_similarity(word, box_word_clean)
                if similarity > best_similarity and similarity >= 0.65 and box not in found_boxes:
                    best_similarity = similarity
                    best_match_box = box
            
            if best_match_box:
                found_boxes.append(best_match_box)
        
        if len(found_boxes) < len(expected_words):
            whole_phrase_boxes = []
            for i, box_word in enumerate(boxes['text']):
                box_word_clean = box_word.strip().lower()
                if not box_word_clean:
                    continue
                
                box = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                
                if expected_text.lower() == "roaring tiger":
                    if TextMatcher._match_roaring_tiger_variants(box_word_clean):
                        whole_phrase_boxes.append(box)
                else:
                    if all(word in box_word_clean for word in expected_words):
                        whole_phrase_boxes.append(box)
            
            if whole_phrase_boxes:
                found_boxes = whole_phrase_boxes
        
        if found_boxes:
            x1 = min(box[0] for box in found_boxes)
            y1 = min(box[1] for box in found_boxes)
            x2 = max(box[0] + box[2] for box in found_boxes)
            y2 = max(box[1] + box[3] for box in found_boxes)
            return [x1, y1, x2-x1, y2-y1]
        
        return None
    
    def _detect_text(self, rule: DetectionRule, frame: np.ndarray, 
                    timestamp: float, frame_number: int) -> DetectionResult:
        """OCR-based text detection pipeline."""
        params = rule.parameters
        
        try:
            roi, roi_offset = self._extract_roi_from_frame(frame, params)
            if roi is None:
                return DetectionResult(
                    rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                    detected=False, details={'error': 'Invalid region bounds', 'roi_size': (0, 0)}
                )
            
            ocr_start_time = time.time()
            text, boxes = self._process_ocr_pipeline(roi, params)
            ocr_time = time.time() - ocr_start_time
            self.performance_metrics['ocr_processing_time'] += ocr_time
            
            detected, text_bounding_box, detection_method = self._analyze_ocr_results(
                text, boxes, params, roi_offset
            )
            
            screenshot_path = None
            if params.get('save_screenshot', True) and (detected or 'pro_incorrect' in detection_method):
                if params.get('expected_text') == TargetTexts.FLASH_TEXT:
                    if 'pro_incorrect' in detection_method:
                        screenshot_path = self._create_flash_pro_screenshot(
                            frame, text_bounding_box, TargetTexts.PRO_TEXT, rule.name, timestamp, is_correct=False
                        )
                    elif 'flash_correct' in detection_method:
                        screenshot_path = self._create_flash_pro_screenshot(
                            frame, text_bounding_box, TargetTexts.FLASH_TEXT, rule.name, timestamp, is_correct=True
                        )
                    else:
                        screenshot_path = self._create_text_screenshot(
                            frame, text_bounding_box, params.get('expected_text', ''), rule.name, timestamp
                        )
                else:
                    screenshot_path = self._create_text_screenshot(
                        frame, text_bounding_box, params.get('expected_text', ''), rule.name, timestamp
                    )
            
            actual_detected_text = text
            if params.get('expected_text') == TargetTexts.FLASH_TEXT:
                if 'pro_incorrect' in detection_method:
                    actual_detected_text = f"{text} (2.5 Pro detected - incorrect model)"
                elif 'flash_correct' in detection_method:
                    actual_detected_text = f"{text} (2.5 Flash detected - correct model)"
            
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=detected, 
                details={
                    'detected_text': actual_detected_text,
                    'expected_text': params.get('expected_text', ''),
                    'full_match': text.lower() == params.get('expected_text', '').lower() if params.get('expected_text') else False,
                    'text_bounding_box': text_bounding_box,
                    'ocr_words': [word.strip() for word in boxes['text'] if word.strip()],
                    'detection_method': detection_method,
                    'roi_offset': roi_offset,
                    'word_count': len([word for word in boxes['text'] if word.strip()]),
                    'model_validation': self._get_model_validation_info(detection_method, params.get('expected_text'))
                }, 
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            logger.error(f"OCR failed for rule {rule.name}: {e}")
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=False, details={'error': str(e)}
            )
    
    def _get_model_validation_info(self, detection_method: str, expected_text: str) -> Dict[str, Any]:
        """Get model validation information for Flash/Pro detection."""
        if expected_text != TargetTexts.FLASH_TEXT:
            return {}
        
        if 'flash_correct' in detection_method:
            return {
                'status': 'correct_model',
                'found_text': '2.5 Flash',
                'validation_result': 'pass'
            }
        elif 'pro_incorrect' in detection_method:
            return {
                'status': 'incorrect_model', 
                'found_text': '2.5 Pro',
                'validation_result': 'fail_incorrect_model'
            }
        else:
            return {
                'status': 'not_found',
                'found_text': None,
                'validation_result': 'fail_not_found'
            }
    
    def _extract_roi_from_frame(self, frame: np.ndarray, params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """Extract region of interest from frame."""
        if 'region' not in params:
            return frame, (0, 0)
        
        x1, y1, x2, y2 = params['region']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None, (x1, y1)
    
    def _process_ocr_pipeline(self, roi: np.ndarray, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process OCR pipeline: preprocessing + OCR."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        preprocess_config = params.get('preprocess', {})
        if preprocess_config.get('denoise', Config.OCR_DENOISING_ENABLED):
            gray = cv2.fastNlMeansDenoising(gray)
        if preprocess_config.get('threshold', Config.OCR_THRESHOLD_ENABLED):
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        text = pytesseract.image_to_string(gray, config=Config.OCR_CONFIG).strip()
        boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=Config.OCR_CONFIG)
        return text, boxes
    
    def _analyze_ocr_results(self, text: str, boxes: Dict[str, Any], params: Dict[str, Any], 
                            roi_offset: Tuple[int, int]) -> Tuple[bool, Optional[List[int]], str]:
        """Analyze OCR results for text detection."""
        expected_text = params.get('expected_text', '')
        
        if expected_text:
            if expected_text == TargetTexts.FLASH_TEXT:
                return self._analyze_flash_pro_detection(text, boxes, roi_offset)
            else:
                detected, detection_method = TextMatcher.match_text(text, expected_text)
                text_bounding_box = None
                if detected:
                    text_bounding_box = self._find_text_bounding_box(expected_text, boxes)
                    if text_bounding_box and roi_offset != (0, 0):
                        text_bounding_box[0] += roi_offset[0]
                        text_bounding_box[1] += roi_offset[1]
                return detected, text_bounding_box, detection_method
        elif 'unexpected_text' in params:
            unexpected = params['unexpected_text']
            detected = unexpected.lower() in text.lower()
            return detected, None, 'unexpected'
        else:
            detected = len(text) > 0
            return detected, None, 'presence'
    
    def _analyze_flash_pro_detection(self, text: str, boxes: Dict[str, Any], 
                                   roi_offset: Tuple[int, int]) -> Tuple[bool, Optional[List[int]], str]:
        """Special analysis for Flash/Pro detection."""
        flash_detected, flash_method = TextMatcher.match_text(text, TargetTexts.FLASH_TEXT)
        
        pro_detected, pro_method = TextMatcher.match_text(text, TargetTexts.PRO_TEXT)
        
        if flash_detected:
            text_bounding_box = self._find_text_bounding_box(TargetTexts.FLASH_TEXT, boxes)
            if text_bounding_box and roi_offset != (0, 0):
                text_bounding_box[0] += roi_offset[0]
                text_bounding_box[1] += roi_offset[1]
            return True, text_bounding_box, f'flash_correct_{flash_method}'
        elif pro_detected:
            text_bounding_box = self._find_text_bounding_box(TargetTexts.PRO_TEXT, boxes)
            if text_bounding_box and roi_offset != (0, 0):
                text_bounding_box[0] += roi_offset[0]
                text_bounding_box[1] += roi_offset[1]
            return False, text_bounding_box, f'pro_incorrect_{pro_method}'
        else:
            return False, None, 'neither_found'
    
    def _create_text_screenshot(self, frame: np.ndarray, text_bounding_box: Optional[List[int]], 
                              expected_text: str, rule_name: str, timestamp: float) -> Optional[str]:
        """Create annotated screenshot for text detection."""
        try:
            frame_copy = frame.copy()
            
            if text_bounding_box:
                if expected_text == TargetTexts.FLASH_TEXT:
                    detected_text = expected_text
                    self._apply_detection_annotations(frame_copy, text_bounding_box, detected_text, is_correct=True)
                else:
                    self._apply_detection_annotations(frame_copy, text_bounding_box, expected_text, is_correct=True)
            else:
                self._apply_search_annotations(frame_copy, expected_text, timestamp)
            
            screenshot_path = self._save_frame(frame_copy, rule_name, timestamp)
            if screenshot_path and os.path.exists(screenshot_path):
                return screenshot_path
            else:
                logger.error(f"Failed to save text screenshot for {rule_name}")
                return None
            
        except Exception as e:
            logger.error(f"Screenshot creation failed for {rule_name}: {e}")
            return None
    
    def _create_flash_pro_screenshot(self, frame: np.ndarray, text_bounding_box: Optional[List[int]], 
                                   detected_text: str, rule_name: str, timestamp: float, is_correct: bool = True) -> Optional[str]:
        """Create annotated screenshot for Flash/Pro detection."""
        try:
            frame_copy = frame.copy()
            
            if text_bounding_box:
                self._apply_detection_annotations(frame_copy, text_bounding_box, detected_text, is_correct)
            else:
                self._apply_search_annotations(frame_copy, detected_text, timestamp)
            
            screenshot_path = self._save_frame(frame_copy, rule_name, timestamp)
            if screenshot_path and os.path.exists(screenshot_path):
                return screenshot_path
            else:
                logger.error(f"Failed to save Flash/Pro screenshot for {rule_name}")
                return None
            
        except Exception as e:
            logger.error(f"Flash/Pro screenshot creation failed for {rule_name}: {e}")
            return None
    
    def _apply_detection_annotations(self, frame: np.ndarray, text_bounding_box: List[int], detected_text: str, is_correct: bool = True) -> None:
        """Apply annotations for text detection (correct or incorrect)."""
        x1, y1, w, h = text_bounding_box
        x2, y2 = x1 + w, y1 + h
        
        color = Config.ANNOTATION_COLORS['GREEN'] if is_correct else Config.ANNOTATION_COLORS['RED']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Config.ANNOTATION_THICKNESS)
        
        text_label = f"Detected: {detected_text}"
        if not is_correct:
            text_label = f"Wrong Model: {detected_text}"
            
        cv2.putText(frame, text_label, (x1, y1-10), 
                   Config.ANNOTATION_FONT, Config.ANNOTATION_FONT_SCALE, 
                   color, Config.ANNOTATION_THICKNESS)
    
    def _apply_search_annotations(self, frame: np.ndarray, expected_text: str, timestamp: float) -> None:
        """Apply annotations for search context (no detection)."""
        cv2.putText(frame, f"Searching for: {expected_text}", (10, 30), 
                   Config.ANNOTATION_FONT, Config.ANNOTATION_FONT_SCALE, 
                   Config.ANNOTATION_COLORS['YELLOW'], Config.ANNOTATION_THICKNESS)
        cv2.putText(frame, f"Frame @ {timestamp:.2f}s", (10, 60), 
                   Config.ANNOTATION_FONT, 0.5, 
                   Config.ANNOTATION_COLORS['WHITE'], 1)
    
    def export_results(self, output_path: str) -> bool:
        """Export analysis results to session directory."""
        if not os.path.dirname(output_path) or output_path == os.path.basename(output_path):
            output_path = os.path.join(self.session_dir, output_path)
        
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            qa_checker = QualityAssuranceChecker(self.results)
            
            data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_frames_processed': self.total_frames_processed,
                    'analysis_duration': self.performance_metrics['total_analysis_time'],
                    'active_rules': len(self.rules),
                    'video_duration': self.video_duration,
                    'session_id': self.session_id
                },
                'performance_metrics': {
                    'total_analysis_time': self.performance_metrics['total_analysis_time'],
                    'flash_detection_time': self.performance_metrics['flash_detection_time'],
                    'eval_mode_detection_time': self.performance_metrics['eval_mode_detection_time'],
                    'audio_analysis_time': self.performance_metrics['audio_analysis_time'],
                    'audio_extraction_time': self.performance_metrics['audio_extraction_time'],
                    'ocr_processing_time': self.performance_metrics['ocr_processing_time'],
                    'frames_analyzed': self.performance_metrics['frames_analyzed'],
                    'analysis_efficiency': {
                        'frames_per_second': self.total_frames_processed / self.performance_metrics['total_analysis_time'] if self.performance_metrics['total_analysis_time'] > 0 else 0,
                        'video_to_analysis_ratio': self.video_duration / self.performance_metrics['total_analysis_time'] if self.performance_metrics['total_analysis_time'] > 0 else 0,
                        'ocr_time_per_frame': self.performance_metrics['ocr_processing_time'] / max(1, self.total_frames_processed)
                    }
                },
                'results': [result.to_dict() for result in self.results],
                'qa_results': qa_checker.qa_results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _get_video_duration(self) -> float:
        """Get the stored video duration."""
        return self.video_duration


def create_detection_rules(target_language: str) -> List[DetectionRule]:
    """Create the standard detection rules for video analysis."""
    target_language = target_language.strip()
    language_name = Config.get_language_display_name(target_language)

    rules = []
    rules.append(DetectionRule(
        name=f"Text Detection: {TargetTexts.FLASH_TEXT}",
        detection_type=DetectionType.TEXT,
        parameters={
            'expected_text': TargetTexts.FLASH_TEXT,
            'save_screenshot': True,
            'preprocess': {
                'denoise': Config.OCR_DENOISING_ENABLED,
                'threshold': Config.OCR_THRESHOLD_ENABLED
            }
        }
    ))
    rules.append(DetectionRule(
        name=f"Text Detection: Alias Name",
        detection_type=DetectionType.TEXT,
        parameters={
            'expected_text': TargetTexts.ALIAS_NAME_TEXT,
            'save_screenshot': True,
            'preprocess': {
                'denoise': Config.OCR_DENOISING_ENABLED,
                'threshold': Config.OCR_THRESHOLD_ENABLED
            }
        }
    ))
    rules.append(DetectionRule(
        name=f"Text Detection: {TargetTexts.EVAL_MODE_TEXT}",
        detection_type=DetectionType.TEXT,
        parameters={
            'expected_text': TargetTexts.EVAL_MODE_TEXT,
            'save_screenshot': True,
            'preprocess': {
                'denoise': Config.OCR_DENOISING_ENABLED,
                'threshold': Config.OCR_THRESHOLD_ENABLED
            }
        }
    ))
    rules.append(DetectionRule(
        name=f"Language Detection: Fluent {language_name}",
        detection_type=DetectionType.LANGUAGE_FLUENCY,
        parameters={
            'target_language': target_language,
            'min_fluency_score': 0.6
        }
    ))
    rules.append(DetectionRule(
        name="Voice Audibility: Both Voices Audible",
        detection_type=DetectionType.VOICE_AUDIBILITY,
        parameters={
            'min_confidence': 0.3,
            'min_duration': 3.0
        }
    ))
    return rules


class StreamlitInterface:
    """Streamlit web interface with three-screen navigation."""
    @staticmethod
    def setup_page():
        """Configure Streamlit page and apply custom styles."""
        st.set_page_config(
            page_title="Gemini Live Video Verifier",
            page_icon="🎥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        StreamlitInterface._apply_custom_styles()

    @staticmethod
    def _apply_custom_styles():
        """Apply custom CSS styles for multi-screen interface."""
        css = StreamlitInterface._get_custom_css()
        st.markdown(css, unsafe_allow_html=True)

    @staticmethod
    def _get_custom_css() -> str:
        """Return custom CSS for the app UI."""
        return """
        <style>
        .main > div { padding-top: 1rem; }
        .stAlert { margin-top: 1rem; }
        .screen-nav { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
        .step-indicator { display: flex; justify-content: center; margin-bottom: 2rem; padding: 1rem; background: #16213e; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .step { display: flex; align-items: center; padding: 0.5rem 1.5rem; margin: 0 0.8rem; border-radius: 10px; background: linear-gradient(135deg, #e6e6fa 0%, #d4d4ff 100%); color: #333; font-weight: 500; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; border: 2px solid transparent; }
        .step.active { background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); color: white; box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3); transform: translateY(-2px); border: 2px solid #1565C0; }
        .step.completed { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3); border: 2px solid #2e7d32; }
        .css-1d391kg { background-color: #fafbfc; }
        .css-1d391kg .css-1v0mbdj { padding-top: 1rem; }
        .sidebar-section { background: white; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sidebar-header { color: #1f77b4; font-weight: bold; margin-bottom: 10px; font-size: 1.1em; }
        .css-1cpxqw2 { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px; }
        </style>
        """

    @staticmethod
    def render_progress_indicator():
        """Render step progress indicator for navigation."""
        current_screen = ScreenManager.get_current_screen()
        input_state = "completed" if current_screen in ['analysis', 'qa'] else ("active" if current_screen == 'input' else "")
        analysis_state = "completed" if current_screen == 'qa' else ("active" if current_screen == 'analysis' else "")
        
        submission_completed = st.session_state.get('submission_locked', False)
        if submission_completed:
            qa_state = "completed"
        else:
            qa_state = "active" if current_screen == 'qa' else ""
            
        st.markdown(f"""
        <div class="step-indicator">
            <div class="step {input_state}"><span>1️⃣ Input Parameters</span></div>
            <div class="step {analysis_state}"><span>2️⃣ Video Analysis</span></div>
            <div class="step {qa_state}"><span>3️⃣ Submit Video</span></div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_temp_video(video_file, session_id: str = None):
        """Create temporary video file with optimized chunking for production. Returns (path, [path]) or (None, [])."""
        if not video_file:
            return None, []
        try:
            if hasattr(video_file, 'size') and video_file.size > Config.MAX_FILE_SIZE:
                st.error(f"File too large: {video_file.size / 1024 / 1024:.1f}MB (max: {Config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB)")
                return None, []
            video_suffix = Path(video_file.name).suffix.lower()
            if video_suffix not in Config.SUPPORTED_VIDEO_FORMATS:
                st.error(f"Unsupported file format: {video_suffix}")
                return None, []
            session_id = session_id or get_session_manager().generate_session_id()
            
            safe_filename = re.sub(r'[^\w.-]', '_', video_file.name)[:100]
            
            file_size = getattr(video_file, 'size', 0)
            if file_size > 100 * 1024 * 1024:  # Files > 100MB
                chunk_size = 4 * 1024 * 1024  # 4MB chunks
            elif file_size > 50 * 1024 * 1024:  # Files > 50MB
                chunk_size = 2 * 1024 * 1024  # 2MB chunks
            else:
                chunk_size = 1024 * 1024  # 1MB chunks for smaller files
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix, prefix=f"video_{session_id}_") as tmp:
                total_written = 0
                video_file.seek(0)
                
                if file_size > 50 * 1024 * 1024:
                    progress_bar = st.progress(0, text="Uploading video...")
                    
                while True:
                    chunk = video_file.read(chunk_size)
                    if not chunk:
                        break
                    total_written += len(chunk)
                    if total_written > Config.MAX_FILE_SIZE:
                        os.unlink(tmp.name)
                        st.error("File size exceeded during upload")
                        return None, []
                    tmp.write(chunk)
                    
                    if file_size > 50 * 1024 * 1024:
                        progress = min(1.0, total_written / file_size)
                        progress_bar.progress(progress, text=f"Uploading video... {progress*100:.1f}%")
                
                if file_size > 50 * 1024 * 1024:
                    progress_bar.empty()
                
                tmp.flush()
                os.fsync(tmp.fileno())
                return tmp.name, [tmp.name]
        except Exception as e:
            logger.error(f"Temp video creation failed: {e}")
            st.error(f"Failed to process uploaded video: {str(e)}")
            return None, []


class QualityAssuranceChecker:
    """Quality Assurance checker for video analysis results."""
    def __init__(self, results: List[DetectionResult]):
        self.results = results
        self.qa_results = self._perform_qa_checks()

    def _perform_qa_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform all QA checks and return results as a dict."""
        if not self.results:
            return self._create_empty_qa_results()
        checks = [
            ('flash_presence', self._check_flash_presence),
            ('alias_name_presence', self._check_alias_name_presence),
            ('eval_mode_presence', self._check_eval_mode_presence),
            ('language_fluency', self._check_language_fluency),
            ('voice_audibility', self._check_voice_audibility)
        ]
        qa_checks = {name: func() for name, func in checks}
        total_checks = len(checks)
        passed_checks = sum(1 for check in qa_checks.values() if check['passed'])
        qa_checks['overall'] = {
            'passed': passed_checks == total_checks,
            'score': passed_checks / total_checks,
            'checks_passed': passed_checks,
            'total_checks': total_checks,
            'status': 'PASS' if passed_checks == total_checks else 'FAIL'
        }
        return qa_checks

    def _create_empty_qa_results(self) -> Dict[str, Dict[str, Any]]:
        """Return empty QA results for no detections."""
        empty_check = {
            'passed': False,
            'score': 0.0,
            'details': 'No detections found'
        }
        return {
            'flash_presence': empty_check.copy(),
            'alias_name_presence': empty_check.copy(),
            'eval_mode_presence': empty_check.copy(),
            'language_fluency': empty_check.copy(),
            'voice_audibility': empty_check.copy(),
            'overall': {
                'passed': False,
                'score': 0.0,
                'checks_passed': 0,
                'total_checks': 5,
                'status': 'FAIL'
            }
        }

    def _check_flash_presence(self) -> Dict[str, Any]:
        """Check if '2.5 Flash' appears in any text detection with Pro model validation."""
        flash_results = [r for r in self.results if '2.5 Flash' in r.rule_name]
        
        if not flash_results:
            return {
                'passed': False,
                'score': 0.0,
                'details': "❌ '2.5 Flash' model was not found in any OCR text detections at the start of the video. Please ensure to use the correct model and try again.",
                'flash_found': False,
                'flash_count': 0,
                'validation_status': 'not_found'
            }
        
        positive_flash_results = [r for r in flash_results if r.detected]
        
        if positive_flash_results:
            return {
                'passed': True,
                'score': 1.0,
                'details': "✅ '2.5 Flash' model found with OCR text detections. Correct usage confirmed.",
                'flash_found': True,
                'flash_count': len(positive_flash_results),
                'validation_status': 'correct_model'
            }
        
        pro_incorrect_results = []
        for result in flash_results:
            model_validation = result.details.get('model_validation', {})
            if model_validation.get('status') == 'incorrect_model':
                pro_incorrect_results.append(result)
        
        if pro_incorrect_results:
            return {
                'passed': False,
                'score': 0.0,
                'details': "❌ Incorrect Model: '2.5 Pro' was detected instead of '2.5 Flash'. Please ensure you are using the correct Gemini model and try again.",
                'flash_found': False,
                'flash_count': 0,
                'pro_found': True,
                'pro_count': len(pro_incorrect_results),
                'validation_status': 'incorrect_model'
            }
        else:
            return {
                'passed': False,
                'score': 0.0,
                'details': "❌ '2.5 Flash' model was not found in any OCR text detections at the start of the video. Please ensure to use the correct model and try again.",
                'flash_found': False,
                'flash_count': 0,
                'validation_status': 'not_found'
            }

    def _check_alias_name_presence(self) -> Dict[str, Any]:
        """Check if 'Roaring tiger' appears in any text detection."""
        alias_name_results = [r for r in self.results if 'Alias Name' in r.rule_name and r.detected]
        
        if not alias_name_results:
            return {
                'passed': False,
                'score': 0.0,
                'details': "❌ 'Roaring tiger' text was not found in any OCR text detections. Please ensure the correct alias name is visible and try again.",
                'alias_name_found': False,
                'alias_name_count': 0,
                'total_text_detections': len([r for r in self.results if 'Alias Name' in r.rule_name])
            }
        
        return {
            'passed': True,
            'score': 1.0,
            'details': "✅ 'Roaring tiger' text found with OCR text detections. Alias name confirmed.",
            'alias_name_found': True,
            'alias_name_count': len(alias_name_results),
            'total_text_detections': len([r for r in self.results if 'Alias Name' in r.rule_name])
        }

    def _extract_detected_text(self, result):
        if not result.detected or not result.details:
            return ''
        if 'detected_text' in result.details:
            return result.details.get('detected_text', '').lower().strip()
        elif isinstance(result.details, dict):
            detected_text = ''
            for key, value in result.details.items():
                if key != 'rule_name' and isinstance(value, str) and value.strip():
                    if key in ['expected_text', 'full_match', 'text_bounding_box', 'ocr_words', 'detection_method', 'roi_offset', 'word_count']:
                        continue
                    detected_text += f" {value}"
            return detected_text.lower().strip()
        return ''

    def _check_text_detection(self, filter_keywords, target_keywords, patterns, 
                             success_message, failure_message, result_key, count_key,
                             custom_pattern_check=None) -> Dict[str, Any]:
        """Generic method for checking text detection patterns."""
        all_text_detections = [r for r in self.results if self._is_text_detection(r, filter_keywords, target_keywords)]
        if not all_text_detections:
            return {
                'passed': False,
                'score': 0.0,
                'details': 'No text detection results found',
                result_key: False
            }
        
        found, detections = self._find_patterns_in_detections(all_text_detections, patterns, custom_pattern_check)
        details = success_message if found else failure_message
        
        return {
            'passed': found,
            'score': 1.0 if found else 0.0,
            'details': details,
            result_key: found,
            count_key: len(detections),
            'total_text_detections': len(all_text_detections)
        }

    def _is_text_detection(self, r, filter_keywords, target_keywords) -> bool:
        """Generic method to check if a result is a text detection of specific type."""
        if any(keyword in r.rule_name.lower() for keyword in filter_keywords):
            return True
        if r.details and 'detected_text' in r.details:
            return True
        if r.details and isinstance(r.details, dict):
            for key, value in r.details.items():
                if isinstance(value, str) and any(target in value.lower() for target in target_keywords):
                    return True
        return False

    def _find_patterns_in_detections(self, detections, patterns, custom_pattern_check=None):
        """Generic method to find patterns in detection results."""
        found = False
        matching_detections = []
        for result in detections:
            detected_text = self._extract_detected_text(result)
            if detected_text:
                if custom_pattern_check:
                    if custom_pattern_check(detected_text, patterns):
                        found = True
                        matching_detections.append(result)
                else:
                    if any(pattern in detected_text for pattern in patterns):
                        found = True
                        matching_detections.append(result)
        return found, matching_detections

    def _eval_mode_pattern_check(self, detected_text, patterns) -> bool:
        """Custom pattern check for eval mode that requires both 'eval mode' and 'native audio'."""
        if any(pattern in detected_text for pattern in patterns):
            if 'eval mode' in detected_text and 'native audio' in detected_text:
                return True
            elif 'eval mode: native audio output' in detected_text:
                return True
        return False

    def _check_eval_mode_presence(self) -> Dict[str, Any]:
        """Check if 'Eval Mode: Native Audio Output' appears in any text detection."""
        return self._check_text_detection(
            filter_keywords=['text', 'ocr', 'eval'],
            target_keywords=['eval', 'mode', 'native', 'audio'],
            patterns=['eval mode', 'native audio', 'eval mode: native audio output'],
            success_message="✅ 'Eval Mode: Native Audio Output' mode found with OCR text detections. Correct usage confirmed.",
            failure_message="❌ 'Eval Mode: Native Audio Output' mode was not found in any OCR text detections. Please ensure to use the correct mode and try again.",
            result_key='eval_mode_found',
            count_key='eval_mode_count',
            custom_pattern_check=self._eval_mode_pattern_check
        )

    def _check_language_fluency(self) -> Dict[str, Any]:
        """Check if language fluency requirements are met in the video."""
        language_results = [r for r in self.results if 'Language Detection' in r.rule_name]
        
        if not language_results:
            return {
                'passed': False,
                'score': 0.0,
                'details': '❌ No language analysis performed',
                'detected_language': 'unknown',
                'transcription_preview': ''
            }
        
        result = language_results[0]
        
        if not isinstance(result.details, dict):
            return {
                'passed': False,
                'score': 0.0,
                'details': '❌ Invalid language analysis result',
                'detected_language': 'unknown',
                'transcription_preview': ''
            }
        
        detected_language = result.details.get('detected_language', 'unknown')
        transcription = result.details.get('transcription', '')
        fluency_score = result.details.get('fluency_score', 0.0)
        total_words = result.details.get('total_words', 0)
        is_fluent = result.detected
        
        if is_fluent:
            details = "✅ The analysis detected the expected language. Fluency in spoken language has been confirmed."
        else:
            details = "❌ The analysis detected incorrect language usage. The fluency in the spoken language is not accurate."
        
        if transcription:
            preview = transcription[:50] + "..." if len(transcription) > 50 else transcription
        
        return {
            'passed': is_fluent,
            'score': fluency_score,
            'details': details,
            'detected_language': detected_language,
            'total_words': total_words,
            'transcription_preview': transcription[:100] if transcription else '',
            'avg_fluency_score': fluency_score
        }

    def _check_voice_audibility(self) -> Dict[str, Any]:
        """Check if both user and model voices are audible in the video."""
        voice_results = [r for r in self.results if 'Voice Audibility' in r.rule_name]
        
        if not voice_results:
            return {
                'passed': False,
                'score': 0.0,
                'details': '❌ No voice audibility analysis performed',
                'both_voices_audible': False,
                'issues_detected': ['No voice audibility data available'],
                'quality_summary': 'Voice audibility analysis not available'
            }
        
        latest_result = voice_results[-1]
        details = latest_result.details
        
        num_voices = details.get('num_audible_voices', 0)
        both_voices_audible = details.get('both_voices_audible', False)
        voice_ratio = details.get('voice_ratio', 0.0)
        total_voice_duration = details.get('total_voice_duration', 0.0)
        has_multiple_speakers = details.get('has_multiple_speakers', False)
        
        passed = both_voices_audible and num_voices == 2
        
        if num_voices == 2:
            score = 1.0
        elif num_voices == 1:
            score = 0.5
        else:
            score = 0.0
        
        if voice_ratio < 0.1:
            score *= 0.5
        
        issues = []
        if num_voices == 0:
            issues.append('No audible voices detected in the audio')
        elif num_voices == 1:
            issues.append('Only one voice detected - expected both user and model voices')
        elif num_voices > 2:
            issues.append(f'Too many voices detected ({num_voices}) - expected exactly 2')
        
        if voice_ratio < 0.1:
            issues.append(f'Very low voice activity ({voice_ratio:.1%} of audio)')
        
        if total_voice_duration < 3.0:
            issues.append(f'Insufficient voice duration ({total_voice_duration:.1f}s)')
        
        if passed:
            quality_summary = 'Both user and model voices are clearly audible'
        elif num_voices == 1:
            quality_summary = 'Only one voice is audible - missing either user or model voice'
        elif num_voices == 0:
            quality_summary = 'No voices detected in the audio'
        else:
            quality_summary = f'{num_voices} voices detected - audio quality may be compromised'
        
        if num_voices == 0:
            detailed_desc = '❌ No audible voices were detected in the video. Please ensure that both user and model voices are present.'
        elif num_voices == 1:
            detailed_desc = '❌ Only one voice was detected; either the user\'s or the model\'s voice is absent in the video. Please ensure that both the user and model voices are clearly audible.'
        elif num_voices == 2:
            detailed_desc = '✅ The analysis identified two distinct audible voices in the video. Both user and model voices are present'
        else:
            detailed_desc = f'❌ {num_voices} voices detected, which may indicate background noise or more speakers than just the user and model. Please review your video and try again.'
        
        return {
            'passed': passed,
            'score': score,
            'details': detailed_desc,
            'both_voices_audible': both_voices_audible,
            'num_voices_detected': num_voices,
            'voice_ratio': voice_ratio,
            'total_voice_duration': total_voice_duration,
            'has_multiple_speakers': has_multiple_speakers,
            'issues_detected': issues,
            'quality_summary': quality_summary
        }

    def get_qa_summary(self) -> Dict[str, Any]:
        """Get QA check summary."""
        return self.qa_results['overall']

    def get_detailed_results(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed QA results."""
        return {k: v for k, v in self.qa_results.items() if k != 'overall'}


class ApplicationRunner:
    """Main application runner with three-screen interface."""
    
    @staticmethod
    def run_streamlit_app():
        """Main application entry point with three-screen navigation."""
        try:
            ApplicationRunner._setup_signal_handlers()
            
            StreamlitInterface.setup_page()
            ScreenManager.initialize_session_state()
            
            ApplicationRunner._render_sidebar()
            
            st.markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
                <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎥 Gemini Live Video Verifier</h1>
                <p style="color: white; font-size: 1.1rem; margin: 10px 0 0 0; opacity: 0.9;">
                    Multi-modal video analysis tool for content detection, language fluency, and quality assurance validation
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            StreamlitInterface.render_progress_indicator()

            if st.session_state.get('submission_locked', False):
                st.markdown("""
                <div style="background-color: #fff8dc; padding: 15px; border-radius: 8px; margin: 20px 0; border: 1px solid #f0e68c;">
                    <p style="color: #b8860b; margin: 0; font-size: 16px; text-align: center;">
                        🔒 This session is now locked. Start a new session for another video analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            current_screen = ScreenManager.get_current_screen()
            
            main_content_area = st.container()
            
            with main_content_area:
                if current_screen == 'input':
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #e8f4f8 0%, #f0f9ff 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
                        <h3 style="color: #1565C0; margin-top: 0; display: flex; align-items: center;">
                            📖 Analysis Information
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **What will be analyzed from your video:**
                        - **Text Detection**: Keyword recognition using OCR for "2.5 Flash", "Roaring Tiger", and "Eval Mode: Native Audio Output" to verify proper model and alias usage
                        - **Language Fluency**: Multi-language speech verification ensures the spoken language matches the expected language, with a minimum fluency score required
                        - **Voice Audibility**: Detection of multiple distinct voices to confirm both user and model voices are clearly audible
                        """)
                    
                    with col2:
                        st.markdown("""
                        **Process:**
                        1. Input your unique Question ID and Alias Email
                        2. Upload your video file in MP4 format with a maximum size of 1GB, a minimum duration of 30 seconds, and a portrait mobile resolution
                        3. The system will automatically analyze the video for text, language, and audio quality
                        4. View detailed results and quality assurance checks
                        5. A Google Drive link will be provided for you to submit your video for further processing
                        6. If any issues are detected, you will be prompted to re-record and upload a new video
                        """)
                    
                    st.divider()
                
                active_count = len(get_session_manager().get_active_sessions())
                
                if current_screen == 'input':
                    InputScreen.render()
                elif current_screen == 'analysis':
                    main_content_area.empty()
                    AnalysisScreen.render()
                elif current_screen == 'qa':
                    main_content_area.empty()
                    VideoSubmissionScreen.render()
                else:
                    st.error("❌ Invalid screen state. Redirecting to input.")
                    ScreenManager.navigate_to_screen('input')
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"❌ System error: {str(e)}")
            
            st.markdown("""
            **Recovery Options:**
            - Refresh the page to start a new session
            - Try with a smaller video file
            - Contact support if the problem persists
            """)
    
    @staticmethod
    def _setup_signal_handlers():
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            try:
                session_manager = get_session_manager()
                session_manager.cleanup_old_sessions()
            except:
                pass
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except Exception:
            pass
    
    @staticmethod
    def _render_sidebar():
        """Render the main sidebar with help and session information."""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 10px; margin-bottom: 20px;">
                <h2 style="color: #1f77b4; margin: 0;">🎥 Gemini Live Video Verifier</h2>
                <p style="color: #666; margin: 5px 0 0 0; font-size: 0.9em;">Multi-Modal Analysis Tool</p>
            </div>
            """, unsafe_allow_html=True)
            
            ApplicationRunner._render_sidebar_help()
            
            st.divider()
            
            ApplicationRunner._render_sidebar_session_info()
    
    @staticmethod
    def _render_sidebar_help():
        """Render help dropdown with program usage instructions."""
        with st.expander("❓ Help - How to Use", expanded=False):
            st.markdown("""
            ### 📋 How to Use This Tool
            
            **Step 1: Input Parameters**
            - Enter a unique **Question ID** for identification (target language will be automatically inferred)
            - Provide the respective **Alias Email Address** for the Question ID
            - **Upload a video file** (supported format: MP4 up to 1GB, minimum duration 30 seconds, portrait mobile resolution)

            **Step 2: Video Analysis**
            - The system will automatically process your video with progress will be displayed in real-time
            - **Text detection** will look for specific content ("2.5 Flash", "Roaring Tiger", "Eval Mode: Native Audio Output") using OCR techniques to verify correct model and alias usage
            - **Audio analysis** will check language fluency and voice audibility to ensure both user and model voices are clearly heard and the spoken language matches the expected language
            - Review the **detailed analysis results** with screenshots and audio summaries
            - If any issues are detected, detailed feedback will be provided to help you understand and you will be prompted to re-record and upload a new video

            **Step 3: Submit Video**
            - Once all quality checks are passed, a **Google Drive link** will be provided for you to submit your video for further processing
            
            ### 🎯 What Gets Analyzed
            
            **Text Detection:**
            - "2.5 Flash" text recognition to verify correct model usage and avoid "2.5 Pro" misuse
            - "Roaring Tiger" alias name recognition to confirm correct alias usage
            - "Eval Mode: Native Audio Output" detection to ensure proper eval mode
            
            **Audio Analysis:**
            - Language detection and fluency scoring (based on expected language)
            - Voice audibility checks to confirm both user and model voices are clearly audible

            **Quality Assurance:**
            - Flash text presence check
            - Alias name presence check
            - Eval mode presence check
            - Language fluency verification
            - Voice audibility confirmation
            
            ### 📊 Understanding Results
            
            **QA Status:**
            - ✅ **PASS**: All quality checks successful. You can proceed to submit your video.
            - ❌ **FAIL**: Some quality checks failed. Review the feedback and re-record if necessary.
            
            ### 💡 Tips for Best Results
            
            - Follow the **model and alias usage guidelines**
            - Use **high-quality video files** (good resolution, clear audio)
            - Ensure **proper lighting** for text detection
            - Speak clearly in the **expected language**
            - Minimize **background noise** for better voice audibility
            - Verify all **input parameters** before starting analysis
            """)
    
    @staticmethod
    def _render_sidebar_session_info():
        """Render current session information in sidebar."""
        current_screen = ScreenManager.get_current_screen()
        
        has_session_data = (
            hasattr(st.session_state, 'question_id') and 
            st.session_state.question_id and
            hasattr(st.session_state, 'alias_email') and
            st.session_state.alias_email and
            current_screen != 'input'
        )
        
        if has_session_data:
            st.markdown("### 📋 Current Session")
            
            session_info = ApplicationRunner._get_session_display_info()
            
            session_html = f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%); padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: #2e7d32;">Session ID: <code style="background: #e0e0e0; padding: 2px 4px; border-radius: 3px; font-size: 0.7em;">{session_info['session_id']}</code></h4>
                <p style="margin: 5px 0; color: #333;"><strong>Question ID:</strong><br>{session_info['question_id']}</p>"""
            
            session_html += f"""
                <p style="margin: 5px 0; color: #333;"><strong>Email:</strong><br>{session_info['email']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Video File:</strong><br>{session_info['video_file']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Language:</strong><br>{session_info['language']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Task Type:</strong><br>{session_info['task_type']}</p>"""
            
            session_html += "</div>"
            
            st.markdown(session_html, unsafe_allow_html=True)
            
            if current_screen in ['analysis', 'qa']:
                ApplicationRunner._render_sidebar_analysis_status()
    
    @staticmethod
    def _get_session_display_info():
        """Get formatted session information for display."""
        language_code = st.session_state.get('selected_language', 'en-US')
        language_display = Config.get_language_display_name(language_code)
        
        task_type = st.session_state.get('task_type', '')
        if not task_type or task_type == 'Unknown':
            task_type = 'Not specified'
        
        session_info = {
            'question_id': st.session_state.get('question_id', 'Not specified'),
            'email': st.session_state.get('alias_email', 'Not specified'),
            'video_file': st.session_state.get('video_file').name if st.session_state.get('video_file') else 'No file uploaded',
            'language': language_display,
            'task_type': task_type,
            'frame_interval': f"{st.session_state.get('frame_interval', Config.DEFAULT_FRAME_INTERVAL)}s",
            'session_id': st.session_state.get('session_id', 'Unknown')[:12] + '...' if st.session_state.get('session_id') else 'Unknown'
        }
        
        video_validation = st.session_state.get('video_validation', {})
        if video_validation and 'duration' in video_validation:
            session_info['video_properties'] = {
                'duration': f"{video_validation.get('duration', 0):.1f}s",
                'resolution': f"{video_validation.get('width', 0)}x{video_validation.get('height', 0)}",
                'audio_properties': video_validation.get('audio_properties', {})
            }
        
        return session_info
    
    @staticmethod
    def _render_sidebar_analysis_status():
        """Render analysis status information in sidebar."""
        if st.session_state.get('analysis_results'):
            results = st.session_state.analysis_results
            results_count = len(results)
            positive_detections = sum(1 for r in results if r.detected)
            
            qa_color = "#2196F3"
            qa_status = "✅ Completed"
            if st.session_state.get('qa_checker'):
                qa_summary = st.session_state.qa_checker.get_qa_summary()
                if qa_summary['passed']:
                    qa_color = "#4CAF50"
                    qa_status = "✅ PASS"
                else:
                    qa_color = "#f44336"
                    qa_status = "❌ FAIL"
            
            text_results = [r for r in results if 'Text Detection' in r.rule_name]
            language_results = [r for r in results if 'Language Detection' in r.rule_name]
            voice_results = [r for r in results if 'Voice Audibility' in r.rule_name]
            
            analysis_details = []
            
            flash_results = [r for r in text_results if '2.5 Flash' in r.rule_name]
            alias_results = [r for r in text_results if 'Alias Name' in r.rule_name]
            eval_results = [r for r in text_results if 'Eval Mode' in r.rule_name]
            
            if flash_results:
                flash_detected = any(r.detected for r in flash_results)
                flash_qa_info = ApplicationRunner._get_qa_info_for_rule_type('flash_presence')
                flash_status = "✅" if flash_qa_info and flash_qa_info.get('passed', False) else "❌"
                
                if flash_detected:
                    status_text = "Detected"
                elif flash_qa_info and flash_qa_info.get('validation_status') == 'incorrect_model':
                    status_text = "Incorrect Model"
                else:
                    status_text = "Not Found"
                
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>2.5 Flash:</strong><br>{flash_status} {status_text}</p>")
            
            if alias_results:
                alias_detected = any(r.detected for r in alias_results)
                alias_qa_info = ApplicationRunner._get_qa_info_for_rule_type('alias_name_presence')
                alias_status = "✅" if alias_qa_info and alias_qa_info.get('passed', False) else "❌"
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Alias Name:</strong><br>{alias_status} {'Detected' if alias_detected else 'Not Found'}</p>")
            
            if eval_results:
                eval_detected = any(r.detected for r in eval_results)
                eval_qa_info = ApplicationRunner._get_qa_info_for_rule_type('eval_mode_presence')
                eval_status = "✅" if eval_qa_info and eval_qa_info.get('passed', False) else "❌"
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Eval Mode:</strong><br>{eval_status} {'Detected' if eval_detected else 'Not Found'}</p>")
            
            if language_results:
                language_result = language_results[0]
                language_qa_info = ApplicationRunner._get_qa_info_for_rule_type('language_fluency')
                language_status = "✅" if language_qa_info and language_qa_info.get('passed', False) else "❌"
                
                if language_result.details and 'analysis_failed_reason' in language_result.details:
                    analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Language Fluency:</strong><br>{language_status} No Voices Detected</p>")
                else:
                    detected_lang = language_result.details.get('detected_language', 'unknown') if language_result.details else 'unknown'
                    target_lang = language_result.details.get('target_language', 'unknown') if language_result.details else 'unknown'
                    if detected_lang != 'unknown':
                        locale_format = Config.whisper_language_to_locale(detected_lang, target_lang)
                        display_name = Config.get_language_display_name(locale_format)

                        if display_name is None:
                            display_name = detected_lang if detected_lang else "Unknown"
                        analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Language Fluency:</strong><br>{language_status} {display_name}</p>")
                    else:
                        analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Language Fluency:</strong><br>{language_status} Unknown</p>")
            
            if voice_results:
                voice_result = voice_results[0]
                voice_qa_info = ApplicationRunner._get_qa_info_for_rule_type('voice_audibility')
                voice_status = "✅" if voice_qa_info and voice_qa_info.get('passed', False) else "❌"
                
                if voice_result.details:
                    num_voices = voice_result.details.get('num_audible_voices', 0)
                    analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Voice Audibility:</strong><br>{voice_status} {num_voices} Voice{'s' if num_voices != 1 else ''}</p>")
                else:
                    analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Voice Audibility:</strong><br>{voice_status} Unknown</p>")
            
            details_html = "".join(analysis_details) if analysis_details else "<p style=\"margin: 5px 0; color: #333;\">No detailed analysis available</p>"
            
            # Add total analysis time
            total_analysis_time = ApplicationRunner._get_total_analysis_time_for_sidebar()
            if total_analysis_time > 0:
                details_html += f"<p style=\"margin: 5px 0; color: #333;\"><strong>Analysis Time:</strong><br>⏱️ {total_analysis_time:.2f} seconds</p>"
            
            if qa_status == "✅ PASS":
                bg_gradient = "linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%)"
                header_color = "#2e7d32"
            elif qa_status == "❌ FAIL":
                bg_gradient = "linear-gradient(135deg, #ffebee 0%, #fce4ec 100%)"
                header_color = "#c62828"
            else:
                bg_gradient = "linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%)"
                header_color = "#1565C0"
            
            st.markdown("### 📊 Analysis Report")
            
            st.markdown(f"""
            <div style="background: {bg_gradient}; padding: 15px; border-radius: 8px; border-left: 4px solid {qa_color}; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: {header_color};">Overall Status: {qa_status}</h4>
                {details_html}
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _get_qa_info_for_rule_type(rule_type: str):
        """Get QA information for a specific rule type."""
        if not st.session_state.get('qa_checker'):
            return None
        
        qa_results = st.session_state.qa_checker.get_detailed_results()
        return qa_results.get(rule_type)

    @staticmethod
    def _get_total_analysis_time_for_sidebar() -> float:
        """Get the total analysis time from the analyzer instance for sidebar display."""
        try:
            analyzer = st.session_state.get('analyzer_instance')
            if analyzer and hasattr(analyzer, 'performance_metrics'):
                return analyzer.performance_metrics.get('total_analysis_time', 0.0)
            return 0.0
        except Exception as e:
            logger.debug(f"Could not retrieve total analysis time for sidebar: {e}")
            return 0.0


if __name__ == "__main__":
    try:
        ApplicationRunner.run_streamlit_app()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal application error: {e}")
        raise
    finally:
        try:
            session_manager = get_session_manager()
            session_manager.cleanup_old_sessions()
        except:
            pass
