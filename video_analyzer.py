"""
Gemini Live Video Analysis System for Multi-Modal Content Detection.

A video analysis system that provides text recognition, language fluency analysis, and speaker diarization.
The system includes quality assurance validation and a multi-screen Streamlit interface.
"""

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
import concurrent.futures
import psutil
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


class Config:
    """
    Application configuration constants.
    
    This class centralizes all configuration parameters to improve maintainability.
    """
    
    # Frame processing parameters
    DEFAULT_FRAME_INTERVAL: float = 5.0
    MIN_FRAME_INTERVAL: float = 1.0
    MAX_FRAME_INTERVAL: float = 10.0
    
    # OCR configuration
    OCR_CONFIG: str = '--psm 6'
    OCR_DENOISING_ENABLED: bool = True
    OCR_THRESHOLD_ENABLED: bool = True
    
    # Directory structure
    OUTPUT_DIR: str = "analysis_output"
    TEMP_DIR: str = "temp"
    SUPPORTED_VIDEO_FORMATS: List[str] = ['.mp4']
    
    # UI annotation settings
    ANNOTATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
    ANNOTATION_FONT_SCALE: float = 0.6
    ANNOTATION_THICKNESS: int = 2
    
    # Resource limits (always optimized)
    MAX_FILE_SIZE: int = 500 * 1024 * 1024 # 500MB
    MAX_VIDEO_DURATION: int = 1200  # 20 minutes
    MAX_CONCURRENT_ANALYSES: int = 2  # Optimized for performance
    SESSION_TIMEOUT: int = 3600  # 1 hour
    
    # Performance optimizations (always enabled)
    STREAMLIT_CLOUD_MODE: bool = True
    ENABLE_PERFORMANCE_MONITORING: bool = True
    AGGRESSIVE_MEMORY_CLEANUP: bool = True

    # Memory limits calculated dynamically based on system capacity (always optimized)
    @staticmethod
    def get_memory_limits() -> Dict[str, float]:
        """Get memory limits appropriate for the current system with optimizations always enabled."""
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Always use optimized settings regardless of environment
            if total_memory_gb <= 2:  # Very limited environment
                return {
                    "min_available_gb": 0.1,
                    "max_usage_percent": 85,
                    "per_session_gb": min(0.8, available_memory_gb * 0.8)  # Conservative
                }
            elif available_memory_gb >= 4.0:  # If we have 4GB+ available
                per_session_gb = min(2.0, available_memory_gb * 0.6)  # Use up to 60% of available
            elif available_memory_gb >= 3.0:  # If we have 3-4GB available
                per_session_gb = min(1.5, available_memory_gb * 0.65)  # Use up to 65% of available
            elif available_memory_gb >= 2.0:  # If we have 2-3GB available
                per_session_gb = min(1.2, available_memory_gb * 0.7)  # Use up to 70% of available
            else:  # Less than 2GB available
                per_session_gb = max(0.8, available_memory_gb * 0.75)  # Use up to 75% of available
            
            if total_memory_gb <= 8:  # Small environment (always use optimized settings)
                return {
                    "min_available_gb": 0.15,
                    "max_usage_percent": 90,
                    "per_session_gb": per_session_gb
                }
            elif total_memory_gb <= 16:  # Medium environment (always use optimized settings)
                return {
                    "min_available_gb": 0.3,
                    "max_usage_percent": 88,
                    "per_session_gb": per_session_gb
                }
            else:  # Large environment (always use optimized settings)
                return {
                    "min_available_gb": 0.5,
                    "max_usage_percent": 85,
                    "per_session_gb": per_session_gb
                }
        except:
            # Fallback - always use optimized settings
            return {
                "min_available_gb": 0.1,
                "max_usage_percent": 85,
                "per_session_gb": 0.8
            }
    
    # Security validation
    ALLOWED_FILENAME_PATTERN: re.Pattern = re.compile(r'^[a-zA-Z0-9._\-\s:]+$')
    MAX_FILENAME_LENGTH: int = 255
    
    # Color definitions for UI annotations and bounding boxes
    COLORS: Dict[str, Tuple[int, int, int]] = {
        'GREEN': (0, 255, 0),
        'BLUE': (255, 0, 0),
        'RED': (0, 0, 255),
        'YELLOW': (0, 255, 255),
        'WHITE': (255, 255, 255),
        'BLACK': (0, 0, 0)
    }
    
    # Supported languages for audio analysis
    SUPPORTED_LANGUAGES: Dict[str, str] = {
        'es-419': 'es-419',
        'hi-IN': 'hi-IN',
        'ja-JP': 'ja-JP',
        'ko-KR': 'ko-KR',
        'de-DE': 'de-DE',
        'en-IN': 'en-IN',
        'fr-FR': 'fr-FR',
        'ar-EG': 'ar-EG',
        'pt-BR': 'pt-BR',
        'id-ID': 'id-ID',
        'ko-JA': 'ko-JA',
        'zh-CN': 'zh-CN',
        'ru-RU': 'ru-RU',
        'ml-IN': 'ml-IN',
        'sv-SE': 'sv-SE',
        'te-IN': 'te-IN',
        'vi-VN': 'vi-VN',
        'tr-TR': 'tr-TR',
        'bn-IN': 'bn-IN',
        'it-IT': 'it-IT',
        'zh-TW': 'zh-TW',
        'pl-PL': 'pl-PL',
        'nl-NL': 'nl-NL',
        'th-TH': 'th-TH',
        'ko-ZH': 'ko-ZH',
    }
    
    @staticmethod
    def get_language_options() -> Dict[str, str]:
        """Get language options for UI selectors."""
        return Config.SUPPORTED_LANGUAGES.copy()

    @staticmethod
    def get_language_display_name(language_code: str) -> str:
        """Get display name for a language code."""
        return Config.SUPPORTED_LANGUAGES.get(language_code, language_code.upper())
    
    @staticmethod
    def locale_to_whisper_language(locale_code: str) -> str:
        """Convert locale code (e.g., 'en-US') to Whisper language code (e.g., 'en'). Only supports configured languages."""
        locale_to_whisper = {
            'es-419': 'es',
            'hi-IN': 'hi',
            'ja-JP': 'ja',
            'ko-KR': 'ko',
            'de-DE': 'de',
            'en-IN': 'en',
            'fr-FR': 'fr',
            'ar-EG': 'ar',
            'pt-BR': 'pt',
            'id-ID': 'id',
            'ko-JA': 'ko',
            'zh-CN': 'zh',
            'ru-RU': 'ru',
            'ml-IN': 'ml',
            'sv-SE': 'sv',
            'te-IN': 'te',
            'vi-VN': 'vi',
            'tr-TR': 'tr',
            'bn-IN': 'bn',
            'it-IT': 'it',
            'zh-TW': 'zh',
            'pl-PL': 'pl',
            'nl-NL': 'nl',
            'th-TH': 'th',
            'ko-ZH': 'ko',
        }
        # Return mapped value if present, else fallback to language part
        return locale_to_whisper.get(locale_code, locale_code.split('-')[0] if '-' in locale_code else locale_code)


class TargetTexts:
    """
    Target text definitions for detection.
    """
    
    FLASH_TEXTS: List[str] = ["2.5 Flash"]
    EVAL_MODE_TEXT: str = "Eval Mode: Native Audio Output"


class Logger:
    """
    Centralized logging management with configuration support.
    """
    
    _logger_instance: Optional[logging.Logger] = None
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_format: Optional[str] = None) -> logging.Logger:
        """
        Initialize application logging with configurable level and format.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Custom log format string
            
        Returns:
            Configured logger instance
        """
        if Logger._logger_instance is not None:
            return Logger._logger_instance
            
        if log_format is None:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
        
        Logger._logger_instance = logging.getLogger(__name__)
        
        return Logger._logger_instance


logger = Logger.setup_logging()


class DetectionType(Enum):
    """
    Content detection types enumeration.
    
    Uses auto() for automatic value assignment to prevent manual value management.
    """
    TEXT = auto()
    AUDIO_LANGUAGE = auto()
    VOICE_AUDIBILITY = auto()


@dataclass(frozen=True)
class DetectionRule:
    """
    Detection rule configuration.
    """
    name: str
    detection_type: DetectionType
    parameters: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Validate rule configuration after initialization."""
        self._validate_rule()
    
    def _validate_rule(self) -> None:
        """
        Rule validation.
        
        Raises:
            ValueError: If rule configuration is invalid
        """
        if not self.name or not self.name.strip():
            raise ValueError("Rule name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Rule name too long (max 100 characters)")
        
        validation_methods = {
            DetectionType.TEXT: self._validate_text_rule,
            DetectionType.AUDIO_LANGUAGE: self._validate_audio_language_rule,
            DetectionType.VOICE_AUDIBILITY: self._validate_voice_audibility_rule
        }
        
        validator = validation_methods.get(self.detection_type)
        if validator:
            validator()
    
    def _validate_text_rule(self) -> None:
        """Validate text detection rule parameters."""
        required_keys = ['expected_text', 'unexpected_text']
        if not any(key in self.parameters for key in required_keys):
            raise ValueError("Text rule requires at least one of: expected_text or unexpected_text")
    
    def _validate_audio_language_rule(self) -> None:
        """Validate audio language detection rule parameters."""
        if 'target_language' not in self.parameters:
            raise ValueError("Audio language rule requires target_language parameter")
        
        target_lang = self.parameters['target_language']
        if target_lang not in Config.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {target_lang}")
    
    def _validate_voice_audibility_rule(self) -> None:
        """Validate voice audibility rule parameters."""
        min_confidence = self.parameters.get('min_confidence', 0.3)
        min_duration = self.parameters.get('min_duration', 3.0)
        
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        
        if min_duration < 0.0:
            raise ValueError("min_duration must be non-negative")


@dataclass
class DetectionResult:
    """
    Detection operation result with metadata.
    
    Provides detailed information about detection operations including
    performance metrics and debugging information.
    """
    rule_name: str
    timestamp: float
    frame_number: int
    detected: bool
    confidence: float
    details: Dict[str, Any]
    screenshot_path: Optional[Union[str, Path]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Validate result data after initialization."""
        self._validate_result()
    
    def _validate_result(self) -> None:
        """Validate detection result data."""
        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be a number between 0.0 and 1.0")
        
        if self.timestamp < 0:
            raise ValueError("Timestamp cannot be negative")
        
        if self.frame_number < 0:
            raise ValueError("Frame number cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert detection result to dictionary representation.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'rule_name': self.rule_name,
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'detected': self.detected,
            'confidence': self.confidence,
            'details': self.details,
            'screenshot_path': str(self.screenshot_path) if self.screenshot_path else None,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def is_valid(self) -> bool:
        """Check if the detection result is valid."""
        try:
            self._validate_result()
            return self.error_message is None
        except ValueError:
            return False


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class GoogleSheetsVerifier:
    """
    Verifies question IDs, alias emails, and languages against authorized list in SoT.
    
    This class implements secure verification of video analysis requests by checking
    question IDs, alias emails, and language IDs against a SoT.
    """
    
    CACHE_DURATION = 300  # 5 minutes cache duration
    
    def __init__(self):
        self._cache = {}
        self._last_cache_time = 0
        
        # Load configuration from secrets
        config = ConfigurationManager.get_secure_config()
        self.SHEET_URL = config.get("verifier_sheet_url", "")
        self.SHEET_ID = config.get("verifier_sheet_id", "")
        self.SHEET_NAME = config.get("verifier_sheet_name", "SoT Video Verifier")
    
    def verify_authorization(self, question_id: str, alias_email: str, language_id: str) -> Tuple[bool, str]:
        """
        Verify if question ID, alias email, and language are authorized for video analysis.
        
        Args:
            question_id: The question identifier to verify
            alias_email: The alias email to verify
            language_id: The language ID to verify (e.g., 'en-US', 'id-ID', 'es-MX')
            
        Returns:
            Tuple of (is_authorized, reason) where reason explains the result
        """
        try:
            # Input validation
            if not question_id or not question_id.strip():
                return False, "Question ID cannot be empty"
            
            if not alias_email or not alias_email.strip():
                return False, "Alias email cannot be empty"
            
            # Language ID validation
            if not language_id or not language_id.strip():
                return False, "Language ID is required and cannot be empty"
            clean_language_id = InputValidator.sanitize_user_input(language_id.strip())
            # Validate language ID format
            if not re.match(r'^[a-z]{2,3}-[A-Z0-9]{2,4}$', clean_language_id):
                return False, f"Invalid language ID format: {language_id}. Expected format: 'en-US', 'id-ID', 'es-419', etc."
            
            # Sanitize inputs
            clean_question_id = InputValidator.sanitize_user_input(question_id.strip())
            clean_alias_email = InputValidator.sanitize_user_input(alias_email.strip()).lower()
            
            # Get authorized entries from SoT
            authorized_entries = self._get_cached_data(
                'authorized_entries',
                lambda: self._fetch_from_sheets("getAuthorizedEntries", self.SHEET_NAME)
            )
            if not authorized_entries:
                return False, "Unable to retrieve authorization data from SoT"
            
            # Check if combination is authorized
            is_authorized = self._check_authorization(
                clean_question_id, clean_alias_email, clean_language_id, authorized_entries
            )
            
            if is_authorized:
                auth_msg = f"question_id={clean_question_id}, email={clean_alias_email}, language={clean_language_id}"
                return True, "Authorization verified successfully"
            else:
                auth_msg = f"Question ID '{clean_question_id}' with email '{clean_alias_email}' and language '{clean_language_id}' is not authorized for video verification"
                logger.warning(f"Authorization failed: {auth_msg}")
                return False, auth_msg
        
        except Exception as e:
            logger.error(f"Authorization verification error: {e}")
            return False, f"Authorization check failed: {str(e)}"
    
    def _get_cached_data(self, cache_key: str, fetch_function) -> any:
        """
        Method to get cached data with fallback to fetch function.
        
        Args:
            cache_key: The key to use for caching the data
            fetch_function: Function to call to fetch fresh data
            
        Returns:
            Cached or freshly fetched data
        """
        current_time = time.time()
        
        # Check cache validity
        if (current_time - self._last_cache_time) < self.CACHE_DURATION and cache_key in self._cache:
            return self._cache.get(cache_key, [])
        
        try:
            # Fetch fresh data
            data = fetch_function()
            
            # Update cache
            self._cache[cache_key] = data
            self._last_cache_time = current_time
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {cache_key}: {e}")
            # Return cached data if available, even if stale
            if cache_key in self._cache:
                logger.warning(f"Using stale cached data for {cache_key} due to fetch error")
                return self._cache.get(cache_key, [])
            return []
    
    def _fetch_from_sheets(self, action: str, sheet_name: str, data_processor=None):
        """
        Method to fetch data from Google Sheets via Google Apps Script.
        
        Args:
            action: The action to perform (e.g., "getAuthorizedEntries", "getQAEmails")
            sheet_name: The name of the sheet to query
            data_processor: Optional function to process the raw data
            
        Returns:
            Processed data from the sheet
        """
        try:
            # Use the existing Google Apps Script infrastructure
            config = ConfigurationManager.get_secure_config()
            gas_url = config["apps_script_url"]
            
            payload = {
                "action": action,
                "sheetName": sheet_name,
                "data": {}  # Empty data object since we're just fetching
            }
            
            response = requests.post(
                gas_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("success") or result.get("status") == "success":
                raw_data = result.get("data", [])
                if data_processor:
                    return data_processor(raw_data)
                return raw_data
            else:
                error_msg = result.get('message', result.get('error', 'Unknown error'))
                logger.error(f"Google Sheets API error for {action}: {error_msg}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Network error fetching from Google Sheets ({action}): {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching from Google Sheets ({action}): {e}")
            return []
    
    def _check_membership(self, item, authorized_list, match_function=None) -> bool:
        """
        Method to check if an item is in an authorized list.
        
        Args:
            item: The item to check for authorization
            authorized_list: List of authorized items
            match_function: Optional custom function to determine matches
            
        Returns:
            True if authorized, False otherwise
        """
        if match_function:
            return match_function(item, authorized_list)
        
        # Default behavior for simple list membership
        return item in authorized_list
    
    def _check_authorization(self, question_id: str, alias_email: str, language_id: str,
                           authorized_entries: List[Dict[str, str]] = None) -> bool:
        """
        Check if question ID, alias email, and language combination is authorized.
        
        Args:
            question_id: Clean question ID
            alias_email: Clean alias email
            language_id: Clean language ID
            authorized_entries: List of authorized entries
            
        Returns:
            True if authorized, False otherwise
        """
        def match_triple(params, entries):
            q_id, email, lang_id = params
            for entry in entries:
                entry_question_id = str(entry.get('question_id', '')).strip()
                entry_alias_email = str(entry.get('alias_email', '')).strip().lower()
                entry_language_id = str(entry.get('language_id', '')).strip()

                # All three must match for authorization
                question_match = entry_question_id == q_id
                email_match = entry_alias_email == email
                language_match = entry_language_id.lower() == lang_id.lower()
                if question_match and email_match and language_match:
                    return True
            return False
        
        return self._check_membership(
            (question_id, alias_email, language_id),
            authorized_entries,
            match_triple
        )
    
    def verify_qa_email(self, qa_email: str) -> Tuple[bool, str]:
        """
        Verify if QA email is authorized in the QA sheet.
        
        Args:
            qa_email: The QA email address to verify
            
        Returns:
            Tuple of (is_authorized, reason) where reason explains the result
        """
        try:
            # Input validation
            if not qa_email or not qa_email.strip():
                return False, "QA email cannot be empty"
            
            # Sanitize input
            clean_qa_email = InputValidator.sanitize_user_input(qa_email.strip()).lower()
            
            # Get authorized QA emails from SoT
            def process_qa_data(raw_data):
                qa_emails = []
                for entry in raw_data:
                    if isinstance(entry, dict) and 'qa_email' in entry:
                        email = str(entry['qa_email']).strip().lower()
                        if email:
                            qa_emails.append(email)
                    elif isinstance(entry, str):
                        email = entry.strip().lower()
                        if email:
                            qa_emails.append(email)
                return qa_emails
            
            authorized_qa_emails = self._get_cached_data(
                'authorized_qa_emails',
                lambda: self._fetch_from_sheets("getQAEmails", "QA", process_qa_data)
            )
            if not authorized_qa_emails:
                return False, "Unable to retrieve QA authorization data from SoT"

            # Check if QA email is authorized
            is_authorized = self._check_membership(clean_qa_email, authorized_qa_emails)
            
            if is_authorized:
                return True, "QA email verified successfully"
            else:
                logger.warning(f"QA authorization failed: {clean_qa_email} is not authorized")
                return False, f"QA email '{clean_qa_email}' is not authorized for video verification"
        
        except Exception as e:
            logger.error(f"QA authorization verification error: {e}")
            return False, f"QA authorization check failed: {str(e)}"


class InputValidator:
    """
    Input validation utilities with compiled patterns.
    
    Handles input sanitization and validation with pre-compiled regex patterns.
    """
    
    _DANGEROUS_PATTERNS = [
        re.compile(r'<[^>]*>', re.IGNORECASE),  # HTML tags
        re.compile(r'javascript:', re.IGNORECASE),  # JavaScript protocol
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'expression\s*\(', re.IGNORECASE),  # CSS expressions
        re.compile(r'(drop|delete|insert|update|select|union|exec|execute)\s+', re.IGNORECASE),  # SQL keywords
        re.compile(r'["\';\\]'),  # Quote characters
        re.compile(r'\.\./', re.IGNORECASE),  # Path traversal
        re.compile(r'\$\('),  # Command substitution
        re.compile(r'`[^`]*`'),  # Backticks
    ]
    
    _SESSION_SANITIZE_PATTERN = re.compile(r'[^\w-]')
    
    @staticmethod
    def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
        """
        Sanitize user input to prevent injection attacks using pre-compiled patterns.
        
        Args:
            user_input: Raw user input to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized input string
        """
        if not isinstance(user_input, str):
            return ""
        
        # Truncate to maximum length
        sanitized = user_input[:max_length]
        
        # Remove dangerous patterns
        for pattern in InputValidator._DANGEROUS_PATTERNS:
            sanitized = pattern.sub('', sanitized)
        
        # Remove control characters except common whitespace
        sanitized = ''.join(
            char for char in sanitized 
            if ord(char) >= 32 or char in '\t\n\r'
        )
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_session_id(session_id: str, max_length: int = 64) -> str:
        """
        Sanitize session ID for filesystem use with pre-compiled pattern.
        
        Args:
            session_id: Session ID to sanitize
            max_length: Maximum length to truncate to
            
        Returns:
            Sanitized session ID safe for filesystem use
        """
        if not isinstance(session_id, str):
            return ""
        
        return InputValidator._SESSION_SANITIZE_PATTERN.sub('_', session_id)[:max_length]
    
    @staticmethod
    def validate_input_type_and_content(input_value: str, field_name: str) -> bool:
        """
        Common input validation pattern.
        
        Args:
            input_value: Value to validate
            field_name: Name of field for logging
            
        Returns:
            True if input is valid, False otherwise
        """
        if not input_value or not isinstance(input_value, str):
            logger.warning(f"Invalid {field_name} type or empty value")
            return False
        return True


class PathValidator:
    """
    File path validation utilities.
    
    Separates path validation concerns from general security validation.
    """
    
    _allowed_prefixes_cache: Optional[List[str]] = None
    _cache_lock = threading.Lock()
    
    @classmethod
    def _get_allowed_prefixes(cls) -> List[str]:
        """Get cached allowed path prefixes."""
        if cls._allowed_prefixes_cache is None:
            with cls._cache_lock:
                if cls._allowed_prefixes_cache is None:
                    cls._allowed_prefixes_cache = [
                        '/tmp/',
                        '/var/tmp/',
                        tempfile.gettempdir(),
                        str(Path(Config.OUTPUT_DIR).resolve()),
                        str(Path(Config.TEMP_DIR).resolve())
                    ]
        return cls._allowed_prefixes_cache
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate file path to prevent path traversal and other attacks.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        if not InputValidator.validate_input_type_and_content(file_path, "file path"):
            return False
        
        try:
            resolved_path = Path(file_path).resolve()
            
            # Check for path traversal attempts
            if '..' in file_path:
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return False
            
            # Validate absolute paths are in allowed locations
            if file_path.startswith('/'):
                allowed_prefixes = PathValidator._get_allowed_prefixes()
                if not any(file_path.startswith(prefix) for prefix in allowed_prefixes):
                    logger.warning(f"Absolute path not in allowed locations: {file_path}")
                    return False
            
            # Validate filename
            filename = resolved_path.name
            if (len(filename) > Config.MAX_FILENAME_LENGTH or 
                not Config.ALLOWED_FILENAME_PATTERN.match(filename)):
                logger.warning(f"Invalid filename: {filename}")
                return False
                
            return True
            
        except (OSError, ValueError) as e:
            logger.error(f"Path validation error: {e}")
            return False
    
    @staticmethod
    def validate_file_size(file_path: str) -> bool:
        """
        Validate file size is within configured limits.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file size is acceptable, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist for size validation: {file_path}")
                return True
            
            file_size = os.path.getsize(file_path)
            if file_size > Config.MAX_FILE_SIZE:
                logger.warning(
                    f"File too large: {file_size} bytes (max: {Config.MAX_FILE_SIZE})"
                )
                return False
                
            return True
            
        except OSError as e:
            logger.error(f"File size validation error: {e}")
            return False


class URLValidator:
    """
    URL validation utilities for trusted domain verification.
    """
    
    # Allowed URL domains for external requests
    _TRUSTED_DOMAINS: List[str] = [
        'script.google.com',
        'docs.google.com',
        'drive.google.com',
        'googleapis.com'
    ]
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL to ensure it's from trusted domains.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is from trusted domain, False otherwise
        """
        if not InputValidator.validate_input_type_and_content(url, "URL"):
            return False
            
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Check if domain is in trusted list
            if parsed.hostname not in URLValidator._TRUSTED_DOMAINS:
                logger.warning(f"Untrusted domain in URL: {parsed.hostname}")
                return False
                
            # Ensure HTTPS for security
            if parsed.scheme != 'https':
                logger.warning(f"Non-HTTPS URL rejected: {parsed.scheme}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False


class SessionManager:
    """
    Session management utilities with proper resource tracking.
    
    Handles session ID generation and secure temporary directory management.
    """
    
    # Track created directories for cleanup
    _created_directories: Set[str] = set()
    _directory_lock = threading.Lock()
    
    @staticmethod
    def generate_session_id() -> str:
        """
        Generate cryptographically secure session ID.
        
        Returns:
            Unique session identifier
        """
        return str(uuid.uuid4())
    
    @staticmethod
    def create_secure_temp_dir(session_id: str) -> str:
        """
        Create secure temporary directory for user session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Path to created directory
            
        Raises:
            SecurityError: If directory creation fails
        """
        if not InputValidator.validate_input_type_and_content(session_id, "session ID"):
            raise SecurityError("Invalid session ID provided")
        
        # Sanitize session ID for filesystem use
        safe_session_id = InputValidator.sanitize_session_id(session_id)
        
        base_temp = Path(tempfile.gettempdir())
        session_dir = base_temp / f"video_analyzer_{safe_session_id}"
        
        try:
            session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            
            # Track created directory for cleanup
            with SessionManager._directory_lock:
                SessionManager._created_directories.add(str(session_dir))
            
            return str(session_dir)
            
        except OSError as e:
            logger.error(f"Failed to create secure temp dir: {e}")
            raise SecurityError(f"Cannot create secure temporary directory: {e}") from e
    
    @staticmethod
    def cleanup_session_directories() -> None:
        """Clean up all tracked session directories."""
        with SessionManager._directory_lock:
            directories_to_clean = SessionManager._created_directories.copy()
            SessionManager._created_directories.clear()
        
        for directory in directories_to_clean:
            try:
                if Path(directory).exists():
                    shutil.rmtree(directory)
            except Exception as e:
                logger.warning(f"Failed to cleanup session directory {directory}: {e}")


class ConfigurationManager:
    """
    Centralized configuration management.
    """
    
    _config_cache: Optional[Dict[str, str]] = None
    _cache_lock = threading.Lock()
    _cache_timestamp: float = 0
    _cache_ttl: float = 300  # 5 minutes TTL
    
    @classmethod
    def get_secure_config(cls, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get secure configuration from Streamlit secrets.
        
        Args:
            force_refresh: Force refresh of cached configuration
            
        Returns:
            Dictionary with validated configuration values
            
        Raises:
            SecurityError: If configuration is invalid or missing
        """
        current_time = time.time()
        
        # Check cache validity
        if (not force_refresh and cls._config_cache is not None and 
            (current_time - cls._cache_timestamp) < cls._cache_ttl):
            return cls._config_cache.copy()
        
        with cls._cache_lock:
            # Double-check pattern
            if (not force_refresh and cls._config_cache is not None and 
                (current_time - cls._cache_timestamp) < cls._cache_ttl):
                return cls._config_cache.copy()
            
            try:
                import streamlit as st
                
                # Validate required secrets exist
                if "google" not in st.secrets:
                    raise SecurityError("Google configuration missing from secrets")
                    
                google_config = st.secrets["google"]
                required_keys = ["apps_script_url", "sheets_url"]
                
                for key in required_keys:
                    if key not in google_config:
                        raise SecurityError(f"Required configuration missing: {key}")
                        
                    # Validate URLs
                    url_value = google_config[key]
                    if not URLValidator.validate_url(url_value):
                        raise SecurityError(f"Invalid or untrusted URL in {key}: {url_value}")
                
                config = {
                    "apps_script_url": google_config["apps_script_url"],
                    "sheets_url": google_config["sheets_url"],
                    "sheet_id": google_config.get("sheet_id", ""),
                    "default_sheet_name": st.secrets.get("settings", {}).get("default_sheet_name", "Submissions"),
                    "max_retries": st.secrets.get("settings", {}).get("max_retries", 2),
                    "timeout_base": st.secrets.get("settings", {}).get("timeout_base", 45)
                }
                
                # Update cache
                cls._config_cache = config
                cls._cache_timestamp = current_time
                
                return config.copy()
                
            except Exception as e:
                logger.error(f"Secure configuration error: {e}")
                raise SecurityError(f"Cannot load secure configuration: {e}") from e


class ResourceManager:
    """
    Manages system resources and limits for concurrent operations.
    """
    _CLEANUP_BATCH_SIZE = 50
    _UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]+$')
    _SESSION_PATTERN = re.compile(r'.*session.*', re.IGNORECASE)
    
    # Class-level coordination for concurrent operations
    _global_lock = threading.RLock()
    _active_sessions_global: Set[str] = set()
    _export_pending_sessions: Set[str] = set()

    def __init__(self) -> None:
        self._active_analyses: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stats = {'files_cleaned': 0, 'directories_cleaned': 0, 'errors': 0}
        self._start_cleanup_monitor()

    def __enter__(self) -> 'ResourceManager':
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        self.shutdown()

    @classmethod
    def register_active_session(cls, session_id: str) -> None:
        """Register a session as active globally to prevent premature cleanup."""
        with cls._global_lock:
            cls._active_sessions_global.add(session_id)

    @classmethod
    def unregister_active_session(cls, session_id: str) -> None:
        """Unregister a session as active globally."""
        with cls._global_lock:
            cls._active_sessions_global.discard(session_id)

    @classmethod
    def mark_session_for_export(cls, session_id: str) -> None:
        """Mark a session as pending export to prevent cleanup."""
        with cls._global_lock:
            cls._export_pending_sessions.add(session_id)

    @classmethod
    def unmark_session_for_export(cls, session_id: str) -> None:
        """Remove export pending status for a session."""
        with cls._global_lock:
            cls._export_pending_sessions.discard(session_id)

    def _start_cleanup_monitor(self) -> None:
        try:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_monitor,
                daemon=True,
                name="ResourceCleanupMonitor"
            )
            self._cleanup_thread.start()
        except Exception as e:
            logger.error(f"Failed to start cleanup monitor: {e}")

    def _cleanup_monitor(self) -> None:
        cleanup_interval = 300  # 5 minutes
        while not self._shutdown_event.wait(cleanup_interval):
            try:
                self._cleanup_expired_sessions()
                self._check_memory()
                self._check_cpu()
                self._log_resource_stats(self.get_active_count())
            except Exception as e:
                logger.error(f"Cleanup monitor error: {e}")

    def _cleanup_expired_sessions(self) -> None:
        current_time = datetime.now().timestamp()
        expired_sessions = []
        with self._lock:
            for session_id, session_data in self._active_analyses.items():
                session_age = current_time - session_data['start_time']
                if session_age > Config.SESSION_TIMEOUT:
                    expired_sessions.append((session_id, session_age))
        for session_id, age in expired_sessions:
            try:
                self.end_analysis(session_id, force_cleanup=True)
            except Exception as e:
                logger.error(f"Failed to cleanup expired session {session_id}: {e}")

    def can_start_analysis(self, video_duration: float = None) -> Tuple[bool, str]:
        """Check if a new analysis can be started considering video characteristics."""
        with self._lock:
            active_count = len(self._active_analyses)
            if active_count >= Config.MAX_CONCURRENT_ANALYSES:
                return False, (f"Maximum concurrent analyses "
                             f"({Config.MAX_CONCURRENT_ANALYSES}) reached")
            
            try:
                memory_info = psutil.virtual_memory()
                memory_limits = Config.get_memory_limits()
                min_available_gb = memory_limits["min_available_gb"]
                max_usage_percent = memory_limits["max_usage_percent"]
                per_session_gb = memory_limits["per_session_gb"]
                
                if video_duration is not None:
                    # Longer videos require more memory for processing
                    duration_factor = min(2.0, max(1.0, video_duration / 120.0))
                    adjusted_per_session_gb = per_session_gb * duration_factor
                else:
                    adjusted_per_session_gb = per_session_gb
                
                available_gb = memory_info.available / (1024**3)
                total_gb = memory_info.total / (1024**3)
                
                if available_gb < min_available_gb:
                    return False, f"Low memory: {available_gb:.1f}GB available, need {min_available_gb:.1f}GB minimum (system: {total_gb:.1f}GB)"
                
                if memory_info.percent > max_usage_percent:
                    return False, f"Memory usage too high: {memory_info.percent:.1f}% (max {max_usage_percent}% for {total_gb:.1f}GB system)"
                
                # Check if we have enough memory for this specific session
                required_for_session = adjusted_per_session_gb * (1024**3)
                
                min_required_for_session = required_for_session * 0.7
                if memory_info.available < min_required_for_session:
                    return False, f"Insufficient memory for session: need {adjusted_per_session_gb:.1f}GB, have {available_gb:.1f}GB"
                
                # Ensure we're not overloading the system with concurrent operations
                if active_count > 0:
                    estimated_memory_per_active = total_gb * 0.1
                    estimated_total_usage = estimated_memory_per_active * active_count + adjusted_per_session_gb
                    if estimated_total_usage > (total_gb * 0.8):
                        return False, f"System memory would be overloaded with {active_count + 1} concurrent sessions"
                
            except Exception as e:
                logger.warning(f"Memory check failed: {e}")
                if active_count < 3:
                    return True, "Resource check passed (fallback)"
                return False, f"Memory check failed and {active_count} sessions already active"
            
            return True, "Resource check passed"

    def start_analysis(self, session_id: str, analyzer_instance, video_duration: float = None) -> bool:
        """Start a new analysis session"""
        can_start, reason = self.can_start_analysis(video_duration)
        if not can_start:
            logger.warning(f"Cannot start analysis {session_id}: {reason}")
            return False
        
        with self._lock:
            self._active_analyses[session_id] = {
                'start_time': datetime.now().timestamp(),
                'analyzer': analyzer_instance,
                'pid': os.getpid(),
                'preserve_files': True,
                'video_duration': video_duration
            }
        
        # Register session globally to prevent premature cleanup
        ResourceManager.register_active_session(session_id)

        logger.info(f"Started analysis session: {session_id}")
        return True
    
    def end_analysis(self, session_id: str, force_cleanup: bool = False):
        """Unregister analysis session, optionally preserve files for UI."""
        with self._lock:
            if session_id in self._active_analyses:
                session_data = self._active_analyses[session_id]
                
                if force_cleanup:
                    try:
                        if hasattr(session_data['analyzer'], 'cleanup_screenshots'):
                            has_results = ('analysis_results' in session_data and 
                                         session_data.get('analysis_results'))
                            session_data['analyzer'].cleanup_screenshots(preserve_for_export=has_results)
                    except Exception as e:
                        logger.error(f"Screenshot cleanup error for {session_id}: {e}")
                else:
                    logger.debug(f"Preserving session files for UI: {session_id}")
                
                del self._active_analyses[session_id]
                logger.info(f"Ended analysis session: {session_id}")
        
        # Only unregister from global sessions if force cleanup is requested
        if force_cleanup:
            ResourceManager.unregister_active_session(session_id)
    
    def force_cleanup_session(self, session_id: str):
        """Force cleanup of a specific session."""
        self.end_analysis(session_id, force_cleanup=True)
    
    def get_active_count(self) -> int:
        """Get number of active analyses."""
        with self._lock:
            return len(self._active_analyses)
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data for a specific session ID."""
        with self._lock:
            return self._active_analyses.get(session_id)
    
    def cleanup_session_directories(self, session_id: str):
        """Clean up all directories and files for a specific session."""
        cleanup_stats = {'files': 0, 'directories': 0, 'errors': 0}
        
        try:
            # Clean session-specific directories
            self._cleanup_session_directory(Config.OUTPUT_DIR, session_id, cleanup_stats)
            self._cleanup_session_directory(Config.TEMP_DIR, session_id, cleanup_stats)
            
            # Clean system temp files
            self._cleanup_system_temp_files(session_id, cleanup_stats)
            
            # Update global stats
            with self._lock:
                self._cleanup_stats['files_cleaned'] += cleanup_stats['files']
                self._cleanup_stats['directories_cleaned'] += cleanup_stats['directories']
                self._cleanup_stats['errors'] += cleanup_stats['errors']
            
            # Unregister session from global tracking since it's fully cleaned up
            ResourceManager.unregister_active_session(session_id)
            ResourceManager.unmark_session_for_export(session_id)
            
        except Exception as e:
            logger.error(f"Session directories cleanup error for {session_id}: {e}")
    
    def _cleanup_session_directory(self, base_dir: str, session_id: str, 
                                  cleanup_stats: Dict[str, int]) -> None:
        """Clean up session directory in specified base directory."""
        try:
            base_path = Path(base_dir)
            if not base_path.exists():
                return
                
            session_dir = base_path / session_id
            if session_dir.exists():
                try:
                    # Get size info before deletion for memory tracking
                    dir_size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
                    if dir_size > 100 * 1024 * 1024:  # 100MB
                        logger.debug(f"Cleaning large directory ({dir_size / (1024*1024):.1f}MB): {session_dir}")
                    
                    shutil.rmtree(str(session_dir))
                    cleanup_stats['directories'] += 1
                except Exception as e:
                    cleanup_stats['errors'] += 1
                    logger.error(f"Failed to remove directory {session_dir}: {e}")
        except Exception as e:
            cleanup_stats['errors'] += 1
            logger.error(f"Error accessing base directory {base_dir}: {e}")
    
    def _cleanup_system_temp_files(self, session_id: str, cleanup_stats: Dict[str, int]) -> None:
        """Clean up system temp files."""
        try:
            system_temp = Path(tempfile.gettempdir())
            session_pattern = f"*{session_id}*"
            
            # Process files in batches to avoid memory spikes with large temp directories
            temp_files = list(system_temp.glob(session_pattern))
            
            for i in range(0, len(temp_files), self._CLEANUP_BATCH_SIZE):
                batch = temp_files[i:i + self._CLEANUP_BATCH_SIZE]
                
                for temp_file in batch:
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                            cleanup_stats['files'] += 1
                        elif temp_file.is_dir():
                            shutil.rmtree(str(temp_file))
                            cleanup_stats['directories'] += 1
                    except Exception as e:
                        cleanup_stats['errors'] += 1
                        logger.warning(f"Failed to remove temp item {temp_file}: {e}")
                
                # Small delay between batches to prevent overwhelming the system
                if i + self._CLEANUP_BATCH_SIZE < len(temp_files):
                    time.sleep(0.01)
                    
        except Exception as e:
            cleanup_stats['errors'] += 1
            logger.error(f"System temp cleanup error: {e}")
    
    def shutdown(self):
        """Shutdown resource manager and cleanup all sessions.""" 
        # Check for preserved sessions
        preserved_sessions = self._get_preserved_sessions()
        
        self._shutdown_event.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Process session cleanup in batches to avoid memory spikes
        with self._lock:
            session_ids = list(self._active_analyses.keys())
        
        self._cleanup_active_sessions(session_ids, preserved_sessions)
        
        # Cleanup directories based on preservation status
        if preserved_sessions:
            self._cleanup_session_directories_selective(preserved_sessions)
        else:
            self._cleanup_session_directories_all()
        
        # Log final cleanup statistics
        self._log_cleanup_statistics()
    
    # Integrated utility methods
    def _check_memory(self, threshold_percent=85):
        """Monitor system memory usage."""
        try:
            memory_info = psutil.virtual_memory()
            if memory_info.percent > threshold_percent:
                logger.warning(f"High memory usage: {memory_info.percent:.1f}%")
            return memory_info
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
            return None

    def _check_cpu(self, threshold_percent=90):
        """Monitor system CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > threshold_percent:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            return cpu_percent
        except Exception as e:
            logger.error(f"CPU monitoring error: {e}")
            return None

    def _log_resource_stats(self, active_count):
        """Log current resource statistics."""
        try:
            if active_count > 0:
                memory_info = psutil.virtual_memory()
        except Exception as e:
            logger.debug(f"Failed to log resource statistics: {e}")

    def _get_preserved_sessions(self) -> set:
        """Get set of sessions that should be preserved during shutdown."""
        preserved_sessions = set()
        
        # Include globally registered active sessions
        with ResourceManager._global_lock:
            preserved_sessions.update(ResourceManager._active_sessions_global)
            preserved_sessions.update(ResourceManager._export_pending_sessions)
        
        # First, check active sessions in memory that haven't been exported yet
        with self._lock:
            for session_id, session_data in self._active_analyses.items():
                if session_data and 'analyzer' in session_data:
                    analyzer = session_data['analyzer']
                    # Preserve sessions that have results but haven't been exported
                    if hasattr(analyzer, 'detection_results') and analyzer.detection_results:
                        preserved_sessions.add(session_id)
        
        # Also check for sessions with existing output directories
        try:
            analysis_output_path = Path(Config.OUTPUT_DIR)
            if analysis_output_path.exists():
                for item in analysis_output_path.iterdir():
                    if item.is_dir() and self._is_session_directory(item.name):
                        # Check if directory has screenshot files
                        screenshot_files = list(item.glob("*.png"))
                        if screenshot_files:
                            preserved_sessions.add(item.name)
        except Exception as e:
            logger.debug(f"Could not check output directories: {e}")
        
        try:
            import streamlit as st
            if (hasattr(st, 'session_state') and 
                hasattr(st.session_state, 'analysis_results') and 
                st.session_state.analysis_results and
                hasattr(st.session_state, 'analysis_session_id') and
                st.session_state.analysis_session_id):
                preserved_sessions.add(st.session_state.analysis_session_id)
        except Exception as e:
            logger.debug(f"Could not check UI session state: {e}")
            
        return preserved_sessions
    
    def _cleanup_active_sessions(self, session_ids: List[str], preserved_sessions: set) -> None:
        """Cleanup active sessions with preservation logic."""
        cleanup_count = 0
        preserved_count = 0
        
        for session_id in session_ids:
            if session_id not in preserved_sessions:
                self.end_analysis(session_id)
                cleanup_count += 1
            else:
                preserved_count += 1
        
        if cleanup_count > 0 or preserved_count > 0:
            logger.info(f"Session cleanup: {cleanup_count} ended, {preserved_count} preserved")
    
    def _cleanup_session_directories_all(self):
        """Clean up all session directories."""
        cleanup_stats = {'directories': 0, 'files': 0, 'errors': 0}
        
        try:
            analysis_output_path = Path(Config.OUTPUT_DIR)
            if analysis_output_path.exists():
                # Process directories in batches
                session_dirs = [item for item in analysis_output_path.iterdir() 
                               if item.is_dir() and self._is_session_directory(item.name)]
                
                self._batch_cleanup_directories(session_dirs, cleanup_stats)
                
                # Clean up results file
                self._cleanup_results_file(analysis_output_path, cleanup_stats)
                
            self._log_directory_cleanup_stats(cleanup_stats, "all")
                        
        except Exception as e:
            logger.warning(f"Session directories cleanup error: {e}")
    
    def _cleanup_session_directories_selective(self, preserved_sessions: set):
        """Clean up session directories except for preserved ones."""
        cleanup_stats = {'directories': 0, 'files': 0, 'errors': 0, 'preserved': 0}
        
        try:
            analysis_output_path = Path(Config.OUTPUT_DIR)
            if analysis_output_path.exists():
                for item in analysis_output_path.iterdir():
                    if item.is_dir() and self._is_session_directory(item.name):
                        if item.name not in preserved_sessions:
                            if self._safe_remove_directory(item):
                                cleanup_stats['directories'] += 1
                            else:
                                cleanup_stats['errors'] += 1
                        else:
                            cleanup_stats['preserved'] += 1
                            
            self._log_directory_cleanup_stats(cleanup_stats, "selective")
                            
        except Exception as e:
            logger.warning(f"Selective session directories cleanup error: {e}")
    
    def _is_session_directory(self, dir_name: str) -> bool:
        """Check if directory name matches session patterns using pre-compiled regexes."""
        return (self._UUID_PATTERN.match(dir_name) or 
                self._SESSION_PATTERN.match(dir_name))
    
    def _batch_cleanup_directories(self, directories: List[Path], 
                                  cleanup_stats: Dict[str, int]) -> None:
        """Clean up directories."""
        for i in range(0, len(directories), self._CLEANUP_BATCH_SIZE):
            batch = directories[i:i + self._CLEANUP_BATCH_SIZE]
            
            for directory in batch:
                if self._safe_remove_directory(directory):
                    cleanup_stats['directories'] += 1
                else:
                    cleanup_stats['errors'] += 1
            
            # Small delay between batches
            if i + self._CLEANUP_BATCH_SIZE < len(directories):
                time.sleep(0.01)
    
    def _safe_remove_directory(self, directory_path: Path) -> bool:
        """Safely remove a directory."""
        try:
            # Check size before removal for large directory warning
            try:
                dir_size = sum(f.stat().st_size for f in directory_path.rglob('*') if f.is_file())
                if dir_size > 100 * 1024 * 1024:  # 100MB
                    logger.debug(f"Removing large directory ({dir_size / (1024*1024):.1f}MB): {directory_path}")
            except Exception:
                pass
            
            shutil.rmtree(str(directory_path))
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup directory {directory_path}: {e}")
            return False
    
    def _cleanup_results_file(self, analysis_output_path: Path, 
                             cleanup_stats: Dict[str, int]) -> None:
        """Clean up analysis results file."""
        analysis_results_file = analysis_output_path / "results.json"
        if analysis_results_file.exists():
            try:
                analysis_results_file.unlink()
                cleanup_stats['files'] += 1
            except Exception as e:
                cleanup_stats['errors'] += 1
                logger.warning(f"Failed to cleanup analysis_output/results.json: {e}")
    
    def _log_directory_cleanup_stats(self, cleanup_stats: Dict[str, int], 
                                    cleanup_type: str) -> None:
        """Log directory cleanup statistics."""
        if any(cleanup_stats.get(key, 0) > 0 for key in ['directories', 'files', 'errors', 'preserved']):
            stats_msg = f"{cleanup_type.capitalize()} cleanup: {cleanup_stats.get('directories', 0)} directories"
            if cleanup_stats.get('files', 0) > 0:
                stats_msg += f", {cleanup_stats['files']} files"
            if cleanup_stats.get('preserved', 0) > 0:
                stats_msg += f", {cleanup_stats['preserved']} preserved"
            if cleanup_stats.get('errors', 0) > 0:
                stats_msg += f", {cleanup_stats['errors']} errors"
            logger.info(stats_msg)
    
    def _log_cleanup_statistics(self) -> None:
        """Log final cleanup statistics."""
        with self._lock:
            if any(self._cleanup_stats.get(key, 0) > 0 for key in self._cleanup_stats):
                logger.info(f"Total cleanup statistics - Files: {self._cleanup_stats['files_cleaned']}, "
                           f"Directories: {self._cleanup_stats['directories_cleaned']}, "
                           f"Errors: {self._cleanup_stats['errors']}")
    
    def get_cleanup_statistics(self) -> Dict[str, int]:
        """Get current cleanup statistics for monitoring."""
        with self._lock:
            return self._cleanup_stats.copy()
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics for monitoring."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            
            with self._lock:
                active_sessions = len(self._active_analyses)
            
            return {
                'total_memory_gb': memory_info.total / (1024**3),
                'available_memory_gb': memory_info.available / (1024**3),
                'used_memory_percent': memory_info.percent,
                'active_sessions': active_sessions,
                'max_concurrent_sessions': Config.MAX_CONCURRENT_ANALYSES,
                'cleanup_stats': self._cleanup_stats.copy(),
                'memory_per_session_limit_gb': Config.get_memory_limits().get('per_session_gb', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'error': str(e)}


_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    return _resource_manager


class TextMatcher:
    """
    Text matching with OCR error correction and fuzzy matching.
    """
    # Confidence thresholds for different matching strategies
    SIMILARITY_THRESHOLDS = {
        'word_match': 0.75,
        'partial_match': 0.75,
        'character_match': 0.6,
        'character_strict': 0.8
    }
    
    # OCR error corrections
    OCR_CORRECTIONS: Dict[str, str] = {
        '2s flash': '2.5 flash', '2.s flash': '2.5 flash', '2,5 flash': '2.5 flash',
        '25 flash': '2.5 flash',
        'fiash': 'flash', 'flasb': 'flash', 'fash': 'flash',
        'flashy': 'flash', 'flast': 'flash', 'flach': 'flash',
        'evai mode': 'eval mode', 'eval rode': 'eval mode',
        'native audio output': 'native audio output',
        'rn': 'm', 'vv': 'w', '1': 'l'
    }

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate text similarity using normalized Levenshtein distance.
        """
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
        """
        Apply common OCR error corrections to improve recognition accuracy.
        """
        if not text:
            return text
        corrected = text.lower()
        for incorrect, correct in cls.OCR_CORRECTIONS.items():
            corrected = corrected.replace(incorrect, correct)
        return corrected

    @classmethod
    def match_text(cls, detected: str, expected: str, enable_fuzzy: bool = True) -> Tuple[bool, str]:
        """
        Precise text matching that requires more exact matches to avoid false positives.
        Returns (match_found, match_method).
        """
        if not detected or not expected:
            return False, 'empty_input'
        detected_lower = detected.lower().strip()
        expected_lower = expected.lower().strip()
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
        if cls._all_words_match(expected_words, detected_words):
            return True, 'all_words_match'
        if len(expected_words) > 1:
            match_result = cls._precise_partial_word_match(expected_words, detected_words)
            if match_result[0]:
                return True, match_result[1]
        overall_similarity = cls.calculate_similarity(detected_lower, expected_lower)
        if overall_similarity >= cls.SIMILARITY_THRESHOLDS['character_strict']:
            return True, f'character_similarity_{overall_similarity:.2f}'
        return False, f'no_match_similarity_{overall_similarity:.2f}'

    @classmethod
    def _exact_phrase_match(cls, detected: str, expected: str) -> bool:
        import re
        expected_words = expected.split()
        if len(expected_words) == 1:
            pattern = r'\b' + re.escape(expected) + r'\b'
            return bool(re.search(pattern, detected))
        else:
            escaped_words = [re.escape(word) for word in expected_words]
            pattern = r'\b' + r'\s+'.join(escaped_words) + r'\b'
            return bool(re.search(pattern, detected))

    @classmethod
    def _all_words_match(cls, expected_words: List[str], detected_words: List[str]) -> bool:
        if not expected_words or not detected_words:
            return False
        if len(expected_words) <= 3 and len(detected_words) <= 5:
            expected_indices = []
            for exp_word in expected_words:
                found_index = -1
                for i, det_word in enumerate(detected_words):
                    similarity = cls.calculate_similarity(exp_word, det_word)
                    if (similarity >= 0.85 or
                        (len(exp_word) >= 3 and exp_word in det_word and len(det_word) <= len(exp_word) + 2)):
                        found_index = i
                        break
                if found_index == -1:
                    return False
                expected_indices.append(found_index)
            if len(expected_indices) >= 2:
                for i in range(len(expected_indices) - 1):
                    if expected_indices[i] >= expected_indices[i + 1]:
                        return False
            return True
        matched_count = 0
        for exp_word in expected_words:
            found_match = False
            for det_word in detected_words:
                similarity = cls.calculate_similarity(exp_word, det_word)
                if similarity >= 0.85:
                    found_match = True
                    break
                elif len(exp_word) >= 3 and exp_word in det_word and len(det_word) <= len(exp_word) + 2:
                    found_match = True
                    break
            if found_match:
                matched_count += 1
        return matched_count == len(expected_words)

    @classmethod
    def _precise_partial_word_match(cls, expected_words: List[str], detected_words: List[str]) -> Tuple[bool, str]:
        if len(expected_words) < 2:
            return False, 'single_word'
        if len(expected_words) <= 3 and len(detected_words) <= 5:
            expected_positions = []
            for exp_word in expected_words:
                found_position = -1
                for i, det_word in enumerate(detected_words):
                    similarity = cls.calculate_similarity(exp_word, det_word)
                    if (similarity >= 0.8 or
                        (len(exp_word) >= 3 and exp_word == det_word) or
                        (len(exp_word) >= 2 and abs(len(exp_word) - len(det_word)) <= 1 and similarity >= 0.7)):
                        found_position = i
                        break
                if found_position == -1:
                    return False, 'word_not_found'
                expected_positions.append(found_position)
            for i in range(len(expected_positions) - 1):
                if expected_positions[i] >= expected_positions[i + 1]:
                    return False, 'wrong_word_order'
        matched_words = 0
        total_similarity = 0.0
        for exp_word in expected_words:
            best_similarity = 0.0
            best_match_found = False
            for det_word in detected_words:
                similarity = cls.calculate_similarity(exp_word, det_word)
                if similarity > best_similarity:
                    best_similarity = similarity
                if (similarity >= 0.8 or
                    (len(exp_word) >= 3 and exp_word == det_word) or
                    (len(exp_word) >= 2 and abs(len(exp_word) - len(det_word)) <= 1 and similarity >= 0.7)):
                    best_match_found = True
                    break
            if best_match_found:
                matched_words += 1
                total_similarity += best_similarity
        match_ratio = matched_words / len(expected_words)
        avg_similarity = total_similarity / len(expected_words) if len(expected_words) > 0 else 0.0
        if match_ratio >= 0.8 and avg_similarity >= 0.75:
            return True, f'precise_partial_match_{match_ratio:.2f}_{avg_similarity:.2f}'
        return False, f'insufficient_partial_match_{match_ratio:.2f}_{avg_similarity:.2f}'


class FileManager:
    """
    Thread-safe file management utilities with security controls.
    
    This class provides secure file operations with:
    - Path validation and sanitization
    - Session-based isolation
    - Thread-safe file operations
    - Automatic cleanup management
    - Security validation at all levels
    
    All file operations are logged and validated to prevent security vulnerabilities.
    """
    
    # Class-level file locks for thread safety
    _file_locks: Dict[str, threading.Lock] = {}
    _lock_manager = threading.Lock()
    
    @classmethod
    def _get_file_lock(cls, file_path: str) -> threading.Lock:
        """
        Get or create file-specific lock for thread safety.
        
        Args:
            file_path: Path to the file needing synchronization
            
        Returns:
            Thread lock for the specific file
        """
        with cls._lock_manager:
            if file_path not in cls._file_locks:
                cls._file_locks[file_path] = threading.Lock()
            return cls._file_locks[file_path]
    
    @staticmethod
    def validate_and_ensure_directory(path: str, session_id: Optional[str] = None) -> Path:
        """
        Create directory with security validation.
        
        Args:
            path: Directory path to create
            session_id: Optional session ID for isolation
            
        Returns:
            Resolved Path object for the created directory
            
        Raises:
            ValueError: If path is invalid or unsafe
            RuntimeError: If directory creation fails
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        if not PathValidator.validate_file_path(path):
            raise ValueError(f"Invalid or unsafe path: {path}")
        
        try:
            # Add session isolation if provided
            if session_id:
                safe_session = InputValidator.sanitize_session_id(session_id)
                path = os.path.join(path, safe_session)
            
            directory = Path(path).resolve()
            
            # Validate directory is within allowed base paths
            allowed_bases = [
                Path(Config.OUTPUT_DIR).resolve(),
                Path(Config.TEMP_DIR).resolve(),
                Path(tempfile.gettempdir()).resolve()
            ]
            
            is_allowed = any(
                directory.is_relative_to(base) or directory == base 
                for base in allowed_bases
            )
            
            if not is_allowed:
                raise ValueError(f"Directory creation outside allowed paths: {directory}")
            
            # Create directory
            directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            
            return directory
            
        except (OSError, ValueError) as e:
            logger.error(f"Directory creation failed for {path}: {e}")
            raise RuntimeError(f"Cannot create directory: {e}") from e
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """
        Backward compatibility wrapper for directory creation.
        
        Args:
            path: Directory path to create
            
        Returns:
            Created directory path
        """
        return FileManager.validate_and_ensure_directory(path)
    
    @classmethod
    def cleanup_files(cls, file_paths: List[str], session_id: Optional[str] = None) -> Dict[str, int]:
        """
        Thread-safe cleanup of temporary files with reporting.
        
        Args:
            file_paths: List of file paths to clean up
            session_id: Optional session ID for additional validation
            
        Returns:
            Dictionary with cleanup statistics
        """
        if not file_paths:
            return {'removed': 0, 'errors': 0, 'skipped': 0}
        
        stats = {'removed': 0, 'errors': 0, 'skipped': 0}
        
        for path in file_paths:
            try:
                file_lock = cls._get_file_lock(path)
                with file_lock:
                    result = cls._safe_remove_file(path, session_id)
                    stats[result] += 1
                    
            except Exception as e:
                logger.error(f"Cleanup error for {path}: {e}")
                stats['errors'] += 1
        
        return stats
    
    @classmethod
    def _safe_remove_file(cls, path: str, session_id: Optional[str] = None) -> str:
        """
        Safely remove a single file.
        
        Args:
            path: File path to remove
            session_id: Optional session ID for validation
            
        Returns:
            Result status: 'removed', 'errors', or 'skipped'
        """
        try:
            if not os.path.exists(path):
                return 'skipped'
            
            # Security validation
            if not PathValidator.validate_file_path(path):
                logger.warning(f"Unsafe file path during cleanup: {path}")
                return 'errors'
            
            # Session validation if provided
            if session_id:
                safe_session = InputValidator.sanitize_session_id(session_id)
                if safe_session not in path:
                    logger.warning(
                        f"Attempting to remove file outside session {session_id}: {path}"
                    )
                    return 'errors'
            
            # Perform the removal
            os.unlink(path)
            return 'removed'
            
        except OSError as e:
            if e.errno == 2:  # File not found
                return 'skipped'
            logger.warning(f"OS error during cleanup of {path}: {e}")
            return 'errors'
            
        except Exception as e:
            logger.error(f"Unexpected cleanup error for {path}: {e}")
            return 'errors'
    
    @classmethod
    def save_frame(cls, frame: np.ndarray, filepath: str, session_id: str = None) -> bool:
        """Thread-safe frame saving with security validation."""
        try:
            if not PathValidator.validate_file_path(filepath):
                logger.error(f"Invalid file path for frame save: {filepath}")
                return False
            
            cls.validate_and_ensure_directory(os.path.dirname(filepath))
            
            file_lock = cls._get_file_lock(filepath)
            with file_lock:
                if frame is None or frame.size == 0:
                    logger.error(f"Invalid frame data for {filepath}")
                    return False
                
                success = cv2.imwrite(filepath, frame)
                if success:
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        return True
                    else:
                        logger.error(f"Frame save verification failed: {filepath}")
                        return False
                else:
                    logger.error(f"cv2.imwrite failed for {filepath}")
                    return False
                    
        except Exception as e:
            logger.error(f"Save frame failed {filepath}: {e}")
            return False
    
    @staticmethod
    def create_temp_filename(prefix: str, suffix: str = ".wav", session_id: str = None) -> str:
        """Create secure temporary filename with session isolation."""
        safe_prefix = InputValidator.sanitize_user_input(prefix, 50)
        safe_prefix = InputValidator.sanitize_session_id(safe_prefix, 50)
        
        if session_id:
            safe_session = InputValidator.sanitize_session_id(session_id)
            safe_prefix = f"{safe_session}_{safe_prefix}"
        
        timestamp = datetime.now().timestamp()
        random_part = str(uuid.uuid4())[:8]
        
        return f"{safe_prefix}_{timestamp}_{random_part}{suffix}"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem compatibility with security."""
        if not filename:
            return "unknown_file"
        
        sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
        
        sanitized = re.sub(r'\.\.+', '.', sanitized)
        
        sanitized = re.sub(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)', 'file_', sanitized, flags=re.IGNORECASE)
        
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 and ord(char) < 127)
        
        sanitized = sanitized.strip('. ')
        
        if len(sanitized) > Config.MAX_FILENAME_LENGTH:
            name, ext = os.path.splitext(sanitized)
            max_name_len = Config.MAX_FILENAME_LENGTH - len(ext) - 10
            sanitized = name[:max_name_len] + "_truncated" + ext
        
        if not sanitized or sanitized.startswith('.') or sanitized in ['', '.', '..']:
            sanitized = "sanitized_file"
        
        return sanitized
    
    @classmethod
    def create_session_temp_dir(cls, session_id: str) -> str:
        """Create isolated temporary directory for session."""
        try:
            base_temp = Path(Config.TEMP_DIR)
            session_temp = cls.validate_and_ensure_directory(str(base_temp), session_id)
            return str(session_temp)
        except Exception as e:
            logger.error(f"Failed to create session temp dir: {e}")
            return SessionManager.create_secure_temp_dir(session_id)


class AudioAnalyzer:
    """Audio processing for voice and language detection with voice separation capabilities."""
    
    # Class-level shared Whisper model to prevent concurrent loading issues
    _shared_whisper_model = None
    _whisper_model_lock = threading.RLock()
    _model_loading_condition = threading.Condition(_whisper_model_lock)
    _model_loading = False
    
    def __init__(self):
        self.whisper_model = None
        self.speech_recognizer = sr.Recognizer()
        self.supported_languages = Config.SUPPORTED_LANGUAGES.copy()
        self.voice_features_cache = {}
        # Performance optimization: cache loaded audio data
        self._audio_cache = {}
        
        # Always use minimal cache size for optimal performance
        self._audio_cache_max_size = 1  # Minimal cache for optimal performance
        logger.info("Using minimal audio cache for optimal performance")
    
    def load_whisper_model(self, model_size: str = "tiny") -> bool:
        """Load Whisper transcription model with thread-safe shared loading."""
        try:
            # Always use tiny model for optimal performance
            model_size = "tiny"  # Always use fastest model for optimal performance
            logger.info("Using 'tiny' Whisper model for optimal performance")
            
            # Use shared model to prevent concurrent loading issues
            with AudioAnalyzer._model_loading_condition:
                # If shared model is already loaded, use it
                if AudioAnalyzer._shared_whisper_model is not None:
                    self.whisper_model = AudioAnalyzer._shared_whisper_model
                    logger.info(f"Using existing shared Whisper model '{model_size}'")
                    return True
                
                # If another thread is loading, wait for it to complete
                while AudioAnalyzer._model_loading:
                    logger.info("Whisper model loading in progress, waiting...")
                    AudioAnalyzer._model_loading_condition.wait()
                    
                    # Check if model was loaded by other thread
                    if AudioAnalyzer._shared_whisper_model is not None:
                        self.whisper_model = AudioAnalyzer._shared_whisper_model
                        logger.info("Using Whisper model loaded by another thread")
                        return True
                
                # Load the model (only one thread will do this)
                AudioAnalyzer._model_loading = True
                try:
                    logger.info(f"Loading Whisper model '{model_size}'...")
                    AudioAnalyzer._shared_whisper_model = whisper.load_model(model_size)
                    self.whisper_model = AudioAnalyzer._shared_whisper_model
                    logger.info(f"Whisper model '{model_size}' loaded successfully")
                    return True
                finally:
                    AudioAnalyzer._model_loading = False
                    # Notify all waiting threads that loading is complete
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
    
    def detect_voice_audibility(self, audio_path: str, 
                               frame_duration: float = 2.0) -> Dict[str, Any]:
        """
        Detect if both user and model voices are audible in the audio.
        Uses simple acoustic features to determine if multiple distinct voices are present.
        """
        try:
            logger.info(f"Analyzing voice audibility: {audio_path}")
            
            # Optimization: Use cached audio data if available
            y, sr = self._get_cached_audio(audio_path)
            duration = len(y) / sr
            
            voice_segments = self.detect_voice_activity(audio_path, frame_duration)
            voice_only_segments = [(start, end) for start, end, has_voice in voice_segments if has_voice]
            
            if len(voice_only_segments) < 2:
                logger.info("Insufficient voice segments for audibility analysis")
                return {
                    'both_voices_audible': False,
                    'voice_segments_count': len(voice_only_segments),
                    'total_voice_duration': sum(end - start for start, end in voice_only_segments),
                    'confidence': 0.0,
                    'analysis_details': 'Insufficient voice segments detected'
                }
            
            segment_features = []
            total_voice_duration = 0
            
            for start_time, end_time in voice_only_segments:
                duration_seg = end_time - start_time
                if duration_seg < 0.5:
                    continue
                    
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = y[start_sample:end_sample]
                
                if len(segment) > 0:
                    features = self._extract_simple_voice_features(segment, sr)
                    if features:
                        segment_features.append(features)
                        total_voice_duration += duration_seg
            
            if len(segment_features) < 2:
                return {
                    'both_voices_audible': False,
                    'voice_segments_count': len(voice_only_segments),
                    'total_voice_duration': total_voice_duration,
                    'confidence': 0.0,
                    'analysis_details': 'Insufficient valid voice segments for analysis'
                }
            
            diversity_analysis = self._analyze_voice_diversity(segment_features)
            
            both_audible = self._determine_voice_audibility(diversity_analysis, total_voice_duration)
            
            result = {
                'both_voices_audible': both_audible['audible'],
                'confidence': both_audible['confidence'],
                'voice_segments_count': len(voice_only_segments),
                'total_voice_duration': total_voice_duration,
                'voice_diversity_score': diversity_analysis['diversity_score'],
                'analysis_details': both_audible['details'],
                'feature_analysis': diversity_analysis
            }
            
            logger.info(f"Voice audibility analysis complete: {both_audible['audible']} (confidence: {both_audible['confidence']:.1%})")
            return result
            
        except Exception as e:
            logger.error(f"Voice audibility detection failed: {e}")
            return {
                'error': str(e),
                'both_voices_audible': False,
                'confidence': 0.0
            }
    
    def _extract_simple_voice_features(self, audio_segment: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract simplified voice features focused on distinguishing different speakers."""
        try:
            features = {}
            
            # Optimization: Skip expensive pitch detection for very short segments
            if len(audio_segment) < sr * 0.1:  # Less than 100ms
                return self._extract_basic_features_only(audio_segment, sr)
            
            # Optimization: Use faster pitch estimation with reduced precision
            try:
                # Reduce frame length for faster processing
                frame_length = min(2048, len(audio_segment) // 4)
                f0 = librosa.yin(audio_segment, fmin=50, fmax=400, sr=sr, frame_length=frame_length)
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    features['pitch_mean'] = np.mean(f0_valid)
                    features['pitch_std'] = np.std(f0_valid)
                else:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
            except:
                # Fallback: Use basic pitch estimation
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
            
            # Optimization: Use more efficient hop length
            hop_length = max(512, len(audio_segment) // 100)
            
            rms = librosa.feature.rms(y=audio_segment, hop_length=hop_length)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=sr, hop_length=hop_length)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            zcr = librosa.feature.zero_crossing_rate(audio_segment, hop_length=hop_length)[0]
            features['zcr_mean'] = np.mean(zcr)
            
            # Optimization: Reduce MFCC computation and use fewer coefficients
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=3, hop_length=hop_length)
            for i in range(3):  # Reduced from 5 to 3
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            
            return features
            
        except Exception as e:
            logger.warning(f"Simple feature extraction failed: {e}")
            return {}

    def _extract_basic_features_only(self, audio_segment: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract only basic features for very short audio segments."""
        try:
            features = {}
            
            # Basic energy features only for short segments
            rms = np.sqrt(np.mean(audio_segment**2))
            features['energy_mean'] = rms
            features['energy_std'] = 0.0
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment)[0])
            features['zcr_mean'] = zcr
            
            # Fill in defaults for missing features
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['spectral_centroid_mean'] = 0.0
            for i in range(3):
                features[f'mfcc_{i}_mean'] = 0.0
            
            return features
        except:
            return {}
    
    def _analyze_voice_diversity(self, segment_features: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze diversity in voice features to determine if multiple speakers are present."""
        try:
            if len(segment_features) < 2:
                return {'diversity_score': 0.0, 'details': 'Insufficient segments'}
            
            feature_names = list(segment_features[0].keys())
            feature_variations = {}
            
            for feature_name in feature_names:
                values = [seg.get(feature_name, 0.0) for seg in segment_features]
                values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = std_val / (abs(mean_val) + 1e-6)
                    feature_variations[feature_name] = cv
                else:
                    feature_variations[feature_name] = 0.0
            
            key_features = ['pitch_mean', 'spectral_centroid_mean', 'mfcc_0_mean', 'mfcc_1_mean']
            key_variations = [feature_variations.get(f, 0.0) for f in key_features if f in feature_variations]
            
            if key_variations:
                diversity_score = np.mean(key_variations)
            else:
                diversity_score = 0.0
            
            return {
                'diversity_score': diversity_score,
                'feature_variations': feature_variations,
                'key_feature_variations': dict(zip(key_features, key_variations)) if key_variations else {},
                'num_segments_analyzed': len(segment_features)
            }
            
        except Exception as e:
            logger.warning(f"Voice diversity analysis failed: {e}")
            return {'diversity_score': 0.0, 'details': f'Analysis error: {str(e)}'}

    def _get_cached_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Get cached audio data or load and cache it."""
        # Create session-aware cache key to prevent conflicts between concurrent sessions
        cache_key = f"{os.path.abspath(audio_path)}_{os.path.getsize(audio_path)}_{os.path.getmtime(audio_path)}"
        
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        # Load audio with optimizations
        try:
            # Load with lower sample rate for faster processing if file is large
            file_size = os.path.getsize(audio_path)
            target_sr = None if file_size < 10 * 1024 * 1024 else 16000  # 10MB threshold
            
            y, sr = librosa.load(audio_path, sr=target_sr)
            
            # Cache management: remove oldest if cache is full
            if len(self._audio_cache) >= self._audio_cache_max_size:
                oldest_key = next(iter(self._audio_cache))
                del self._audio_cache[oldest_key]
            
            self._audio_cache[cache_key] = (y, sr)
            return y, sr
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise

    def _clear_audio_cache(self):
        """Clear audio cache to free memory."""
        self._audio_cache.clear()

    def _determine_voice_audibility(self, diversity_analysis: Dict[str, Any], 
                                   total_voice_duration: float) -> Dict[str, Any]:
        """Determine if both user and model voices are audible based on diversity analysis."""
        try:
            diversity_score = diversity_analysis.get('diversity_score', 0.0)
            
            min_diversity_threshold = 0.12
            good_diversity_threshold = 0.20
            min_duration_threshold = 3.0
            
            sufficient_duration = total_voice_duration >= min_duration_threshold
            
            has_diversity = diversity_score >= min_diversity_threshold
            good_diversity = diversity_score >= good_diversity_threshold
            
            if sufficient_duration and good_diversity:
                audible = True
                confidence = min(0.95, 0.75 + (diversity_score - good_diversity_threshold) * 2)
                details = f"Both voices audible - good diversity ({diversity_score:.3f}) and sufficient duration ({total_voice_duration:.1f}s)"
            elif sufficient_duration and has_diversity:
                audible = True
                confidence = min(0.85, 0.60 + (diversity_score - min_diversity_threshold) * 2.5)
                details = f"Both voices likely audible - moderate diversity ({diversity_score:.3f}) and sufficient duration ({total_voice_duration:.1f}s)"
            elif has_diversity and total_voice_duration >= min_duration_threshold * 0.6:
                audible = True
                confidence = min(0.75, 0.45 + (diversity_score - min_diversity_threshold) * 2)
                details = f"Both voices possibly audible - has diversity ({diversity_score:.3f}) but limited duration ({total_voice_duration:.1f}s)"
            else:
                audible = False
                if not sufficient_duration:
                    confidence = 0.15
                    details = f"Insufficient voice duration ({total_voice_duration:.1f}s < {min_duration_threshold}s)"
                else:
                    confidence = 0.15
                    details = f"Low voice diversity ({diversity_score:.3f} < {min_diversity_threshold})"
            
            return {
                'audible': audible,
                'confidence': max(0.0, min(1.0, confidence)),
                'details': details,
                'diversity_score': diversity_score,
                'total_duration': total_voice_duration,
                'thresholds': {
                    'min_diversity': min_diversity_threshold,
                    'good_diversity': good_diversity_threshold,
                    'min_duration': min_duration_threshold
                }
            }
            
        except Exception as e:
            logger.warning(f"Voice audibility determination failed: {e}")
            return {
                'audible': False,
                'confidence': 0.0,
                'details': f'Analysis error: {str(e)}'
            }
    
    def detect_voice_activity(self, audio_path: str, 
                            frame_duration: float = 1.0) -> List[Tuple[float, float, bool]]:
        """Detect voice activity using energy analysis."""
        try:
            return self._analyze_voice_segments(audio_path, frame_duration)
        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            return []
    
    def _analyze_voice_segments(self, audio_path: str, 
                               frame_duration: float) -> List[Tuple[float, float, bool]]:
        """Analyze audio for voice segments."""
        logger.info(f"Analyzing voice activity: {audio_path}")
        # Optimization: Use cached audio data
        y, sr = self._get_cached_audio(audio_path)
        logger.info(f"Audio loaded: {len(y)/sr:.2f}s @ {sr}Hz")
        
        frame_length = int(frame_duration * sr)
        hop_length = frame_length // 4
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        base_threshold = np.percentile(rms, 15)
        max_rms = np.max(rms)
        relative_threshold = max_rms * 0.1
        threshold = max(base_threshold, relative_threshold)
        logger.info(f"Voice threshold: {threshold:.6f}")
        
        voice_segments = []
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        for time, energy in zip(times, rms):
            end_time = min(time + frame_duration, len(y)/sr)
            has_voice = energy > threshold
            voice_segments.append((time, end_time, has_voice))
        
        voice_count = sum(1 for _, _, has_voice in voice_segments if has_voice)
        logger.info(f"Voice activity: {voice_count}/{len(voice_segments)} segments")
        
        return voice_segments
    
    def _locale_to_whisper_language(self, locale_code: str) -> str:
        """Convert locale code (e.g., 'en-US') to Whisper language code (e.g., 'en')."""
        return Config.locale_to_whisper_language(locale_code)
    
    def analyze_language_fluency(self, audio_path: str, 
                               target_language: str = 'en-US') -> Optional[Dict[str, Any]]:
        """Analyze language and fluency using Whisper."""
        if self.whisper_model is None:
            return {'detected_language': 'unknown', 'is_fluent': False, 'confidence': 0.0}
        
        try:
            # Convert locale code to Whisper language code
            whisper_language = self._locale_to_whisper_language(target_language)
            
            # Optimization: Use faster transcription options
            result = self.whisper_model.transcribe(
                audio_path, 
                task="transcribe", 
                fp16=False,
                condition_on_previous_text=False,  # Faster processing
                temperature=0.0  # Deterministic, faster results
            )
            detected_language = result.get('language', 'unknown')
            transcription = result.get('text', '').strip()
            
            words = transcription.split()
            if len(words) < 3:
                return None
            
            is_target_language = detected_language == whisper_language
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            fluency_indicators = {
                'word_count': len(words),
                'avg_word_length': avg_word_length,
                'language_match': is_target_language,
                'has_transcription': len(transcription) > 0
            }
            
            fluency_score = 1.0 if (is_target_language and avg_word_length > 4 
                                  and len(transcription) > 0) else 0.0
            is_fluent = fluency_score > 0.6 and is_target_language
            
            return {
                'detected_language': detected_language,
                'target_language': target_language,
                'whisper_language': whisper_language,
                'is_fluent': is_fluent,
                'fluency_score': fluency_score,
                'confidence': result.get('avg_logprob', fluency_score),
                'transcription': transcription,
                'fluency_indicators': fluency_indicators
            }
            
        except Exception as e:
            logger.error(f"Language analysis failed: {e}")
            return {
                'detected_language': 'unknown',
                'is_fluent': False,
                'confidence': 0.0,
                'error': str(e)
            }


class VideoContentAnalyzer:
    """
    Thread-safe video content analysis system with modular helpers for frame, audio, and memory management.
    """
    def __init__(self, session_id: Optional[str] = None) -> None:
        import gc
        gc.collect()
        
        # Ensure unique session ID for proper isolation
        if session_id:
            # Validate that the provided session ID is not already active to prevent conflicts
            with ResourceManager._global_lock:
                if session_id in ResourceManager._active_sessions_global:
                    logger.warning(f"Session ID {session_id} already active, generating new ID")
                    session_id = SessionManager.generate_session_id()
        
        self.session_id = session_id or SessionManager.generate_session_id()
        self.rules: List[DetectionRule] = []
        self.results: List[DetectionResult] = []
        self.temp_files: List[str] = []
        self.screenshot_files: List[str] = []
        self._lock = threading.RLock()
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._resource_manager = get_resource_manager()
        self._memory_monitor: Optional[threading.Thread] = None
        self._memory_limit_exceeded = False
        self.analysis_start_time: Optional[float] = None
        self.total_frames_processed: int = 0
        self.video_duration: float = 0.0
        self._processing_stats: Dict[str, Any] = {
            'frames_processed': 0,
            'detection_time': 0.0,
            'ocr_time': 0.0,
            'audio_time': 0.0
        }
        try:
            self.output_dir = FileManager.validate_and_ensure_directory(
                Config.OUTPUT_DIR, self.session_id
            )
            self.temp_dir = FileManager.create_session_temp_dir(self.session_id)
        except Exception as e:
            logger.error(f"Failed to create session directories: {e}")
            raise RuntimeError(f"Cannot initialize session directories: {e}") from e
        try:
            self.audio_analyzer = self._initialize_audio_analyzer()
        except Exception as e:
            logger.warning(f"Audio analyzer initialization failed: {e}")
            self.audio_analyzer = None
        logger.info(f"VideoContentAnalyzer initialized for session: {self.session_id}")

    def _monitor_memory_usage(self) -> None:
        """
        Monitor memory usage during analysis with detailed reporting.
        
        This method runs in a separate thread and monitors:
        - Process memory usage
        - System memory availability
        - Memory growth patterns
        - Automatic cleanup triggers
        """
        memory_samples = []
        sample_interval = 5  # seconds
        
        while self._is_processing and not self._memory_limit_exceeded:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                # Record memory sample
                sample = {
                    'timestamp': time.time(),
                    'process_rss': memory_info.rss,
                    'process_vms': memory_info.vms,
                    'system_available': system_memory.available,
                    'system_percent': system_memory.percent
                }
                memory_samples.append(sample)
                
                # Check process memory limit using dynamic limits
                memory_limits = Config.get_memory_limits()
                per_session_gb = memory_limits["per_session_gb"]
                max_process_memory = (per_session_gb * 1.5) * 1024 * 1024 * 1024
                
                if memory_info.rss > max_process_memory:
                    self._memory_limit_exceeded = True
                    logger.warning(
                        f"Process memory limit exceeded: "
                        f"{memory_info.rss / 1024 / 1024:.1f}MB "
                        f"(limit: {per_session_gb * 1.5:.1f}GB)"
                    )
                    break
                
                # Check system memory pressure
                if system_memory.percent > 95:
                    logger.warning(
                        f"High system memory usage: {system_memory.percent:.1f}%"
                    )
                
                # Log periodic statistics
                if len(memory_samples) % 12 == 0:
                    self._log_memory_statistics(memory_samples[-12:])
                
                time.sleep(sample_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(sample_interval)
    
    def _log_memory_statistics(self, samples: List[Dict[str, Any]]) -> None:
        """
        Log memory usage statistics from recent samples.
        
        Args:
            samples: Recent memory usage samples
        """
        if not samples:
            return
        
        try:
            avg_process_mb = sum(s['process_rss'] for s in samples) / len(samples) / 1024 / 1024
            max_process_mb = max(s['process_rss'] for s in samples) / 1024 / 1024
            avg_system_percent = sum(s['system_percent'] for s in samples) / len(samples)
            
            logger.debug(
                f"Memory stats - Process: avg={avg_process_mb:.1f}MB, "
                f"max={max_process_mb:.1f}MB, System: {avg_system_percent:.1f}%"
            )
        except Exception as e:
            logger.debug(f"Failed to compute memory statistics: {e}")
    
    def _initialize_audio_analyzer(self) -> Optional[AudioAnalyzer]:
        """Initialize audio analyzer."""
        try:
            analyzer = AudioAnalyzer()
            return analyzer
        except Exception as e:
            logger.warning(f"Audio analyzer failed: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry with resource registration."""
        video_duration = self.video_duration if hasattr(self, 'video_duration') else 0.0
        if not self._resource_manager.start_analysis(self.session_id, self, video_duration):
            raise RuntimeError("Cannot start analysis - resource limits exceeded")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with guaranteed cleanup."""
        try:
            self._cleanup_temp_files_only()
        finally:
            pass
    
    def _cleanup_temp_files_only(self):
        """Clean up only temporary files, preserve screenshots for UI display."""
        with self._lock:
            try:
                self._is_processing = False
                
                if self.temp_files:
                    FileManager.cleanup_files(self.temp_files, self.session_id)
                    self.temp_files.clear()
                
                if self.screenshot_files:
                    pass
                
            except Exception as e:
                logger.error(f"Temp files cleanup error for session {self.session_id}: {e}")
    
    def cleanup(self) -> None:
        """Thread-safe cleanup of resources."""
        with self._lock:
            try:
                self._is_processing = False
                
                if self.temp_files:
                    FileManager.cleanup_files(self.temp_files, self.session_id)
                    self.temp_files.clear()
                
                if self.screenshot_files:
                    pass
                
                self._cleanup_session_directory()
                
                try:
                    analysis_results_path = "analysis_output/results.json"
                    if os.path.exists(analysis_results_path):
                        os.unlink(analysis_results_path)
                except Exception as e:
                    logger.warning(f"Failed to auto-clean analysis results file: {e}")
                
            except Exception as e:
                logger.error(f"Cleanup error for session {self.session_id}: {e}")
     
    def cleanup_screenshots(self, preserve_for_export=False):
        """Clean up screenshot files when session truly ends."""
        try:
            if self.screenshot_files:
                if preserve_for_export:
                    return
                FileManager.cleanup_files(self.screenshot_files, self.session_id)
                self.screenshot_files.clear()
        except Exception as e:
            logger.error(f"Error cleaning up screenshots: {e}")

    def add_rule(self, rule: DetectionRule) -> None:
        """Thread-safe rule addition."""
        with self._lock:
            self._validate_rule_input(rule)
            
            if self._is_processing:
                raise RuntimeError("Cannot modify rules during active analysis")
            
            self._replace_existing_rule(rule)
            self.rules.append(rule)
    
    def _validate_rule_input(self, rule: DetectionRule) -> None:
        """Rule validation."""
        if not isinstance(rule, DetectionRule):
            raise ValueError("Rule must be DetectionRule instance")
        
        if not rule.name or len(rule.name.strip()) == 0:
            raise ValueError("Rule name cannot be empty")
        
        # Validate that the name is sanitized
        sanitized_name = InputValidator.sanitize_user_input(rule.name, 100)
        if rule.name != sanitized_name:
            raise ValueError(f"Rule name contains unsafe characters. Use: {sanitized_name}")
        
        if rule.parameters:
            self._validate_rule_parameters(rule.parameters)
    
    def _validate_rule_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate rule parameters for security."""
        dangerous_keys = ['__', 'eval', 'exec', 'import', 'open', 'file']
        
        for key, value in parameters.items():
            if any(dangerous in key.lower() for dangerous in dangerous_keys):
                raise ValueError(f"Potentially dangerous parameter key: {key}")
            
            if isinstance(value, str):
                parameters[key] = InputValidator.sanitize_user_input(value)
            
            elif 'path' in key.lower() and isinstance(value, str):
                if not PathValidator.validate_file_path(value):
                    raise ValueError(f"Invalid file path in parameter {key}: {value}")
    
    def _replace_existing_rule(self, rule: DetectionRule) -> None:
        """Replace existing rule with same name."""
        existing_rule = next((r for r in self.rules if r.name == rule.name), None)
        if existing_rule:
            self.rules.remove(existing_rule)
    
    def get_active_rules(self) -> List[DetectionRule]:
        """Thread-safe access to enabled rules."""
        with self._lock:
            return [rule for rule in self.rules if rule.enabled]
    
    def get_rules_by_type(self, detection_type: DetectionType) -> List[DetectionRule]:
        """Get rules by detection type with thread safety."""
        return [rule for rule in self.get_active_rules() if rule.detection_type == detection_type]
    
    def analyze_video(self, video_path: str, 
                     frame_interval: float = Config.DEFAULT_FRAME_INTERVAL) -> List[DetectionResult]:
        """Thread-safe video analysis."""
        
        if not self._processing_lock.acquire(blocking=False):
            raise RuntimeError("Analysis already in progress for this session")
        
        try:
            with self._lock:
                self._is_processing = True
                self._memory_limit_exceeded = False
            
            if self._memory_monitor is None or not self._memory_monitor.is_alive():
                self._memory_monitor = threading.Thread(
                    target=self._monitor_memory_usage, daemon=True
                )
                self._memory_monitor.start()
            
            self._validate_analysis_inputs(video_path, frame_interval)
            
            active_rules = self.get_active_rules()
            if not active_rules:
                logger.warning(f"No active rules configured (session: {self.session_id})")
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
            self._processing_lock.release()
    
    def _validate_analysis_inputs(self, video_path: str, frame_interval: float) -> None:
        """Input validation with security checks."""
        
        if not PathValidator.validate_file_path(video_path):
            raise ValueError(f"Invalid or unsafe video path: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if not PathValidator.validate_file_size(video_path):
            raise ValueError("Video file exceeds maximum allowed size")
        
        if not (Config.MIN_FRAME_INTERVAL <= frame_interval <= Config.MAX_FRAME_INTERVAL):
            raise ValueError(f"Frame interval must be {Config.MIN_FRAME_INTERVAL}-{Config.MAX_FRAME_INTERVAL}")
    
    def _initialize_analysis(self) -> None:
        """Initialize analysis state with thread safety."""
        with self._lock:
            self.results = []
            self.analysis_start_time = datetime.now().timestamp()
            self.total_frames_processed = 0
    
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
            frame_params = self._calculate_frame_parameters(
                context, frame_interval
            )
            
            if context['audio_path']:
                self._process_audio_analysis(
                    context['audio_path'], 
                    frame_params['start_time'], 
                    frame_params['end_time'], 
                    frame_interval
                )
            
            self._process_video_frames(context, frame_params, frame_interval)
            
        finally:
            if context.get('cap'):
                context['cap'].release()
            
            duration = datetime.now().timestamp() - (self.analysis_start_time or 0)
    
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
        """Process video frames for detection with optimizations."""
        cap = context['cap']
        fps = context['fps']
        
        # Optimization: Pre-filter visual rules once instead of per frame
        visual_rules = [rule for rule in self.get_active_rules() 
                       if rule.detection_type not in [DetectionType.AUDIO_LANGUAGE, 
                                                     DetectionType.VOICE_AUDIBILITY]]
        
        if not visual_rules:
            logger.info("No visual rules to process, skipping frame processing")
            return
        
        processed = 0
        total_frames = (frame_params['end_frame'] - frame_params['start_frame']) // frame_params['frame_step']
        logger.info(f"Processing {total_frames} frames with {len(visual_rules)} visual rules")
        
        for frame_idx in range(frame_params['start_frame'], 
                              frame_params['end_frame'], 
                              frame_params['frame_step']):
            if self._process_single_frame_optimized(cap, frame_idx, fps, visual_rules):
                processed += 1
                # Optimization: Less frequent logging
                if processed % 50 == 0:  # Reduced from 100 to 50 for better feedback
                    logger.info(f"Processed {processed}/{total_frames} frames ({processed/total_frames*100:.1f}%)")
        
        self.total_frames_processed = processed
        logger.info(f"Frame processing completed: {processed} frames")
    
    def _setup_audio_analysis(self, video_path: str) -> Optional[str]:
        """Setup audio extraction for audio rules."""
        audio_rules = [rule for rule in self.get_active_rules() 
                      if rule.detection_type in [DetectionType.AUDIO_LANGUAGE, 
                                               DetectionType.VOICE_AUDIBILITY]]
        
        if not audio_rules or not self.audio_analyzer:
            return None
        
        try:
            audio_filename = FileManager.create_temp_filename("temp_audio", session_id=self.session_id)
            audio_path = os.path.join(self.temp_dir, audio_filename)
            
            if not PathValidator.validate_file_path(audio_path):
                logger.error(f"Invalid audio path: {audio_path}")
                return None
            
            logger.info(f"Extracting audio: {os.path.basename(video_path)} -> {os.path.basename(audio_path)} (session: {self.session_id})")
            
            if self.audio_analyzer.extract_audio(video_path, audio_path):
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    self.temp_files.append(audio_path)
                    
                    try:
                        self.audio_analyzer.load_whisper_model()
                    except Exception as e:
                        logger.warning(f"Whisper model load failed: {e}")
                    
                    return audio_path
                else:
                    logger.warning("Audio extraction produced invalid file")
                    return None
            else:
                logger.warning("Audio extraction failed")
                return None
                
        except Exception as e:
            logger.error(f"Audio setup failed (session: {self.session_id}): {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        try:
            # Clear audio cache to free memory
            if hasattr(self, 'audio_analyzer') and self.audio_analyzer:
                self.audio_analyzer._clear_audio_cache()
            
            # Original cleanup
            super().cleanup() if hasattr(super(), 'cleanup') else None
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def _process_single_frame(self, cap: cv2.VideoCapture, frame_idx: int, fps: float) -> bool:
        """Process single frame with memory monitoring."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return False
            
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame at index {frame_idx}")
                return False
                
            timestamp = frame_idx / fps
            
            visual_rules = [rule for rule in self.get_active_rules() 
                           if rule.detection_type not in [DetectionType.AUDIO_LANGUAGE, 
                                                         DetectionType.VOICE_AUDIBILITY]]
            
            for rule in visual_rules:
                try:
                    result = self._apply_visual_rule(rule, frame, timestamp, frame_idx)
                    if result.detected:
                        logger.info(f"✓ Detection: {rule.name} at {timestamp:.2f}s")
                    
                    with self._lock:
                        self.results.append(result)
                        
                except Exception as e:
                    logger.error(f"Rule {rule.name} failed at frame {frame_idx} (session: {self.session_id}): {e}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Frame processing failed at {frame_idx} (session: {self.session_id}): {e}")
            return False

    def _process_single_frame_optimized(self, cap: cv2.VideoCapture, frame_idx: int, 
                                      fps: float, visual_rules: List[DetectionRule]) -> bool:
        """Optimized frame processing with pre-filtered rules."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                return False
            
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame at index {frame_idx}")
                return False
                
            timestamp = frame_idx / fps
            
            # Optimization: Use pre-filtered rules instead of filtering each time
            for rule in visual_rules:
                try:
                    result = self._apply_visual_rule(rule, frame, timestamp, frame_idx)
                    if result.detected:
                        logger.info(f"✓ Detection: {rule.name} at {timestamp:.2f}s")
                    
                    with self._lock:
                        self.results.append(result)
                        
                except Exception as e:
                    logger.error(f"Rule {rule.name} failed at frame {frame_idx} (session: {self.session_id}): {e}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Frame processing failed at {frame_idx} (session: {self.session_id}): {e}")
            return False
    
    def _save_frame(self, frame: np.ndarray, rule_name: str, timestamp: float) -> Optional[str]:
        """Save frame as screenshot with session isolation."""
        try:
            # Check if session is still active
            with ResourceManager._global_lock:
                if self.session_id not in ResourceManager._active_sessions_global and \
                   self.session_id not in ResourceManager._export_pending_sessions:
                    return None
            
            safe_rule_name = FileManager.sanitize_filename(rule_name)
            filename = f"{safe_rule_name}_{timestamp:.2f}s.png"
            filepath = os.path.join(self.output_dir, filename)
            
            if FileManager.save_frame(frame, filepath, self.session_id):
                with self._lock:
                    self.screenshot_files.append(filepath)
                return filepath
            return None
                
        except Exception as e:
            logger.error(f"Frame save failed for rule {rule_name} (session: {self.session_id}): {e}")
            return None
    
    def _apply_visual_rule(self, rule: DetectionRule, frame: np.ndarray,
                          timestamp: float, frame_number: int) -> DetectionResult:
        """Apply visual detection rule to frame."""
        start_time = datetime.now().timestamp()
        
        try:
            result = self._execute_rule_detection(rule, frame, timestamp, frame_number)
            result.processing_time = datetime.now().timestamp() - start_time
            return result
        except Exception as e:
            logger.error(f"Visual rule '{rule.name}' failed: {e}")
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=False, confidence=0.0, details={'error': str(e)}
            )
    
    def _execute_rule_detection(self, rule: DetectionRule, frame: np.ndarray,
                               timestamp: float, frame_number: int) -> DetectionResult:
        """Execute specific detection type."""        
        detection_methods = {
            DetectionType.TEXT: self._detect_text
        }
        
        if rule.detection_type in detection_methods:
            return detection_methods[rule.detection_type](rule, frame, timestamp, frame_number)
        else:
            logger.error(f"Detection type {rule.detection_type} not implemented")
            raise NotImplementedError(f"Detection type {rule.detection_type} not implemented")
    
    def _process_audio_analysis(self, audio_path: str, start_time: float, 
                              end_time: float, frame_interval: float) -> None:
        """Process audio for language detection and voice audibility analysis."""
        if not self.audio_analyzer:
            return
        
        try:
            audio_rules = [rule for rule in self.get_active_rules() 
                          if rule.detection_type in [DetectionType.AUDIO_LANGUAGE, 
                                                   DetectionType.VOICE_AUDIBILITY]]
            
            if not audio_rules:
                return
            
            # Optimization: Pre-load audio data to cache for reuse
            try:
                self.audio_analyzer._get_cached_audio(audio_path)
                logger.info("Audio data cached for reuse")
            except Exception as e:
                logger.warning(f"Audio caching failed: {e}")
            
            voice_audibility_rules = [r for r in audio_rules if r.detection_type == DetectionType.VOICE_AUDIBILITY]
            
            if voice_audibility_rules:
                voice_audibility_result = self.audio_analyzer.detect_voice_audibility(audio_path, frame_interval)
                
                for rule in voice_audibility_rules:
                    result = self._create_voice_audibility_result(rule, voice_audibility_result, start_time, 0)
                    if result:
                        self.results.append(result)
            
            language_rules = [r for r in audio_rules if r.detection_type == DetectionType.AUDIO_LANGUAGE]
            if language_rules:
                voice_segments = self.audio_analyzer.detect_voice_activity(audio_path, frame_interval)
                logger.info(f"Voice segments found: {len(voice_segments)}")
                self._process_voice_segments(voice_segments, audio_path, start_time)
                        
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
        finally:
            # Clean up audio cache after processing to free memory
            try:
                self.audio_analyzer._clear_audio_cache()
            except:
                pass
    
    def _create_voice_audibility_result(self, rule: DetectionRule, 
                                      audibility_result: Dict[str, Any],
                                      start_time: float, frame_number: int) -> Optional[DetectionResult]:
        """Create detection result for voice audibility analysis."""
        try:
            if 'error' in audibility_result:
                return DetectionResult(
                    rule_name=rule.name,
                    timestamp=start_time,
                    frame_number=frame_number,
                    detected=False,
                    confidence=0.0,
                    details={'error': audibility_result['error']}
                )
            
            both_audible = audibility_result.get('both_voices_audible', False)
            confidence = audibility_result.get('confidence', 0.0)
            
            params = rule.parameters
            min_confidence = params.get('min_confidence', 0.3)
            min_duration = params.get('min_duration', 3.0)
            
            actual_duration = audibility_result.get('total_voice_duration', 0.0)
            detected = (both_audible and 
                       confidence >= min_confidence and 
                       actual_duration >= min_duration)
            
            details = {
                'both_voices_audible': both_audible,
                'confidence': confidence,
                'voice_segments_count': audibility_result.get('voice_segments_count', 0),
                'total_voice_duration': actual_duration,
                'voice_diversity_score': audibility_result.get('voice_diversity_score', 0.0),
                'analysis_details': audibility_result.get('analysis_details', 'No details available'),
                'detection_criteria': {
                    'min_confidence_required': min_confidence,
                    'min_duration_required': min_duration,
                    'confidence_criteria_met': confidence >= min_confidence,
                    'duration_criteria_met': actual_duration >= min_duration,
                    'audibility_criteria_met': both_audible
                }
            }
            
            return DetectionResult(
                rule_name=rule.name,
                timestamp=start_time,
                frame_number=frame_number,
                detected=detected,
                confidence=confidence,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Voice audibility result creation failed: {e}")
            return None
    
    def _process_voice_segments(self, voice_segments: List[Tuple[float, float, bool]],
                               audio_path: str, start_time: float) -> None:
        """Process individual voice segments with batch optimization."""
        audio_rules = self.get_rules_by_type(DetectionType.AUDIO_LANGUAGE)
        
        if not audio_rules:
            return
            
        # Optimization: Group consecutive voice segments for batch processing
        voice_only_segments = [(ts, end_ts) for ts, end_ts, has_voice in voice_segments if has_voice]
        
        if len(voice_only_segments) > 5:
            # For many segments, use batch processing
            self._process_voice_segments_batch(voice_only_segments, audio_path, start_time, audio_rules)
        else:
            # For few segments, process individually
            for timestamp, end_timestamp, has_voice in voice_segments:
                if not has_voice:
                    continue
                
                adjusted_timestamp = timestamp + start_time
                
                for rule in audio_rules:
                    try:
                        result = self._apply_audio_rule(rule, audio_path, timestamp, 
                                                       end_timestamp, adjusted_timestamp)
                        if result is not None:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Audio rule {rule.name} failed: {e}")

    def _process_voice_segments_batch(self, voice_segments: List[Tuple[float, float]], 
                                    audio_path: str, start_time: float, 
                                    audio_rules: List[DetectionRule]) -> None:
        """Process voice segments in batches for better performance."""
        try:
            # Sample key segments instead of processing all
            total_segments = len(voice_segments)
            max_segments = 10  # Process at most 10 segments
            
            if total_segments <= max_segments:
                segments_to_process = voice_segments
            else:
                # Sample evenly across the timeline
                step = total_segments // max_segments
                segments_to_process = [voice_segments[i] for i in range(0, total_segments, step)][:max_segments]
            
            logger.info(f"Processing {len(segments_to_process)} of {total_segments} voice segments (batch mode)")
            
            for timestamp, end_timestamp in segments_to_process:
                adjusted_timestamp = timestamp + start_time
                
                for rule in audio_rules:
                    try:
                        result = self._apply_audio_rule(rule, audio_path, timestamp, 
                                                       end_timestamp, adjusted_timestamp)
                        if result is not None:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Audio rule {rule.name} failed: {e}")
                        
        except Exception as e:
            logger.error(f"Batch voice segment processing failed: {e}")
    
    def _apply_audio_rule(self, rule: DetectionRule, audio_path: str, 
                         segment_start: float, segment_end: float, 
                         timestamp: float) -> Optional[DetectionResult]:
        """Apply audio detection rule."""
        frame_number = int(timestamp)
        start_time = datetime.now().timestamp()
        
        try:
            result = None
            if rule.detection_type == DetectionType.AUDIO_LANGUAGE:
                result = self._detect_language(rule, audio_path, segment_start, 
                                           segment_end, timestamp, frame_number)
            else:
                raise NotImplementedError(f"Audio type {rule.detection_type} not supported")
            
            if result is not None:
                result.processing_time = datetime.now().timestamp() - start_time
            
            return result
                
        except Exception as e:
            logger.error(f"Audio rule '{rule.name}' failed: {e}")
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=False, confidence=0.0, details={'error': str(e)},
                processing_time=datetime.now().timestamp() - start_time
            )
    
    def _detect_language(self, rule: DetectionRule, audio_path: str, 
                        segment_start: float, segment_end: float, 
                        timestamp: float, frame_number: int) -> Optional[DetectionResult]:
        """Detect language fluency."""
        params = rule.parameters
        target_language = params.get('target_language', 'en-US')
        
        try:
            segment_path = self._create_audio_segment(audio_path, segment_start, 
                                                    segment_end, timestamp)
            
            fluency_result = self.audio_analyzer.analyze_language_fluency(segment_path, target_language)
            if fluency_result is None:
                return None
            
            return self._create_language_detection_result(rule, timestamp, frame_number,
                                                        fluency_result, target_language,
                                                        segment_end - segment_start)
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=False, confidence=0.0, details={'error': str(e)}
            )
    
    def _create_audio_segment(self, audio_path: str, segment_start: float,
                             segment_end: float, timestamp: float) -> str:
        """Create audio segment file with optimized caching."""
        from pydub import AudioSegment
        import tempfile
        
        # Optimization: Use memory-efficient temporary files with session isolation
        segment_filename = FileManager.sanitize_filename(f"audio_segment_{timestamp:.2f}_{os.getpid()}.wav")
        segment_path = os.path.join(self.temp_dir, segment_filename)
        
        try:
            # Optimization: Load audio data once and reuse from cache
            y, sr = self.audio_analyzer._get_cached_audio(audio_path)
            
            # Convert to segment directly from numpy array (faster than loading from file)
            start_sample = int(segment_start * sr)
            end_sample = int(segment_end * sr)
            segment_data = y[start_sample:end_sample]
            
            # Convert to AudioSegment more efficiently
            import io
            import wave
            
            # Create in-memory WAV data
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sr)
                wav_file.writeframes((segment_data * 32767).astype(np.int16).tobytes())
            
            buffer.seek(0)
            segment = AudioSegment.from_wav(buffer)
            segment.export(segment_path, format="wav")
            
        except Exception as e:
            # Fallback to original method if optimization fails
            logger.warning(f"Fast segment creation failed, using fallback: {e}")
            audio = AudioSegment.from_wav(audio_path)
            segment = audio[int(segment_start * 1000):int(segment_end * 1000)]
            segment.export(segment_path, format="wav")
        
        self.temp_files.append(segment_path)
        return segment_path
    
    def _create_language_detection_result(self, rule: DetectionRule, timestamp: float,
                                        frame_number: int, fluency_result: Dict[str, Any],
                                        target_language: str, 
                                        segment_duration: float) -> DetectionResult:
        """Create language detection result."""
        detected = fluency_result.get('is_fluent', False)
        confidence = fluency_result.get('fluency_score', 0.0)
        
        details = {
            'target_language': target_language,
            'detected_language': fluency_result.get('detected_language', 'unknown'),
            'is_fluent': detected,
            'fluency_score': confidence,
            'transcription': fluency_result.get('transcription', ''),
            'segment_duration': segment_duration,
            'fluency_indicators': fluency_result.get('fluency_indicators', {})
        }
        
        return DetectionResult(
            rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
            detected=detected, confidence=confidence, details=details
        )
    
    def _find_text_bounding_box(self, expected_text: str, boxes: dict) -> Optional[List[int]]:
        """Find bounding box for expected text using fuzzy matching."""
        expected_words = expected_text.lower().split()
        found_boxes = self._find_word_boxes(expected_words, boxes)
        
        if found_boxes:
            return self._calculate_combined_bounding_box(found_boxes)
        
        return None
    
    def _find_word_boxes(self, expected_words: List[str], boxes: dict) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes for expected words."""
        found_boxes = []
        
        for word in expected_words:
            best_match_box = self._find_best_word_match(word, boxes, found_boxes)
            if best_match_box:
                found_boxes.append(best_match_box)
        
        return found_boxes
    
    def _find_best_word_match(self, word: str, boxes: dict, 
                             existing_boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Find best matching box for a word."""
        best_match_box = None
        best_similarity = 0
        
        for i, box_word in enumerate(boxes['text']):
            box_word_clean = box_word.strip().lower()
            if not box_word_clean:
                continue
            
            box = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            
            if box_word_clean == word:
                return box
            
            similarity = TextMatcher.calculate_similarity(word, box_word_clean)
            if similarity > best_similarity and similarity >= 0.65 and box not in existing_boxes:
                best_similarity = similarity
                best_match_box = box
        
        return best_match_box
    
    def _calculate_combined_bounding_box(self, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """Calculate combined bounding box from multiple boxes."""
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[0] + box[2] for box in boxes)
        y2 = max(box[1] + box[3] for box in boxes)
        return [x1, y1, x2-x1, y2-y1]
    
    def _detect_text(self, rule: DetectionRule, frame: np.ndarray, 
                    timestamp: float, frame_number: int) -> DetectionResult:
        """OCR-based text detection."""
        params = rule.parameters
        
        try:
            roi_info = self._prepare_text_roi(frame, params)
            if roi_info['error']:
                logger.warning(f"ROI preparation failed: {roi_info['error']}")
                return DetectionResult(
                    rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                    detected=False, confidence=0.0, details=roi_info['error']
                )
            
            processed_image = self._preprocess_for_ocr(roi_info['roi'], params)
            
            ocr_results = self._perform_ocr(processed_image)
            
            detection_result = self._analyze_text_detection(ocr_results, params, roi_info['offset'])
            
            screenshot_path = None
            if params.get('save_screenshot', True) and detection_result['detected']:
                screenshot_path = self._create_text_screenshot(
                    frame, detection_result['text_bounding_box'], 
                    detection_result['expected_text'], rule.name, timestamp
                )
            
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=detection_result['detected'], confidence=detection_result['confidence'],
                details=detection_result['details'], screenshot_path=screenshot_path
            )
            
        except Exception as e:
            logger.error(f"OCR failed for rule {rule.name}: {e}")
            return DetectionResult(
                rule_name=rule.name, timestamp=timestamp, frame_number=frame_number,
                detected=False, confidence=0.0, details={'error': str(e)}
            )
    
    def _prepare_text_roi(self, frame: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare region of interest for text detection."""
        if 'region' not in params:
            return {'roi': frame, 'offset': (0, 0), 'error': None}
        
        x1, y1, x2, y2 = params['region']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {
                'roi': None, 'offset': None,
                'error': {'error': 'Invalid region bounds', 'roi_size': (0, 0)}
            }
        
        return {'roi': roi, 'offset': (x1, y1), 'error': None}
    
    def _preprocess_for_ocr(self, roi: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Preprocess image for OCR."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        preprocess_config = params.get('preprocess', {})
        
        if preprocess_config.get('denoise', Config.OCR_DENOISING_ENABLED):
            gray = cv2.fastNlMeansDenoising(gray)
        
        if preprocess_config.get('threshold', Config.OCR_THRESHOLD_ENABLED):
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return gray
    
    def _perform_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform OCR on preprocessed image."""
        text = pytesseract.image_to_string(image, config=Config.OCR_CONFIG).strip()
        boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, 
                                        config=Config.OCR_CONFIG)
        return {'text': text, 'boxes': boxes}
    
    def _analyze_text_detection(self, ocr_results: Dict[str, Any], params: Dict[str, Any],
                               roi_offset: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze OCR results for text detection."""
        text = ocr_results['text']
        boxes = ocr_results['boxes']
        
        expected_text = params.get('expected_text', '')
        detected, confidence, text_bounding_box, detection_method = False, 0.0, None, 'none'
        
        if expected_text:
            detected, detection_method = TextMatcher.match_text(text, expected_text)
            confidence = 1.0 if detected else 0.0
            
            if detected:
                text_bounding_box = self._find_text_bounding_box(expected_text, boxes)
                if text_bounding_box and roi_offset != (0, 0):
                    text_bounding_box[0] += roi_offset[0]
                    text_bounding_box[1] += roi_offset[1]
        elif 'unexpected_text' in params:
            unexpected = params['unexpected_text']
            detected = unexpected.lower() in text.lower()
            confidence, detection_method = 1.0 if detected else 0.0, 'unexpected'
        else:
            detected, confidence, detection_method = len(text) > 0, min(len(text) / 10.0, 1.0), 'presence'
        
        return {
            'detected': detected,
            'confidence': confidence,
            'text_bounding_box': text_bounding_box,
            'expected_text': expected_text,
            'details': {
                'detected_text': text,
                'expected_text': expected_text,
                'full_match': text.lower() == expected_text.lower() if expected_text else False,
                'text_bounding_box': text_bounding_box,
                'ocr_words': [word.strip() for word in boxes['text'] if word.strip()],
                'detection_method': detection_method,
                'roi_offset': roi_offset,
                'word_count': len([word for word in boxes['text'] if word.strip()])
            }
        }
    
    def _create_text_screenshot(self, frame: np.ndarray, text_bounding_box: Optional[List[int]], 
                              expected_text: str, rule_name: str, timestamp: float) -> Optional[str]:
        """Create annotated screenshot for text detection."""
        try:
            frame_copy = frame.copy()
            
            if text_bounding_box:
                self._draw_text_annotation(frame_copy, text_bounding_box, expected_text)
            else:
                cv2.putText(frame_copy, f"Searching for: {expected_text}", (10, 30), 
                           Config.ANNOTATION_FONT, Config.ANNOTATION_FONT_SCALE, 
                           Config.COLORS['YELLOW'], Config.ANNOTATION_THICKNESS)
                cv2.putText(frame_copy, f"Frame @ {timestamp:.2f}s", (10, 60), 
                           Config.ANNOTATION_FONT, 0.5, 
                           Config.COLORS['WHITE'], 1)
            
            screenshot_path = self._save_frame(frame_copy, rule_name, timestamp)
            if screenshot_path and os.path.exists(screenshot_path):
                return screenshot_path
            else:
                logger.error(f"Failed to save text screenshot for {rule_name}")
                return None
            
        except Exception as e:
            logger.error(f"Screenshot creation failed for {rule_name}: {e}")
            return None
    
    def _draw_text_annotation(self, frame: np.ndarray, text_bounding_box: List[int], 
                             expected_text: str) -> None:
        """Draw text detection annotation on frame."""
        x1, y1, w, h = text_bounding_box
        x2, y2 = x1 + w, y1 + h
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), 
                     Config.COLORS['GREEN'], Config.ANNOTATION_THICKNESS)
        
        text_label = f"Detected: {expected_text}"
        cv2.putText(frame, text_label, (x1, y1-10), 
                   Config.ANNOTATION_FONT, Config.ANNOTATION_FONT_SCALE, 
                   Config.COLORS['GREEN'], Config.ANNOTATION_THICKNESS)
    
    def export_results(self, output_path: str, format_type: str = 'json', cleanup_after_export: bool = False) -> bool:
        """Export analysis results with QA data to session-specific directory."""
        # If only filename provided, use session output directory
        if not os.path.dirname(output_path) or output_path == os.path.basename(output_path):
            output_path = os.path.join(str(self.output_dir), output_path)
        
        qa_checker = QualityAssuranceChecker(self.results)
        exporter = ResultExporter(self.results, self._create_export_metadata(), qa_checker.qa_results)
        success = exporter.export(output_path)
        
        if success and cleanup_after_export and os.path.basename(output_path) not in ["results.json", "./results.json"]:
            try:
                root_results_path = "results.json"
                if os.path.exists(root_results_path):
                    os.unlink(root_results_path)
            except Exception as e:
                logger.warning(f"Failed to clean up results file: {e}")
        
        return success
    
    def _create_export_metadata(self) -> Dict[str, Any]:
        """Create metadata for export."""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'total_detections': len(self.results),
            'total_frames_processed': self.total_frames_processed,
            'analysis_duration': datetime.now().timestamp() - (self.analysis_start_time or 0),
            'active_rules': len(self.get_active_rules())
        }
    
    def _get_video_duration(self) -> float:
        """Get the stored video duration."""
        return self.video_duration


class ResultExporter:
    """
    Handles exporting analysis results in JSON format.
    """

    def __init__(self, results: List[DetectionResult], metadata: Dict[str, Any], qa_results: Optional[Dict[str, Dict[str, Any]]] = None):
        self.results = results
        self.metadata = metadata
        self.qa_results = qa_results

    def export(self, output_path: str) -> bool:
        """
        Export results to JSON format.
        Args:
            output_path: Path to export file
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            full_output_path = self._resolve_output_path(output_path)
            self._export_json(full_output_path)
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _resolve_output_path(self, output_path: str) -> str:
        """
        Ensure output directory exists and return full output file path.
        """
        if not os.path.dirname(output_path):
            return output_path
        output_dir = os.path.dirname(output_path) or "."
        FileManager.ensure_directory(output_dir)
        return os.path.join(output_dir, os.path.basename(output_path))

    def _export_json(self, output_path: str) -> None:
        """Export results as JSON."""
        data = {
            'metadata': self.metadata,
            'results': [result.to_dict() for result in self.results]
        }
        
        if self.qa_results:
            data['qa_results'] = self.qa_results
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def create_detection_rules(target_language: str = 'en-US') -> List[DetectionRule]:
    """
    Create the standard detection rules for video analysis.
    
    Always creates the same set of rules:
    - Text Detection for "2.5 Flash" 
    - Text Detection for "Eval Mode: Native Audio Output"
    - Audio Language Detection
    - Voice Audibility Detection
    
    Args:
        target_language: Language code for audio detection (e.g., 'en-US', 'es-MX')
        
    Returns:
        List of configured detection rules
    """
    rules = []
    
    # Sanitize input
    safe_target_language = InputValidator.sanitize_user_input(target_language, 10)
    language_name = Config.SUPPORTED_LANGUAGES.get(safe_target_language, safe_target_language.upper())
    
    # Text Detection Rule: "2.5 Flash"
    for text in TargetTexts.FLASH_TEXTS:
        safe_text = InputValidator.sanitize_user_input(text, 100)
        rules.append(DetectionRule(
            name=f"Text Detection: {safe_text}",
            detection_type=DetectionType.TEXT,
            parameters={
                'expected_text': safe_text,
                'save_screenshot': True,
                'preprocess': {
                    'denoise': Config.OCR_DENOISING_ENABLED,
                    'threshold': Config.OCR_THRESHOLD_ENABLED
                }
            },
            enabled=True
        ))
    
    # Text Detection Rule: "Eval Mode: Native Audio Output"
    safe_eval_text = InputValidator.sanitize_user_input(TargetTexts.EVAL_MODE_TEXT, 100)
    rules.append(DetectionRule(
        name=f"Text Detection: {safe_eval_text}",
        detection_type=DetectionType.TEXT,
        parameters={
            'expected_text': safe_eval_text,
            'save_screenshot': True,
            'preprocess': {
                'denoise': Config.OCR_DENOISING_ENABLED,
                'threshold': Config.OCR_THRESHOLD_ENABLED
            }
        },
        enabled=True
    ))
    
    # Audio Language Detection Rule
    rules.append(DetectionRule(
        name=f"Language Detection: Fluent {language_name}",
        detection_type=DetectionType.AUDIO_LANGUAGE,
        parameters={
            'target_language': safe_target_language,
            'min_fluency_score': 0.6
        },
        enabled=True
    ))
    
    # Voice Audibility Detection Rule
    rules.append(DetectionRule(
        name="Voice Audibility: Both Voices Audible",
        detection_type=DetectionType.VOICE_AUDIBILITY,
        parameters={
            'min_confidence': 0.3,
            'min_duration': 3.0
        },
        enabled=True
    ))

    return rules


class ScreenManager:
    """Manages the three-screen interface state."""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables for screen management."""
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 'input'
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = SessionManager.generate_session_id()
        
        if 'question_id' not in st.session_state:
            st.session_state.question_id = ""
        if 'alias_email' not in st.session_state:
            st.session_state.alias_email = ""
        if 'is_authorized' not in st.session_state:
            st.session_state.is_authorized = False
        if 'video_file' not in st.session_state:
            st.session_state.video_file = None
        if 'selected_language' not in st.session_state:
            st.session_state.selected_language = 'en-US'
        if 'frame_interval' not in st.session_state:
            st.session_state.frame_interval = Config.DEFAULT_FRAME_INTERVAL
        if 'assigned_qa' not in st.session_state:
            st.session_state.assigned_qa = None
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analyzer_instance' not in st.session_state:
            st.session_state.analyzer_instance = None
        if 'qa_checker' not in st.session_state:
            st.session_state.qa_checker = None
        if 'export_in_progress' not in st.session_state:
            st.session_state.export_in_progress = False
        if 'export_completed' not in st.session_state:
            st.session_state.export_completed = False
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
        # Store the current screen value
        target_screen = 'input'
        
        # Get all session state keys to clear
        keys_to_clear = list(st.session_state.keys())
        
        # Clear all session state
        for key in keys_to_clear:
            try:
                del st.session_state[key]
            except:
                logger.warning(f"Failed to clear session state key: {key}")
        
        # Re-initialize essential session state with a NEW session ID
        st.session_state.current_screen = target_screen
        st.session_state.session_id = SessionManager.generate_session_id()  # Generate new session ID
        st.session_state.question_id = ""
        st.session_state.alias_email = ""
        st.session_state.video_file = None
        st.session_state.selected_language = 'en-US'
        st.session_state.frame_interval = Config.DEFAULT_FRAME_INTERVAL
        st.session_state.analysis_results = None
        st.session_state.analyzer_instance = None
        st.session_state.qa_checker = None
        st.session_state.export_in_progress = False
        st.session_state.export_completed = False
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
        """Clean up screenshots and files from the current session only."""
        try:
            current_session_id = st.session_state.get('session_id')
            if current_session_id:
                # Clean up only the current session, not all sessions
                from video_analyzer import ResourceManager
                resource_manager = ResourceManager()
                resource_manager.force_cleanup_session(current_session_id)
                
                session_dir = os.path.join('analysis_output', current_session_id)
                if os.path.exists(session_dir):
                    from pathlib import Path
                    # Only clean up screenshots from THIS session
                    for file_path in Path(session_dir).glob("*.png"):
                        try:
                            file_path.unlink()
                            logger.debug(f"Removed screenshot: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove screenshot {file_path}: {e}")
                    
                    # Remove session directory if empty
                    try:
                        if not any(Path(session_dir).iterdir()):
                            os.rmdir(session_dir)
                            logger.debug(f"Removed empty session directory: {session_dir}")
                    except OSError:
                        pass
            else:
                logger.debug("No current session ID to clean up")
                        
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")


class InputScreen:
    """
    First screen: Input form for analysis parameters.
    """
    _verifier = None

    @classmethod
    def _get_verifier(cls):
        """Get or create the SoT verifier instance."""
        if cls._verifier is None:
            cls._verifier = GoogleSheetsVerifier()
        return cls._verifier

    @staticmethod
    def render():
        """Render the input screen UI."""
        # Clear all containers first
        main_container = st.container()
        
        # Only render if on input screen
        if ScreenManager.get_current_screen() != 'input':
            main_container.empty()
            return
            
        with main_container:
            InputScreen._render_title_and_divider()
            InputScreen._render_form_fields()
            InputScreen._render_config_section()
            st.divider()
            InputScreen._render_validation_and_navigation()

    @staticmethod
    def _render_title_and_divider():
        """Render title and divider for input screen."""
        with st.container():
            st.title("1️⃣ Video Content Analyzer")
            st.divider()

    @staticmethod
    def _render_form_fields():
        """Render form input fields."""
        # Only render if on input screen
        if ScreenManager.get_current_screen() != 'input':
            return
            
        col1, col2, col3 = st.columns(3)
        with col1:
            question_id = st.text_input(
                "Question ID *",
                value=st.session_state.question_id,
                placeholder="Enter question identifier",
                help="Unique identifier for this analysis session (must be authorized in SoT Video Verifier sheet)",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.question_id = question_id
        with col2:
            alias_email = st.text_input(
                "Alias Email Address *",
                value=st.session_state.alias_email,
                placeholder="alias-email@gmail.com",
                help="Email address for this analysis session (must be authorized in SoT Video Verifier sheet)",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.alias_email = alias_email
        with col3:
            qa_email = st.text_input(
                "QA Email Address *",
                value=st.session_state.get('assigned_qa', '') or '',
                placeholder="qa.analyst@invisible.email",
                help="Enter QA email address (must be authorized in QA sheet)",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.assigned_qa = qa_email
        st.divider()
        st.subheader("📁 Video File")
        video_file = st.file_uploader(
            "Upload Video File *",
            type=Config.SUPPORTED_VIDEO_FORMATS,
            help=f"Supported formats: {', '.join(Config.SUPPORTED_VIDEO_FORMATS)} (Max: {Config.MAX_FILE_SIZE // 1024 // 1024}MB)",
            disabled=st.session_state.get('analysis_in_progress', False)
        )
        st.session_state.video_file = video_file

    @staticmethod
    def _render_config_section():
        """Render configuration section with language and frame interval settings."""
        if ScreenManager.get_current_screen() != 'input':
            return
        st.subheader("⚙️ Analysis Configuration")
        col1, col2 = st.columns(2)
        with col1:
            language_options = InputScreen._get_language_options()
            language_keys = list(language_options.keys())
            # Use the first supported language as default if current is not in the list
            current_language = st.session_state.get('selected_language')
            if current_language not in language_keys:
                current_language = language_keys[0]
                st.session_state.selected_language = current_language
            selected_language = st.selectbox(
                "Target Language *",
                options=language_keys,
                format_func=lambda x: language_options[x],
                index=language_keys.index(current_language),
                help="Expected language for fluency analysis and SoT verification",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.selected_language = selected_language
        with col2:
            frame_interval = st.slider(
                "Frame Interval (seconds)",
                min_value=Config.MIN_FRAME_INTERVAL,
                max_value=10.0,
                value=st.session_state.frame_interval,
                step=1.0,
                help="Time interval between analyzed frames (less interval more accuracy but slower analysis)",
                disabled=st.session_state.get('analysis_in_progress', False)
            )
            st.session_state.frame_interval = frame_interval

    @staticmethod
    def _get_language_options() -> Dict[str, str]:
        """Get language options without flags, using locale codes."""
        return Config.get_language_options()

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
            # Check if analysis is already in progress
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
        """Handle the Start Analysis button click: authorization and state cleanup."""
        question_id = st.session_state.question_id.strip()
        alias_email = st.session_state.alias_email.strip()
        qa_email = st.session_state.get('assigned_qa', '').strip()
        selected_language = st.session_state.get('selected_language', 'en-US')
        
        try:
            if question_id and alias_email and qa_email:
                verifier = InputScreen._get_verifier()
                is_main_authorized, main_message = verifier.verify_authorization(
                    question_id, alias_email, selected_language
                )
                is_qa_authorized, qa_message = verifier.verify_qa_email(qa_email)
                
                authorization_errors = []
                if not is_main_authorized:
                    authorization_errors.append(f"**Main Authorization Failed**: {main_message}")
                if not is_qa_authorized:
                    authorization_errors.append(f"**QA Authorization Failed**: {qa_message}")
                
                if authorization_errors:
                    st.session_state.analysis_in_progress = False
                    st.session_state.analysis_started = False
                    st.session_state.validation_error_shown = True
                    for error in authorization_errors:
                        st.error(f"❌ {error}")
                    st.rerun()
                    return
                
                time.sleep(1)
                
                InputScreen._cleanup_previous_analysis_state()
                import gc
                gc.collect()
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                
                logger.info("Memory cleanup completed before starting new analysis")
                
                st.session_state.analysis_in_progress = False
                st.session_state.analysis_started = False
                ScreenManager.navigate_to_screen('analysis')
            
        except Exception as e:
            st.session_state.analysis_in_progress = False
            st.session_state.analysis_started = False
            st.session_state.validation_error_shown = True
            logger.error(f"Authorization check error: {e}")
            st.error(f"❌ **Authorization Check Failed**: Unable to verify credentials - {str(e)}")
            st.rerun()
            return

    @staticmethod
    def _cleanup_previous_analysis_state():
        """Clean up previous analysis state from session."""
        st.session_state.analysis_results = None
        st.session_state.analyzer_instance = None
        st.session_state.qa_checker = None
        st.session_state.export_in_progress = False
        st.session_state.export_completed = False
        st.session_state.analysis_in_progress = False
        st.session_state.analysis_started = False
        st.session_state.validation_error_shown = False
        if hasattr(st.session_state, 'analysis_session_id'):
            del st.session_state.analysis_session_id
        if hasattr(st.session_state, 'preserved_screenshots'):
            del st.session_state.preserved_screenshots

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
        qa_email = st.session_state.get('assigned_qa', '')
        if not qa_email or not qa_email.strip():
            errors.append("validation_error")
        return errors

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        """Basic email validation."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


class AnalysisScreen:
    """
    Second screen: Analysis progress and results.
    """
    @staticmethod
    def render():
        """Render the analysis screen."""
        st.title("2️⃣ Video Analysis in Progress")
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
            video_path, temp_files = StreamlitInterface.create_temp_video(
                st.session_state.video_file, session_id
            )
            if not video_path:
                st.error("❌ Failed to process video file")
                return
            overall_progress.progress(10, text="Video processed, initializing analyzer...")
            with VideoContentAnalyzer(session_id=session_id) as analyzer:
                AnalysisScreen._setup_and_run_analysis(analyzer, video_path, overall_progress)
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}")
            AnalysisScreen._render_analysis_error_buttons()

    @staticmethod
    def _setup_and_run_analysis(analyzer, video_path, overall_progress):
        default_rules = create_detection_rules(target_language=st.session_state.selected_language)
        active_count = 0
        for rule in default_rules:
            logger.info(f"Rule created: {rule.name} (type: {rule.detection_type.value})")
        for rule in default_rules:
            try:
                analyzer.add_rule(rule)
                active_count += 1
            except Exception as e:
                logger.error(f"Failed to add rule {rule.name}: {e}")
        active_rules = analyzer.get_active_rules()
        rule_types = {}
        for rule in active_rules:
            rule_type = rule.detection_type.value
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        rule_summary = ", ".join(f"{count} {type}" for type, count in rule_types.items())
        overall_progress.progress(20, text=f"Added {active_count} analysis rules...")
        overall_progress.progress(30, text="Starting video analysis...")
        results = analyzer.analyze_video(
            video_path,
            frame_interval=st.session_state.frame_interval
        )
        overall_progress.progress(90, text="Analysis complete, generating report...")
        analyzer.export_results("results.json")
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
        screenshot_count = sum(1 for result in results if result.screenshot_path and os.path.exists(result.screenshot_path))
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
        st.subheader("📊 Analysis Results")
        AnalysisScreen._preserve_screenshots()
        results = st.session_state.analysis_results
        AnalysisScreen._render_metrics(results)
        st.divider()
        if results:
            AnalysisScreen._render_individual_detections(results)
        st.divider()
        AnalysisScreen._render_navigation_buttons()

    @staticmethod
    def _preserve_screenshots():
        if hasattr(st.session_state, 'analysis_results') and st.session_state.analysis_results:
            if not hasattr(st.session_state, 'preserved_screenshots'):
                st.session_state.preserved_screenshots = set()
            for result in st.session_state.analysis_results:
                if hasattr(result, 'screenshot_path') and result.screenshot_path:
                    st.session_state.preserved_screenshots.add(result.screenshot_path)

    @staticmethod
    def _render_metrics(results):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Detections", len(results))
        with col2:
            positive_detections = sum(1 for r in results if r.detected)
            st.metric("Positive Detections", positive_detections)
        with col3:
            detection_rate = positive_detections / len(results) if results else 0
            st.metric("Detection Rate", f"{detection_rate:.1%}")
        with col4:
            avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    @staticmethod
    def _render_individual_detections(results):
        text_results = [r for r in results if 'Text Detection' in r.rule_name]
        audio_results = [r for r in results if 'Language Detection' in r.rule_name or 'Voice Audibility' in r.rule_name]
        if text_results:
            with st.expander(f"📝 Text Detections ({len(text_results)})", expanded=True):
                for result in text_results[:15]:
                    AnalysisScreen._render_text_detection_result(result)
        if audio_results:
            with st.expander(f"🎵 Audio Analysis ({len(audio_results)})", expanded=True):
                for result in audio_results[:15]:
                    AnalysisScreen._render_audio_detection_result(result)

    @staticmethod
    def _render_navigation_buttons():
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ New Analysis", use_container_width=True):
                AnalysisScreen._cleanup_and_reset_for_new_analysis()
        with col2:
            if st.button("📋 View QA Report", type="primary", use_container_width=True):
                ScreenManager.navigate_to_screen('qa')

    @staticmethod
    def _cleanup_and_reset_for_new_analysis():
        # Only clean up the CURRENT session, not all sessions
        current_session_id = st.session_state.get('analysis_session_id', st.session_state.get('session_id'))
        resource_manager = get_resource_manager()
        
        # Clean up only the current session
        if current_session_id:
            try:
                resource_manager.force_cleanup_session(current_session_id)
                resource_manager.cleanup_session_directories(current_session_id)
            except Exception as e:
                logger.error(f"❌ Error cleaning up current session {current_session_id}: {e}")
        
        # Only clean up orphaned sessions
        try:
            if os.path.exists(Config.OUTPUT_DIR):
                current_time = time.time()
                for item in os.listdir(Config.OUTPUT_DIR):
                    item_path = os.path.join(Config.OUTPUT_DIR, item)
                    if os.path.isdir(item_path) and item != current_session_id:
                        try:
                            # Only clean up directories older than 1 hour
                            dir_mtime = os.path.getmtime(item_path)
                            if current_time - dir_mtime > 3600:  # 1 hour
                                resource_manager.cleanup_session_directories(item)
                        except Exception as e:
                            logger.debug(f"Skipped cleanup of session {item}: {e}")
        except Exception as e:
            logger.debug(f"Error scanning for orphaned sessions: {e}")
        
        # Session-specific cleanup is handled elsewhere
        
        ScreenManager._cleanup_previous_session()
        import gc
        gc.collect()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        ScreenManager.reset_session_for_new_analysis()

    @staticmethod
    def _render_text_detection_result(result):
        status_icon = "✅" if result.detected else "❌"
        confidence_color = "green" if result.confidence > 0.7 else "orange" if result.confidence > 0.4 else "red"
        with st.container():
            st.markdown(f"### {status_icon} {result.rule_name}")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**Timestamp:** {result.timestamp:.2f}s")
                st.write(f"**Frame:** {result.frame_number}")
            with col2:
                st.markdown(f"**Confidence**")
                st.markdown(f":{confidence_color}[{result.confidence:.1%}]")
            with col3:
                st.markdown("**Status**")
                st.write("✅ Detected" if result.detected else "❌ Not Found")
            if hasattr(result, 'details') and result.details:
                with st.expander("🔍 Detection Details", expanded=False):
                    if isinstance(result.details, dict):
                        if 'detected_text' in result.details:
                            st.markdown("**Detected Text:**")
                            st.code(result.details['detected_text'])
                        if 'bbox' in result.details:
                            st.markdown("**Bounding Box:**")
                            bbox = result.details['bbox']
                            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                st.write(f"x: {bbox[0]}, y: {bbox[1]}, width: {bbox[2]}, height: {bbox[3]}")
                        if 'text_confidence' in result.details:
                            st.markdown(f"**OCR Confidence:** {result.details['text_confidence']:.2f}")
                        other_details = {k: v for k, v in result.details.items() if k not in ['detected_text', 'bbox', 'text_confidence']}
                        if other_details:
                            st.json(other_details)
                    else:
                        st.write(str(result.details))
            AnalysisScreen._render_screenshot_section(result)
        st.divider()

    @staticmethod
    def _render_screenshot_section(result):
        """Render screenshot section."""
        if result.screenshot_path:
            if os.path.isabs(result.screenshot_path):
                screenshot_path = result.screenshot_path
            else:
                if hasattr(st.session_state, 'analysis_session_id') and st.session_state.analysis_session_id:
                    session_dir = os.path.join(Config.OUTPUT_DIR, st.session_state.analysis_session_id)
                    screenshot_path = os.path.join(session_dir, result.screenshot_path)
                else:
                    screenshot_path = result.screenshot_path
            exists = os.path.exists(screenshot_path)
            if exists:
                try:
                    file_size = os.path.getsize(result.screenshot_path)
                    is_readable = os.access(result.screenshot_path, os.R_OK)
                    st.markdown("**Screenshot**")
                    with st.expander("🖼️ View", expanded=False):
                        if is_readable and file_size > 0:
                            st.image(result.screenshot_path, caption=f"Detection at {result.timestamp:.2f}s", width=300)
                        else:
                            st.error(f"File exists but not readable (size: {file_size}, readable: {is_readable})")
                except Exception as e:
                    st.markdown("**Screenshot**")
                    st.error(f"Error accessing file: {e}")
            else:
                st.markdown("**Screenshot**")
                st.write("❌ *File not found*")
                st.caption(f"Path: {result.screenshot_path}")
        else:
            st.markdown("**Screenshot**")
            st.write("❌ *Not generated*")
            if not result.detected:
                st.caption("No detection found")
            else:
                st.caption("Detection found but screenshot missing")

    @staticmethod
    def _render_audio_detection_result(result):
        status_icon = "✅" if result.detected else "❌"
        confidence_color = "green" if result.confidence > 0.7 else "orange" if result.confidence > 0.4 else "red"
        with st.container():
            st.markdown(f"### {status_icon} {result.rule_name}")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**Timestamp:** {result.timestamp:.2f}s")
                st.write(f"**Segment:** {result.frame_number}")
            with col2:
                st.markdown(f"**Confidence**")
                st.markdown(f":{confidence_color}[{result.confidence:.1%}]")
            with col3:
                st.markdown("**Status**")
                st.write("✅ Detected" if result.detected else "❌ Not Found")
            if hasattr(result, 'details') and result.details:
                with st.expander("🎵 Audio Analysis Details", expanded=False):
                    if isinstance(result.details, dict):
                        if 'Language Detection' in result.rule_name:
                            if 'detected_language' in result.details:
                                st.markdown(f"**Detected Language:** {result.details['detected_language']}")
                            if 'language_confidence' in result.details:
                                st.markdown(f"**Language Confidence:** {result.details['language_confidence']:.2f}")
                            if 'fluency_score' in result.details:
                                st.markdown(f"**Fluency Score:** {result.details['fluency_score']:.2f}")
                            if 'details' in result.details:
                                nested_details = result.details['details']
                                if isinstance(nested_details, dict):
                                    if 'transcript' in nested_details:
                                        st.markdown("**Transcript:**")
                                        st.code(nested_details['transcript'])
                                    if 'word_count' in nested_details:
                                        st.markdown(f"**Word Count:** {nested_details['word_count']}")
                                    if 'avg_confidence' in nested_details:
                                        st.markdown(f"**Avg Word Confidence:** {nested_details['avg_confidence']:.2f}")
                        elif 'Voice Audibility' in result.rule_name:
                            if 'audio_features' in result.details:
                                features = result.details['audio_features']
                                if isinstance(features, dict):
                                    st.markdown("**Audio Features:**")
                                    for feature, value in features.items():
                                        if isinstance(value, (int, float)):
                                            st.write(f"• {feature}: {value:.3f}")
                                        else:
                                            st.write(f"• {feature}: {value}")
                            if 'voice_detected' in result.details:
                                st.markdown(f"**Voice Activity:** {'Yes' if result.details['voice_detected'] else 'No'}")
                            if 'rms_energy' in result.details:
                                st.markdown(f"**RMS Energy:** {result.details['rms_energy']:.4f}")
                        excluded_keys = ['detected_language', 'language_confidence', 'fluency_score', 'details', 'audio_features', 'voice_detected', 'rms_energy']
                        other_details = {k: v for k, v in result.details.items() if k not in excluded_keys}
                        if other_details:
                            st.markdown("**Additional Details:**")
                            st.json(other_details)
                    else:
                        st.write(str(result.details))
        st.divider()


class QAScreen:
    """
    Third screen: Quality Assurance results and export.
    """
    @staticmethod
    def render():
        """Render the QA screen."""
        st.title("🛡️ Quality Assurance Report")
        if st.session_state.qa_checker:
            QAScreen._render_qa_results()
        else:
            st.error("❌ No QA data available. Please run analysis first.")
            if st.button("⬅️ Back to Analysis"):
                ScreenManager.navigate_to_screen('analysis')

    @staticmethod
    def _render_qa_results():
        """Render detailed QA results."""
        qa_checker = st.session_state.qa_checker
        overall = qa_checker.get_qa_summary()
        QAScreen._render_overall_status(overall)
        st.progress(overall['score'], text=f"Overall Quality Score: {overall['score']:.1%}")
        st.divider()
        detailed_results = qa_checker.get_detailed_results()
        QAScreen._render_detailed_checks(detailed_results)
        st.divider()
        QAScreen._render_navigation_and_export()

    @staticmethod
    def _render_overall_status(overall):
        if overall['passed']:
            st.success(f"✅ **Quality Assurance: PASSED** - All requirements met!")
        else:
            st.error(f"❌ **Quality Assurance: FAILED** - {overall['checks_passed']}/{overall['total_checks']} checks passed")

    @staticmethod
    def _render_detailed_checks(detailed_results):
        checks = [
            ('🔥 Flash Text Detection', 'flash_predominance', 'Video should contain "2.5 Flash" text'),
            ('⚙️ Eval Mode Detection', 'eval_mode_predominance', 'Video should show "Eval Mode: Native Audio Output"'),
            ('🗣️ Language Fluency', 'language_fluency', 'Audio should demonstrate fluent speech in target language'),
            ('👥 Voice Audibility', 'voice_audibility', 'Both user and model voices should be clearly audible')
        ]
        for icon_title, check_key, description in checks:
            if check_key in detailed_results:
                QAScreen._render_single_check(icon_title, check_key, description, detailed_results[check_key])

    @staticmethod
    def _render_single_check(icon_title, check_key, description, check_result):
        status_icon = "✅ PASS" if check_result['passed'] else "❌ FAIL"
        with st.expander(f"{icon_title} - {status_icon}", expanded=not check_result['passed']):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Description:** {description}")
                st.markdown(f"**Result:** {check_result['details']}")
                if 'percentage' in check_result:
                    st.progress(
                        min(check_result['percentage'] / 100, 1.0),
                        text=f"Coverage: {check_result['percentage']:.1f}%"
                    )
                if 'confidence' in check_result and check_result['confidence'] > 0:
                    st.progress(
                        check_result['confidence'],
                        text=f"Confidence: {check_result['confidence']:.1%}"
                    )
            with col2:
                if check_result['passed']:
                    st.success("✅ PASS")
                else:
                    st.error("❌ FAIL")
                st.metric("Score", f"{check_result['score']:.1%}")
    
    @staticmethod
    def _render_navigation_and_export():
        """Render navigation and export options."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Back to Results", use_container_width=True):
                ScreenManager.navigate_to_screen('analysis')
        
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True):
                # Only clean up the current session, not all sessions
                current_session_id = st.session_state.get('analysis_session_id', st.session_state.get('session_id'))
                
                # Clean up only the current session
                if current_session_id:
                    # Clean up current session directories
                    analysis_session_dir = os.path.join(Config.OUTPUT_DIR, current_session_id)
                    if os.path.exists(analysis_session_dir):
                        try:
                            shutil.rmtree(analysis_session_dir)
                        except Exception as e:
                            logger.error(f"❌ QA Screen: Failed to delete analysis_output directory {analysis_session_dir}: {e}")
                    
                    temp_session_dir = os.path.join(Config.TEMP_DIR, current_session_id)
                    if os.path.exists(temp_session_dir):
                        try:
                            shutil.rmtree(temp_session_dir)
                        except Exception as e:
                            logger.error(f"❌ QA Screen: Failed to delete temp directory {temp_session_dir}: {e}")
                # Only clean up orphaned sessions that are old
                try:
                    current_time = time.time()
                    if os.path.exists(Config.OUTPUT_DIR):
                        for item in os.listdir(Config.OUTPUT_DIR):
                            if item != current_session_id and os.path.isdir(os.path.join(Config.OUTPUT_DIR, item)):
                                item_path = os.path.join(Config.OUTPUT_DIR, item)
                                try:
                                    # Only clean up directories older than 1 hour
                                    dir_mtime = os.path.getmtime(item_path)
                                    if current_time - dir_mtime > 3600:  # 1 hour
                                        shutil.rmtree(item_path)
                                except Exception as e:
                                    logger.debug(f"QA Screen: Skipped cleanup of session {item}: {e}")
                    
                    if os.path.exists(Config.TEMP_DIR):
                        for item in os.listdir(Config.TEMP_DIR):
                            if item != current_session_id and os.path.isdir(os.path.join(Config.TEMP_DIR, item)):
                                item_path = os.path.join(Config.TEMP_DIR, item)
                                try:
                                    # Only clean up directories older than 1 hour
                                    dir_mtime = os.path.getmtime(item_path)
                                    if current_time - dir_mtime > 3600:  # 1 hour
                                        shutil.rmtree(item_path)
                                except Exception as e:
                                    logger.debug(f"QA Screen: Skipped cleanup of temp session {item}: {e}")
                except Exception as e:
                    logger.debug(f"QA Screen: Error scanning for orphaned sessions: {e}")
                
                # Session-specific cleanup
                
                ScreenManager._cleanup_previous_session()
                
                # Perform memory cleanup
                import gc
                gc.collect()
                
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                
                ScreenManager.reset_session_for_new_analysis()
        
        with col3:
            # Determine button state
            export_disabled = st.session_state.get('export_in_progress', False) or st.session_state.get('export_completed', False)
            export_button_text = "📤 Export Report"
            
            if st.session_state.get('export_in_progress', False):
                export_button_text = "⏳ Exporting..."
            elif st.session_state.get('export_completed', False):
                export_button_text = "✅ Export Completed"
            
            if st.button(export_button_text, type="primary", use_container_width=True, disabled=export_disabled):
                try:
                    st.session_state.export_in_progress = True
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.export_in_progress = False
                    st.error(f"❌ Export failed: {str(e)}")
                    logger.error(f"Export exception: {e}")
                    st.rerun()
            
            # Handle the actual export process if in progress
            if st.session_state.get('export_in_progress', False):
                try:
                        # Mark session for export to prevent cleanup
                        session_id = st.session_state.get('analysis_session_id')
                        if session_id:
                            ResourceManager.mark_session_for_export(session_id)
                        
                        # Use secure configuration
                        config = ConfigurationManager.get_secure_config()
                        gas_url = config["apps_script_url"]
                        sheet_name = config["default_sheet_name"]
                        overwrite_data = False
                        
                        export_data = QAScreen._prepare_export_data()
                        
                        screenshot_count = len(export_data.get('screenshots', []))
                        if screenshot_count > 0:
                            st.info(f"📷 Exporting to Google Sheets and uploading {screenshot_count} screenshots to Google Drive...")
                            if screenshot_count > 25:
                                st.info(f"⏰ Large screenshot set detected. Extended timeout will be used to prevent timeouts...")
                        
                        success, message = QAScreen._send_to_google_sheets(gas_url, sheet_name, export_data, overwrite_data)
                        
                        # Update status based on result
                        st.session_state.export_in_progress = False
                        
                        # Unmark session for export and allow cleanup
                        if session_id:
                            ResourceManager.unmark_session_for_export(session_id)
                            ResourceManager.unregister_active_session(session_id)
                        
                        if success:
                            st.session_state.export_completed = True
                            st.rerun()
                        else:
                            st.error(f"❌ Export failed: {message}")
                            logger.error(f"Export error: {message}")
                            st.rerun()
                            
                except Exception as e:
                    st.session_state.export_in_progress = False
                    # Ensure cleanup in case of exception
                    session_id = st.session_state.get('analysis_session_id')
                    if session_id:
                        ResourceManager.unmark_session_for_export(session_id)
                        ResourceManager.unregister_active_session(session_id)
                    st.error(f"❌ Export failed: {str(e)}")
                    logger.error(f"Export exception: {e}")
                    st.rerun()

    @staticmethod
    def _prepare_export_data() -> Dict[str, Any]:
        """Prepare analysis data for SoT export."""
        try:
            # Verify session integrity before export
            session_id = st.session_state.get('analysis_session_id')
            if session_id:
                # Check if session still has output directory
                session_output_dir = Path(Config.OUTPUT_DIR) / session_id
                if not session_output_dir.exists():
                    logger.warning(f"Session output directory missing: {session_output_dir}")
                else:
                    logger.debug(f"Session output directory exists: {session_output_dir}")
            
            qa_checker = st.session_state.qa_checker
            analysis_results = st.session_state.analysis_results
            
            if not qa_checker:
                logger.error("No QA checker available in session state")
                raise ValueError("QA checker not available")
                
            if not analysis_results:
                logger.error("No analysis results available in session state")
                raise ValueError("Analysis results not available")
            
            session_info = {
                "timestamp": datetime.now().isoformat(),
                "export_time": datetime.now().isoformat(),
                "session_id": st.session_state.get('analysis_session_id', 'unknown'),
                "question_id": st.session_state.get('question_id', ''),
                "assigned_qa": st.session_state.get('assigned_qa', ''),
                "alias_email": st.session_state.get('alias_email', ''),
                "video_file": st.session_state.video_file.name if st.session_state.video_file else 'unknown',
                "video_length": f"{st.session_state.get('video_duration', 0):.1f}s" if 'video_duration' in st.session_state else 'unknown',
                "selected_language": st.session_state.get('selected_language', 'en-US'),
                "frame_interval": st.session_state.get('frame_interval', 2.0)
            }
            
            qa_summary = qa_checker.get_qa_summary()
            
            detailed_qa = qa_checker.get_detailed_results()
            
            total_results = len(analysis_results)
            positive_detections = sum(1 for r in analysis_results if r.detected)
            text_detections = len([r for r in analysis_results if 'Text Detection' in r.rule_name and r.detected])
            audio_detections = len([r for r in analysis_results if ('Language Detection' in r.rule_name or 'Voice Audibility' in r.rule_name) and r.detected])
            unique_rules = len(set(r.rule_name for r in analysis_results))
            
            analysis_stats = {
                "total_detections": int(total_results),
                "positive_detections": int(positive_detections),
                "detection_rate": positive_detections / total_results if total_results > 0 else 0,
                "text_detections": int(text_detections),
                "audio_detections": int(audio_detections),
                "avg_confidence": sum(r.confidence for r in analysis_results) / total_results if total_results > 0 else 0
            }
            
            detection_details = []
            screenshot_data = []
            
            for i, result in enumerate(analysis_results):
                if hasattr(result, 'screenshot_path') and result.screenshot_path and os.path.exists(result.screenshot_path):
                    try:
                        import base64
                        import cv2
                        import numpy as np
                        
                        img = cv2.imread(result.screenshot_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            
                            max_dimension = 1200
                            if max(height, width) > max_dimension:
                                scale = max_dimension / max(height, width)
                                new_width = int(width * scale)
                                new_height = int(height * scale)
                                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                            
                            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
                            success, buffer = cv2.imencode('.png', img, encode_param)
                            
                            if success:
                                img_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
                                
                                screenshot_info = {
                                    "filename": os.path.basename(result.screenshot_path),
                                    "data": img_data,
                                    "original_path": result.screenshot_path,
                                    "rule_name": result.rule_name,
                                    "timestamp": result.timestamp
                                }
                                screenshot_data.append(screenshot_info)
                            else:
                                logger.error(f"Failed to encode screenshot: {result.screenshot_path}")
                        else:
                            logger.error(f"Failed to read image: {result.screenshot_path}")
                    except Exception as e:
                        logger.error(f"Failed to prepare screenshot {result.screenshot_path}: {e}")
                elif hasattr(result, 'screenshot_path') and result.screenshot_path:
                    logger.warning(f"Screenshot file not found: {result.screenshot_path}")
            
            for i, result in enumerate(analysis_results[:20]):
                detail = {
                    "rule_name": result.rule_name,
                    "detected": result.detected,
                    "confidence": result.confidence,
                    "timestamp": result.timestamp,
                    "frame_number": result.frame_number
                }
                
                if hasattr(result, 'screenshot_path') and result.screenshot_path:
                    detail["screenshot_path"] = result.screenshot_path
                
                if hasattr(result, 'detected_text') and result.detected_text:
                    detail["detected_text"] = result.detected_text
                
                if hasattr(result, 'details') and result.details:
                    if isinstance(result.details, dict):
                        if 'detected_language' in result.details:
                            detail["detected_language"] = result.details['detected_language']
                        if 'language_confidence' in result.details:
                            detail["language_confidence"] = result.details['language_confidence']
                        if 'fluency_score' in result.details:
                            detail["fluency_score"] = result.details['fluency_score']
                
                detection_details.append(detail)
            
            export_data = {
                "session_info": session_info,
                "qa_summary": qa_summary,
                "qa_details": detailed_qa,
                "analysis_stats": analysis_stats,
                "detection_details": detection_details,
                "screenshots": screenshot_data
            }
            
            export_data = QAScreen._make_json_serializable(export_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error preparing export data: {e}")
            raise

    @staticmethod
    def _make_json_serializable(obj):
        """Convert any non-JSON serializable objects to JSON serializable format."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: QAScreen._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [QAScreen._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return QAScreen._make_json_serializable(obj.__dict__)
        elif obj is None:
            return None
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    @staticmethod
    def _send_to_google_sheets(gas_url: str, sheet_name: str, data: Dict[str, Any], overwrite: bool) -> tuple[bool, str]:
        """Send data to Google Sheets via Google Apps Script with security validation."""
        try:
            # Validate the URL before making the request
            if not URLValidator.validate_url(gas_url):
                return False, "Invalid or untrusted URL provided"
            
            screenshot_count = len(data.get('screenshots', []))
            
            try:
                test_json = json.dumps(data)
            except Exception as json_error:
                logger.error(f"JSON serialization failed: {json_error}")
                return False, f"Data contains non-serializable objects: {str(json_error)}"
            
            payload = {
                "action": "writeData",
                "sheetName": sheet_name,
                "overwrite": overwrite,
                "data": data
            }
            
            # Extended timeout based on screenshot count
            try:
                config = ConfigurationManager.get_secure_config()
                base_timeout = config["timeout_base"]
                max_retries = config["max_retries"]
            except SecurityError:
                base_timeout = 45
                max_retries = 2
                
            if screenshot_count > 50:
                timeout_seconds = 300  # 5 minutes for 50+ screenshots
            elif screenshot_count > 40:
                timeout_seconds = 240  # 4 minutes for 40-50 screenshots
            elif screenshot_count > 30:
                timeout_seconds = 180  # 3 minutes for 30-40 screenshots
            elif screenshot_count > 20:
                timeout_seconds = 150  # 2.5 minutes for 20-30 screenshots
            elif screenshot_count > 15:
                timeout_seconds = 120  # 2 minutes for 15-20 screenshots
            elif screenshot_count > 10:
                timeout_seconds = 90   # 1.5 minutes for 10-15 screenshots
            elif screenshot_count > 5:
                timeout_seconds = 60   # 1 minute for 5-10 screenshots
            else:
                timeout_seconds = 45   # 45 seconds for ≤5 screenshots
            
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        import time
                        time.sleep(retry_delay)
                    
                    response = requests.post(
                        gas_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=timeout_seconds
                    )
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt == max_retries:
                        logger.warning(f"Request timed out after {timeout_seconds}s (final attempt), but Google Apps Script may still be processing...")
                        if screenshot_count > 0:
                            return True, f"Export initiated successfully! {screenshot_count} screenshots are being uploaded to Google Drive. Check the Google Sheet in a few minutes."
                        else:
                            return False, f"Request timeout - Google Apps Script may be slow to respond"
                    else:
                        logger.warning(f"Request timed out on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries:
                        logger.error(f"Request failed after {max_retries} retries: {e}")
                        return False, f"Network error after retries: {str(e)}"
                    else:
                        logger.warning(f"Request failed on attempt {attempt + 1}/{max_retries + 1}, retrying: {e}")
                        continue
            
            response.raise_for_status()
            
            try:
                result = response.json()
            except Exception as json_error:
                logger.error(f"Failed to parse JSON response: {json_error}")
                logger.error(f"Raw response text: {response.text}")
                return False, f"Invalid JSON response from Google Apps Script"
            
            if result.get("success") or result.get("status") == "success":
                logger.info("Export successful!")
                
                try:
                    # Clean up session data after successful export
                    resource_manager = get_resource_manager()
                    session_id = st.session_state.get('analysis_session_id') or st.session_state.get('session_id')
                    
                    if session_id:
                        session_data = resource_manager.get_session_data(session_id)
                        if session_data and 'analyzer' in session_data:
                            session_data['analyzer'].cleanup_screenshots(preserve_for_export=False)
                            resource_manager.cleanup_session_directories(session_id)
                        else:
                            # Clean up by session ID even if not in active sessions
                            resource_manager.cleanup_session_directories(session_id)
                    else:
                        logger.warning("No session ID found for cleanup")
                        
                except Exception as e:
                    logger.warning(f"Error cleaning up session data after export: {e}")
                
                folder_link = result.get("folderLink", "")
                if folder_link:
                    return True, f"Successfully exported to Google Sheets! Screenshots uploaded to Drive folder."
                else:
                    return True, f"Successfully exported to Google Sheets! Sheet: {sheet_name}"
            else:
                error_msg = result.get('message', result.get('error', 'Unknown error'))
                logger.error(f"Export failed with message: {error_msg}")
                return False, f"Export failed: {error_msg}"
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error during export: {e}")
            screenshot_count = len(data.get('screenshots', [])) if data else 0
            if screenshot_count > 0:
                return True, f"Export initiated! {screenshot_count} screenshots are being uploaded to Google Drive. The process may take a few minutes to complete. Check your Google Sheet for the Drive folder link."
            else:
                return False, f"Request timeout - Google Apps Script may be slow to respond"
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during export: {e}")
            return False, f"Network error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected export error: {e}")
            return False, f"Export failed: {str(e)}"


class StreamlitInterface:
    """
    Streamlit web interface with three-screen navigation.
    """
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
        qa_state = "active" if current_screen == 'qa' else ""
        st.markdown(f"""
        <div class="step-indicator">
            <div class="step {input_state}"><span>1️⃣ Input Parameters</span></div>
            <div class="step {analysis_state}"><span>2️⃣ Video Analysis</span></div>
            <div class="step {qa_state}"><span>3️⃣ QA Report</span></div>
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
            session_id = session_id or SessionManager.generate_session_id()
            safe_filename = FileManager.sanitize_filename(video_file.name)
            
            # Optimize chunk size for production environments
            # Larger chunks for faster uploads in cloud environments
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
                
                # Add progress indication for large files
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
                    
                    # Update progress for large files
                    if file_size > 50 * 1024 * 1024:
                        progress = min(1.0, total_written / file_size)
                        progress_bar.progress(progress, text=f"Uploading video... {progress*100:.1f}%")
                
                if file_size > 50 * 1024 * 1024:
                    progress_bar.empty()
                
                tmp.flush()
                os.fsync(tmp.fileno())  # Force write to disk
                if not PathValidator.validate_file_size(tmp.name):
                    os.unlink(tmp.name)
                    return None, []
                return tmp.name, [tmp.name]
        except Exception as e:
            logger.error(f"Temp video creation failed: {e}")
            st.error(f"Failed to process uploaded video: {str(e)}")
            return None, []


class QualityAssuranceChecker:
    """
    Quality Assurance checker for video analysis results.
    """
    def __init__(self, results: List[DetectionResult]):
        self.results = results
        self.qa_results = self._perform_qa_checks()

    def _perform_qa_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform all QA checks and return results as a dict."""
        if not self.results:
            return self._create_empty_qa_results()
        checks = [
            ('flash_predominance', self._check_flash_predominance),
            ('eval_mode_predominance', self._check_eval_mode_predominance),
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
            'flash_predominance': empty_check.copy(),
            'eval_mode_predominance': empty_check.copy(),
            'language_fluency': empty_check.copy(),
            'voice_audibility': empty_check.copy(),
            'overall': {
                'passed': False,
                'score': 0.0,
                'checks_passed': 0,
                'total_checks': 4,
                'status': 'FAIL'
            }
        }

    def _check_flash_predominance(self) -> Dict[str, Any]:
        """Check if '2.5 Flash' appears in any text detection."""
        all_text_detections = [r for r in self.results if self._is_flash_text_detection(r)]
        if not all_text_detections:
            return {
                'passed': False,
                'score': 0.0,
                'details': 'No text detection results found',
                'flash_found': False
            }
        flash_found, flash_detections = self._find_flash_in_detections(all_text_detections)
        details = (f"✅ '2.5 Flash' found in OCR text ({len(flash_detections)} detections)"
                   if flash_found else "❌ '2.5 Flash' not found in any OCR text detections")
        return {
            'passed': flash_found,
            'score': 1.0 if flash_found else 0.0,
            'details': details,
            'flash_found': flash_found,
            'flash_count': len(flash_detections),
            'total_text_detections': len(all_text_detections)
        }

    def _is_flash_text_detection(self, r) -> bool:
        if any(keyword in r.rule_name.lower() for keyword in ['text', 'ocr', '2.5']):
            return True
        if r.details and 'detected_text' in r.details:
            return True
        if r.details and isinstance(r.details, dict):
            for key, value in r.details.items():
                if isinstance(value, str) and any(target in value.lower() for target in ['2.5', 'flash']):
                    return True
        return False

    def _find_flash_in_detections(self, detections):
        flash_found = False
        flash_detections = []
        for result in detections:
            detected_text = self._extract_detected_text(result)
            if detected_text:
                flash_patterns = ['2.5 flash', '2.5flash', '2.5_flash', '2.5-flash']
                if any(pattern in detected_text for pattern in flash_patterns):
                    flash_found = True
                    flash_detections.append(result)
        return flash_found, flash_detections

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

    def _check_eval_mode_predominance(self) -> Dict[str, Any]:
        """Check if 'Eval Mode: Native Audio Output' appears in any text detection."""
        all_text_detections = [r for r in self.results if self._is_eval_mode_text_detection(r)]
        if not all_text_detections:
            return {
                'passed': False,
                'score': 0.0,
                'details': 'No text detection results found',
                'eval_mode_found': False
            }
        eval_mode_found, eval_mode_detections = self._find_eval_mode_in_detections(all_text_detections)
        details = (f"✅ 'Eval Mode: Native Audio Output' found in OCR text ({len(eval_mode_detections)} detections)"
                   if eval_mode_found else "❌ 'Eval Mode: Native Audio Output' not found in any OCR text detections")
        return {
            'passed': eval_mode_found,
            'score': 1.0 if eval_mode_found else 0.0,
            'details': details,
            'eval_mode_found': eval_mode_found,
            'eval_mode_count': len(eval_mode_detections),
            'total_text_detections': len(all_text_detections)
        }

    def _is_eval_mode_text_detection(self, r) -> bool:
        if any(keyword in r.rule_name.lower() for keyword in ['text', 'ocr', 'eval']):
            return True
        if r.details and 'detected_text' in r.details:
            return True
        if r.details and isinstance(r.details, dict):
            for key, value in r.details.items():
                if isinstance(value, str) and any(target in value.lower() for target in ['eval', 'mode', 'native', 'audio']):
                    return True
        return False

    def _find_eval_mode_in_detections(self, detections):
        eval_mode_found = False
        eval_mode_detections = []
        for result in detections:
            detected_text = self._extract_detected_text(result)
            if detected_text:
                eval_mode_patterns = ['eval mode', 'native audio', 'eval mode: native audio output']
                if any(pattern in detected_text for pattern in eval_mode_patterns):
                    if 'eval mode' in detected_text and 'native audio' in detected_text:
                        eval_mode_found = True
                        eval_mode_detections.append(result)
                    elif 'eval mode: native audio output' in detected_text:
                        eval_mode_found = True
                        eval_mode_detections.append(result)
        return eval_mode_found, eval_mode_detections

    def _check_language_fluency(self) -> Dict[str, Any]:
        """Check if language is fluent for the majority of the duration."""
        fluency_detections = [r for r in self.results if r.detected and r.rule_name.startswith('Language Detection')]
        total_audio_segments = len([r for r in self.results if 'Language Detection' in r.rule_name])
        fluent_count = len(fluency_detections)
        if total_audio_segments == 0:
            return {
                'passed': False,
                'score': 0.0,
                'details': 'No audio analysis performed',
                'fluent_count': 0,
                'total_segments': 0,
                'percentage': 0.0,
                'avg_fluency_score': 0.0
            }
        percentage = (fluent_count / total_audio_segments) * 100
        threshold = 60.0
        avg_fluency_score = 0.0
        fluency_scores = []
        all_language_results = [r for r in self.results if 'Language Detection' in r.rule_name]
        for detection in all_language_results:
            score = None
            if isinstance(detection.details, dict) and 'details' in detection.details:
                nested_details = detection.details['details']
                if isinstance(nested_details, dict) and 'fluency_score' in nested_details:
                    score = nested_details['fluency_score']
            if score is None and isinstance(detection.details, dict) and 'fluency_score' in detection.details:
                score = detection.details['fluency_score']
            if score is None and detection.detected:
                score = detection.confidence
            if score is None:
                score = 0.0
            if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
                fluency_scores.append(score)
        if fluency_scores:
            avg_fluency_score = sum(fluency_scores) / len(fluency_scores)
        coverage_pass = percentage >= threshold
        fluency_pass = avg_fluency_score >= 0.6
        passed = (coverage_pass and avg_fluency_score >= 0.5) or (fluency_pass and percentage >= 50.0)
        coverage_score = min(percentage / threshold, 1.0)
        fluency_score_normalized = avg_fluency_score
        composite_score = (coverage_score * 0.6) + (fluency_score_normalized * 0.4)
        return {
            'passed': passed,
            'score': composite_score,
            'details': f"Fluent speech detected in {fluent_count}/{total_audio_segments} segments ({percentage:.1f}%)",
            'fluent_count': fluent_count,
            'total_segments': total_audio_segments,
            'percentage': percentage,
            'avg_fluency_score': avg_fluency_score,
            'threshold': threshold
        }

    def _check_voice_audibility(self) -> Dict[str, Any]:
        """Check if both user and model voices are audible in the video."""
        voice_audibility_results = [r for r in self.results if 'Voice Audibility' in r.rule_name]
        if not voice_audibility_results:
            return {
                'passed': False,
                'score': 0.0,
                'details': 'No voice audibility analysis performed',
                'both_voices_audible': False,
                'confidence': 0.0
            }
        best_result = max(voice_audibility_results, key=lambda x: x.confidence)
        details = best_result.details if best_result.details else {}
        both_voices_audible = details.get('both_voices_audible', False)
        confidence = best_result.confidence
        passed = both_voices_audible and confidence >= 0.3
        return {
            'passed': passed,
            'score': confidence if both_voices_audible else 0.0,
            'details': f"Both voices audible: {both_voices_audible} (confidence: {confidence:.1%})",
            'both_voices_audible': both_voices_audible,
            'confidence': confidence,
            'voice_diversity_score': details.get('voice_diversity_score', 0.0),
            'audio_duration': details.get('audio_duration', 0.0)
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
            
            # Render sidebar
            ApplicationRunner._render_sidebar()
            
            st.markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
                <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎥 Gemini Live Video Verifier</h1>
                <p style="color: white; font-size: 1.1rem; margin: 10px 0 0 0; opacity: 0.9;">
                    Multi-modal video analysis system for content detection, language fluency, and quality assurance validation
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            StreamlitInterface.render_progress_indicator()
            
            current_screen = ScreenManager.get_current_screen()
            
            main_content_area = st.container()
            
            with main_content_area:
                # Analysis Information section
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
                        **What will be analyzed:**
                        - **Text Detection**: OCR-based content recognition (2.5 Flash, Eval Mode)
                        - **Language Fluency**: Multi-language speech verification
                        - **Voice Audibility**: Detection of multiple distinct voices
                        - **Quality Assurance**: Automated validation of content requirements
                        """)
                    
                    with col2:
                        st.markdown("""
                        **Process:**
                        1. Video frames will be processed at the specified interval
                        2. Audio will be extracted and analyzed for language and voice patterns
                        3. Text content will be extracted using OCR
                        4. Results will be compiled and quality-checked
                        5. Comprehensive report will be generated
                        """)
                    
                    st.divider()
                
                resource_manager = get_resource_manager()
                active_count = resource_manager.get_active_count()
                
                if active_count >= Config.MAX_CONCURRENT_ANALYSES:
                    st.error(f"🚫 System at capacity ({active_count}/{Config.MAX_CONCURRENT_ANALYSES} active analyses). Please try again later.")
                    return
                
                if current_screen == 'input':
                    InputScreen.render()
                elif current_screen == 'analysis':
                    main_content_area.empty()
                    AnalysisScreen.render()
                elif current_screen == 'qa':
                    main_content_area.empty()
                    QAScreen.render()
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
            get_resource_manager().shutdown()
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
            
            # Help section
            ApplicationRunner._render_sidebar_help()
            
            st.divider()
            
            # Current session information
            ApplicationRunner._render_sidebar_session_info()
    
    @staticmethod
    def _render_sidebar_help():
        """Render help dropdown with program usage instructions."""
        with st.expander("❓ Help - How to Use", expanded=False):
            st.markdown("""
            ### 📋 How to Use This Tool
            
            **Step 1: Input Parameters**
            - Enter a unique **Question ID** for identification
            - Provide the respective **Alias Email Address** for the Question ID
            - As a QA, enter your **Invisible Email Address** to assign yourself to the task
            - **Upload a video file** (supported format: MP4)
            - Select the **target language** for analysis
            - Adjust the **frame interval** (time between analyzed frames)
            
            **Step 2: Video Analysis**
            - The system will automatically process your video
            - **Text detection** will look for specific content ("2.5 Flash", "Eval Mode")
            - **Audio analysis** will check language fluency and voice audibility
            - **Screenshots** will be captured at detection points
            - Progress will be displayed in real-time
            
            **Step 3: QA Report**
            - Review automated **quality assurance checks**
            - View **detailed detection results** with screenshots
            - **Export results** to Google Sheets if needed
            
            ### 🎯 What Gets Analyzed
            
            **Text Detection:**
            - "2.5 Flash" text recognition
            - "Eval Mode: Native Audio Output" detection
            - OCR accuracy validation
            
            **Audio Analysis:**
            - Language detection and fluency scoring
            - Voice audibility (user and model voices)
            - Speech quality assessment
            
            **Quality Assurance:**
            - Flash text predominance check
            - Eval mode text validation
            - Language fluency verification
            - Voice audibility confirmation
            
            ### 📊 Understanding Results
            
            **Confidence Levels:**
            - 🟢 High (90%+): Very reliable detection
            - 🟡 Good (70-89%): Reliable detection
            - 🟠 Fair (50-69%): Moderate confidence
            - 🔴 Low (<50%): Uncertain detection
            
            **QA Status:**
            - ✅ **PASS**: All quality checks successful
            - ❌ **FAIL**: Some quality checks failed
            
            ### 💡 Tips for Best Results
            
            - Use **high-quality video files** (good resolution, clear audio)
            - Ensure **proper lighting** for text detection
            - Choose the **correct target language** for analysis
            - Use **shorter frame intervals** for more accurate analysis
            - Verify all **input parameters** before starting analysis
            """)
    
    @staticmethod
    def _render_sidebar_session_info():
        """Render current session information in sidebar."""
        current_screen = ScreenManager.get_current_screen()
        
        # Only show session information if we have actual session data
        has_session_data = (
            hasattr(st.session_state, 'question_id') and 
            st.session_state.question_id and
            hasattr(st.session_state, 'alias_email') and
            st.session_state.alias_email and
            current_screen != 'input'
        )
        
        if has_session_data:
            st.markdown("### 📋 Current Session")
            
            # Show filled information from step 1
            session_info = ApplicationRunner._get_session_display_info()
            
            # Build the session info HTML
            session_html = f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%); padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: #2e7d32;">Session ID: <code style="background: #e0e0e0; padding: 2px 4px; border-radius: 3px; font-size: 0.7em;">{session_info['session_id']}</code></h4>
                <p style="margin: 5px 0; color: #333;"><strong>Question ID:</strong><br>{session_info['question_id']}</p>"""
            
            if session_info.get('assigned_qa'):
                session_html += f"""<p style="margin: 5px 0; color: #333;"><strong>Assigned QA:</strong><br>{session_info['assigned_qa']}</p>"""
            
            session_html += f"""
                <p style="margin: 5px 0; color: #333;"><strong>Email:</strong><br>{session_info['email']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Video File:</strong><br>{session_info['video_file']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Language:</strong><br>{session_info['language']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Frame Interval:</strong><br>{session_info['frame_interval']}</p>
            </div>
            """
            
            st.markdown(session_html, unsafe_allow_html=True)
            
            # Show analysis status if applicable
            if current_screen in ['analysis', 'qa']:
                ApplicationRunner._render_sidebar_analysis_status()
    
    @staticmethod
    def _get_session_display_info():
        """Get formatted session information for display."""
        # Language display
        language_code = st.session_state.get('selected_language', 'en-US')
        language_display = ApplicationRunner._get_language_display(language_code)
        
        session_info = {
            'question_id': st.session_state.get('question_id', 'Not specified'),
            'email': st.session_state.get('alias_email', 'Not specified'),
            'video_file': st.session_state.get('video_file').name if st.session_state.get('video_file') else 'No file uploaded',
            'language': language_display,
            'frame_interval': f"{st.session_state.get('frame_interval', Config.DEFAULT_FRAME_INTERVAL)}s",
            'session_id': st.session_state.get('session_id', 'Unknown')[:12] + '...' if st.session_state.get('session_id') else 'Unknown'
        }
        
        # Add assigned QA
        if st.session_state.get('assigned_qa'):
            session_info['assigned_qa'] = st.session_state.assigned_qa
        
        return session_info
    
    @staticmethod
    def _get_language_display(language_code: str) -> str:
        """Get language display name without flags, using locale codes."""
        return Config.get_language_display_name(language_code)
    
    @staticmethod
    def _render_sidebar_analysis_status():
        """Render analysis status information in sidebar."""
        if st.session_state.get('analysis_results'):
            # Analysis completed
            results_count = len(st.session_state.analysis_results)
            positive_detections = sum(1 for r in st.session_state.analysis_results if r.detected)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%); padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: #1565C0;">📊 Analysis Status</h4>
                <p style="margin: 5px 0; color: #333;"><strong>Status:</strong> ✅ Completed</p>
                <p style="margin: 5px 0; color: #333;"><strong>Total Detections:</strong> {results_count}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Positive Detections:</strong> {positive_detections}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Detection Rate:</strong> {(positive_detections/results_count*100) if results_count > 0 else 0:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show QA status
            if st.session_state.get('qa_checker'):
                qa_summary = st.session_state.qa_checker.get_qa_summary()
                qa_status = "✅ PASS" if qa_summary['passed'] else "❌ FAIL"
                qa_color = "#4CAF50" if qa_summary['passed'] else "#f44336"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fff3e0 0%, #fafafa 100%); padding: 15px; border-radius: 8px; border-left: 4px solid {qa_color}; margin-bottom: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: {qa_color};">🛡️ Quality Assurance</h4>
                    <p style="margin: 5px 0; color: #333;"><strong>QA Status:</strong> {qa_status}</p>
                    <p style="margin: 5px 0; color: #333;"><strong>Score:</strong> {qa_summary['score']:.1%}</p>
                    <p style="margin: 5px 0; color: #333;"><strong>Checks Passed:</strong> {qa_summary['checks_passed']}/{qa_summary['total_checks']}</p>
                </div>
                """, unsafe_allow_html=True)


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
            get_resource_manager().shutdown()
        except:
            pass
