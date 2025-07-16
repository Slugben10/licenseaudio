#!/usr/bin/env python3
# --- LICENSE VALIDATION WRAPPER (DO NOT MODIFY EXISTING APP LOGIC) ---
import sys
import os

# License validation: import and call check_license before any app logic
try:
    from license_client import check_license
except ImportError:
    # Try to import from current directory if not in path
    import importlib.util
    spec = importlib.util.spec_from_file_location("license_client", os.path.join(os.path.dirname(__file__), "license_client.py"))
    license_client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(license_client)
    check_license = license_client.check_license

LICENSE_API_URL = "https://demo.freshlook.hu/license-api/verify_license.php"

# Allow up to 3 attempts for license validation
_attempts = 0
while _attempts < 3:
    if check_license(LICENSE_API_URL):
        break
    _attempts += 1
    if _attempts < 3:
        print(f"\nLicense validation failed. Attempts remaining: {3 - _attempts}\n")
    else:
        print("\nLicense validation failed after 3 attempts. Exiting application.")
        sys.exit(1)
# --- END LICENSE VALIDATION WRAPPER ---

# Disable screen access check
import os
import sys
import platform
# Load .env for local development
try:
    from dotenv import load_dotenv
    import os
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path, override=True)
    # Fallback: manually parse .env if variable is still missing
    if os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is None:
        with open(dotenv_path, 'r') as f:
            for line in f:
                if line.strip().startswith('AZURE_STORAGE_CONNECTION_STRING='):
                    # Remove possible quotes and whitespace
                    value = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = value
                    print('DEBUG: Fallback loaded AZURE_STORAGE_CONNECTION_STRING from .env')
                    break
except ImportError:
    pass

# For macOS, implement comprehensive screen access check bypass
if platform.system() == 'darwin':
    # Set all required environment variables
    os.environ['PYTHONFRAMEWORK'] = '1'
    os.environ['DISPLAY'] = ':0'
    os.environ['WX_NO_DISPLAY_CHECK'] = '1'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['WXMAC_NO_NATIVE_MENUBAR'] = '1'
    os.environ['PYOBJC_DISABLE_CONFIRMATION'] = '1'
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['PYTHONHASHSEED'] = '1' 
    os.environ['WX_NO_NATIVE'] = '1'
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    # Function to handle uncaught exceptions in the app
    def handle_exception(exc_type, exc_value, exc_traceback):
        import traceback
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Try to show a dialog if possible, otherwise print to stderr
        try:
            # Check global WX_AVAILABLE flag which will be defined later
            # This will be caught in the except block if not yet defined
            if 'WX_AVAILABLE' in globals() and WX_AVAILABLE:
                import wx
                app = wx.App(False)
                wx.MessageBox(f"An error occurred:\n\n{error_msg}", "Application Error", wx.OK | wx.ICON_ERROR)
                app.MainLoop()
            else:
                # wx is not available or not imported yet, fall back to stderr
                sys.stderr.write(f"FATAL ERROR: {error_msg}\n")
        except:
            sys.stderr.write(f"FATAL ERROR: {error_msg}\n")
        
        # Exit with error code
        sys.exit(1)
    
    # Set the exception handler
    sys.excepthook = handle_exception

    # Note: wxPython patching will be done after the WX_AVAILABLE flag is defined

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
import requests
import base64
from io import BytesIO
import openai
from openai import AzureOpenAI
import wave
import uuid
import re
import io
import subprocess
import hashlib
import pickle
import types
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import math
import concurrent.futures
import logging

# Make wx imports without errors
try:
    import wx
    import wx.adv
    WX_AVAILABLE = True
    
    # Now that wx is successfully imported, apply patches for macOS compatibility
    try:
        # Patch wx.App to avoid screen check
        if hasattr(wx, 'App'):
            original_init = wx.App.__init__
            
            def patched_init(self, *args, **kwargs):
                # Force redirect to False to avoid screen check issues
                kwargs['redirect'] = False
                return original_init(self, *args, **kwargs)
            
            wx.App.__init__ = patched_init
        
        # Try to patch _core directly
        if hasattr(wx, '_core') and hasattr(wx._core, '_macIsRunningOnMainDisplay'):
            # Replace with function that always returns True
            wx._core._macIsRunningOnMainDisplay = lambda: True
            
        print("Successfully applied wxPython patches for macOS compatibility")
    except Exception as e:
        # Not a fatal error, just log it
        print(f"Warning: Could not apply wxPython patches: {e}")
        
except ImportError:
    WX_AVAILABLE = False
    print("Could not import wxPython. GUI will not be available.")

# Try to import other dependencies with graceful fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Check if pydub is available for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Check if pyannote is available for speaker diarization
try:
    import torch
    import pyannote.audio
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.audio import Audio
    from pyannote.core import Segment, Annotation
    
    # Fix for SpeechBrain when running as frozen application
    if getattr(sys, 'frozen', False):
        # Add a custom import hook to fix SpeechBrain paths
        import importlib.abc
        import importlib.machinery
        
        class SpeechBrainFixer(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.startswith('speechbrain.utils.importutils'):
                    # Return empty list for find_imports to avoid file system access
                    return importlib.machinery.ModuleSpec(fullname, None)
                return None
        
        # Add the finder to sys.meta_path
        sys.meta_path.insert(0, SpeechBrainFixer())
    
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Make PyAudio optional with silent failure
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except:
    PYAUDIO_AVAILABLE = False

# Ensure required directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    import os
    import platform
    import sys
    from pathlib import Path
    
    # First determine if app is running as a frozen executable
    is_frozen = getattr(sys, 'frozen', False)
    is_windows = platform.system() == 'windows' or platform.system() == 'Windows'
    is_macos = platform.system() == 'darwin'
    
    # Get application name
    app_name = "KeszAudio"
    
    # If frozen, determine the application directory
    if is_frozen:
        try:
            # For frozen applications, use platform-specific data locations
            if is_macos:
                # For macOS, use ~/Documents for user data
                home_dir = Path.home()
                
                # Create app directory in Documents
                app_dir = home_dir / "Documents" / app_name
                
            elif is_windows:
                # For Windows, use AppData/Local
                app_dir = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser("~"))) / app_name
            else:
                # For Linux, use standard XDG data location
                app_dir = Path.home() / ".local" / "share" / app_name.lower()
            
            # Define required directories
            directories = [
                app_dir,
                app_dir / "Transcripts", 
                app_dir / "Summaries",
                app_dir / "diarization_cache"
            ]
            
            # Create the directories if they don't exist
            for directory in directories:
                if not directory.exists():
                    try:
                        directory.mkdir(parents=True, exist_ok=True)
                        print(f"Created directory: {directory}")
                    except Exception as e:
                        print(f"Error creating directory {directory}: {e}")
            
            # Return the app directory path for reference
            return str(app_dir)
        except Exception as e:
            # If we can't create directories in standard locations, use a fallback
            print(f"Error creating directories in standard location: {e}")
            try:
                # Use home directory as fallback
                fallback_dir = Path.home() / f".{app_name}"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                (fallback_dir / "Transcripts").mkdir(exist_ok=True)
                (fallback_dir / "Summaries").mkdir(exist_ok=True)
                (fallback_dir / "diarization_cache").mkdir(exist_ok=True)
                
                print(f"Created fallback directories in {fallback_dir}")
                return str(fallback_dir)
            except Exception as e2:
                print(f"Failed to create directories in home directory: {e2}")
                
                # Last resort: use temp directory
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / app_name
                temp_dir.mkdir(exist_ok=True)
                print(f"Using temporary directory as last resort: {temp_dir}")
                return str(temp_dir)
    else:
        # For normal terminal execution, use relative paths but first check if they can be created
        try:
            directories = ["Transcripts", "Summaries", "diarization_cache"]
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            # Return current directory for reference
            return os.path.abspath('.')
        except OSError as e:
            # If we can't create in current directory, try user's home directory
            print(f"Error creating directories in current path: {e}")
            try:
                home_dir = Path.home()
                app_dir = home_dir / app_name
                app_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories
                (app_dir / "Transcripts").mkdir(exist_ok=True)
                (app_dir / "Summaries").mkdir(exist_ok=True)
                (app_dir / "diarization_cache").mkdir(exist_ok=True)
                
                print(f"Created directories in {app_dir}")
                return str(app_dir)
            except Exception as e2:
                print(f"Failed to create directories in home directory: {e2}")
                # Return current directory as a last resort, even if we can't write to it
                return os.path.abspath('.')

# Global variables
APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_LANGUAGES = ["en", "hu"]  # Add more as needed
LANGUAGE_DISPLAY_NAMES = {
    "en": "English",
    "hu": "Hungarian"
}
app_name = "KeszAudio"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # Azure OpenAI deployment name for chat
WHISPER_MODEL = "gpt-4o-transcribe"  # Azure OpenAI deployment name for whisper
client = None  # Azure OpenAI client instance

# Disable GUI requirement check for wxPython
os.environ['WXSUPPRESS_SIZER_FLAGS_CHECK'] = '1'
os.environ['WXSUPPRESS_APP_NAME_WARNING'] = '1'

# Add new imports for speaker diarization with graceful fallback
DIARIZATION_AVAILABLE = False
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Timeline, Annotation
    import torch
    DIARIZATION_AVAILABLE = True
except ImportError:
    # Silently set flag without showing warning
    pass

# --- LANGUAGE CODE MAPPING FOR AZURE SPEECH SDK ---
AZURE_SPEECH_LANGUAGE_CODES = {
    "en": "en-US",
    "hu": "hu-HU"
}

# Simple CLI version for when GUI is not available
def run_cli():
    print("\n========= AI Assistant (CLI Mode) =========")
    print("1. Set Azure OpenAI API Key")
    print("2. Transcribe Audio")
    print("3. Chat with AI")
    print("4. Exit")
    
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "1Dhj1dCS5baiMwSLQ5IUp7jdMXCs9ja6jRbHDU2ThRLwg0N3rxr9JQQJ99BFACHYHv6XJ3w3AAAAACOGEpcL")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://slugb-mbkt1eqz-eastus2.cognitiveservices.azure.com")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    client = None
    if api_key and endpoint:
        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            print("Azure OpenAI configuration found in environment.")
        except Exception as e:
            print(f"Error initializing Azure OpenAI client: {e}")
    
    while True:
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            api_key = input("Enter your Azure OpenAI API Key: ").strip()
            endpoint = input("Enter your Azure OpenAI Endpoint URL: ").strip()
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
            try:
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version="2025-03-01-preview",
                    azure_endpoint=endpoint
                )
                print("Azure OpenAI configuration set successfully.")
            except Exception as e:
                print(f"Error setting API key: {e}")
        
        elif choice == "2":
            if not client:
                print("Please set your Azure OpenAI API Key first (option 1).")
                continue
                
            audio_path = input("Enter the path to your audio file: ").strip()
            if not os.path.exists(audio_path):
                print(f"File not found: {audio_path}")
                continue
                
            print("Transcribing audio...")
            try:
                with open(audio_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        file=audio_file,
                        model=whisper_deployment,
                        api_version="2025-03-01-preview"
                    )
                print("\n--- Transcription ---")
                print(response.text)
                print("---------------------")
            except Exception as e:
                print(f"Error transcribing audio: {e}")
        
        elif choice == "3":
            if not client:
                print("Please set your Azure OpenAI API Key first (option 1).")
                continue
                
            print("\nChat with AI (type 'exit' to end conversation)")
            chat_history = []
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    break
                    
                chat_history.append({"role": "user", "content": user_input})
                
                try:
                    response = client.chat.completions.create(
                        model=chat_deployment,
                        messages=chat_history,
                        temperature=0.7,
                        max_tokens=1000,
                        api_version="2025-03-01-preview"
                    )
                    
                    assistant_message = response.choices[0].message.content
                    chat_history.append({"role": "assistant", "content": assistant_message})
                    
                    print(f"\nAssistant: {assistant_message}")
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == "4":
            print("Exiting AI Assistant. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


def main():
    # Check if we're running in CLI mode explicitly
    if "--cli" in sys.argv:
        run_cli()
        return 0
    
    # Check if wxPython is available
    if not WX_AVAILABLE:
        print("wxPython is not available. Running in CLI mode.")
        run_cli()
        return 0
    
    # Try to run in GUI mode
    try:
        app = MainApp()
        app.MainLoop()
        return 0
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Falling back to CLI mode.")
        run_cli()
        return 1

class AudioProcessor:
    """Audio processing functionality for transcription and diarization."""
    def __init__(self, client, update_callback=None, config_manager=None):
        self.client = client
        self.update_callback = update_callback
        self.config_manager = config_manager
        self.transcript = None  # Initialize transcript attribute
        self.speech_client = None
        self.initialize_speech_client()
    
    def initialize_speech_client(self):
        """Initialize Azure Speech client."""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Get API key and region from config
            speech_config = self.config_manager.get_azure_speech_config()
            if speech_config is None:
                print("Failed to get Azure Speech configuration")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
                
            api_key = speech_config.get("api_key", "")
            region = speech_config.get("region", "")
            
            # Print diagnostic information
            print("Initializing Speech client with configuration:")
            print(f"Region: {region}")
            print(f"API Key: {'[SET]' if api_key else '[NOT SET]'}")
            
            # Validate basic requirements
            if not api_key:
                print("No API key found in configuration")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
                
            if not region:
                print("No region found in configuration")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
            
            # Create speech config using subscription key and region
            try:
                print(f"Creating SpeechConfig with region: {region}")
                self.speech_client = speechsdk.SpeechConfig(subscription=api_key, region=region)
                print("Successfully created SpeechConfig")
                
                # Test the configuration by creating a recognizer
                print("Testing speech configuration...")
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
                test_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_client, audio_config=audio_config)
                print("Successfully created test recognizer")
                
                # Set language if specified
                language = self.config_manager.get_language()
                if language:
                    print(f"Setting recognition language to: {language}")
                    self.speech_client.speech_recognition_language = language
                
                print("Speech client initialized successfully")
                return True
                
            except Exception as e:
                print(f"Error creating SpeechConfig: {str(e)}")
                if isinstance(self.update_callback, MainFrame):
                    self.update_callback.show_azure_speech_config_dialog()
                return False
            
        except Exception as e:
            error_msg = f"Error initializing Azure Speech client: {str(e)}"
            print(error_msg)
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            self.speech_client = None
            
            # Show configuration dialog if there's an error and we have a UI
            if isinstance(self.update_callback, MainFrame):
                self.update_callback.show_azure_speech_config_dialog()
            return False
    
    def update_status(self, message, percent=None):
        """Update status with message and optional progress percentage."""
        if self.update_callback:
            self.update_callback(message, percent)
    
    def convert_audio_file(self, file_path, target_format=".wav"):
        """Convert audio file to a different format using FFmpeg."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Get ffmpeg command - check for bundled version first
        ffmpeg_cmd = "ffmpeg"  # Default command
        
        # Check for bundled FFmpeg
        if hasattr(sys, '_MEIPASS'):
            # Running as a PyInstaller bundle
            base_dir = sys._MEIPASS
            bundled_ffmpeg_dir = os.path.join(base_dir, 'ffmpeg')
            
            if os.path.exists(bundled_ffmpeg_dir):
                # Choose executable based on platform
                if platform.system() == 'Windows':
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg.exe')
                else:
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg')
                
                if os.path.exists(bundled_ffmpeg):
                    ffmpeg_cmd = bundled_ffmpeg
                    # Update PATH environment variable to include bundled ffmpeg
                    os.environ["PATH"] = bundled_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        # Check if FFmpeg is available
        try:
            subprocess.run(
                [ffmpeg_cmd, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("FFmpeg is required for audio conversion but was not found.")
        
        # Create output file path with new extension
        output_path = os.path.splitext(file_path)[0] + target_format
        
        # Run conversion
        self.update_status(f"Converting audio to {target_format} format...", percent=10)
        try:
            subprocess.run(
                [ffmpeg_cmd, "-i", file_path, "-y", output_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            self.update_status(f"Conversion complete: {os.path.basename(output_path)}", percent=20)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error during audio conversion: {e.stderr.decode('utf-8', errors='replace')}")
    
    def transcribe_audio(self, audio_path, language=None):
        """Transcribe audio file using Azure Speech SDK with strict 10-minute chunking and parallel processing for large files."""
        import os
        import wave
        import math
        import tempfile
        import shutil
        import threading
        import time
        from dataclasses import dataclass
        from typing import List, Tuple
        import concurrent.futures

        # --- DEBUG LOGGING FOR FILE PATH ---
        print(f"[DEBUG] Trying to open audio file: {audio_path}")
        print(f"[DEBUG] os.path.exists: {os.path.exists(audio_path)}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # --- END DEBUG LOGGING ---

        @dataclass
        class TranscriptionSegment:
            """Represents a transcribed segment with timing information."""
            text: str
            start_time: float
            end_time: float
            chunk_index: int
            words: list = None  # List of dicts: {word, start, end}

        def get_wav_duration_and_size(path):
            """Get duration in seconds and size in MB of a WAV file."""
            with wave.open(path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            return duration, size_mb

        def convert_to_wav(src_path):
            """Convert audio file to 16kHz mono WAV format."""
            out_path = tempfile.mktemp(suffix='.wav')
            # --- Use bundled ffmpeg if available ---
            import sys, os, platform
            ffmpeg_cmd = 'ffmpeg'
            if hasattr(sys, '_MEIPASS'):
                ffmpeg_dir = os.path.join(sys._MEIPASS, 'ffmpeg')
                if platform.system() == 'Windows':
                    ffmpeg_path = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
                else:
                    ffmpeg_path = os.path.join(ffmpeg_dir, 'ffmpeg')
                if os.path.exists(ffmpeg_path):
                    ffmpeg_cmd = ffmpeg_path
            cmd = [
                ffmpeg_cmd, '-y', '-i', src_path,
                '-ar', '16000', '-ac', '1', out_path
            ]
            print(f"[DEBUG] Running FFmpeg command: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                print(f"[DEBUG] FFmpeg stdout: {result.stdout.decode(errors='ignore')}")
                print(f"[DEBUG] FFmpeg stderr: {result.stderr.decode(errors='ignore')}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] FFmpeg failed with return code {e.returncode}")
                print(f"[ERROR] FFmpeg stderr: {e.stderr.decode(errors='ignore')}")
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode(errors='ignore')}")
            return out_path

        def split_audio_to_chunks(wav_path, chunk_sec):
            """Split audio into chunks of specified duration in seconds."""
            with wave.open(wav_path, 'rb') as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                duration = nframes / float(framerate)
                chunk_frames = int(chunk_sec * framerate)
                num_chunks = math.ceil(duration / chunk_sec)
                chunk_paths = []
                for i in range(num_chunks):
                    start_frame = i * chunk_frames
                    wf.setpos(start_frame)
                    frames_to_read = min(chunk_frames, nframes - start_frame)
                    frames = wf.readframes(frames_to_read)
                    chunk_path = tempfile.mktemp(suffix=f'_chunk_{i+1:03d}.wav')
                    with wave.open(chunk_path, 'wb') as out_wf:
                        out_wf.setnchannels(nchannels)
                        out_wf.setsampwidth(sampwidth)
                        out_wf.setframerate(framerate)
                        out_wf.writeframes(frames)
                    chunk_paths.append(chunk_path)
                return chunk_paths

        def transcribe_chunk_with_speech_sdk(chunk_path, chunk_index, chunk_start_time, language=None):
            """Transcribe a single chunk using Azure Speech SDK."""
            try:
                import azure.cognitiveservices.speech as speechsdk
                speech_config = self.config_manager.get_azure_speech_config()
                api_key = speech_config["api_key"]
                region = speech_config["region"]
                speech_config_obj = speechsdk.SpeechConfig(subscription=api_key, region=region)
                # --- LANGUAGE CODE PATCH (must be BEFORE recognizer is created) ---
                if language:
                    azure_lang_code = AZURE_SPEECH_LANGUAGE_CODES.get(language, "en-US")
                    speech_config_obj.speech_recognition_language = azure_lang_code
                # Enable word-level timestamps
                speech_config_obj.request_word_level_timestamps()
                audio_input = speechsdk.AudioConfig(filename=chunk_path)
                recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config_obj, audio_config=audio_input)
                segments = []
                done = threading.Event()
                def handle_final(evt):
                    if evt.result.text:
                        segment_start = chunk_start_time
                        segment_end = chunk_start_time + (len(segments) * 2.0)  # Approximate timing
                        # Parse word-level info from evt.result.json
                        words = []
                        try:
                            result_json = json.loads(evt.result.json)
                            nbest = result_json.get("NBest", [])
                            if nbest and "Words" in nbest[0]:
                                for w in nbest[0]["Words"]:
                                    # Offset and Duration are in ticks (100ns), convert to seconds
                                    start = (w["Offset"] / 1e7) + segment_start
                                    end = start + (w["Duration"] / 1e7)
                                    words.append({
                                        "word": w["Word"],
                                        "start": start,
                                        "end": end
                                    })
                                print(f"[DEBUG] Chunk {chunk_index}: Extracted {len(words)} words with timestamps.")
                                if words:
                                    print(f"[DEBUG] First 3 words: {words[:3]}")
                            else:
                                print(f"[DEBUG] Chunk {chunk_index}: No word-level timing found in NBest.")
                        except Exception as e:
                            print(f"[DEBUG] Chunk {chunk_index}: Error parsing word-level timing: {e}")
                            words = []
                        segments.append(TranscriptionSegment(
                            text=evt.result.text,
                            start_time=segment_start,
                            end_time=segment_end,
                            chunk_index=chunk_index,
                            words=words
                        ))
                def handle_session_stopped(evt):
                    done.set()
                def handle_canceled(evt):
                    done.set()
                recognizer.recognized.connect(handle_final)
                recognizer.session_stopped.connect(handle_session_stopped)
                recognizer.canceled.connect(handle_canceled)
                recognizer.start_continuous_recognition()
                done.wait()
                recognizer.stop_continuous_recognition()
                return segments
            except Exception as e:
                print(f"[DEBUG] Error in transcribe_chunk_with_speech_sdk: {e}")
                self.update_status(f"Error transcribing chunk {chunk_index}: {str(e)}", percent=None)
                return []

        def combine_transcript_segments(segments):
            if not segments:
                return ""
            sorted_segments = sorted(segments, key=lambda s: (s.chunk_index, s.start_time))
            combined_text = []
            for segment in sorted_segments:
                combined_text.append(segment.text)
            return " ".join(combined_text)

        # Step 1: Convert to WAV if needed
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext != '.wav':
            self.update_status("Converting audio to WAV format...", percent=5)
            wav_path = convert_to_wav(audio_path)
            cleanup_wav = True
        else:
            wav_path = audio_path
            cleanup_wav = False

        try:
            self.update_status("Analyzing audio file...", percent=10)
            duration, size_mb = get_wav_duration_and_size(wav_path)
            self.update_status(f"Audio: {duration:.1f}s, {size_mb:.1f}MB", percent=15)
            needs_chunking = duration > 600  # 10 minutes
            if needs_chunking:
                chunk_duration = 600  # 10 minutes
                num_chunks = math.ceil(duration / chunk_duration)
                self.update_status(f"Splitting audio into {num_chunks} chunks of 10 minutes each...", percent=20)
                chunk_paths = split_audio_to_chunks(wav_path, chunk_duration)
                all_segments = []
                all_words = []
                total_chunks = len(chunk_paths)
                # Parallel transcription (up to 6 workers)
                def transcribe_one(args):
                    chunk_path, i = args
                    chunk_start_time = i * chunk_duration
                    return transcribe_chunk_with_speech_sdk(chunk_path, i, chunk_start_time, language)
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    futures = {executor.submit(transcribe_one, (chunk_path, i)): (chunk_path, i) for i, chunk_path in enumerate(chunk_paths)}
                    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                        chunk_path, i = futures[future]
                        progress = 20 + ((idx + 1) / total_chunks) * 70  # Progress from 20% to 90%
                        self.update_status(f"Transcribing chunk {i+1}/{total_chunks}...", percent=int(progress))
                        try:
                            chunk_segments = future.result()
                            all_segments.extend(chunk_segments)
                            for seg in chunk_segments:
                                if seg.words:
                                    all_words.extend(seg.words)
                        except Exception as e:
                            print(f"[DEBUG] Error in chunk {i+1}: {e}")
                            self.update_status(f"Error in chunk {i+1}: {str(e)}", percent=None)
                        # Clean up chunk file
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass
                self.update_status("Combining transcription results...", percent=95)
                self.transcript = combine_transcript_segments(all_segments)
                print(f"[DEBUG] Total words with timestamps extracted: {len(all_words)}")
                if all_words:
                    # Sort all_words by start time to ensure correct order
                    all_words.sort(key=lambda w: w.get('start', 0))
                    print(f"[DEBUG] First 5 word timings: {all_words[:5]}")
                    print(f"[DEBUG] Last 5 word timings: {all_words[-5:]}")
                    print(f"[DEBUG] Min word start: {all_words[0]['start'] if all_words else 'N/A'}")
                    print(f"[DEBUG] Max word end: {all_words[-1]['end'] if all_words else 'N/A'}")
                self.word_by_word = all_words  # Store for diarization
                print(f"[DEBUG] self.word_by_word set with {len(self.word_by_word)} words.")
            else:
                self.update_status("Transcribing with Azure Speech SDK...", percent=30)
                segments = transcribe_chunk_with_speech_sdk(wav_path, 0, 0, language)
                self.transcript = combine_transcript_segments(segments)
                all_words = []
                for seg in segments:
                    if seg.words:
                        all_words.extend(seg.words)
                print(f"[DEBUG] Total words with timestamps extracted: {len(all_words)}")
                if all_words:
                    # Sort all_words by start time to ensure correct order
                    all_words.sort(key=lambda w: w.get('start', 0))
                    print(f"[DEBUG] First 5 word timings: {all_words[:5]}")
                    print(f"[DEBUG] Last 5 word timings: {all_words[-5:]}")
                    print(f"[DEBUG] Min word start: {all_words[0]['start'] if all_words else 'N/A'}")
                    print(f"[DEBUG] Max word end: {all_words[-1]['end'] if all_words else 'N/A'}")
                self.word_by_word = all_words
                print(f"[DEBUG] self.word_by_word set with {len(self.word_by_word)} words.")
            self.update_status("Transcription complete", percent=100)
            return self.transcript
        finally:
            if cleanup_wav and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
    
    def _transcribe_with_azure_openai(self, audio_path, language=None):
        """DEPRECATED: This method has been replaced by smart chunking with Azure Speech SDK.
        Use transcribe_audio() method instead which handles all file sizes automatically."""
        raise NotImplementedError(
            "Azure OpenAI transcription has been deprecated. "
            "Use the transcribe_audio() method which automatically handles chunking for large files."
        )

    def _get_ffmpeg_install_instructions(self):
        """Return platform-specific FFmpeg installation instructions."""
        import platform
        system = platform.system().lower()
        
        if system == 'darwin':  # macOS
            return "On macOS:\n1. Install Homebrew from https://brew.sh if you don't have it\n2. Run: brew install ffmpeg"
        elif system == 'windows':
            return "On Windows:\n1. Download from https://ffmpeg.org/download.html\n2. Add to PATH or use a package manager like Chocolatey (choco install ffmpeg)"
        elif system == 'linux':
            return "On Linux:\n- Ubuntu/Debian: sudo apt install ffmpeg\n- Fedora: sudo dnf install ffmpeg\n- Arch: sudo pacman -S ffmpeg"
        else:
            return "Please download FFmpeg from https://ffmpeg.org/download.html"

class MainApp(wx.App):
    def OnInit(self):
        try:
            # Force GUI to work on macOS without Framework build
            self.SetExitOnFrameDelete(True)
            self.frame = MainFrame(None, title="AI Assistant", base_dir=APP_BASE_DIR)
            self.frame.Show()
            # Set top window explicitly for macOS
            self.SetTopWindow(self.frame)
            return True
        except AttributeError as e:
            # Add missing methods to MainFrame that might be referenced but don't exist
            if "'MainFrame' object has no attribute" in str(e):
                attr_name = str(e).split("'")[-2]
                print(f"Adding missing attribute: {attr_name}")
                setattr(MainFrame, attr_name, lambda self, *args, **kwargs: None)
                # Try again
                return self.OnInit()
            else:
                print(f"Error initializing main frame: {e}")
                return False
        except Exception as e:
            print(f"Error initializing main frame: {e}")
            return False

class MainFrame(wx.Frame):
    def __init__(self, parent, title, base_dir):
        super(MainFrame, self).__init__(parent, title=title, size=(1200, 800))
        
        # Initialize config manager
        self.config_manager = ConfigManager(base_dir)
        
        # Initialize attributes
        self.client = None
        self.api_key = self.config_manager.get_azure_api_key()
        self.language = self.config_manager.get_language()  # This will always return a valid language code
        self.hf_token = self.config_manager.get_pyannote_token()
        
        # Initialize other attributes that might be referenced
        self.identify_speakers_btn = None
        self.speaker_id_help_text = None
        self.transcript = None
        self.last_audio_path = None
        self.word_by_word = None  # Will store word timestamps for diarization
        
        # Check for API key and initialize client
        self.initialize_openai_client()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(client, self.update_status, self.config_manager)
        self.llm_processor = LLMProcessor(client, self.config_manager, self.update_status)
        
        # Set up the UI - use either create_ui or init_ui, not both
        # Initialize menus and status bar using create_ui
        self.create_ui() # Create notebook and panels
        
        # Event bindings
        self.bind_events()
        
        # Center the window
        self.Centre()
        
        # Status update
        self.update_status("Application ready.", percent=0)
        
        # Display info about supported audio formats
        wx.CallLater(1000, self.show_format_info)
        
        # Check for PyAnnote and display installation message if needed
        wx.CallLater(1500, self.check_pyannote)
    
    def initialize_openai_client(self):
        """Initialize Azure OpenAI client."""
        global client
        api_key = self.config_manager.get_azure_api_key()
        endpoint = self.config_manager.get_azure_endpoint()
        
        if not api_key or not endpoint or endpoint == "https://your-resource-name.openai.azure.com":
            dlg = wx.Dialog(self, title="Azure OpenAI Configuration Required", size=(400, 200))
            panel = wx.Panel(dlg)
            sizer = wx.BoxSizer(wx.VERTICAL)
            
            # API Key input
            key_label = wx.StaticText(panel, label="Azure OpenAI API Key:")
            key_input = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
            key_input.SetValue(api_key)
            
            # Endpoint input
            endpoint_label = wx.StaticText(panel, label="Azure OpenAI Endpoint:")
            endpoint_input = wx.TextCtrl(panel)
            endpoint_input.SetValue(endpoint)
            
            # Add to sizer
            sizer.Add(key_label, 0, wx.ALL, 5)
            sizer.Add(key_input, 0, wx.EXPAND | wx.ALL, 5)
            sizer.Add(endpoint_label, 0, wx.ALL, 5)
            sizer.Add(endpoint_input, 0, wx.EXPAND | wx.ALL, 5)
            
            # Buttons
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            ok_btn = wx.Button(panel, wx.ID_OK, "OK")
            cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
            btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
            btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
            sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            
            panel.SetSizer(sizer)
            
            if dlg.ShowModal() == wx.ID_OK:
                api_key = key_input.GetValue().strip()
                endpoint = endpoint_input.GetValue().strip()
                self.config_manager.set_azure_api_key(api_key)
                self.config_manager.set_azure_endpoint(endpoint)
            dlg.Destroy()
        
        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=self.config_manager.get_azure_api_version(),
                azure_endpoint=endpoint
            )
        except Exception as e:
            wx.MessageBox(f"Error initializing Azure OpenAI client: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
            
    def create_ui(self):
        """Create the user interface."""
        # Create status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")
        
        # Create menu bar
        menu_bar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        
        # Audio submenu
        audio_menu = wx.Menu()
        upload_audio_item = audio_menu.Append(wx.ID_ANY, "&Upload Audio File", "Upload audio file for transcription")
        self.Bind(wx.EVT_MENU, self.on_upload_audio, upload_audio_item)
        
        file_menu.AppendSubMenu(audio_menu, "&Audio")
        
        # Document submenu
        doc_menu = wx.Menu()
        upload_doc_item = doc_menu.Append(wx.ID_ANY, "&Upload Document", "Upload document for LLM context")
        select_docs_item = doc_menu.Append(wx.ID_ANY, "&Select Documents", "Select documents to load into context")
        
        self.Bind(wx.EVT_MENU, self.on_upload_document, upload_doc_item)
        self.Bind(wx.EVT_MENU, self.on_select_documents, select_docs_item)
        
        file_menu.AppendSubMenu(doc_menu, "&Documents")
        
        # Settings menu item
        settings_item = file_menu.Append(wx.ID_ANY, "&Settings", "Application settings")
        self.Bind(wx.EVT_MENU, self.on_settings, settings_item)
        
        # Exit menu item
        exit_item = file_menu.Append(wx.ID_EXIT, "E&xit", "Exit application")
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        
        menu_bar.Append(file_menu, "&File")
        
        # Help menu
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "&About", "About this application")
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        
        menu_bar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menu_bar)
        
        # Create notebook for tabbed interface
        self.notebook = wx.Notebook(self)
        
        # Create panels for each tab
        self.audio_panel = wx.Panel(self.notebook)
        self.chat_panel = wx.Panel(self.notebook)
        self.settings_panel = wx.Panel(self.notebook)
        
        # Add panels to notebook
        self.notebook.AddPage(self.audio_panel, "Audio Processing")
        self.notebook.AddPage(self.chat_panel, "Chat")
        self.notebook.AddPage(self.settings_panel, "Settings")
        
        # Bind the notebook page change event
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_notebook_page_changed)
        
        # Create UI for each panel
        if hasattr(self, 'create_audio_panel'):
            self.create_audio_panel()
        
        # Add placeholder method if not exists
        if not hasattr(self, 'create_chat_panel'):
            def create_chat_panel(self):
                chat_sizer = wx.BoxSizer(wx.VERTICAL)
                placeholder = wx.StaticText(self.chat_panel, label="Chat panel")
                chat_sizer.Add(placeholder, 1, wx.EXPAND | wx.ALL, 5)
                self.chat_panel.SetSizer(chat_sizer)
            self.create_chat_panel = types.MethodType(create_chat_panel, self)
        self.create_chat_panel()
        
        # Add placeholder method if not exists
        if not hasattr(self, 'create_settings_panel'):
            def create_settings_panel(self):
                import wx
                import copy
                try:
                    from wx.lib.scrolledpanel import ScrolledPanel
                    ScrolledPanelClass = ScrolledPanel
                except ImportError:
                    ScrolledPanelClass = wx.ScrolledWindow

                # Remove previous children
                for child in self.settings_panel.GetChildren():
                    child.Destroy()

                # Create a scrolled panel inside the settings tab
                self._settings_scroll = ScrolledPanelClass(self.settings_panel, style=wx.TAB_TRAVERSAL)
                self._settings_scroll.SetBackgroundColour(self.settings_panel.GetBackgroundColour())
                if hasattr(self._settings_scroll, 'SetupScrolling'):
                    self._settings_scroll.SetupScrolling(scroll_x=False, scroll_y=True)
                else:
                    self._settings_scroll.SetScrollRate(0, 20)

                def make_label(key):
                    return wx.StaticText(self._settings_scroll, label=key.replace('_', ' ').capitalize() + ':')

                self._settings_widgets = {}

                def build_fields(parent_sizer, config_dict, widgets_dict, path=[]):
                    for key, value in config_dict.items():
                        field_path = path + [key]
                        if isinstance(value, dict):
                            box = wx.StaticBox(self._settings_scroll, label=key.replace('_', ' ').capitalize())
                            box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
                            widgets_dict[key] = {}
                            build_fields(box_sizer, value, widgets_dict[key], field_path)
                            parent_sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)
                        elif isinstance(value, bool):
                            cb = wx.CheckBox(self._settings_scroll, label=key.replace('_', ' ').capitalize())
                            cb.SetValue(value)
                            widgets_dict[key] = cb
                            parent_sizer.Add(cb, 0, wx.ALL, 5)
                        elif isinstance(value, (int, float)):
                            parent_sizer.Add(make_label(key), 0, wx.ALL, 5)
                            tc = wx.TextCtrl(self._settings_scroll)
                            tc.SetValue(str(value))
                            widgets_dict[key] = tc
                            parent_sizer.Add(tc, 0, wx.EXPAND | wx.ALL, 5)
                        elif isinstance(value, str):
                            parent_sizer.Add(make_label(key), 0, wx.ALL, 5)
                            style = wx.TE_PASSWORD if 'key' in key.lower() or 'token' in key.lower() else 0
                            tc = wx.TextCtrl(self._settings_scroll, style=style)
                            tc.SetValue(value)
                            widgets_dict[key] = tc
                            parent_sizer.Add(tc, 0, wx.EXPAND | wx.ALL, 5)
                        elif isinstance(value, list):
                            parent_sizer.Add(make_label(key), 0, wx.ALL, 5)
                            tc = wx.TextCtrl(self._settings_scroll)
                            tc.SetValue(json.dumps(value))
                            widgets_dict[key] = tc
                            parent_sizer.Add(tc, 0, wx.EXPAND | wx.ALL, 5)
                        else:
                            parent_sizer.Add(make_label(key), 0, wx.ALL, 5)
                            tc = wx.TextCtrl(self._settings_scroll)
                            tc.SetValue(str(value))
                            widgets_dict[key] = tc
                            parent_sizer.Add(tc, 0, wx.EXPAND | wx.ALL, 5)

                def build_templates_section(parent_sizer, templates_dict, widgets_dict):
                    box = wx.StaticBox(self._settings_scroll, label='Templates')
                    box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
                    widgets_dict.clear()
                    self._template_rows = []
                    for name, content in templates_dict.items():
                        row_sizer = wx.BoxSizer(wx.HORIZONTAL)
                        name_tc = wx.TextCtrl(self._settings_scroll)
                        name_tc.SetValue(name)
                        content_tc = wx.TextCtrl(self._settings_scroll, style=wx.TE_MULTILINE)
                        content_tc.SetValue(content)
                        row_sizer.Add(name_tc, 1, wx.EXPAND | wx.ALL, 2)
                        row_sizer.Add(content_tc, 2, wx.EXPAND | wx.ALL, 2)
                        remove_btn = wx.Button(self._settings_scroll, label='Remove')
                        row_sizer.Add(remove_btn, 0, wx.ALL, 2)
                        def make_remove_handler(n=name):
                            def handler(evt):
                                del widgets_dict[n]
                                # Also remove from config before rebuild
                                if 'templates' in self.config_manager.config and n in self.config_manager.config['templates']:
                                    del self.config_manager.config['templates'][n]
                                self.create_settings_panel()
                            return handler
                        remove_btn.Bind(wx.EVT_BUTTON, make_remove_handler(name))
                        box_sizer.Add(row_sizer, 0, wx.EXPAND | wx.ALL, 2)
                        widgets_dict[name] = (name_tc, content_tc)
                        self._template_rows.append((row_sizer, name_tc, content_tc, remove_btn))
                    new_row = wx.BoxSizer(wx.HORIZONTAL)
                    new_name = wx.TextCtrl(self._settings_scroll)
                    new_content = wx.TextCtrl(self._settings_scroll, style=wx.TE_MULTILINE)
                    add_btn = wx.Button(self._settings_scroll, label='Add')
                    def on_add(evt):
                        n = new_name.GetValue().strip()
                        c = new_content.GetValue().strip()
                        if n and c:
                            # Add to config immediately so it persists on refresh
                            if 'templates' not in self.config_manager.config:
                                self.config_manager.config['templates'] = {}
                            self.config_manager.config['templates'][n] = c
                            self.create_settings_panel()
                    add_btn.Bind(wx.EVT_BUTTON, on_add)
                    new_row.Add(new_name, 1, wx.EXPAND | wx.ALL, 2)
                    new_row.Add(new_content, 2, wx.EXPAND | wx.ALL, 2)
                    new_row.Add(add_btn, 0, wx.ALL, 2)
                    box_sizer.Add(new_row, 0, wx.EXPAND | wx.ALL, 2)
                    parent_sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

                settings_sizer = wx.BoxSizer(wx.VERTICAL)
                config = copy.deepcopy(self.config_manager.config)
                if 'templates' in config:
                    templates = config.pop('templates')
                else:
                    templates = None
                build_fields(settings_sizer, config, self._settings_widgets)
                if templates is not None:
                    self._settings_widgets['templates'] = {}
                    build_templates_section(settings_sizer, templates, self._settings_widgets['templates'])
                save_btn = wx.Button(self._settings_scroll, label='Save All Settings')
                def on_save(evt):
                    def extract(widgets, orig):
                        result = type(orig)() if isinstance(orig, dict) else None
                        for k, v in widgets.items():
                            if isinstance(v, dict):
                                result[k] = extract(v, orig[k])
                            elif isinstance(v, wx.TextCtrl):
                                val = v.GetValue()
                                if isinstance(orig[k], bool):
                                    result[k] = val.lower() in ('1', 'true', 'yes', 'on')
                                elif isinstance(orig[k], int):
                                    try: result[k] = int(val)
                                    except: result[k] = 0
                                elif isinstance(orig[k], float):
                                    try: result[k] = float(val)
                                    except: result[k] = 0.0
                                elif isinstance(orig[k], list):
                                    try: result[k] = json.loads(val)
                                    except: result[k] = []
                                else:
                                    result[k] = val
                            elif isinstance(v, wx.CheckBox):
                                result[k] = v.GetValue()
                        return result
                    new_config = extract(self._settings_widgets, self.config_manager.config)
                    if 'templates' in self._settings_widgets:
                        templates = {}
                        for name, (name_tc, content_tc) in self._settings_widgets['templates'].items():
                            n = name_tc.GetValue().strip()
                            c = content_tc.GetValue().strip()
                            if n and c:
                                templates[n] = c
                        new_config['templates'] = templates
                    self.config_manager.save_config(new_config)
                    self.config_manager.config = new_config
                    if hasattr(self, 'client'):
                        try:
                            self.client = AzureOpenAI(
                                api_key=new_config.get('azure_api_key', ''),
                                api_version=new_config.get('azure_api_version', '2023-05-15'),
                                azure_endpoint=new_config.get('azure_endpoint', '')
                            )
                        except Exception as e:
                            self.show_error(f"Error setting Azure OpenAI client: {e}")
                    if hasattr(self, 'audio_processor'):
                        self.audio_processor.config_manager = self.config_manager
                        self.audio_processor.initialize_speech_client()
                    # Refresh template dropdowns in the app if present
                    if hasattr(self, 'template_choice'):
                        templates = list(new_config.get('templates', {}).keys())
                        self.template_choice.SetItems(["None"] + templates)
                        self.template_choice.SetSelection(0)
                    wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                    self.status_bar.SetStatusText("Settings saved successfully")
                save_btn.Bind(wx.EVT_BUTTON, on_save)
                settings_sizer.Add(save_btn, 0, wx.EXPAND | wx.ALL, 10)
                self._settings_scroll.SetSizer(settings_sizer)
                self._settings_scroll.Layout()
                panel_sizer = wx.BoxSizer(wx.VERTICAL)
                panel_sizer.Add(self._settings_scroll, 1, wx.EXPAND | wx.ALL, 0)
                self.settings_panel.SetSizer(panel_sizer)
                self.settings_panel.Layout()
            self.create_settings_panel = types.MethodType(create_settings_panel, self)
        self.create_settings_panel()
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        
    def on_notebook_page_changed(self, event):
        """Handle notebook page change event."""
        old_page = event.GetOldSelection()
        new_page = event.GetSelection()
        
        # If user switched from settings to audio tab, update the speaker ID button styling
        if old_page == 2 and new_page == 0:  # 2 = settings, 0 = audio
            self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
            self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
            self.update_speaker_id_button_style()
            self.audio_panel.Layout()
        
        # If user switched to settings tab, refresh the settings values
        if new_page == 2:  # 2 = settings
            # Update HuggingFace token in settings tab from config
            if hasattr(self, 'hf_input'):
                self.hf_input.SetValue(self.config_manager.get_pyannote_token())
            
            # Update Azure OpenAI API key in settings tab from config
            if hasattr(self, 'azure_api_key_input'):
                self.azure_api_key_input.SetValue(self.config_manager.get_azure_api_key())
        
        event.Skip()  # Allow default event processing
    
    def get_speaker_id_button_label(self):
        """Get label for speaker identification button based on token availability."""
        has_token = bool(self.config_manager.get_pyannote_token())
        return "Identify Speakers (Advanced)" if has_token else "Identify Speakers (Basic)"
    
    def get_speaker_id_help_text(self):
        """Get help text for speaker identification based on token availability."""
        has_token = bool(self.config_manager.get_pyannote_token())
        if has_token:
            return "Using PyAnnote for advanced speaker identification"
        else:
            return "Using basic speaker identification (Add PyAnnote token in Settings for better results)"
            
    def update_speaker_id_button_style(self):
        """Update the style of the speaker identification button based on token availability."""
        if hasattr(self, 'identify_speakers_btn'):
            has_token = bool(self.config_manager.get_pyannote_token())
            if has_token:
                self.identify_speakers_btn.SetBackgroundColour(wx.Colour(50, 200, 50))
            else:
                self.identify_speakers_btn.SetBackgroundColour(wx.NullColour)
    
    def check_api_key(self):
        """Check if Azure OpenAI API key and endpoint are available and initialize the client."""
        api_key = self.config_manager.get_azure_api_key()
        endpoint = self.config_manager.get_azure_endpoint()
        
        if api_key and endpoint and endpoint != "https://your-resource-name.openai.azure.com":
            try:
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=self.config_manager.get_azure_api_version(),
                    azure_endpoint=endpoint
                )
                self.status_bar.SetStatusText("Azure OpenAI configuration loaded")
                return
            except Exception as e:
                print(f"Error loading Azure OpenAI configuration: {e}")
        
        # If API key or endpoint is not in config or invalid, show the configuration dialog
        self.show_azure_config_dialog()

    def show_azure_config_dialog(self):
        """Show dialog to enter Azure OpenAI configuration."""
        dlg = wx.Dialog(self, title="Azure OpenAI Configuration Required", size=(400, 200))
        panel = wx.Panel(dlg)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # API Key input
        key_label = wx.StaticText(panel, label="Azure OpenAI API Key:")
        key_input = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
        key_input.SetValue(self.config_manager.get_azure_api_key())
        
        # Endpoint input
        endpoint_label = wx.StaticText(panel, label="Azure OpenAI Endpoint:")
        endpoint_input = wx.TextCtrl(panel)
        endpoint_input.SetValue(self.config_manager.get_azure_endpoint())
        
        # Add to sizer
        sizer.Add(key_label, 0, wx.ALL, 5)
        sizer.Add(key_input, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(endpoint_label, 0, wx.ALL, 5)
        sizer.Add(endpoint_input, 0, wx.EXPAND | wx.ALL, 5)
        
        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(panel, wx.ID_OK, "OK")
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(ok_btn, 0, wx.ALL, 5)
        btn_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        
        if dlg.ShowModal() == wx.ID_OK:
            api_key = key_input.GetValue().strip()
            endpoint = endpoint_input.GetValue().strip()
            if api_key and endpoint:
                # Save the configuration
                self.config_manager.set_azure_api_key(api_key)
                self.config_manager.set_azure_endpoint(endpoint)
                
                try:
                    self.client = AzureOpenAI(
                        api_key=api_key,
                        api_version=self.config_manager.get_azure_api_version(),
                        azure_endpoint=endpoint
                    )
                    self.status_bar.SetStatusText("Azure OpenAI configuration saved")
                except Exception as e:
                    wx.MessageBox(f"Error initializing Azure OpenAI client: {e}", "Error", wx.OK | wx.ICON_ERROR)
                    self.show_azure_config_dialog()
            else:
                wx.MessageBox("Both API Key and Endpoint are required to use this application.", "Error", wx.OK | wx.ICON_ERROR)
                self.show_azure_config_dialog()
        else:
            wx.MessageBox("Azure OpenAI configuration is required to use this application.", "Error", wx.OK | wx.ICON_ERROR)
            self.show_azure_config_dialog()
        dlg.Destroy()
    
    def init_ui(self):
        # Create status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready")
        
        # Create menu bar
        menu_bar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        
        # Audio submenu
        audio_menu = wx.Menu()
        upload_audio_item = audio_menu.Append(wx.ID_ANY, "&Upload Audio File", "Upload audio file for transcription")
        self.Bind(wx.EVT_MENU, self.on_upload_audio, upload_audio_item)
        
        file_menu.AppendSubMenu(audio_menu, "&Audio")
        
        # Document submenu
        doc_menu = wx.Menu()
        upload_doc_item = doc_menu.Append(wx.ID_ANY, "&Upload Document", "Upload document for LLM context")
        select_docs_item = doc_menu.Append(wx.ID_ANY, "&Select Documents", "Select documents to load into context")
        
        self.Bind(wx.EVT_MENU, self.on_upload_document, upload_doc_item)
        self.Bind(wx.EVT_MENU, self.on_select_documents, select_docs_item)
        
        file_menu.AppendSubMenu(doc_menu, "&Documents")
        
        # Settings menu item
        settings_item = file_menu.Append(wx.ID_ANY, "&Settings", "Application settings")
        self.Bind(wx.EVT_MENU, self.on_settings, settings_item)
        
        # Exit menu item
        exit_item = file_menu.Append(wx.ID_EXIT, "E&xit", "Exit application")
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        
        menu_bar.Append(file_menu, "&File")
        
        # Help menu
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "&About", "About this application")
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        
        menu_bar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menu_bar)
        
        # Main panel with notebook
        self.panel = wx.Panel(self)
        self.notebook = wx.Notebook(self.panel)
        
        # Chat tab
        self.chat_tab = wx.Panel(self.notebook)
        chat_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Chat history
        self.chat_display = wx.TextCtrl(self.chat_tab, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        
        # Input area
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chat_input = wx.TextCtrl(self.chat_tab, style=wx.TE_MULTILINE)
        send_button = wx.Button(self.chat_tab, label="Send")
        
        input_sizer.Add(self.chat_input, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        input_sizer.Add(send_button, proportion=0, flag=wx.EXPAND)
        
        chat_sizer.Add(self.chat_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        chat_sizer.Add(input_sizer, proportion=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=5)
        
        self.chat_tab.SetSizer(chat_sizer)
        
        # Transcription tab
        self.transcription_tab = wx.Panel(self.notebook)
        transcription_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Transcription display
        self.transcription_display = wx.TextCtrl(self.transcription_tab, style=wx.TE_MULTILINE | wx.TE_RICH2)
        
        # Speaker panel
        speaker_panel = wx.Panel(self.transcription_tab)
        speaker_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.speaker_list = wx.ListCtrl(speaker_panel, style=wx.LC_REPORT)
        self.speaker_list.InsertColumn(0, "Speaker")
        self.speaker_list.InsertColumn(1, "Name")
        
        speaker_button_sizer = wx.BoxSizer(wx.VERTICAL)
        rename_speaker_button = wx.Button(speaker_panel, label="Rename Speaker")
        regenerate_button = wx.Button(speaker_panel, label="Regenerate Transcript")
        
        speaker_button_sizer.Add(rename_speaker_button, flag=wx.EXPAND | wx.BOTTOM, border=5)
        speaker_button_sizer.Add(regenerate_button, flag=wx.EXPAND)
        
        speaker_sizer.Add(self.speaker_list, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        speaker_sizer.Add(speaker_button_sizer, proportion=0, flag=wx.EXPAND)
        
        speaker_panel.SetSizer(speaker_sizer)
        
        # Summarization panel
        summary_panel = wx.Panel(self.transcription_tab)
        summary_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        templates_label = wx.StaticText(summary_panel, label="Template:")
        self.templates_combo = wx.ComboBox(summary_panel, choices=["Meeting Notes", "Interview Summary", "Lecture Notes"])
        summarize_button = wx.Button(summary_panel, label="Summarize")
        
        summary_sizer.Add(templates_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        summary_sizer.Add(self.templates_combo, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        summary_sizer.Add(summarize_button, proportion=0, flag=wx.EXPAND)
        
        summary_panel.SetSizer(summary_sizer)
        
        transcription_sizer.Add(self.transcription_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        transcription_sizer.Add(speaker_panel, proportion=0, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)
        transcription_sizer.Add(summary_panel, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
        
        self.transcription_tab.SetSizer(transcription_sizer)
        
        # Settings tab (NEW)
        self.settings_tab = wx.Panel(self.notebook)
        settings_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # API Keys section
        api_box = wx.StaticBox(self.settings_tab, label="API Keys")
        api_box_sizer = wx.StaticBoxSizer(api_box, wx.VERTICAL)
        
        # Azure OpenAI API Key
        azure_sizer = wx.BoxSizer(wx.HORIZONTAL)
        azure_label = wx.StaticText(self.settings_tab, label="Azure OpenAI API Key:")
        self.azure_api_key_input = wx.TextCtrl(self.settings_tab, value=self.api_key, style=wx.TE_PASSWORD)
        
        azure_sizer.Add(azure_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        azure_sizer.Add(self.azure_api_key_input, proportion=1)
        
        # HuggingFace API Key
        hf_sizer = wx.BoxSizer(wx.HORIZONTAL)
        hf_label = wx.StaticText(self.settings_tab, label="HuggingFace Token:")
        self.hf_input = wx.TextCtrl(self.settings_tab, value=self.hf_token, style=wx.TE_PASSWORD)
        
        hf_sizer.Add(hf_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        hf_sizer.Add(self.hf_input, proportion=1)
        
        api_box_sizer.Add(azure_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        api_box_sizer.Add(hf_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=5)
        
        # Language settings section
        lang_box = wx.StaticBox(self.settings_tab, label="Language Settings")
        lang_box_sizer = wx.StaticBoxSizer(lang_box, wx.VERTICAL)
        
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(self.settings_tab, label="Transcription Language:")
        self.lang_combo = wx.ComboBox(self.settings_tab, 
                                     choices=[LANGUAGE_DISPLAY_NAMES[lang] for lang in SUPPORTED_LANGUAGES],
                                     style=wx.CB_READONLY)
        
        # Set initial selection based on saved language
        current_lang = self.config_manager.get_language()
        self.lang_combo.SetSelection(SUPPORTED_LANGUAGES.index(current_lang))
        
        lang_sizer.Add(lang_label, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=5)
        lang_sizer.Add(self.lang_combo, proportion=1)
        
        lang_box_sizer.Add(lang_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        
        # Save button for settings
        save_button = wx.Button(self.settings_tab, label="Save Settings")
        save_button.Bind(wx.EVT_BUTTON, self.on_save_settings)
        
        # Add all sections
        settings_sizer.Add(api_box_sizer, flag=wx.EXPAND | wx.ALL, border=10)
        settings_sizer.Add(lang_box_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)
        settings_sizer.Add(save_button, flag=wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)
        
        self.settings_tab.SetSizer(settings_sizer)
        
        # Add tabs to notebook
        self.notebook.AddPage(self.chat_tab, "Chat")
        self.notebook.AddPage(self.transcription_tab, "Transcription")
        self.notebook.AddPage(self.settings_tab, "Settings")
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.notebook, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        
        self.panel.SetSizer(main_sizer)
        
        # Bind events
        send_button.Bind(wx.EVT_BUTTON, self.on_send_message)
        self.chat_input.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        rename_speaker_button.Bind(wx.EVT_BUTTON, self.on_rename_speaker)
        regenerate_button.Bind(wx.EVT_BUTTON, self.on_regenerate_transcript)
        summarize_button.Bind(wx.EVT_BUTTON, self.on_summarize)
    
    def on_key_down(self, event):
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_RETURN and event.ShiftDown():
            # Allow Shift+Enter to insert a newline
            event.Skip()
        elif key_code == wx.WXK_RETURN:
            # Enter key sends the message
            self.on_send_message(event)
        else:
            event.Skip()
    
    def on_send_message(self, event):
        """Handle sending a message in the chat."""
        user_input = self.user_input.GetValue()
        if not user_input:
            return
            
        # Generate response
        response = self.llm_processor.generate_response(user_input)
        
        # Update chat history
        self.chat_history_text.AppendText(f"You: {user_input}\n")
        self.chat_history_text.AppendText(f"Assistant: {response}\n\n")
        
        # Clear user input
        self.user_input.SetValue("")
        
    def on_clear_chat_history(self, event):
        """Clear the chat history."""
        self.llm_processor.clear_chat_history()
        self.chat_history_text.SetValue("")
        
    def on_save_api_key(self, event):
        """Save the Azure OpenAI API key."""
        api_key = self.azure_api_key_input.GetValue()
        endpoint = self.azure_endpoint_input.GetValue()
        
        self.config_manager.set_azure_api_key(api_key)
        self.config_manager.set_azure_endpoint(endpoint)
        
        # Reinitialize the client with new settings
        self.initialize_openai_client()
        
        wx.MessageBox("Azure OpenAI settings saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
    
    def on_save_pyannote_token(self, event):
        """Save the PyAnnote token."""
        token = self.pyannote_token_input.GetValue()
        self.config_manager.set_pyannote_token(token)
        
        # Update the speaker identification button style
        self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
        self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
        self.update_speaker_id_button_style()
        self.audio_panel.Layout()
        
        wx.MessageBox("PyAnnote token saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
    
    def on_save_model(self, event):
        """Save the selected Azure OpenAI deployment."""
        chat_deployment = self.chat_deployment_input.GetValue()
        whisper_deployment = self.whisper_deployment_input.GetValue()
        
        self.config_manager.set_azure_deployment("chat", chat_deployment)
        self.config_manager.set_azure_deployment("whisper", whisper_deployment)
        
        wx.MessageBox("Azure OpenAI deployments saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
    
    def on_save_temperature(self, event):
        """Save the temperature value."""
        temperature = self.temperature_slider.GetValue() / 10.0
        self.config_manager.set_temperature(temperature)
        wx.MessageBox("Temperature saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
    
    def on_save_language(self, event):
        """Save the selected language."""
        language = self.language_settings_choice.GetString(self.language_settings_choice.GetSelection()).lower()
        self.config_manager.set_language(language)
        wx.MessageBox("Language saved successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
        
    def populate_template_list(self):
        """Populate the template list with available templates."""
        if not hasattr(self, 'template_list'):
            return
            
        self.template_list.Clear()
        templates = self.config_manager.get_templates()
        for name in templates.keys():
            self.template_list.Append(name)
            
    def on_add_template(self, event):
        """Add a new template."""
        name = self.template_name_input.GetValue()
        content = self.template_content_input.GetValue()
        
        if not name or not content:
            wx.MessageBox("Please enter both name and content for the template.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        self.config_manager.add_template(name, content)
        self.populate_template_list()
        self.template_name_input.SetValue("")
        self.template_content_input.SetValue("")
        
    def on_remove_template(self, event):
        """Remove the selected template."""
        if not hasattr(self, 'template_list'):
            return
            
        selected = self.template_list.GetSelection()
        if selected == wx.NOT_FOUND:
            wx.MessageBox("Please select a template to remove.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        template_name = self.template_list.GetString(selected)
        
        # Confirm deletion
        dlg = wx.MessageDialog(self, f"Are you sure you want to delete the template '{template_name}'?",
                              "Confirm Deletion", wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            # Delete template
            self.config_manager.remove_template(template_name)
            
            # Update lists
            self.populate_template_list()
            
            # Update template choice in audio panel
            templates = list(self.config_manager.get_templates().keys())
            self.template_choice.SetItems(["None"] + templates)
            self.template_choice.SetSelection(0)
        
        dlg.Destroy()
    
    def on_upload_audio(self, event):
        # Check if PyAudio is available
        if not PYAUDIO_AVAILABLE:
            wx.MessageBox("PyAudio is not available. Recording functionality will be limited.", 
                         "PyAudio Missing", wx.OK | wx.ICON_WARNING)
        
        # File dialog to select audio file - fix wildcard and dialog settings
        wildcard = "Audio files (*.mp3;*.wav;*.m4a)|*.mp3;*.wav;*.m4a|All files (*.*)|*.*"
        with wx.FileDialog(
            self, 
            message="Choose an audio file",
            defaultDir=os.path.expanduser("~"),  # Start in user's home directory
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            
            # Show the dialog and check if user clicked OK
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return  # User cancelled the dialog
            
            # Get the selected file path
            audio_path = file_dialog.GetPath()
            self.last_audio_path = audio_path  # Store for potential retry
            
            # Show a message that we're processing
            wx.MessageBox(f"Selected file: {audio_path}\n\nStarting transcription...", 
                         "Transcription Started", wx.OK | wx.ICON_INFORMATION)
            
            # Disable UI during processing
            self.notebook.Disable()
            self.status_bar.SetStatusText(f"Transcribing audio...")
            
            # Start transcription in a thread
            threading.Thread(target=self.transcribe_audio, args=(audio_path,), daemon=True).start()
    
    
    def _fallback_speaker_detection(self):
        """Use a basic approach to detect speakers when diarization is not available"""
        paragraphs = self.transcript.split("\n\n")
        speaker_count = min(len(paragraphs), 3)
        
        self.speakers = []
        self.speaker_names = {}
        
        # Use language-appropriate speaker names
        current_lang = self.config_manager.get_language()
        if current_lang == "hu":
            speaker_prefix = "Beszélő"  # Hungarian for "Speaker"
        else:
            speaker_prefix = "Speaker"
            
        for i in range(speaker_count):
            speaker_id = f"{speaker_prefix} {i+1}"
            self.speakers.append(speaker_id)
            self.speaker_names[speaker_id] = speaker_id
    
    def combine_transcript_with_speakers(self, whisper_response, speaker_segments):
        """
        Combine the word-level transcription from Whisper with speaker information from diarization.
        
        Args:
            whisper_response: The response from Whisper API with timestamps
            speaker_segments: Dictionary of speaker segments {speaker_id: [(start_time, end_time), ...]}
            
        Returns:
            Formatted transcript with speaker labels
        """
        try:
            # Get words with timestamps from Whisper response
            segments = whisper_response.segments
            
            # Build a new transcript with speaker information
            formatted_lines = []
            current_speaker = None
            current_line = []
            
            for segment in segments:
                for word_info in segment.words:
                    word = word_info.word
                    start_time = word_info.start
                    
                    # Find which speaker was talking at this time
                    speaker_at_time = None
                    for speaker, time_segments in speaker_segments.items():
                        for start, end in time_segments:
                            if start <= start_time <= end:
                                speaker_at_time = speaker
                                break
                        if speaker_at_time:
                            break
                    
                    # If no speaker found or couldn't determine, use the first speaker
                    if not speaker_at_time and self.speakers:
                        speaker_at_time = self.speakers[0]
                    
                    # Start a new line if the speaker changes
                    if speaker_at_time != current_speaker:
                        if current_line:
                            formatted_lines.append(f"{self.speaker_names.get(current_speaker, 'Unknown')}: {' '.join(current_line)}")
                            current_line = []
                        current_speaker = speaker_at_time
                    
                    # Add the word to the current line
                    current_line.append(word)
            
            # Add the last line
            if current_line:
                formatted_lines.append(f"{self.speaker_names.get(current_speaker, 'Unknown')}: {' '.join(current_line)}")
            
            return "\n\n".join(formatted_lines)
            
        except Exception as e:
            print(f"Error combining transcript with speakers: {e}")
            return whisper_response.text  # Fall back to the original transcript
    
    def show_hf_token_dialog(self):
        # Localize dialog text based on language
        current_lang = self.config_manager.get_language()
        if current_lang == "hu":
            dialog_title = "HuggingFace Token Szükséges"
            dialog_message = "Kérjük, add meg a HuggingFace hozzáférési tokened a beszélők azonosításához:\n" \
                            "(Szerezz egyet innen: https://huggingface.co/settings/tokens)"
            error_message = "A HuggingFace token szükséges a beszélők azonosításához."
        else:
            dialog_title = "HuggingFace Token Required"
            dialog_message = "Please enter your HuggingFace Access Token for speaker identification:\n" \
                            "(You can get one from https://huggingface.co/settings/tokens)"
            error_message = "HuggingFace token is required for speaker identification."
        
        dialog = wx.TextEntryDialog(
            self, 
            dialog_message,
            dialog_title
        )
        
        if dialog.ShowModal() == wx.ID_OK:
            self.hf_token = dialog.GetValue().strip()
            if self.hf_token:
                # Save the token to environment
                os.environ["HF_TOKEN"] = self.hf_token
                
                # Save to config manager
                self.config_manager.set_pyannote_token(self.hf_token)
                
                # Update the speaker identification button style if it exists
                if hasattr(self, 'identify_speakers_btn') and self.identify_speakers_btn:
                    self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
                    self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
                    self.update_speaker_id_button_style()
                    self.audio_panel.Layout()
                
                # Retry transcription
                self.notebook.Disable()
                self.status_bar.SetStatusText("Retrying transcription...")
                threading.Thread(target=self.transcribe_audio, args=(self.last_audio_path,), daemon=True).start()
            else:
                self.show_error(error_message)
                self.notebook.Enable()
        else:
            self.show_error(error_message)
            self.notebook.Enable()
        
        dialog.Destroy()
    
    def update_transcript_display(self):
        if hasattr(self, 'transcript_text'):
            self.transcript_text.Clear()
            
            # Set the transcript to the display control
            self.transcript_text.SetValue(self.transcript)
    
    def update_speaker_list(self):
        self.speaker_list.DeleteAllItems()
        if hasattr(self, 'speakers') and self.speakers:
            # --- FIX: Always ensure speaker_names is up to date with all global speakers ---
            if not hasattr(self, 'speaker_names') or not isinstance(self.speaker_names, dict):
                self.speaker_names = {}
            unique_speakers = []
            for speaker_data in self.speakers:
                speaker_id = speaker_data["speaker"] if isinstance(speaker_data, dict) and "speaker" in speaker_data else speaker_data
                if speaker_id not in unique_speakers:
                    unique_speakers.append(speaker_id)
                if speaker_id not in self.speaker_names:
                    self.speaker_names[speaker_id] = speaker_id
            for i, speaker_id in enumerate(unique_speakers):
                index = self.speaker_list.InsertItem(i, speaker_id)
                self.speaker_list.SetItem(index, 1, self.speaker_names.get(speaker_id, speaker_id))

    def on_rename_speaker(self, event):
        # Get selected speaker
        selected = self.speaker_list.GetFirstSelected()
        if selected == -1:
            self.show_error("Please select a speaker to rename")
            return
        speaker_id = self.speaker_list.GetItemText(selected, 0)
        current_name = self.speaker_list.GetItemText(selected, 1)
        dialog = wx.TextEntryDialog(self, f"Enter new name for {speaker_id}:", "Rename Speaker", value=current_name)
        if dialog.ShowModal() == wx.ID_OK:
            new_name = dialog.GetValue().strip()
            if new_name:
                # --- FIX: Only update mapping, never overwrite speaker_id itself ---
                self.speaker_names[speaker_id] = new_name
                self.speaker_list.SetItem(selected, 1, new_name)
                for i in range(self.speaker_list.GetItemCount()):
                    if i != selected and self.speaker_list.GetItemText(i, 0) == speaker_id:
                        self.speaker_list.SetItem(i, 1, new_name)
                # --- FIX: Regenerate transcript using assign_speaker_names mapping ---
                if hasattr(self, 'speakers') and self.speakers:
                    self.transcript = self.assign_speaker_names(self.speaker_names)
                    self.update_transcript_display()
                    self.status_bar.SetStatusText(f"Speaker '{speaker_id}' renamed to '{new_name}'")
        dialog.Destroy()
    
    def on_regenerate_transcript(self, event):
        if not self.transcript or not self.speakers:
            self.show_error("No transcript available to regenerate")
            return
        # --- FIX: Always regenerate using assign_speaker_names and current mapping ---
        self.transcript = self.assign_speaker_names(self.speaker_names)
        self.update_transcript_display()
        self.status_bar.SetStatusText("Transcript regenerated with speaker names")
    
    def on_summarize(self, event):
        """Generate a summary of the transcript."""
        if not self.audio_processor.transcript:
            wx.MessageBox("Please transcribe an audio file first.", "No Transcript", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get selected template
        template_idx = self.template_choice.GetSelection()
        template_name = None
        if template_idx > 0:  # 0 is "None"
            template_name = self.template_choice.GetString(template_idx)
            
        # Disable button during processing
        self.summarize_btn.Disable()
        
        # Start summarization in a separate thread
        transcript = self.transcript_text.GetValue()
        threading.Thread(target=self.summarize_thread, args=(transcript, template_name)).start()
        
    def summarize_thread(self, transcript, template_name):
        """Thread function for transcript summarization."""
        try:
            summary = self.llm_processor.summarize_transcript(transcript, template_name)
            
            # Show summary in a dialog
            wx.CallAfter(self.show_summary_dialog, summary)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Summarization error: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.summarize_btn.Enable)
            
    def show_summary_dialog(self, summary):
        """Show summary in a dialog."""
        self.last_summary_text = summary  # Store the summary for later saving
        dlg = wx.Dialog(self, title="Summary", size=(600, 400))
        sizer = wx.BoxSizer(wx.VERTICAL)
        text_ctrl = wx.TextCtrl(dlg, style=wx.TE_MULTILINE | wx.TE_READONLY)
        text_ctrl.SetValue(summary)
        sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        # Add Close button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        close_btn = wx.Button(dlg, wx.ID_CLOSE)
        btn_sizer.Add(close_btn, 0, wx.ALL, 5)
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        dlg.SetSizer(sizer)
        close_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(wx.ID_CLOSE))
        dlg.ShowModal()
        dlg.Destroy()
    
    def update_button_states(self):
        """Update the enabled/disabled states of buttons based on current state."""
        has_audio_file = bool(self.audio_file_path.GetValue())
        has_transcript = hasattr(self.audio_processor, 'transcript') and bool(self.audio_processor.transcript)
        has_speakers = hasattr(self.audio_processor, 'speakers') and bool(self.audio_processor.speakers)
        
        if hasattr(self, 'transcribe_btn'):
            self.transcribe_btn.Enable(has_audio_file)
            
        if hasattr(self, 'identify_speakers_btn'):
            self.identify_speakers_btn.Enable(has_transcript)
            
        if hasattr(self, 'apply_speaker_names_btn'):
            self.apply_speaker_names_btn.Enable(has_speakers)
            
        if hasattr(self, 'summarize_btn'):
            self.summarize_btn.Enable(has_transcript)
    
    def on_upload_document(self, event):
        # Improved file dialog to select document
        wildcard = "Text files (*.txt)|*.txt|PDF files (*.pdf)|*.pdf|All files (*.*)|*.*"
        with wx.FileDialog(
            self, 
            message="Choose a document to add",
            defaultDir=os.path.expanduser("~"),  # Start in user's home directory
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as file_dialog:
            
            # Show the dialog and check if user clicked OK
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return  # User cancelled the dialog
            
            # Get the selected file path
            doc_path = file_dialog.GetPath()
            filename = os.path.basename(doc_path)
            dest_path = os.path.join(self.documents_folder, filename)
            
            # Check if file already exists
            if os.path.exists(dest_path):
                dialog = wx.MessageDialog(self, f"File {filename} already exists. Replace it?",
                                         "File exists", wx.YES_NO | wx.ICON_QUESTION)
                if dialog.ShowModal() == wx.ID_NO:
                    dialog.Destroy()
                    return
                dialog.Destroy()
            
            # Copy file to documents folder
            try:
                shutil.copy2(doc_path, dest_path)
                self.status_bar.SetStatusText(f"Document {filename} uploaded")
                
                # Show success message
                wx.MessageBox(f"Document '{filename}' has been successfully added.", 
                             "Document Added", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                self.show_error(f"Error uploading document: {str(e)}")
    
    def on_select_documents(self, event):
        # Get list of documents
        try:
            files = os.listdir(self.documents_folder)
            files = [f for f in files if os.path.isfile(os.path.join(self.documents_folder, f))]
        except Exception as e:
            self.show_error(f"Error listing documents: {str(e)}")
            return
        
        if not files:
            self.show_error("No documents found. Please upload documents first.")
            return
        
        # Create a dialog with checkboxes for each document
        dialog = wx.Dialog(self, title="Select Documents", size=(400, 300))
        panel = wx.Panel(dialog)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Instruction text
        instructions = wx.StaticText(panel, label="Select documents to load into context:")
        sizer.Add(instructions, flag=wx.ALL, border=10)
        
        # Checkboxes for each document
        checkboxes = {}
        for filename in files:
            checkbox = wx.CheckBox(panel, label=filename)
            checkbox.SetValue(filename in self.loaded_documents)
            checkboxes[filename] = checkbox
            sizer.Add(checkbox, flag=wx.ALL, border=5)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(panel, wx.ID_OK, "OK")
        cancel_button = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        
        button_sizer.Add(ok_button, flag=wx.RIGHT, border=5)
        button_sizer.Add(cancel_button)
        
        sizer.Add(button_sizer, flag=wx.ALIGN_RIGHT | wx.ALL, border=10)
        
        panel.SetSizer(sizer)
        
        # Show dialog
        if dialog.ShowModal() == wx.ID_OK:
            # Load selected documents
            selected = [filename for filename, checkbox in checkboxes.items() if checkbox.GetValue()]
            
            # Clear previously loaded documents
            self.loaded_documents = {}
            
            # Load new selections
            for filename in selected:
                try:
                    with open(os.path.join(self.documents_folder, filename), 'r', encoding='utf-8') as f:
                        self.loaded_documents[filename] = f.read()
                except Exception as e:
                    self.show_error(f"Error loading {filename}: {str(e)}")
            
            self.status_bar.SetStatusText(f"Loaded {len(self.loaded_documents)} documents")
        
        dialog.Destroy()
    
    def on_settings(self, event):
        # Switch to the Settings tab
        self.notebook.SetSelection(2)  # Index 2 is the Settings tab
    
    def on_exit(self, event):
        self.Close()
    
    def on_about(self, event):
        info = wx.adv.AboutDialogInfo()
        info.SetName("AI Assistant")
        info.SetVersion("1.0")
        info.SetDescription("An AI assistant application for transcription, summarization, and document processing.")
        info.SetCopyright("(C) 2023")
        
        wx.adv.AboutBox(info)
    
    def show_error(self, message):
        wx.MessageBox(message, "Error", wx.OK | wx.ICON_ERROR)

    def on_save_settings(self, event):
        """Save settings from the Settings tab"""
        # Save Azure OpenAI settings
        new_api_key = self.azure_api_key_input.GetValue().strip()
        new_endpoint = self.azure_endpoint_input.GetValue().strip()
        new_hf_token = self.hf_input.GetValue().strip()
        
        # Get language selection
        lang_selection = self.lang_combo.GetSelection()
        if 0 <= lang_selection < len(SUPPORTED_LANGUAGES):
            new_language = SUPPORTED_LANGUAGES[lang_selection]  # Get language code directly from selection
        else:
            new_language = "en"  # Default to English if selection is invalid
        
        # Update Azure OpenAI settings if changed
        if new_api_key != self.config_manager.get_azure_api_key() or new_endpoint != self.config_manager.get_azure_endpoint():
            os.environ["AZURE_OPENAI_API_KEY"] = new_api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = new_endpoint
            self.config_manager.set_azure_api_key(new_api_key)
            self.config_manager.set_azure_endpoint(new_endpoint)
            
            # Update client
            if new_api_key and new_endpoint:
                try:
                    self.client = AzureOpenAI(
                        api_key=new_api_key,
                        api_version=self.config_manager.get_azure_api_version(),
                        azure_endpoint=new_endpoint
                    )
                except Exception as e:
                    self.show_error(f"Error setting Azure OpenAI client: {e}")
        
        # Update HuggingFace token if changed
        if new_hf_token != self.hf_token:
            self.hf_token = new_hf_token
            os.environ["HF_TOKEN"] = self.hf_token
            self.config_manager.set_pyannote_token(new_hf_token)
            
            # Update the speaker identification button style
            if hasattr(self, 'identify_speakers_btn') and self.identify_speakers_btn:
                self.identify_speakers_btn.SetLabel(self.get_speaker_id_button_label())
                self.speaker_id_help_text.SetLabel(self.get_speaker_id_help_text())
                self.update_speaker_id_button_style()
                if hasattr(self, 'audio_panel'):
                    self.audio_panel.Layout()
        
        # Update language if changed
        if new_language != self.language:
            self.language = new_language
            os.environ["TRANSCRIPTION_LANGUAGE"] = self.language
            self.config_manager.set_language(new_language)
        
        wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.status_bar.SetStatusText("Settings saved successfully")

    def _identify_speakers_chunked(self, paragraphs, chunk_size):
        """Process long transcripts in chunks for speaker identification."""
        self.update_status("Processing transcript in chunks...", percent=0.1)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for p in paragraphs:
            if current_length + len(p) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [p]
                current_length = len(p)
            else:
                current_chunk.append(p)
                current_length += len(p)
                
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        self.update_status(f"Processing transcript in {len(chunks)} chunks...", percent=0.15)
        
        # Process first chunk to establish speaker patterns
        model_to_use = self.config_manager.get_azure_deployment("chat")
        
        # Initialize result container
        all_results = []
        speaker_characteristics = {}
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Calculate progress percentage (0-1)
            progress = (i / len(chunks)) * 0.7 + 0.2  # 20% to 90% of total progress
            
            self.update_status(f"Processing chunk {i+1}/{len(chunks)}...", percent=progress)
            
            # For first chunk, get detailed analysis
            if i == 0:
                prompt = f"""
                Analyze this transcript segment and identify exactly two speakers (A and B).
                
                TASK:
                1. Determine which paragraphs belong to which speaker
                2. Identify each speaker's characteristics and speaking style
                3. Ensure logical conversation flow
                
                Return JSON in this exact format:
                {{
                    "analysis": {{
                        "speaker_a_characteristics": ["characteristic 1", "characteristic 2"],
                        "speaker_b_characteristics": ["characteristic 1", "characteristic 2"]
                    }},
                    "paragraphs": [
                        {{
                            "id": {len(all_results)},
                            "speaker": "A",
                            "text": "paragraph text"
                        }},
                        ...
                    ]
                }}
                
                Transcript paragraphs:
                {json.dumps([{"id": len(all_results) + j, "text": p} for j, p in enumerate(chunk)])}
                """
            else:
                # For subsequent chunks, use characteristics from first analysis
                prompt = f"""
                Continue assigning speakers to this transcript segment.
                
                Speaker A characteristics: {json.dumps(speaker_characteristics.get("speaker_a_characteristics", []))}
                Speaker B characteristics: {json.dumps(speaker_characteristics.get("speaker_b_characteristics", []))}
                
                Return JSON with speaker assignments:
                {{
                    "paragraphs": [
                        {{
                            "id": {len(all_results)},
                            "speaker": "A or B",
                            "text": "paragraph text"
                        }},
                        ...
                    ]
                }}
                
                Transcript paragraphs:
                {json.dumps([{"id": len(all_results) + j, "text": p} for j, p in enumerate(chunk)])}
                """
            
            # Make API call for this chunk
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyst who identifies speaker turns in transcripts with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Robust JSON parsing with error handling
            try:
                result = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as json_error:
                # Handle malformed JSON (e.g., single quotes instead of double quotes)
                content = response.choices[0].message.content
                self.update_status(f"Fixing malformed JSON response...", percent=0.5)
                
                # Try to fix common JSON issues
                try:
                    # Replace single quotes with double quotes for property names
                    import re
                    # Fix property names: 'key': -> "key":
                    content = re.sub(r"'([^']+)':", r'"\1":', content)
                    # Fix string values: 'value' -> "value" (but be careful with nested quotes)
                    content = re.sub(r':\s*\'([^\']*)\'(?=\s*[,}])', r': "\1"', content)
                    # Fix array values: ['item'] -> ["item"]
                    content = re.sub(r'\[\s*\'([^\']*)\'\s*\]', r'["\1"]', content)
                    
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # If still fails, create a fallback result
                    self.update_status(f"JSON parsing failed, using fallback speaker assignment...", percent=0.5)
                    result = {
                        "paragraphs": [
                            {"id": len(all_results) + j, "speaker": "A" if j % 2 == 0 else "B", "text": p}
                            for j, p in enumerate(chunk)
                        ]
                    }
            
            # Save speaker characteristics from first chunk
            if i == 0 and "analysis" in result:
                speaker_characteristics = result["analysis"]
            
            # Add results from this chunk
            if "paragraphs" in result:
                all_results.extend(result["paragraphs"])
            
            # Update progress
            after_progress = (i + 0.5) / len(chunks) * 0.7 + 0.2
            self.update_status(f"Processed chunk {i+1}/{len(chunks)}...", percent=after_progress)
        
        # Map Speaker A/B to Speaker 1/2
        speaker_map = {
            "A": "Speaker 1", 
            "B": "Speaker 2",
            "Speaker A": "Speaker 1", 
            "Speaker B": "Speaker 2"
        }
        
        self.update_status("Finalizing speaker assignments...", percent=0.95)
        
        # Create final speakers list
        self.speakers = []
        for item in sorted(all_results, key=lambda x: x.get("id", 0)):
            speaker_label = item.get("speaker", "Unknown")
            mapped_speaker = speaker_map.get(speaker_label, speaker_label)
            
            self.speakers.append({
                "speaker": mapped_speaker,
                "text": item.get("text", "")
            })
        
        # Ensure we have the right number of paragraphs
        if len(self.speakers) != len(paragraphs):
            self.update_status(f"Warning: Received {len(self.speakers)} segments but expected {len(paragraphs)}. Fixing...", percent=0.98)
            self.speakers = [
                {"speaker": self.speakers[min(i, len(self.speakers)-1)]["speaker"] if self.speakers else f"Speaker {i % 2 + 1}", 
                 "text": p}
                for i, p in enumerate(paragraphs)
            ]
        
        self.update_status(f"Speaker identification complete. Found 2 speakers across {len(chunks)} chunks.", percent=1.0)
        return self.speakers

    def identify_speakers_simple(self, transcript):
        """Identify speakers using a simplified and optimized approach."""
        self.update_status("Analyzing transcript for speaker identification...", percent=0.1)
        
        # First, split transcript into paragraphs
        paragraphs = self._create_improved_paragraphs(transcript)
        self.speaker_segments = paragraphs
        
        # Setup model
        model_to_use = self.config_manager.get_azure_deployment("chat")
        
        # For very long transcripts, we'll analyze in chunks
        MAX_CHUNK_SIZE = 8000  # characters per chunk
        
        if len(transcript) > MAX_CHUNK_SIZE:
            self.update_status("Long transcript detected. Processing in chunks...", percent=0.15)
            return self._identify_speakers_chunked(paragraphs, MAX_CHUNK_SIZE)
        
        # Enhanced single-pass approach for shorter transcripts
        prompt = f"""
        Analyze this transcript and identify exactly two speakers (A and B).
        
        TASK:
        1. Determine which paragraphs belong to which speaker
        2. Focus on conversation pattern and speaking style
        3. Ensure logical conversation flow (e.g., questions are followed by answers)
        4. Maintain consistency in first-person statements
        
        Return JSON in this exact format:
        {{
            "analysis": {{
                "speaker_a_characteristics": ["characteristic 1", "characteristic 2"],
                "speaker_b_characteristics": ["characteristic 1", "characteristic 2"],
                "speaker_count": 2,
                "conversation_type": "interview/discussion/etc"
            }},
            "paragraphs": [
                {{
                    "id": 0,
                    "speaker": "A",
                    "text": "paragraph text"
                }},
                ...
            ]
        }}
        
        Transcript paragraphs:
        {json.dumps([{"id": i, "text": p} for i, p in enumerate(paragraphs)])}
        """
        
        try:
            # Single API call to assign speakers
            self.update_status("Sending transcript for speaker analysis...", percent=0.3)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert conversation analyst who identifies speaker turns in transcripts with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            self.update_status("Processing speaker identification results...", percent=0.7)
            
            # Robust JSON parsing with error handling
            try:
                result = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as json_error:
                # Handle malformed JSON (e.g., single quotes instead of double quotes)
                content = response.choices[0].message.content
                self.update_status(f"Fixing malformed JSON response...", percent=0.7)
                
                # Try to fix common JSON issues
                try:
                    # Replace single quotes with double quotes for property names
                    import re
                    # Fix property names: 'key': -> "key":
                    content = re.sub(r"'([^']+)':", r'"\1":', content)
                    # Fix string values: 'value' -> "value" (but be careful with nested quotes)
                    content = re.sub(r':\s*\'([^\']*)\'(?=\s*[,}])', r': "\1"', content)
                    # Fix array values: ['item'] -> ["item"]
                    content = re.sub(r'\[\s*\'([^\']*)\'\s*\]', r'["\1"]', content)
                    
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # If still fails, create a fallback result
                    self.update_status(f"JSON parsing failed, using fallback speaker assignment...", percent=0.7)
                    result = {
                        "paragraphs": [
                            {"id": i, "speaker": "A" if i % 2 == 0 else "B", "text": p}
                            for i, p in enumerate(paragraphs)
                        ]
                    }
            
            # Get paragraph assignments
            assignments = result.get("paragraphs", [])
            
            # Map Speaker A/B to Speaker 1/2 for compatibility with existing system
            speaker_map = {
                "A": "Speaker 1", 
                "B": "Speaker 2",
                "Speaker A": "Speaker 1", 
                "Speaker B": "Speaker 2"
            }
            
            # Create speakers list with proper mapping
            self.speakers = []
            for item in sorted(assignments, key=lambda x: x.get("id", 0)):
                speaker_label = item.get("speaker", "Unknown")
                mapped_speaker = speaker_map.get(speaker_label, speaker_label)
                
                self.speakers.append({
                    "speaker": mapped_speaker,
                    "text": item.get("text", "")
                })
            
            # Ensure we have the right number of paragraphs
            if len(self.speakers) != len(paragraphs):
                self.update_status(f"Warning: Received {len(self.speakers)} segments but expected {len(paragraphs)}. Fixing...", percent=0.9)
                self.speakers = [
                    {"speaker": self.speakers[min(i, len(self.speakers)-1)]["speaker"] if self.speakers else f"Speaker {i % 2 + 1}", 
                     "text": p}
                    for i, p in enumerate(paragraphs)
                ]
            
            self.update_status(f"Speaker identification complete. Found {2} speakers.", percent=1.0)
            return self.speakers
            
        except Exception as e:
            self.update_status(f"Error in speaker identification: {str(e)}", percent=0)
            # Fallback to basic alternating speaker assignment
            self.speakers = [
                {"speaker": f"Speaker {i % 2 + 1}", "text": p}
                for i, p in enumerate(paragraphs)
            ]
            return self.speakers
            
    def _create_improved_paragraphs(self, transcript):
        """Create more intelligent paragraph breaks based on semantic analysis."""
        import re
        # Split transcript into sentences
        sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into paragraphs
        paragraphs = []
        current_para = []
        
        # These phrases often signal the start of a new speaker's turn
        new_speaker_indicators = [
            "yes", "no", "I think", "I believe", "so,", "well,", "actually", 
            "to be honest", "in my opinion", "I agree", "I disagree",
            "let me", "I'd like to", "I would", "you know", "um", "uh", 
            "hmm", "but", "however", "from my perspective", "wait", "okay",
            "right", "sure", "exactly", "absolutely", "definitely", "perhaps",
            "look", "listen", "basically", "frankly", "honestly", "now", "so",
            "thank you", "thanks", "good point", "interesting", "true", "correct",
            "first of all", "firstly", "secondly", "finally", "in conclusion"
        ]
        
        # Words/phrases that indicate continuation by the same speaker
        continuation_indicators = [
            "and", "also", "additionally", "moreover", "furthermore", "plus",
            "then", "after that", "next", "finally", "lastly", "in addition",
            "consequently", "as a result", "therefore", "thus", "besides",
            "for example", "specifically", "in particular", "especially",
            "because", "since", "due to", "as such", "which means"
        ]
        
        for i, sentence in enumerate(sentences):
            # Start a new paragraph if:
            start_new_para = False
            
            # 1. This is the first sentence
            if i == 0:
                start_new_para = True
                
            # 2. Previous sentence ended with a question mark
            elif i > 0 and sentences[i-1].endswith('?'):
                start_new_para = True
                
            # 3. Current sentence begins with a common new speaker phrase
            elif any(sentence.lower().startswith(indicator.lower()) for indicator in new_speaker_indicators):
                start_new_para = True
                
            # 4. Not a continuation and not a pronoun reference
            elif (i > 0 and 
                  not any(sentence.lower().startswith(indicator.lower()) for indicator in continuation_indicators) and
                  not re.match(r'^(It|This|That|These|Those|They|He|She|We|I)\b', sentence, re.IGNORECASE) and
                  len(current_para) >= 2):
                start_new_para = True
                
            # 5. Natural length limit to avoid overly long paragraphs
            elif len(current_para) >= 4:
                start_new_para = True
            
            # Start a new paragraph if needed
            if start_new_para and current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
            
            current_para.append(sentence)
        
        # Add the last paragraph
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs

    def assign_speaker_names(self, speaker_map):
        """Apply custom speaker names to the transcript."""
        if not hasattr(self, 'speakers') or not self.speakers:
            return self.transcript
        # --- FIX: Always update self.speaker_names with the new mapping ---
        if not hasattr(self, 'speaker_names') or not isinstance(self.speaker_names, dict):
            self.speaker_names = {}
        for orig, new in speaker_map.items():
            self.speaker_names[orig] = new
        # Create a formatted transcript with the new speaker names
        formatted_text = []
        for segment in self.speakers:
            original_speaker = segment.get("speaker", "Unknown")
            new_speaker = self.speaker_names.get(original_speaker, original_speaker)
            text = segment.get("text", "")
            formatted_text.append(f"{new_speaker}: {text}")
        return "\n\n".join(formatted_text)

    def create_audio_panel(self):
        """Create the audio processing panel."""
        panel = self.audio_panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # File upload section
        file_box = wx.StaticBox(panel, label="Audio File")
        file_sizer = wx.StaticBoxSizer(file_box, wx.VERTICAL)
        
        file_select_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.audio_file_path = wx.TextCtrl(panel, style=wx.TE_READONLY)
        browse_btn = wx.Button(panel, label="Browse")
        browse_btn.Bind(wx.EVT_BUTTON, self.on_browse_audio)
        
        file_select_sizer.Add(self.audio_file_path, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        file_select_sizer.Add(browse_btn, proportion=0, flag=wx.EXPAND)
        
        file_sizer.Add(file_select_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(file_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Create transcription controls
        transcribe_box = wx.StaticBox(panel, label="Transcription")
        transcribe_sizer = wx.StaticBoxSizer(transcribe_box, wx.VERTICAL)
        
        # Language selector
        lang_sizer = wx.BoxSizer(wx.HORIZONTAL)
        lang_label = wx.StaticText(panel, label="Language:")
        self.language_choice = wx.Choice(panel, choices=[LANGUAGE_DISPLAY_NAMES[lang] for lang in SUPPORTED_LANGUAGES])
        current_lang = self.config_manager.get_language()
        self.language_choice.SetSelection(SUPPORTED_LANGUAGES.index(current_lang))
        
        lang_sizer.Add(lang_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        lang_sizer.Add(self.language_choice, 1, wx.EXPAND)
        
        transcribe_sizer.Add(lang_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Transcribe button
        self.transcribe_btn = wx.Button(panel, label="Transcribe Audio")
        self.transcribe_btn.Bind(wx.EVT_BUTTON, self.on_transcribe)
        self.transcribe_btn.Disable()  # Start disabled
        transcribe_sizer.Add(self.transcribe_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        sizer.Add(transcribe_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Speaker identification
        speaker_box = wx.StaticBox(panel, label="Speaker Identification")
        speaker_sizer = wx.StaticBoxSizer(speaker_box, wx.VERTICAL)
        
        self.identify_speakers_btn = wx.Button(panel, label=self.get_speaker_id_button_label())
        self.identify_speakers_btn.Bind(wx.EVT_BUTTON, self.on_identify_speakers)
        self.identify_speakers_btn.Disable()  # Start disabled
        
        # Set button styling based on PyAnnote availability
        self.update_speaker_id_button_style()
        
        speaker_sizer.Add(self.identify_speakers_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Add help text
        self.speaker_id_help_text = wx.StaticText(panel, label=self.get_speaker_id_help_text())
        self.speaker_id_help_text.SetForegroundColour(wx.Colour(100, 100, 100))  # Gray text
        speaker_sizer.Add(self.speaker_id_help_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        sizer.Add(speaker_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Transcript display
        transcript_box = wx.StaticBox(panel, label="Transcript")
        transcript_sizer = wx.StaticBoxSizer(transcript_box, wx.VERTICAL)
        
        self.transcript_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        self.transcript_text.SetMinSize((400, 200))
        transcript_sizer.Add(self.transcript_text, 1, wx.EXPAND | wx.ALL, 5)
        
        # Add Save and Recall buttons below transcript
        transcript_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_transcript_btn = wx.Button(panel, label="Save Transcript")
        recall_transcript_btn = wx.Button(panel, label="Recall Transcript")
        transcript_btn_sizer.Add(save_transcript_btn, 0, wx.RIGHT, 5)
        transcript_btn_sizer.Add(recall_transcript_btn, 0)
        transcript_sizer.Add(transcript_btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        # Save transcript handler
        def on_save_transcript(event):
            transcript = self.transcript_text.GetValue()
            if not transcript.strip():
                wx.MessageBox("No transcript to save.", "Info", wx.OK | wx.ICON_INFORMATION)
                return
            import os
            import datetime
            base_dir = APP_BASE_DIR if 'APP_BASE_DIR' in globals() else os.path.abspath('.')
            save_dir = os.path.join(base_dir, "Transcripts")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ask user for a name
            dlg = wx.TextEntryDialog(panel, "Enter a name for the transcript file (no extension):", "Save Transcript", f"transcript_{timestamp}")
            if dlg.ShowModal() == wx.ID_OK:
                custom_name = dlg.GetValue().strip()
                if not custom_name:
                    custom_name = f"transcript_{timestamp}"
                filename = os.path.join(save_dir, f"{custom_name}.txt")
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    self.last_saved_transcript_path = filename
                    wx.MessageBox(f"Transcript saved to:\n{filename}", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to save transcript: {e}", "Error", wx.OK | wx.ICON_ERROR)
            dlg.Destroy()

        # Recall transcript handler
        def on_recall_transcript(event):
            import os
            with wx.FileDialog(panel, "Open Transcript", defaultDir=os.path.join(APP_BASE_DIR if 'APP_BASE_DIR' in globals() else os.path.abspath('.'), "Transcripts"),
                              wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
                              style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
                if file_dialog.ShowModal() == wx.ID_CANCEL:
                    return
                path = file_dialog.GetPath()
                if not path or not os.path.exists(path):
                    wx.MessageBox("Selected transcript file does not exist.", "Error", wx.OK | wx.ICON_ERROR)
                    return
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.transcript_text.SetValue(content)
                    wx.MessageBox(f"Transcript loaded from:\n{path}", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to load transcript: {e}", "Error", wx.OK | wx.ICON_ERROR)

        save_transcript_btn.Bind(wx.EVT_BUTTON, on_save_transcript)
        recall_transcript_btn.Bind(wx.EVT_BUTTON, on_recall_transcript)
        
        # Speaker list
        speaker_list_box = wx.StaticBox(panel, label="Speakers")
        speaker_list_sizer = wx.StaticBoxSizer(speaker_list_box, wx.VERTICAL)
        
        self.speaker_list = wx.ListCtrl(panel, style=wx.LC_REPORT)
        self.speaker_list.InsertColumn(0, "ID", width=50)
        self.speaker_list.InsertColumn(1, "Name", width=150)
        speaker_list_sizer.Add(self.speaker_list, 1, wx.EXPAND | wx.ALL, 5)
        
        # Edit speakers button
        rename_speaker_btn = wx.Button(panel, label="Rename Speaker")
        rename_speaker_btn.Bind(wx.EVT_BUTTON, self.on_rename_speaker)
        speaker_list_sizer.Add(rename_speaker_btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Create a horizontal sizer for transcript and speaker list
        transcript_speaker_sizer = wx.BoxSizer(wx.HORIZONTAL)
        transcript_speaker_sizer.Add(transcript_sizer, 2, wx.EXPAND | wx.RIGHT, 5)
        transcript_speaker_sizer.Add(speaker_list_sizer, 1, wx.EXPAND)
        
        sizer.Add(transcript_speaker_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        # Summarization section
        summary_box = wx.StaticBox(panel, label="Summarization")
        summary_sizer = wx.StaticBoxSizer(summary_box, wx.VERTICAL)
        
        # Template selection
        template_sizer = wx.BoxSizer(wx.HORIZONTAL)
        template_label = wx.StaticText(panel, label="Template:")
        
        # Get templates from config
        template_names = list(self.config_manager.get_templates().keys())
        self.template_choice = wx.Choice(panel, choices=["None"] + template_names)
        self.template_choice.SetSelection(0)
        self.template_choice.Bind(wx.EVT_CHOICE, self.on_template_selected)
        
        template_sizer.Add(template_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        template_sizer.Add(self.template_choice, 1, wx.EXPAND)
        
        summary_sizer.Add(template_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Summarize button
        self.summarize_btn = wx.Button(panel, label="Summarize Transcript")
        self.summarize_btn.Bind(wx.EVT_BUTTON, self.on_summarize)
        self.summarize_btn.Disable()  # Start disabled
        summary_sizer.Add(self.summarize_btn, 0, wx.EXPAND | wx.ALL, 5)

        # Add Save and Recall buttons for summary
        summary_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_summary_btn = wx.Button(panel, label="Save Summary")
        recall_summary_btn = wx.Button(panel, label="Recall Summary")
        summary_btn_sizer.Add(save_summary_btn, 0, wx.RIGHT, 5)
        summary_btn_sizer.Add(recall_summary_btn, 0)
        summary_sizer.Add(summary_btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        # Save summary handler
        def on_save_summary(event):
            # Try to get summary from the last summary dialog if possible, else from transcript_text
            summary = None
            if hasattr(self, 'last_summary_text') and self.last_summary_text:
                summary = self.last_summary_text
            else:
                summary = self.transcript_text.GetValue()
            if not summary.strip():
                wx.MessageBox("No summary to save.", "Info", wx.OK | wx.ICON_INFORMATION)
                return
            import os
            import datetime
            base_dir = APP_BASE_DIR if 'APP_BASE_DIR' in globals() else os.path.abspath('.')
            save_dir = os.path.join(base_dir, "Summaries")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ask user for a name
            dlg = wx.TextEntryDialog(panel, "Enter a name for the summary file (no extension):", "Save Summary", f"summary_{timestamp}")
            if dlg.ShowModal() == wx.ID_OK:
                custom_name = dlg.GetValue().strip()
                if not custom_name:
                    custom_name = f"summary_{timestamp}"
                filename = os.path.join(save_dir, f"{custom_name}.txt")
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    self.last_saved_summary_path = filename
                    wx.MessageBox(f"Summary saved to:\n{filename}", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to save summary: {e}", "Error", wx.OK | wx.ICON_ERROR)
            dlg.Destroy()

        # Recall summary handler
        def on_recall_summary(event):
            import os
            with wx.FileDialog(panel, "Open Summary", defaultDir=os.path.join(APP_BASE_DIR if 'APP_BASE_DIR' in globals() else os.path.abspath('.'), "Summaries"),
                              wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
                              style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
                if file_dialog.ShowModal() == wx.ID_CANCEL:
                    return
                path = file_dialog.GetPath()
                if not path or not os.path.exists(path):
                    wx.MessageBox("Selected summary file does not exist.", "Error", wx.OK | wx.ICON_ERROR)
                    return
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Store for later use
                    self.last_summary_text = content
                    # Show in a dialog (like show_summary_dialog)
                    dlg = wx.Dialog(panel, title="Summary", size=(600, 400))
                    sizer = wx.BoxSizer(wx.VERTICAL)
                    text_ctrl = wx.TextCtrl(dlg, style=wx.TE_MULTILINE | wx.TE_READONLY)
                    text_ctrl.SetValue(content)
                    sizer.Add(text_ctrl, 1, wx.EXPAND | wx.ALL, 10)
                    btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
                    close_btn = wx.Button(dlg, wx.ID_CLOSE)
                    btn_sizer.Add(close_btn, 0, wx.ALL, 5)
                    sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 5)
                    dlg.SetSizer(sizer)
                    close_btn.Bind(wx.EVT_BUTTON, lambda event: dlg.EndModal(wx.ID_CLOSE))
                    dlg.ShowModal()
                    dlg.Destroy()
                    wx.MessageBox(f"Summary loaded from:\n{path}", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Failed to load summary: {e}", "Error", wx.OK | wx.ICON_ERROR)

        save_summary_btn.Bind(wx.EVT_BUTTON, on_save_summary)
        recall_summary_btn.Bind(wx.EVT_BUTTON, on_recall_summary)
        
        sizer.Add(summary_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        
        return panel

    def bind_events(self):
        """Bind events to handlers."""
        # Enter key in prompt input
        if hasattr(self, 'prompt_input'):
            self.prompt_input.Bind(wx.EVT_TEXT_ENTER, self.on_send_prompt)
        
    def on_close(self, event):
        """Handle application close event."""
        self.Destroy()
        
    def update_status(self, message, percent=None):
        """Update status bar with message and optional progress percentage."""
        if percent is not None:
            self.status_bar.SetStatusText(f"{message} ({percent:.0f}%)")
        else:
            self.status_bar.SetStatusText(message)
            
    def on_identify_speakers(self, event):
        """Handle speaker identification button click."""
        if hasattr(self, 'transcript') and self.transcript:
            # Make sure we have a client available
            if self.client is None:
                global client
                self.client = client

            has_token = bool(self.config_manager.get_pyannote_token())
            if has_token and hasattr(self, 'last_audio_path') and self.last_audio_path:
                # Use advanced speaker identification with diarization
                threading.Thread(
                    target=self.identify_speakers_with_diarization,
                    args=(self.last_audio_path, self.transcript),
                    daemon=True
                ).start()
                
                # Add a timer to check for completion and update UI
                self.speaker_id_timer = wx.Timer(self)
                self.Bind(wx.EVT_TIMER, self.check_speaker_id_complete, self.speaker_id_timer)
                self.speaker_id_timer.Start(1000)  # Check every second
            else:
                # Use basic speaker identification
                speakers = self.identify_speakers_simple(self.transcript)
                
                # Update UI with results
                if hasattr(self, 'speakers') and self.speakers:
                    # Format transcript with speaker names
                    speaker_transcript = self.assign_speaker_names({s["speaker"]: s["speaker"] for s in self.speakers})
                    self.transcript = speaker_transcript
                    
                    # Update the UI
                    self.update_transcript_display()
                    self.update_speaker_list()
        else:
            wx.MessageBox("No transcript available. Please transcribe audio first.", 
                         "No Transcript", wx.OK | wx.ICON_INFORMATION)
    
    def check_speaker_id_complete(self, event):
        """Check if speaker identification is complete and update UI if it is."""
        if hasattr(self, 'speakers') and self.speakers:
            # Stop the timer
            self.speaker_id_timer.Stop()
            
            # Format transcript with speaker names
            speaker_transcript = self.assign_speaker_names({s["speaker"]: s["speaker"] for s in self.speakers})
            self.transcript = speaker_transcript
            
            # Update the UI
            self.update_transcript_display()
            self.update_speaker_list()
    
    def on_browse_audio(self, event):
        """Handle audio file browse button."""
        wildcard = (
            "Audio files|*.flac;*.m4a;*.mp3;*.mp4;*.mpeg;*.mpga;*.oga;*.ogg;*.wav;*.webm|"
            "FLAC files (*.flac)|*.flac|"
            "M4A files (*.m4a)|*.m4a|"
            "MP3 files (*.mp3)|*.mp3|"
            "MP4 files (*.mp4)|*.mp4|"
            "OGG files (*.ogg;*.oga)|*.ogg;*.oga|"
            "WAV files (*.wav)|*.wav|"
            "All files (*.*)|*.*"
        )
        
        with wx.FileDialog(self, "Choose an audio file", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
                
            path = file_dialog.GetPath()
            
            # Validate file extension
            file_ext = os.path.splitext(path)[1].lower()
            supported_formats = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
            
            if file_ext not in supported_formats:
                # If user selected "All files" and chose an unsupported format
                wx.MessageBox(
                    f"The selected file has an unsupported format: {file_ext}\n"
                    f"Supported formats are: {', '.join(supported_formats)}", 
                    "Unsupported Format", 
                    wx.OK | wx.ICON_WARNING
                )
                return
                
            # Check file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            # Removed 25MB Whisper API limit check to allow large files for Azure Speech SDK
            self.audio_file_path.SetValue(path)
            self.update_status(f"Selected audio file: {os.path.basename(path)} ({file_size_mb:.1f}MB)", percent=0)
            self.update_button_states()
            
    def on_transcribe(self, event):
        """Handle audio transcription."""
        if not self.audio_file_path.GetValue():
            wx.MessageBox("Please select an audio file first.", "No File Selected", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Check if Azure Speech configuration is set
        if not self.config_manager.get_azure_speech_api_key():
            self.show_azure_speech_config_dialog()
            return
            
        # Check if Azure OpenAI configuration is set
        if not self.config_manager.get_azure_api_key() or not self.config_manager.get_azure_endpoint():
            wx.MessageBox("Please set your Azure OpenAI configuration in the Settings tab.", "Configuration Required", wx.OK | wx.ICON_INFORMATION)
            return
            
        # Get the whisper deployment name
        whisper_deployment = self.config_manager.get_azure_deployment("whisper")
        if not whisper_deployment:
            wx.MessageBox("Whisper deployment name not configured in Azure OpenAI settings.", "Configuration Error", wx.OK | wx.ICON_ERROR)
            return
            
        # Get language selection and ensure it's valid
        selection = self.language_choice.GetSelection()
        if 0 <= selection < len(SUPPORTED_LANGUAGES):
            language = SUPPORTED_LANGUAGES[selection]
        else:
            language = "en"  # Default to English if selection is invalid
        
        # Save language choice to config
        self.config_manager.set_language(language)
        
        # Store the audio file path in the AudioProcessor
        # This ensures it's available for diarization later
        self.audio_processor.audio_file_path = self.audio_file_path.GetValue()
        
        # Update status message with display name
        self.update_status(f"Transcribing in {LANGUAGE_DISPLAY_NAMES[language]}...", percent=0)
        
        # Disable buttons during processing
        self.transcribe_btn.Disable()
        self.identify_speakers_btn.Disable()
        self.summarize_btn.Disable()
        
        # Start transcription in a separate thread
        threading.Thread(target=self.transcribe_thread, args=(self.audio_file_path.GetValue(), language)).start()
        
    def transcribe_thread(self, file_path, language):
        """Thread function for audio transcription using Azure Speech SDK with smart chunking."""
        try:
            # Check if Azure Speech configuration is set
            if not self.config_manager.get_azure_speech_api_key():
                wx.CallAfter(self.show_azure_speech_config_dialog)
                return
                
            # Get file extension for better error reporting
            file_ext = os.path.splitext(file_path)[1].lower()
            supported_formats = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
            
            # Check if file format is supported
            if file_ext not in supported_formats:
                # Attempt to convert to WAV if FFmpeg is available
                if self._is_ffmpeg_available():
                    wx.CallAfter(self.update_status, f"Converting {file_ext} file to WAV format...", percent=10)
                    try:
                        output_path = os.path.splitext(file_path)[0] + ".wav"
                        subprocess.run(
                            ["ffmpeg", "-i", file_path, "-y", output_path],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            check=True
                        )
                        wx.CallAfter(self.update_status, f"Conversion complete. Starting transcription...", percent=20)
                        file_path = output_path
                    except Exception as e:
                        error_msg = f"Error converting file: {str(e)}"
                        wx.CallAfter(wx.MessageBox, error_msg, "Conversion Error", wx.OK | wx.ICON_ERROR)
                        wx.CallAfter(self.update_status, "Ready", percent=0)
                        wx.CallAfter(self.transcribe_btn.Enable)
                        return
                else:
                    error_msg = f"The file format {file_ext} is not supported.\n\nSupported formats: {', '.join(supported_formats)}"
                    wx.CallAfter(wx.MessageBox, error_msg, "Unsupported Format", wx.OK | wx.ICON_ERROR)
                    wx.CallAfter(self.update_status, "Ready", percent=0)
                    wx.CallAfter(self.transcribe_btn.Enable)
                    return
            
            # For M4A files that often have issues, try to convert to WAV if FFmpeg is available
            if file_ext == '.m4a' and self._is_ffmpeg_available():
                wx.CallAfter(self.update_status, "Converting M4A to WAV format for better compatibility...", percent=10)
                try:
                    output_path = os.path.splitext(file_path)[0] + ".wav"
                    subprocess.run(
                        ["ffmpeg", "-i", file_path, "-y", output_path],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    wx.CallAfter(self.update_status, "Conversion complete. Starting transcription...", percent=20)
                    file_path = output_path
                except Exception as e:
                    # Continue with original file, just log the warning
                    wx.CallAfter(self.update_status, f"Warning: Could not convert M4A file. Attempting to use original file.", percent=20)
            
            # Use the smart chunking transcription method
            response = self.audio_processor.transcribe_audio(file_path, language)
            
            # Transfer word timestamps from audio_processor to MainFrame for diarization
            if hasattr(self.audio_processor, 'word_by_word') and self.audio_processor.word_by_word:
                self.word_by_word = self.audio_processor.word_by_word
                print(f"[DEBUG] Transferred {len(self.word_by_word)} word timestamps to MainFrame for diarization")
                print(f"[DEBUG] Sample word data: {self.word_by_word[:3]}")
            else:
                print(f"[DEBUG] No word timestamps found in audio_processor")
                print(f"[DEBUG] audio_processor.word_by_word exists: {hasattr(self.audio_processor, 'word_by_word')}")
                if hasattr(self.audio_processor, 'word_by_word'):
                    print(f"[DEBUG] audio_processor.word_by_word content: {self.audio_processor.word_by_word}")
            
            # Add a note about speaker identification at the top of the transcript
            transcription_notice = "--- TRANSCRIPTION COMPLETE ---\n" + \
                                  "To identify speakers in this transcript, click the 'Identify Speakers' button below.\n\n"
            
            # Ensure transcript is not None before concatenation
            transcript_text = self.audio_processor.transcript if self.audio_processor.transcript is not None else ""
            wx.CallAfter(self.transcript_text.SetValue, transcription_notice + transcript_text)
            # Set the transcript attribute in the MainFrame instance
            wx.CallAfter(self.set_transcript, transcript_text)
            wx.CallAfter(self.update_button_states)
            wx.CallAfter(self.update_status, f"Transcription complete: {len(transcript_text)} characters", percent=100)
            
            # Show a dialog informing the user to use speaker identification
            wx.CallAfter(self.show_speaker_id_hint)
            
        except FileNotFoundError as e:
            wx.CallAfter(wx.MessageBox, f"File not found: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)
        except ValueError as e:
            error_msg = str(e)
            title = "Format Error"
            
            # Special handling for common error cases
            if 'ffprobe' in error_msg or 'ffmpeg' in error_msg:
                title = "FFmpeg Missing"
                error_msg = error_msg.replace('[Errno 2] No such file or directory:', 'Missing required component:')
                # Installation instructions are already in the error message from _get_ffmpeg_install_instructions
            elif file_ext == '.m4a' and 'Invalid file format' in error_msg:
                error_msg = (
                    "There was an issue with your M4A file. Some M4A files have compatibility issues.\n\n"
                    "Possible solutions:\n"
                    "1. Install FFmpeg on your system (required for m4a processing)\n"
                    "2. Convert the file to WAV or MP3 format manually\n"
                    "3. Try a different M4A file (some are more compatible than others)"
                )
                title = "M4A Compatibility Issue"
                wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
            else:
                wx.CallAfter(wx.MessageBox, error_msg, title, wx.OK | wx.ICON_ERROR)
        except Exception as e:
            error_msg = str(e)
            if 'ffprobe' in error_msg or 'ffmpeg' in error_msg:
                # Handle FFmpeg-related errors not caught by previous handlers
                install_instructions = self.audio_processor._get_ffmpeg_install_instructions()
                error_msg = f"FFmpeg/FFprobe is required but not found. Please install it to process audio files.\n\n{install_instructions}"
                wx.CallAfter(wx.MessageBox, error_msg, "FFmpeg Required", wx.OK | wx.ICON_ERROR)
            else:
                wx.CallAfter(wx.MessageBox, f"Transcription error: {error_msg}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            wx.CallAfter(self.transcribe_btn.Enable)
            wx.CallAfter(self.update_status, "Ready", percent=0)

    def show_speaker_id_hint(self):
        """Show a hint dialog about using speaker identification."""
        # Check if PyAnnote is available
        if PYANNOTE_AVAILABLE:
            message = (
                "Transcription is complete!\n\n"
                "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
                "This system will use advanced audio-based speaker diarization to detect different "
                "speakers by analyzing voice characteristics (pitch, tone, speaking style) from the "
                "original audio file.\n\n"
                "This approach is significantly more accurate than text-based analysis since it "
                "uses the actual voice patterns to distinguish between speakers."
            )
        else:
            message = (
                "Transcription is complete!\n\n"
                "To identify different speakers in this transcript, click the 'Identify Speakers' button.\n\n"
                "Currently, the system will analyze the text patterns to detect different speakers.\n\n"
                "For more accurate speaker identification, consider installing PyAnnote which uses "
                "audio analysis to distinguish speakers based on their voice characteristics. "
                "Click 'Yes' for installation instructions."
            )
            
        dlg = wx.MessageDialog(
            self,
            message,
            "Speaker Identification",
            wx.OK | (wx.CANCEL | wx.YES_NO if not PYANNOTE_AVAILABLE else wx.OK) | wx.ICON_INFORMATION
        )
        
        result = dlg.ShowModal()
        dlg.Destroy()
        
        # If user wants to install PyAnnote
        if result == wx.ID_YES:
            self.show_pyannote_setup_guide()
        
        # Highlight the identify speakers button
        self.identify_speakers_btn.SetFocus()

    def show_format_info(self):
        """Show information about supported audio formats and requirements."""
        # Check if we already showed this info
        if self.config_manager.config.get("shown_format_info", False):
            return
            
        # Check if FFmpeg is available
        ffmpeg_available = self._is_ffmpeg_available()
        ffmpeg_missing = not ffmpeg_available
        
        # List required tools
        needed_tools = []
        if ffmpeg_missing:
            needed_tools.append("FFmpeg")
            
        # No tools needed, we're all set
        if not needed_tools:
            if hasattr(sys, '_MEIPASS'):
                # Running as a PyInstaller bundle - FFmpeg is bundled
                self.update_status("FFmpeg is bundled with this application", percent=0)
                return
            else:
                # Running from source with FFmpeg installed
                self.update_status("All audio tools available", percent=0)
                return
        
        # Generate installation instructions
        ffmpeg_install = self._get_ffmpeg_install_instructions() if hasattr(self, '_get_ffmpeg_install_instructions') else self.audio_processor._get_ffmpeg_install_instructions()
        
        # Prepare message
        msg = (
            "Audio Format Compatibility Information:\n\n"
            "• Directly supported formats: WAV, MP3, FLAC, OGG\n"
            "• M4A files require FFmpeg for conversion\n\n"
            "For better audio file compatibility, especially with M4A files, "
            f"you need to install the following tools:\n\n{', '.join(needed_tools)}\n\n"
        )
        
        if ffmpeg_missing:
            if hasattr(sys, '_MEIPASS'):
                # Running as a PyInstaller bundle, but FFmpeg directory not found
                msg = (
                    "There was an issue with the bundled FFmpeg. This is unusual and indicates "
                    "a problem with the application package. Please contact support.\n\n"
                    "As a workaround, you can install FFmpeg manually:\n\n"
                )
                msg += ffmpeg_install
            else:
                # Running from source without FFmpeg
                msg += f"FFmpeg installation instructions:\n{ffmpeg_install}\n\n"
                msg += "FFmpeg is required for processing M4A files. Without it, M4A transcription will likely fail."
        
        self.update_status("FFmpeg required for M4A support - please install it", percent=0)
        
        # Always show FFmpeg warning because it's critical
        if ffmpeg_missing:
            wx.MessageBox(msg, "FFmpeg Required for M4A Files", wx.OK | wx.ICON_WARNING)
            self.config_manager.config["shown_format_info"] = True
            self.config_manager.save_config()
        # Only show other warnings if not shown before
        elif not self.config_manager.config.get("shown_format_info", False):
            wx.MessageBox(msg, "Audio Format Information", wx.OK | wx.ICON_INFORMATION)
            self.config_manager.config["shown_format_info"] = True
            self.config_manager.save_config()

    def _is_ffmpeg_available(self):
        """Check if ffmpeg is available on the system."""
        # First try the bundled FFmpeg
        bundled_ffmpeg = None
        
        # Get the application directory
        if hasattr(sys, '_MEIPASS'):
            # Running as a PyInstaller bundle
            base_dir = sys._MEIPASS
            bundled_ffmpeg_dir = os.path.join(base_dir, 'ffmpeg')
            
            if os.path.exists(bundled_ffmpeg_dir):
                # Check for ffmpeg executable based on platform
                if platform.system() == 'Windows':
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg.exe')
                else:
                    bundled_ffmpeg = os.path.join(bundled_ffmpeg_dir, 'ffmpeg')
                
                if bundled_ffmpeg and os.path.exists(bundled_ffmpeg):
                    # Update PATH environment variable to include bundled ffmpeg
                    os.environ["PATH"] = bundled_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                    return True
        
        # Then try the standard way (using PATH)
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # On macOS, try common Homebrew path
            if platform.system() == 'darwin':
                common_mac_paths = [
                    "/opt/homebrew/bin/ffmpeg",
                    "/usr/local/bin/ffmpeg",
                    "/opt/local/bin/ffmpeg"  # MacPorts
                ]
                
                for path in common_mac_paths:
                    try:
                        subprocess.run(
                            [path, "-version"],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            check=True
                        )
                        # If found, update the PATH environment variable
                        os.environ["PATH"] = os.path.dirname(path) + ":" + os.environ.get("PATH", "")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
            
            return False

    def check_pyannote(self):
        """Check if PyAnnote is available and show installation instructions if not."""
        if not PYANNOTE_AVAILABLE:
            dlg = wx.MessageDialog(
                self,
                "PyAnnote is not installed. PyAnnote provides more accurate speaker diarization "
                "by analyzing audio directly, rather than just text.\n\n"
                "To install PyAnnote and set it up, click 'Yes' for detailed instructions.",
                "Speaker Diarization Enhancement",
                wx.YES_NO | wx.ICON_INFORMATION
            )
            if dlg.ShowModal() == wx.ID_YES:
                self.show_pyannote_setup_guide()
            dlg.Destroy()
    
    def show_pyannote_setup_guide(self):
        """Show detailed setup instructions for PyAnnote."""
        dlg = wx.Dialog(self, title="PyAnnote Setup Guide", size=(650, 550))
        
        panel = wx.Panel(dlg)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create a styled text control for better formatting
        text = wx.TextCtrl(
            panel, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 400)
        )
        
        # Set up the instructions
        guide = """PYANNOTE SETUP GUIDE

Step 1: Install Required Dependencies
--------------------------------------
Run the following commands in your terminal:

pip install torch torchaudio
pip install pyannote.audio

Step 2: Get HuggingFace Access Token
------------------------------------
1. Create a HuggingFace account at https://huggingface.co/join
2. Go to https://huggingface.co/pyannote/speaker-diarization
3. Accept the user agreement
4. Go to https://huggingface.co/settings/tokens
5. Create a new token with READ access
6. Copy the token

Step 3: Configure the Application
--------------------------------
1. After installing, restart this application
2. Go to the Settings tab
3. Paste your token in the "PyAnnote Speaker Diarization" section
4. Click "Save Token"
5. Return to the Audio Processing tab
6. Click "Identify Speakers" to use audio-based speaker identification

Important Notes:
---------------
• PyAnnote requires at least 4GB of RAM
• GPU acceleration (if available) will make processing much faster
• For best results, use high-quality audio with minimal background noise
• The first run may take longer as models are downloaded

Troubleshooting:
---------------
• If you get CUDA errors, try installing a compatible PyTorch version for your GPU
• If you get "Access Denied" errors, check that your token is valid and you've accepted the license agreement
• For long audio files (>10 min), processing may take several minutes
"""
        
        # Add the text with some styling
        text.SetValue(guide)
        
        # Style the headers
        text.SetStyle(0, 19, wx.TextAttr(wx.BLUE, wx.NullColour, wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)))
        
        # Find all the section headers and style them
        for section in ["Step 1:", "Step 2:", "Step 3:", "Important Notes:", "Troubleshooting:"]:
            start = guide.find(section)
            if start != -1:
                end = start + len(section)
                text.SetStyle(start, end, wx.TextAttr(wx.Colour(128, 0, 128), wx.NullColour, wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)))
        
        # Add to sizer
        sizer.Add(text, 1, wx.EXPAND | wx.ALL, 10)
        
        # Add buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Add a button to copy installation commands
        copy_btn = wx.Button(panel, label="Copy Installation Commands")
        copy_btn.Bind(wx.EVT_BUTTON, lambda e: self.copy_to_clipboard("pip install torch torchaudio\npip install pyannote.audio"))
        btn_sizer.Add(copy_btn, 0, wx.RIGHT, 10)
        
        # Add a button to open HuggingFace token page
        hf_btn = wx.Button(panel, label="Open HuggingFace Token Page")
        hf_btn.Bind(wx.EVT_BUTTON, lambda e: wx.LaunchDefaultBrowser("https://huggingface.co/settings/tokens"))
        btn_sizer.Add(hf_btn, 0, wx.RIGHT, 10)
        
        # Add button to go to settings tab
        settings_btn = wx.Button(panel, label="Go to Settings Tab")
        settings_btn.Bind(wx.EVT_BUTTON, lambda e: (self.notebook.SetSelection(2), dlg.EndModal(wx.ID_CLOSE)))
        btn_sizer.Add(settings_btn, 0, wx.RIGHT, 10)
        
        # Add close button
        close_btn = wx.Button(panel, wx.ID_CLOSE)
        close_btn.Bind(wx.EVT_BUTTON, lambda e: dlg.EndModal(wx.ID_CLOSE))
        btn_sizer.Add(close_btn, 0)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        panel.SetSizer(sizer)
        dlg.ShowModal()
        dlg.Destroy()
    
    def copy_to_clipboard(self, text):
        """Copy text to clipboard."""
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(text))
            wx.TheClipboard.Close()
            wx.MessageBox("Commands copied to clipboard", "Copied", wx.OK | wx.ICON_INFORMATION)

    def on_template_selected(self, event):
        """Handle selection of a template."""
        if not hasattr(self, 'template_list'):
            return
            
        selected = self.template_list.GetSelection()
        if selected == wx.NOT_FOUND:
            return
            
        template_name = self.template_list.GetString(selected)
        templates = self.config_manager.get_templates()
        
        if template_name in templates:
            self.template_content_input.SetValue(templates[template_name])
        else:
            self.template_content_input.Clear()

    def _quick_consistency_check(self):
        """Ultra-quick consistency check for short files"""
        if len(self.speakers) < 3:
            return
            
        # Look for isolated speaker segments
        for i in range(1, len(self.speakers) - 1):
            prev_speaker = self.speakers[i-1]["speaker"]
            curr_speaker = self.speakers[i]["speaker"]
            next_speaker = self.speakers[i+1]["speaker"]
            
            # If current speaker is sandwiched between different speakers
            if prev_speaker == next_speaker and curr_speaker != prev_speaker:
                # Fix the segment only if very short (likely error)
                if len(self.speakers[i]["text"].split()) < 15:
                    self.speakers[i]["speaker"] = prev_speaker

    def _process_audio_in_chunks(self, pipeline, audio_file, total_duration, chunk_size):
        """Process long audio files in chunks to optimize memory usage and speed."""
        from pyannote.core import Segment, Annotation
        import concurrent.futures
        from threading import Lock
        
        # Initialize a combined annotation object
        combined_diarization = Annotation()
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(total_duration / chunk_size))
        self.update_status(f"Processing audio in {num_chunks} chunks...", percent=0.1)
        
        # Optimize number of workers based on file length and available memory
        # More chunks = more workers (up to cpu_count), but limit for very long files
        # to avoid excessive memory usage
        cpu_count = os.cpu_count() or 4
        
        # Determine optimal number of workers based on audio duration
        if total_duration > 7200:  # > 2 hours
            max_workers = max(2, min(cpu_count - 1, 4))  # For very long files, use fewer workers to avoid memory issues
        else:
            max_workers = max(2, min(cpu_count, 6))  # Use more workers for shorter files
        
        # Function to process a single chunk
        def process_chunk(chunk_idx, start_time):
            end_time = min(start_time + chunk_size, total_duration)
            chunk_segment = Segment(start=start_time, end=end_time)
            
            # Create optimized pipeline parameters for this chunk
            pipeline_params = {
                "segmentation": {
                    "min_duration_on": 0.25,
                    "min_duration_off": 0.25,
                },
                "clustering": {
                    "min_cluster_size": 6,
                    "method": "centroid"
                },
                "segmentation_batch_size": 64,  # Larger batch size for speed
                "embedding_batch_size": 64      # Larger batch size for speed
            }
            
            # Apply optimized parameters
            pipeline.instantiate(pipeline_params)
            
            # Apply diarization to this chunk
            chunk_result = pipeline(audio_file, segmentation=chunk_segment)
            
            return chunk_idx, chunk_result
        
        # Create thread lock for combining results
        lock = Lock()
        chunk_results = [None] * num_chunks
        progress_counter = [0]
        
        # Function to update progress
        def update_progress():
            with lock:
                progress_counter[0] += 1
                progress = 0.15 + (progress_counter[0] / num_chunks) * 0.7
                self.update_status(f"Processed {progress_counter[0]}/{num_chunks} chunks...", percent=progress)
        
        try:
            # Process chunks in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks for processing
                futures = {}
                for i in range(num_chunks):
                    start_time = i * chunk_size
                    future = executor.submit(process_chunk, i, start_time)
                    futures[future] = i
                
                # Handle results as they complete
                for future in concurrent.futures.as_completed(futures):
                    chunk_idx, chunk_result = future.result()
                    chunk_results[chunk_idx] = chunk_result
                    update_progress()
            
            # Combine results
            self.update_status(f"Combining results from {num_chunks} chunks...", percent=0.85)
            
            # Create a new combined annotation
            combined_diarization = Annotation()
            
            # Merge annotations while preserving speaker identities across chunks
            speaker_mapping = {}  # Map to track speakers across chunks
            global_speaker_count = 0
            
            # First pass: collect all speaker embeddings
            all_speakers = {}
            for chunk_idx, chunk_result in enumerate(chunk_results):
                if chunk_result is None:
                    continue
                    
                # Get speakers from this chunk
                for segment, track, speaker in chunk_result.itertracks(yield_label=True):
                    if speaker not in all_speakers:
                        all_speakers[speaker] = []
                        
                    # Store segment information
                    all_speakers[speaker].append((chunk_idx, segment, track))
            
            # Second pass: establish global speaker mapping based on temporal proximity
            for local_speaker, segments in all_speakers.items():
                # Sort segments by time
                segments.sort(key=lambda x: x[1].start)
                
                # Check if this speaker should be mapped to an existing global speaker
                mapped = False
                for global_speaker, global_segments in speaker_mapping.items():
                    # Check if any segment overlaps or is very close to an existing global speaker
                    for chunk_idx, segment, track in segments:
                        for g_chunk_idx, g_segment, g_track in global_segments:
                            # Consider speakers the same if segments are close in time
                            if (abs(segment.start - g_segment.end) < 2.0 or 
                                abs(g_segment.start - segment.end) < 2.0):
                                mapped = True
                                break
                                
                        if mapped:
                            break
                        
                    if mapped:
                        # Add segments to existing global speaker
                        speaker_mapping[global_speaker].extend(segments)
                        break
                        
                if not mapped:
                    # Create new global speaker
                    global_speaker = f"SPEAKER_{global_speaker_count}"
                    global_speaker_count += 1
                    speaker_mapping[global_speaker] = segments
            
            # Third pass: add all segments to the combined diarization
            for global_speaker, segments in speaker_mapping.items():
                for chunk_idx, segment, track in segments:
                    combined_diarization[segment, track] = global_speaker
            
            # Return the combined diarization
            return combined_diarization
            
        except Exception as e:
            self.update_status(f"Error in parallel processing: {str(e)}", percent=0.5)
            
            # Fall back to simpler sequential processing
            self.update_status("Falling back to sequential processing...", percent=0.5)
            
            # Process chunks sequentially as fallback
            combined_diarization = Annotation()
            for i in range(num_chunks):
                start_time = i * chunk_size
                end_time = min(start_time + chunk_size, total_duration)
                segment = Segment(start=start_time, end=end_time)
                
                # Update progress
                progress = 0.5 + (i / num_chunks) * 0.4
                self.update_status(f"Processing chunk {i+1}/{num_chunks} (sequential mode)...", percent=progress)
                
                # Process this chunk
                chunk_result = pipeline(audio_file, segmentation=segment)
                
                # Add results to combined annotation
                for s, t, spk in chunk_result.itertracks(yield_label=True):
                    # Create a global speaker ID
                    global_spk = f"SPEAKER_{spk.split('_')[-1]}"
                    combined_diarization[s, t] = global_spk
            
            return combined_diarization

    def identify_speakers_with_diarization(self, audio_file_path, transcript):
        """Professional Multi-Speaker Diarization using pyannote.audio 3.1+ (fully automatic, robust, global clustering, word-level mapping)."""
        import numpy as np
        import torch
        from pyannote.audio import Pipeline
        from sklearn.cluster import AgglomerativeClustering
        import soundfile as sf
        import math
        import traceback
        import logging

        # --- ENSURE WAV FOR DIARIZATION ---
        wav_path, is_temp_wav = self.ensure_wav_for_diarization(audio_file_path)

        # Setup logger
        logger = logging.getLogger("DiarizationLogger")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("diarization_debug.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        if not logger.hasHandlers():
            logger.addHandler(handler)
        else:
            logger.handlers.clear()
            logger.addHandler(handler)

        self.update_status("Starting advanced speaker diarization...", percent=0.05)
        try:
            token = self.config_manager.get_pyannote_token() if self.config_manager else None
            if not token:
                self.update_status("No HuggingFace token for pyannote. Falling back to basic logic.", percent=0)
                return self.identify_speakers_simple(transcript)
            
            # Validate that we have the required data
            if not hasattr(self, 'word_by_word') or not self.word_by_word:
                logger.warning("No word timestamps available, but continuing with diarization")
            
            if not wav_path or not os.path.exists(wav_path):
                logger.error(f"Audio file not found: {wav_path}")
                raise RuntimeError(f"Audio file not found: {wav_path}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            pipeline.to(device)

            # Step 1: Chunking (reuse existing logic)
            chunk_duration = 600  # 10 minutes
            with sf.SoundFile(wav_path) as f:
                total_duration = f.frames / f.samplerate
                sr = f.samplerate
            num_chunks = math.ceil(total_duration / chunk_duration)
            chunk_infos = []
            for i in range(num_chunks):
                start = i * chunk_duration
                end = min((i + 1) * chunk_duration, total_duration)
                chunk_infos.append((start, end))

            diarizations = []
            all_embeddings = []
            all_segments = []
            self.update_status(f"Processing {num_chunks} chunks for diarization...", percent=0.10)

            logger.info(f"Total duration: {total_duration:.2f}s, Sample rate: {sr}, Num chunks: {num_chunks}")

            for idx, (start, end) in enumerate(chunk_infos):
                self.update_status(f"Diarizing chunk {idx+1}/{num_chunks}...", percent=0.10 + 0.7 * (idx / num_chunks))
                start_frame = int(start * sr)
                end_frame = int(end * sr)
                waveform, _ = sf.read(wav_path, start=start_frame, stop=end_frame)
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)
                waveform = waveform.astype(np.float32)
                diarization, embeddings = pipeline({
                    "waveform": torch.tensor(waveform).unsqueeze(0).to(device),
                    "sample_rate": sr
                }, return_embeddings=True)
                diarizations.append(diarization)
                segs = []
                embs = []
                local_speaker_labels = list(diarization.labels())
                logger.info(f"Chunk {idx}: start={start:.2f}, end={end:.2f}, num_local_speakers={len(local_speaker_labels)}")
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    segs.append((segment.start + start, segment.end + start, speaker))
                    logger.debug(f"  Local speaker: {speaker}, segment=({segment.start+start:.2f}, {segment.end+start:.2f})")
                if embeddings is not None:
                    for i, speaker in enumerate(local_speaker_labels):
                        # Find all segments for this speaker in this chunk
                        speaker_segments = [(segment.start + start, segment.end + start) for segment, _, spk in diarization.itertracks(yield_label=True) if spk == speaker]
                        # Use the first segment for logging
                        seg_start, seg_end = speaker_segments[0] if speaker_segments else (start, end)
                        segment_id = (idx, speaker, seg_start, seg_end)
                        embs.append((segment_id, embeddings[i]))
                        logger.debug(f"    Embedding for (chunk={idx}, local_speaker={speaker}, seg=({seg_start:.2f},{seg_end:.2f}))")
                all_segments.extend([(s, e, spk) for s, e, spk in segs])
                all_embeddings.extend(embs)

            # Step 2: Global clustering of all embeddings
            self.update_status("Clustering speaker embeddings globally...", percent=0.85)
            if not all_embeddings:
                logger.error("No embeddings extracted from diarization")
                raise RuntimeError("No embeddings extracted from diarization.")
            
            try:
                emb_vectors = np.stack([emb for _, emb in all_embeddings])
                logger.info(f"Embedding vectors shape: {emb_vectors.shape}")
                
                if emb_vectors.shape[0] == 0:
                    logger.error("Empty embedding vectors after stacking")
                    raise RuntimeError("Empty embedding vectors")
                
                clustering = AgglomerativeClustering(
                    n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.25
                )
                cluster_labels = clustering.fit_predict(emb_vectors)
                logger.info(f"Global clustering: {len(set(cluster_labels))} clusters found.")
                
                if len(set(cluster_labels)) == 0:
                    logger.error("No clusters found in clustering")
                    raise RuntimeError("Clustering produced no clusters")
                    
            except Exception as e:
                logger.error(f"Error in clustering: {e}")
                logger.error(f"Embeddings info: {len(all_embeddings)} embeddings")
                if all_embeddings:
                    logger.error(f"First embedding shape: {all_embeddings[0][1].shape if hasattr(all_embeddings[0][1], 'shape') else 'No shape'}")
                raise
            
            # Map each unique segment to its global cluster ID
            segment_speaker_map = {}
            cluster_to_segments = {}
            # --- NEW: Build (chunk_idx, local_speaker) -> global label mapping ---
            local_to_global = {}
            if len(all_embeddings) != len(cluster_labels):
                logger.error(f"Mismatch between embeddings ({len(all_embeddings)}) and cluster labels ({len(cluster_labels)})")
                raise RuntimeError("Embedding and cluster label count mismatch")
            for (segment_id, _), cluster_id in zip(all_embeddings, cluster_labels):
                segment_speaker_map[segment_id] = f"Speaker {cluster_id+1}"
                if cluster_id not in cluster_to_segments:
                    cluster_to_segments[cluster_id] = []
                cluster_to_segments[cluster_id].append(segment_id)
                # segment_id = (chunk_idx, local_speaker, seg_start, seg_end)
                local_key = (segment_id[0], segment_id[1])
                local_to_global[local_key] = f"Speaker {cluster_id+1}"
            # Log mapping
            logger.info("Local to global speaker mapping:")
            for k, v in local_to_global.items():
                logger.info(f"  (chunk={k[0]}, local={k[1]}) -> {v}")
            
            # Step 3: Map every word to its correct speaker
            if hasattr(self, 'word_by_word') and self.word_by_word:
                print(f"[DEBUG] Using self.word_by_word for diarization: {len(self.word_by_word)} words available.")
                # Validate word timestamps structure
                valid_words = []
                for w in self.word_by_word:
                    if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                        if isinstance(w["start"], (int, float)) and isinstance(w["end"], (int, float)):
                            valid_words.append((w["word"], w["start"], w["end"]))
                        else:
                            logger.warning(f"Invalid timestamp types for word '{w.get('word', 'unknown')}': start={type(w.get('start'))}, end={type(w.get('end'))}")
                    else:
                        logger.warning(f"Invalid word structure: {w}")
                
                words = valid_words
                logger.info(f"Total valid words with timestamps: {len(words)}")
                
                if not words:
                    logger.warning("No valid word timestamps found in word_by_word data")
                    logger.warning(f"Sample word_by_word data: {self.word_by_word[:3] if self.word_by_word else 'None'}")
                else:
                    logger.info(f"First 3 words: {words[:3]}")
                
                # --- FORCE: Robust word-to-speaker mapping ---
                # Use the correct word list for mapping
                words = valid_words  # previously 'all_words', now fixed
                diar_segments = []
                for chunk_idx, (start, end) in enumerate(chunk_infos):
                    chunk_segments = [(s, e, local_to_global[(chunk_idx, spk)], chunk_idx, spk)
                                      for s, e, spk in all_segments if start <= s < end]
                    diar_segments.extend(chunk_segments)
                logger.info(f"Total diarization segments: {len(diar_segments)}")
                logger.info(f"Sample segments: {diar_segments[:5]}")

                word_speaker = []
                fallback_count = 0
                fallback_samples = []
                for w in words:
                    w_start = w[1]
                    w_end = w[2]
                    best_overlap = 0
                    best_spk = None
                    for seg_start, seg_end, global_spk, chunk_idx, local_spk in diar_segments:
                        overlap = max(0, min(w_end, seg_end) - max(w_start, seg_start))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_spk = (global_spk, seg_start, seg_end, chunk_idx, local_spk)
                    if best_spk and best_overlap > 0:
                        word_speaker.append((best_spk[0], w[0]))
                        logger.debug(f"Word '{w[0]}' ({w_start:.2f}-{w_end:.2f}) mapped to {best_spk[0]} (overlap {best_overlap:.2f}s, segment {best_spk[1]:.2f}-{best_spk[2]:.2f}, chunk={best_spk[3]}, local={best_spk[4]})")
                    else:
                        # No overlap: assign to nearest segment
                        w_mid = (w_start + w_end) / 2
                        min_dist = float('inf')
                        nearest_spk = None
                        for seg_start, seg_end, global_spk, chunk_idx, local_spk in diar_segments:
                            dist = min(abs(w_mid - seg_start), abs(w_mid - seg_end))
                            if dist < min_dist:
                                min_dist = dist
                                nearest_spk = (global_spk, seg_start, seg_end, chunk_idx, local_spk)
                        if nearest_spk:
                            word_speaker.append((nearest_spk[0], w[0]))
                            fallback_count += 1
                            if len(fallback_samples) < 10:
                                fallback_samples.append((w[0], w_start, w_end, nearest_spk[0]))
                            logger.debug(f"Word '{w[0]}' at {w_start:.2f}s not in any segment, assigned to closest speaker {nearest_spk[0]}")
                        else:
                            word_speaker.append(("Unknown", w[0]))
                            logger.warning(f"Word '{w[0]}' at {w_start:.2f}s could not be assigned to any speaker!")
                logger.info(f"Successfully mapped {len(word_speaker)} words to speakers")
                logger.info(f"Words requiring fallback assignment: {fallback_count}")
                logger.info(f"Sample fallback assignments: {fallback_samples}")

                # --- Rebuild transcript with correct speaker turns ---
                lines = []
                last_speaker = None
                current_line = []
                for spk, word in word_speaker:
                    if spk != last_speaker:
                        if current_line:
                            lines.append(f"{last_speaker}: {' '.join(current_line)}")
                        last_speaker = spk
                        current_line = [word]
                    else:
                        current_line.append(word)
                if current_line:
                    lines.append(f"{last_speaker}: {' '.join(current_line)}")
                # Update speaker_names and self.speakers
                self.speaker_names = {v: v for v in set([spk for spk, _ in word_speaker])}
                self.speakers = []
                for line in lines:
                    if ': ' in line:
                        spk, text = line.split(': ', 1)
                        self.speakers.append({"speaker": spk, "text": text})
                self.transcript = '\n'.join(lines)
                logger.info(f"Final transcript contains {len(set([s['speaker'] for s in self.speakers]))} unique speakers.")
                logger.info(f"Final transcript has {len(self.speakers)} speaker segments")
                logger.info(f"Sample transcript lines: {lines[:3]}")
                logger.info(f"Full transcript length: {len(self.transcript)} characters")
                logger.info(f"Total words in final transcript: {len(word_speaker)}")
                
            else:
                # --- NEW: If word-level mapping is missing, use diarization segments ---
                logger.warning("No word-level timestamps found. Using diarization segments for transcript.")
                logger.warning(f"self.word_by_word status: {hasattr(self, 'word_by_word')}")
                if hasattr(self, 'word_by_word'):
                    logger.warning(f"self.word_by_word content: {len(self.word_by_word) if self.word_by_word else 'None'} items")
                    if self.word_by_word:
                        logger.warning(f"Sample word_by_word data: {self.word_by_word[:3]}")
                
                # Build a list of (start, end, global_speaker) for all segments
                diar_segments = []
                for chunk_idx, (start, end) in enumerate(chunk_infos):
                    chunk_segments = [(s, e, spk) for s, e, spk in all_segments if start <= s < end]
                    for s, e, spk in chunk_segments:
                        segment_id = (chunk_idx, spk, s, e)
                        global_spk = segment_speaker_map.get(segment_id, spk)
                        diar_segments.append((s, e, global_spk))
                diar_segments.sort()
                
                if not diar_segments:
                    logger.error("No diarization segments found - cannot create transcript")
                    raise RuntimeError("No diarization segments available")
                
                # Build transcript as sequence of speaker segments
                self.speaker_names = {}
                self.speakers = []
                lines = []
                for s, e, spk in diar_segments:
                    self.speaker_names[spk] = spk
                    lines.append(f"{spk}: [{s:.2f}-{e:.2f}] ...")  # Placeholder for text, as we don't have word-level
                    self.speakers.append({"speaker": spk, "text": f"[{s:.2f}-{e:.2f}] ..."})
                self.transcript = '\n'.join(lines)
            # Validate final results before returning
            if not hasattr(self, 'speakers') or not self.speakers:
                logger.error("No speakers found in final result")
                raise RuntimeError("Diarization failed to produce speaker results")
            
            if not hasattr(self, 'transcript') or not self.transcript:
                logger.error("No transcript found in final result")
                raise RuntimeError("Diarization failed to produce transcript")
            
            # --- FORCE: Immediately update UI after diarization ---
            if hasattr(self, 'update_callback') and hasattr(self.update_callback, 'update_speaker_list'):
                self.update_callback.update_speaker_list()
            if hasattr(self, 'update_callback') and hasattr(self.update_callback, 'update_transcript_display'):
                self.update_callback.update_transcript_display()
            
            logger.info(f"Final transcript contains {len(set([s['speaker'] for s in self.speakers]))} unique speakers.")
            logger.info(f"Final transcript length: {len(self.transcript)} characters")
            self.update_status(f"Diarization complete. Found {len(set(segment_speaker_map.values()))} speakers.", percent=1.0)
            return self.speakers
        except Exception as e:
            self.update_status(f"Diarization failed: {e}", percent=0)
            logger.error(f"Diarization failed: {e}")
            traceback.print_exc()
            return self.identify_speakers_simple(transcript)

    
    def on_send_prompt(self, event):
        """Handle sending a prompt from the chat input."""
        prompt = self.chat_input.GetValue()
        if prompt.strip():
            self.llm_processor.generate_response(prompt)
            self.chat_input.SetValue("")

    def set_transcript(self, transcript_text):
        """Set the transcript attribute directly."""
        self.transcript = transcript_text
        self.last_audio_path = getattr(self.audio_processor, 'audio_file_path', None)

    def _check_diarization_cache(self, audio_file_path):
        """Check if diarization results exist in cache for the given audio file.
        
        If found, it loads the results into self.diarization.
        Returns True if cache was found and loaded, False otherwise.
        """
        try:
            # Create hash of file path and modification time to use as cache key
            file_stats = os.stat(audio_file_path)
            file_hash = hashlib.md5(f"{audio_file_path}_{file_stats.st_mtime}".encode()).hexdigest()
            
            # Use APP_BASE_DIR if available
            if 'APP_BASE_DIR' in globals() and APP_BASE_DIR:
                cache_dir = os.path.join(APP_BASE_DIR, "diarization_cache")
            else:
                cache_dir = "diarization_cache"
                
            if not os.path.exists(cache_dir):
                return False
                
            # Cache file path
            cache_file = os.path.join(cache_dir, f"{file_hash}.diar")
            
            # Check if cache file exists
            if not os.path.exists(cache_file):
                return False
                
            # Load cached results
            self.update_status("Loading cached diarization results...", percent=0.3)
            with open(cache_file, 'rb') as f:
                self.diarization = pickle.load(f)
                
            self.update_status("Successfully loaded cached results", percent=0.35)
            return True
        except Exception as e:
            self.update_status(f"Error loading from cache: {str(e)}", percent=0.05)
            return False
            
    def _save_diarization_cache(self, audio_file_path):
        """Save diarization results to cache for future use."""
        try:
            # Create hash of file path and modification time to use as cache key
            file_stats = os.stat(audio_file_path)
            file_hash = hashlib.md5(f"{audio_file_path}_{file_stats.st_mtime}".encode()).hexdigest()
            
            # Use APP_BASE_DIR if available
            if 'APP_BASE_DIR' in globals() and APP_BASE_DIR:
                cache_dir = os.path.join(APP_BASE_DIR, "diarization_cache")
            else:
                cache_dir = "diarization_cache"
                
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            # Cache file path
            cache_file = os.path.join(cache_dir, f"{file_hash}.diar")
            
            # Save results
            self.update_status("Saving diarization results to cache for future use...", percent=0.95)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.diarization, f)
            
            # Clean up old cache files if there are more than 20
            cache_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.diar')]
            if len(cache_files) > 20:
                # Sort by modification time and remove oldest
                cache_files.sort(key=os.path.getmtime)
                for old_file in cache_files[:-20]:  # Keep the 20 most recent
                    os.unlink(old_file)
                    
            self.update_status("Successfully cached results for future use", percent=0.98)
        except Exception as e:
            self.update_status(f"Error saving to cache: {str(e)}", percent=0.95)
            # Continue without caching - non-critical error

    def show_azure_speech_config_dialog(self):
        """Show dialog to configure Azure Speech settings."""
        dialog = wx.Dialog(self, title="Configure Azure Speech", size=(400, 300))
        
        # Create main panel with padding
        panel = wx.Panel(dialog)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Get current configuration
        speech_config = self.config_manager.get_azure_speech_config()
        current_key = speech_config.get("api_key", "") if speech_config else ""
        current_region = speech_config.get("region", "eastus") if speech_config else "eastus"
        
        # Add description
        description = wx.StaticText(panel, label=(
            "Configure your Azure Speech service settings.\n"
            "These are required for audio transcription.\n\n"
            "1. The API key should be from your Azure Speech service resource\n"
            "2. The region should match your Azure Speech service region (e.g., eastus)\n\n"
            "You can find these in your Azure portal under your Speech service resource."
        ))
        description.Wrap(380)
        sizer.Add(description, 0, wx.ALL | wx.EXPAND, 10)
        
        # Create input fields
        key_label = wx.StaticText(panel, label="API Key:")
        key_input = wx.TextCtrl(panel, value=current_key, size=(300, -1))
        key_input.SetHint("Enter your Azure Speech API key")
        
        region_label = wx.StaticText(panel, label="Region:")
        region_input = wx.TextCtrl(panel, value=current_region, size=(300, -1))
        region_input.SetHint("e.g., eastus, westus, etc.")
        
        # Add fields to sizer
        field_sizer = wx.FlexGridSizer(2, 2, 5, 5)
        field_sizer.Add(key_label, 0, wx.ALIGN_CENTER_VERTICAL)
        field_sizer.Add(key_input, 1, wx.EXPAND)
        field_sizer.Add(region_label, 0, wx.ALIGN_CENTER_VERTICAL)
        field_sizer.Add(region_input, 1, wx.EXPAND)
        
        sizer.Add(field_sizer, 0, wx.ALL | wx.EXPAND, 10)
        
        # Add buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        save_button = wx.Button(panel, label="Save")
        cancel_button = wx.Button(panel, label="Cancel")
        
        button_sizer.Add(save_button, 0, wx.ALL, 5)
        button_sizer.Add(cancel_button, 0, wx.ALL, 5)
        
        sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        # Set up event handlers
        def on_save(event):
            api_key = key_input.GetValue().strip()
            region = region_input.GetValue().strip()
            
            # Validate inputs
            if not api_key:
                wx.MessageBox("Please enter an API key", "Validation Error", wx.OK | wx.ICON_ERROR)
                return
                
            if not region:
                wx.MessageBox("Please enter a region", "Validation Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Save configuration
            if self.config_manager.set_azure_speech_config(api_key, None, region):
                wx.MessageBox("Configuration saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                dialog.EndModal(wx.ID_OK)
            else:
                wx.MessageBox("Failed to save configuration", "Error", wx.OK | wx.ICON_ERROR)
        
        def on_cancel(event):
            dialog.EndModal(wx.ID_CANCEL)
        
        save_button.Bind(wx.EVT_BUTTON, on_save)
        cancel_button.Bind(wx.EVT_BUTTON, on_cancel)
        
        # Set up the dialog
        panel.SetSizer(sizer)
        dialog.Centre()
        
        # Show the dialog
        if dialog.ShowModal() == wx.ID_OK:
            # Reinitialize the speech client with new configuration
            if hasattr(self, 'audio_processor'):
                self.audio_processor.initialize_speech_client()
        
        dialog.Destroy()

    # --- NEW: Ensure WAV for diarization ---
    def ensure_wav_for_diarization(self, audio_path):
        import os
        import subprocess
        import sys
        import shutil
        # If already a .wav file, just return
        if audio_path.lower().endswith('.wav'):
            return audio_path, False
        # --- Always use bundled FFmpeg ---
        ffmpeg_path = None
        # PyInstaller _MEIPASS (bundled) location
        if hasattr(sys, '_MEIPASS'):
            candidate = os.path.join(sys._MEIPASS, 'ffmpeg', 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
            if os.path.isfile(candidate):
                ffmpeg_path = candidate
        # If not PyInstaller, check local ffmpeg/ffmpeg.exe in app dir
        if not ffmpeg_path:
            app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            candidate = os.path.join(app_dir, 'ffmpeg', 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
            if os.path.isfile(candidate):
                ffmpeg_path = candidate
        # If still not found, show error and abort
        if not ffmpeg_path:
            self.show_error(
                "FFmpeg is required for audio conversion but was not found in the bundled app.\n"
                "Please contact support or reinstall the application.\n"
                "(Developer: Make sure ffmpeg is bundled in the 'ffmpeg' folder in the app directory or PyInstaller _MEIPASS.)"
            )
            raise RuntimeError("Bundled FFmpeg not found.")
        # Prepare output path
        base, _ = os.path.splitext(audio_path)
        wav_path = base + '_for_diarization.wav'
        # Convert to 16kHz mono WAV using bundled FFmpeg
        try:
            cmd = [ffmpeg_path, '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            self.show_error(f"Failed to convert {audio_path} to WAV for diarization using bundled FFmpeg: {e}")
            raise RuntimeError(f"Failed to convert {audio_path} to WAV for diarization: {e}")
        return wav_path, True

class LLMProcessor:
    """LLM processing functionality for chat and summarization."""
    def __init__(self, client, config_manager, update_callback=None):
        self.client = client
        self.config_manager = config_manager
        self.update_callback = update_callback
        self.chat_history = []
        
    def update_status(self, message, percent=None):
        if self.update_callback:
            wx.CallAfter(self.update_callback, message, percent)
            
    def generate_response(self, prompt, temperature=None):
        """Generate a response from the LLM."""
        if temperature is None:
            temperature = self.config_manager.get_temperature()
        
        messages = self.prepare_messages(prompt)
        
        try:
            self.update_status("Generating response...", percent=0)
            chat_deployment = self.config_manager.get_azure_deployment("chat")
            response = self.client.chat.completions.create(
                model=chat_deployment,
                messages=messages,
                temperature=temperature,
                api_version=self.config_manager.get_azure_api_version()
            )
            
            response_text = response.choices[0].message.content
            
            # Add to chat history
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            self.update_status("Response generated.", percent=100)
            return response_text
            
        except Exception as e:
            self.update_status(f"Error generating response: {str(e)}", percent=50)
            return f"Error: {str(e)}"
            
    def prepare_messages(self, prompt):
        """Prepare messages for the LLM, including chat history."""
        messages = []
        
        # Add system message
        system_content = "You are a helpful assistant that can analyze transcripts."
        messages.append({"role": "system", "content": system_content})
        
        # Add chat history (limit to last 10 messages to avoid token limits)
        if self.chat_history:
            messages.extend(self.chat_history[-10:])
            
        # Add the current prompt
        if prompt not in [msg["content"] for msg in messages if msg["role"] == "user"]:
            messages.append({"role": "user", "content": prompt})
            
        return messages
        
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []
        self.update_status("Chat history cleared.", percent=0)
        
    def summarize_transcript(self, transcript, template_name=None):
        """Summarize a transcript, optionally using a template."""
        if not transcript:
            return "No transcript to summarize."
            
        self.update_status("Generating summary...", percent=0)
        
        prompt = f"Summarize the following transcript:"
        template = None
        
        if template_name:
            templates = self.config_manager.get_templates()
            if template_name in templates:
                template = templates[template_name]
                prompt += f" Follow this template format:\n\n{template}"
        
        prompt += f"\n\nTranscript:\n{transcript}"
        
        try:
            chat_deployment = self.config_manager.get_azure_deployment("chat")
            # --- PATCH: If transcript is Hungarian, add a system message in Hungarian ---
            language = self.config_manager.get_language()
            if language == "hu":
                system_message = "Te egy magyar nyelvű asszisztens vagy, aki összefoglalja a beszélgetéseket magyarul."
            else:
                system_message = "You are an assistant that specializes in summarizing transcripts."
            response = self.client.chat.completions.create(
                model=chat_deployment,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            summary = response.choices[0].message.content
            
            # Save summary to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use APP_BASE_DIR if available
            if APP_BASE_DIR:
                summary_dir = os.path.join(APP_BASE_DIR, "Summaries")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                summary_filename = os.path.join(summary_dir, f"summary_{timestamp}.txt")
            else:
                summary_filename = f"Summaries/summary_{timestamp}.txt"
                
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            self.update_status(f"Summary generated and saved to {summary_filename}.", percent=100)
            return summary
            
        except Exception as e:
            self.update_status(f"Error generating summary: {str(e)}", percent=50)
            return f"Error: {str(e)}"

class ConfigManager:
    def __init__(self, base_dir):
        """Initialize the config manager with the base directory."""
        print(f"Initializing ConfigManager with base_dir: {base_dir}")
        self.base_dir = base_dir
        self.config_file = os.path.join(base_dir, "config.json")
        print(f"Config file will be at: {self.config_file}")
        self.config = self.load_config()
        # Ensure the config has the correct API keys
        self.ensure_correct_config()
        
    def load_config(self):
        """Load configuration from file."""
        try:
            # Get absolute path to config file
            config_path = os.path.abspath(os.path.join(self.base_dir, 'config.json'))
            print(f"Loading config from: {config_path}")
            
            if not os.path.exists(config_path):
                print(f"Config file not found at {config_path}")
                return self.config if hasattr(self, 'config') else {}
                
            # Read the raw file content first
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                print("Raw file content:", raw_content)
                config = json.loads(raw_content)
                
            # Remove any OpenAI-specific keys that might have been added
            if 'api_key' in config:
                del config['api_key']
            if 'model' in config:
                del config['model']
                
            # Ensure required sections exist
            if 'azure_speech' not in config:
                config['azure_speech'] = {}
                
            # Get the azure_speech section
            speech_config = config.get('azure_speech', {})
            
            # Print raw config for debugging
            print("Raw config from file:", config)
            print("Raw azure_speech section:", speech_config)
            
            # Store the original API key
            original_api_key = speech_config.get('api_key', '')
            print(f"Original API key from file: {original_api_key}")
            
            # Ensure the API key is properly loaded and preserved
            if not original_api_key:
                print("API key is missing or empty in azure_speech config")
            else:
                # Make sure we keep the original API key
                speech_config['api_key'] = original_api_key
                print(f"Preserved API key: {original_api_key}")
                
            if not speech_config.get('region'):
                print("Region is missing or empty in azure_speech config")
                speech_config['region'] = 'eastus'
                
            # Update the config with the validated speech_config
            config['azure_speech'] = speech_config
            
            # Save the config back to ensure it's properly formatted
            self.save_config(config)
                
            print("Successfully loaded configuration")
            print(f"Azure Speech config: {config.get('azure_speech', {})}")
            return config
            
        except json.JSONDecodeError as e:
            print(f"Error decoding config file: {str(e)}")
            if hasattr(self, 'config'):
                print("Returning existing configuration")
                return self.config
            print("No existing configuration found, returning empty dict")
            return {}
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            if hasattr(self, 'config'):
                print("Returning existing configuration")
                return self.config
            print("No existing configuration found, returning empty dict")
            return {}

    def save_config(self, config=None):
        """Save configuration to file."""
        try:
            if config is not None:
                self.config = config
                
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Ensure azure_speech section exists
            if 'azure_speech' not in self.config:
                self.config['azure_speech'] = {}
                
            # Save with proper formatting
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
                
            # Verify the save was successful
            with open(self.config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                if saved_config.get('azure_speech', {}).get('api_key') != self.config.get('azure_speech', {}).get('api_key'):
                    print("Warning: API key may not have been saved correctly")
                    
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False
            
    def ensure_correct_config(self):
        """Ensure the config file has the correct API keys."""
        try:
            # Load the current config
            current_config = self.load_config()
            
            # Check if we need to update the speech API key
            speech_config = current_config.get('azure_speech', {})
            if speech_config.get('api_key') == current_config.get('azure_api_key'):
                print("Fixing incorrect speech API key...")
                # Update with the correct speech API key
                speech_config['api_key'] = "EqEmEYR5wbZF5EVkIB612IigqBsZW5sh3qhU6o97k2CeZc0ITP9NJQQJ99BFACYeBjFXJ3w3AAAYACOGXEbk"
                current_config['azure_speech'] = speech_config
                self.save_config(current_config)
                print("Updated speech API key in config")
                
            return True
        except Exception as e:
            print(f"Error ensuring correct config: {str(e)}")
            return False

    def get_azure_api_key(self):
        """Get Azure API key."""
        return self.config.get("azure_api_key", "")

    def set_azure_api_key(self, api_key):
        """Set Azure API key."""
        self.config["azure_api_key"] = api_key
        self.save_config()

    def get_azure_endpoint(self):
        """Get Azure endpoint."""
        return self.config.get("azure_endpoint", "")

    def set_azure_endpoint(self, endpoint):
        """Set Azure endpoint."""
        self.config["azure_endpoint"] = endpoint
        self.save_config()

    def get_azure_api_version(self):
        """Get Azure API version."""
        return self.config.get("azure_api_version", "2023-05-15")

    def get_azure_deployment(self, deployment_type):
        """Get Azure deployment name for the specified type."""
        deployments = self.config.get("azure_deployments", {})
        return deployments.get(deployment_type, "")

    def set_azure_deployment(self, deployment_type, deployment_name):
        """Set Azure deployment name for the specified type."""
        if "azure_deployments" not in self.config:
            self.config["azure_deployments"] = {}
        self.config["azure_deployments"][deployment_type] = deployment_name
        self.save_config()

    def get_pyannote_token(self):
        """Get Pyannote token."""
        return self.config.get("pyannote_token", "")

    def set_pyannote_token(self, token):
        """Set Pyannote token."""
        self.config["pyannote_token"] = token
        self.save_config()

    def get_temperature(self):
        """Get temperature setting."""
        return float(self.config.get("temperature", 0.7))

    def set_temperature(self, temperature):
        """Set temperature setting."""
        try:
            self.config["temperature"] = float(temperature)
            self.save_config()
            return True
        except ValueError:
            return False

    def get_language(self):
        """Get language setting."""
        return self.config.get("language", "en")

    def set_language(self, language):
        """Set language setting."""
        self.config["language"] = language
        self.save_config()

    def get_templates(self):
        """Get all templates."""
        return self.config.get("templates", {})

    def get_template(self, name):
        """Get specific template by name."""
        templates = self.get_templates()
        return templates.get(name, "")

    def add_template(self, name, content):
        """Add a new template."""
        if "templates" not in self.config:
            self.config["templates"] = {}
        self.config["templates"][name] = content
        self.save_config()

    def remove_template(self, name):
        """Remove a template."""
        if "templates" in self.config and name in self.config["templates"]:
            del self.config["templates"][name]
            self.save_config()

    def get_azure_speech_config(self):
        """Get Azure Speech configuration."""
        try:
            # Get the azure_speech section from the current config
            speech_config = self.config.get('azure_speech', {})
            print("Loading Azure Speech configuration...")
            print(f"Config content: {speech_config}")
            print(f"Available keys in azure_speech: {list(speech_config.keys())}")
            
            # Check if we have the required fields
            if not speech_config:
                print("No Azure Speech configuration found")
                return None
                
            # Get and validate required fields
            api_key = speech_config.get('api_key', '')
            region = speech_config.get('region', '')
            
            # Print diagnostic information
            print(f"API Key length: {len(api_key)}")
            print(f"Region: {region}")
            
            # Validate the API key
            if not api_key or not isinstance(api_key, str):
                print("API key is empty or invalid in azure_speech config")
                # Try to reload from file
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    file_speech_config = file_config.get('azure_speech', {})
                    api_key = file_speech_config.get('api_key', '')
                    if api_key:
                        print("Successfully reloaded API key from file")
                        speech_config['api_key'] = api_key
                        self.config['azure_speech'] = speech_config
                        self.save_config()
                    else:
                        return None
                
            # Validate the region
            if not region or not isinstance(region, str):
                print("Region is empty or invalid in azure_speech config")
                return None
                
            # Return a copy of the configuration to prevent modification
            return {
                'api_key': api_key,
                'region': region,
                'endpoint': speech_config.get('endpoint', '')
            }
        except Exception as e:
            print(f"Error getting Azure Speech configuration: {str(e)}")
            return None

    def set_azure_speech_config(self, api_key, region, endpoint=None):
        """Set Azure Speech configuration."""
        try:
            # Validate inputs
            if not api_key or not isinstance(api_key, str):
                raise ValueError("API key must be a non-empty string")
            if not region or not isinstance(region, str):
                raise ValueError("Region must be a non-empty string")
                
            # Strip whitespace
            api_key = api_key.strip()
            region = region.strip()
            
            # Ensure azure_speech section exists
            if 'azure_speech' not in self.config:
                self.config['azure_speech'] = {}
                
            # Set the configuration
            self.config['azure_speech']['api_key'] = api_key
            self.config['azure_speech']['region'] = region
            if endpoint:
                self.config['azure_speech']['endpoint'] = endpoint
                
            # Save the configuration
            if not self.save_config():
                raise Exception("Failed to save Azure Speech configuration")
                
            # Verify the configuration was saved correctly
            saved_config = self.get_azure_speech_config()
            if not saved_config or saved_config.get('api_key') != api_key:
                raise Exception("Failed to save Azure Speech configuration")
                
            return True
        except Exception as e:
            print(f"Error setting Azure Speech configuration: {str(e)}")
            return False

    def get_azure_speech_api_key(self):
        """Get Azure Speech API key."""
        # Print diagnostic information
        print("Attempting to load Azure Speech API key from config...")
        
        # Get the azure_speech section
        speech_config = self.config.get("azure_speech", {})
        
        # Print diagnostic information about the config
        print(f"Config content: {speech_config}")
        print(f"Available keys in azure_speech: {list(speech_config.keys())}")
        
        # Get the API key
        api_key = speech_config.get("api_key", "")
        
        # Print diagnostic information about the API key
        print(f"Raw API key type: {type(api_key)}")
        print(f"Raw API key length: {len(api_key)}")
        
        # Validate the API key
        if not api_key or not isinstance(api_key, str):
            print("API key is empty or invalid in azure_speech config")
            return ""
            
        # Ensure the API key is properly formatted
        api_key = api_key.strip()
        
        # Additional validation
        if len(api_key) < 10:  # Basic length check
            print("API key appears to be too short")
            return ""
            
        print("Successfully loaded API key from config")
        return api_key

    def get_azure_speech_region(self):
        """Get Azure Speech region."""
        speech_config = self.config.get("azure_speech", {})
        return speech_config.get("region", "eastus")

def add_save_all_settings_button(panel, parent_frame):
    """Create a prominent save all settings button and add it to the panel."""
    
    # Create a function to safely save settings and verify the result
    def on_save_button_click(event):
        try:
            # Get Azure API key and endpoint from appropriate input fields
            if hasattr(parent_frame, 'azure_api_key_input'):
                api_key = parent_frame.azure_api_key_input.GetValue().strip()
            else:
                api_key = parent_frame.config_manager.get_azure_api_key()

            if hasattr(parent_frame, 'azure_endpoint_input'):
                endpoint = parent_frame.azure_endpoint_input.GetValue().strip()
            else:
                endpoint = parent_frame.config_manager.get_azure_endpoint()

            # Get Azure OpenAI deployments
            if hasattr(parent_frame, 'chat_deployment_input'):
                chat_deployment = parent_frame.chat_deployment_input.GetValue().strip()
            else:
                chat_deployment = parent_frame.config_manager.get_azure_deployment("chat")

            if hasattr(parent_frame, 'whisper_deployment_input'):
                whisper_deployment = parent_frame.whisper_deployment_input.GetValue().strip()
            else:
                whisper_deployment = parent_frame.config_manager.get_azure_deployment("whisper")

            # Get HF token from appropriate input field
            if hasattr(parent_frame, 'hf_input'):
                hf_token = parent_frame.hf_input.GetValue().strip()
            elif hasattr(parent_frame, 'pyannote_token_input'):
                hf_token = parent_frame.pyannote_token_input.GetValue().strip()
            else:
                hf_token = parent_frame.hf_token if hasattr(parent_frame, 'hf_token') else ""
                
            # Update parent_frame attributes
            parent_frame.hf_token = hf_token
            
            # Save to config manager
            parent_frame.config_manager.set_azure_api_key(api_key)
            parent_frame.config_manager.set_azure_endpoint(endpoint)
            parent_frame.config_manager.set_azure_deployment("chat", chat_deployment)
            parent_frame.config_manager.set_azure_deployment("whisper", whisper_deployment)
            parent_frame.config_manager.set_pyannote_token(hf_token)
            
            # Set environment variables
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
            os.environ["HF_TOKEN"] = hf_token
            
            # Update client if needed
            if hasattr(parent_frame, 'client') and api_key and endpoint:
                try:
                    parent_frame.client = AzureOpenAI(
                        api_key=api_key,
                        api_version=parent_frame.config_manager.get_azure_api_version(),
                        azure_endpoint=endpoint
                    )
                    wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                except Exception as e:
                    wx.MessageBox(f"Error setting Azure OpenAI client: {e}", "Error", wx.OK | wx.ICON_ERROR)
            else:
                wx.MessageBox("Settings saved successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
                
        except Exception as e:
            wx.MessageBox(f"Error saving settings: {e}", "Error", wx.OK | wx.ICON_ERROR)
            
    # Create button sizer
    button_sizer = wx.BoxSizer(wx.VERTICAL)
    
    # Create save button with custom style
    save_button = wx.Button(panel, label="Save All Settings")
    save_button.SetBackgroundColour(wx.Colour(50, 200, 50))  # Green background
    save_button.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
    save_button.Bind(wx.EVT_BUTTON, on_save_button_click)
    
    button_sizer.Add(save_button, 0, wx.EXPAND | wx.ALL, 10)
    
    # Add a verify button to check settings at any time
    verify_button = wx.Button(panel, label="Verify Saved Settings")
    verify_button.Bind(wx.EVT_BUTTON, lambda e: wx.MessageBox(
        verify_saved_settings(parent_frame.config_manager.config_file),
        "Current Settings", wx.OK | wx.ICON_INFORMATION))
    
    button_sizer.Add(verify_button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
    
    return button_sizer

def verify_saved_settings(config_file_path):
    """Verify that settings are properly saved and return a status message."""
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        status_lines = []
        
        # Check Azure OpenAI settings
        azure_api_key = config.get('azure_api_key', '')
        azure_endpoint = config.get('azure_endpoint', '')
        azure_api_version = config.get('azure_api_version', '')
        azure_deployments = config.get('azure_deployments', {})
        
        if azure_api_key:
            status_lines.append("✓ Azure OpenAI API Key is set")
        else:
            status_lines.append("✗ Azure OpenAI API Key is not set")
            
        if azure_endpoint and azure_endpoint != "https://your-resource-name.openai.azure.com":
            status_lines.append("✓ Azure OpenAI Endpoint is set")
        else:
            status_lines.append("✗ Azure OpenAI Endpoint is not set")
            
        if azure_api_version:
            status_lines.append(f"✓ Azure OpenAI API Version: {azure_api_version}")
        else:
            status_lines.append("✗ Azure OpenAI API Version is not set")
            
        if azure_deployments:
            status_lines.append("\nAzure OpenAI Deployments:")
            for deployment_type, deployment_name in azure_deployments.items():
                status_lines.append(f"✓ {deployment_type}: {deployment_name}")
        else:
            status_lines.append("✗ No Azure OpenAI deployments configured")
        
        # Check PyAnnote token
        pyannote_token = config.get('pyannote_token', '')
        if pyannote_token:
            status_lines.append("\n✓ PyAnnote Token is set")
        else:
            status_lines.append("\n✗ PyAnnote Token is not set")
        
        # Check language setting
        language = config.get('language', '')
        if language:
            status_lines.append(f"✓ Language is set to: {language}")
        else:
            status_lines.append("✗ Language is not set")
        
        # Check templates
        templates = config.get('templates', {})
        if templates:
            status_lines.append(f"\nFound {len(templates)} templates:")
            for name in templates.keys():
                status_lines.append(f"✓ {name}")
        else:
            status_lines.append("\n✗ No templates found")
        
        return "\n".join(status_lines)
        
    except FileNotFoundError:
        return "✗ Config file not found"
    except json.JSONDecodeError:
        return "✗ Config file is not valid JSON"
    except Exception as e:
        return f"✗ Error verifying settings: {str(e)}"

# Fallback: Load AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER from config.json if not set in environment
try:
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        if os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is None:
            storage_conn = config_data.get('AZURE_STORAGE_CONNECTION_STRING')
            if not storage_conn:
                storage_conn = config_data.get('azure_storage_connection_string')
            if not storage_conn:
                storage_conn = config_data.get('azure_speech', {}).get('AZURE_STORAGE_CONNECTION_STRING')
            if not storage_conn:
                storage_conn = config_data.get('azure_speech', {}).get('azure_storage_connection_string')
            if not storage_conn:
                storage_conn = config_data.get('storage_connection_string')
            if storage_conn:
                os.environ['AZURE_STORAGE_CONNECTION_STRING'] = storage_conn
        if os.environ.get('AZURE_STORAGE_CONTAINER') is None:
            storage_container = config_data.get('AZURE_STORAGE_CONTAINER')
            if not storage_container:
                storage_container = config_data.get('azure_storage_container')
            if not storage_container:
                storage_container = config_data.get('azure_speech', {}).get('AZURE_STORAGE_CONTAINER')
            if not storage_container:
                storage_container = config_data.get('azure_speech', {}).get('azure_storage_container')
            if not storage_container:
                storage_container = config_data.get('storage_container')
            if storage_container:
                os.environ['AZURE_STORAGE_CONTAINER'] = storage_container
except Exception as e:
    print(f"Warning: Could not load storage connection info from config.json: {e}")

if __name__ == "__main__":
    try:
        print("Starting KeszAudio...")
        
        # Check if application is frozen (running as executable)
        is_frozen = getattr(sys, 'frozen', False)
        is_windows = platform.system() == 'windows' or platform.system() == 'Windows'
        is_macos = platform.system() == 'darwin'
        
        # For frozen applications, set specific environment variables
        if is_frozen:
            # Set application name for display
            app_name = "KeszAudio"
            
            # For macOS, make sure GUI works correctly in app bundle
            if is_macos:
                # Set additional environment variables for macOS App Bundle
                os.environ['PYTHONFRAMEWORK'] = '1'
                os.environ['DISPLAY'] = ':0'
                os.environ['WX_NO_DISPLAY_CHECK'] = '1'
                os.environ['WXMAC_NO_NATIVE_MENUBAR'] = '1'
            
            # For Windows, set working directory correctly
            if is_windows:
                # Make sure working directory is set to the executable location
                if hasattr(sys, '_MEIPASS'):
                    # PyInstaller sets _MEIPASS
                    os.chdir(sys._MEIPASS)
                else:
                    # Otherwise use executable's directory
                    os.chdir(os.path.dirname(sys.executable))
            
            # --- BEGIN: Ensure bundled config.json is copied to AppData if not present ---
            import shutil
            appdata_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser("~")), app_name)
            appdata_config = os.path.join(appdata_dir, 'config.json')
            # The bundled config.json will be next to the executable or in _MEIPASS
            if hasattr(sys, '_MEIPASS'):
                bundled_config = os.path.join(sys._MEIPASS, 'config.json')
            else:
                bundled_config = os.path.join(os.path.dirname(sys.executable), 'config.json')
            if not os.path.exists(appdata_config) and os.path.exists(bundled_config):
                os.makedirs(appdata_dir, exist_ok=True)
                shutil.copy2(bundled_config, appdata_config)
                print(f"Copied bundled config.json to {appdata_config}")
            # --- END: Ensure bundled config.json is copied to AppData if not present ---
        
        # Handle CLI mode explicitly
        if "--cli" in sys.argv:
            # CLI mode explicitly requested
            main()
        else:
            # Ensure required directories exist and get base directory
            # Critical step: this must succeed before proceeding
            try:
                APP_BASE_DIR = ensure_directories()
                print(f"Using application directory: {APP_BASE_DIR}")
                
                # Verify directories are created and writable
                for subdir in ["Transcripts", "Summaries", "diarization_cache"]:
                    test_dir = os.path.join(APP_BASE_DIR, subdir)
                    if not os.path.exists(test_dir):
                        os.makedirs(test_dir, exist_ok=True)
                    
                    # Verify we can write to the directory
                    test_file = os.path.join(test_dir, ".write_test")
                    try:
                        with open(test_file, 'w') as f:
                            f.write("test")
                        if os.path.exists(test_file):
                            os.remove(test_file)
                        print(f"Directory {test_dir} is writable")
                    except Exception as e:
                        print(f"WARNING: Directory {test_dir} is not writable: {e}")
                        # Try to find an alternative location
                        APP_BASE_DIR = os.path.join(os.path.expanduser("~"), "KeszAudio")
                        os.makedirs(APP_BASE_DIR, exist_ok=True)
                        print(f"Using alternative directory: {APP_BASE_DIR}")
                        break
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"FATAL ERROR setting up directories: {error_msg}")
                
                # Use a simple directory in the user's home folder as a last resort
                APP_BASE_DIR = os.path.join(os.path.expanduser("~"), "KeszAudio")
                os.makedirs(APP_BASE_DIR, exist_ok=True)
                print(f"Using fallback directory: {APP_BASE_DIR}")
            
            # Short delay to ensure filesystem operations complete
            import time
            time.sleep(0.5)
            
            # Use main() function to start the appropriate mode
            sys.exit(main())
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"FATAL ERROR: {error_msg}")
        
        # Try to show error dialog if possible
        try:
            # Check if we can import wx and if it's available
            if WX_AVAILABLE:
                if not hasattr(wx, 'App') or not wx.GetApp():
                    error_app = wx.App(False)
                wx.MessageBox(f"Fatal error starting application:\n\n{error_msg}", 
                             "Application Error", wx.OK | wx.ICON_ERROR)
            else:
                # wx is not available, just print the error
                print("Could not show error dialog because wxPython is not available.")
        except Exception as dialog_error:
            # If we can't even show a dialog, just report the error
            print(f"Could not show error dialog: {dialog_error}")
        
        sys.exit(1)