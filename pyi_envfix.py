import os
import platform
import sys
import pathlib
import shutil
import warnings

# Force copy strategy for all libraries that might use symlinks
os.environ['SB_LINK_STRATEGY'] = 'copy'  # For SpeechBrain
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'  # For HuggingFace Hub
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'

# --- Robust symlink patching for Windows ---
if platform.system().lower() == 'windows':
    # Only patch if not already patched
    if not hasattr(pathlib.Path, '_original_symlink_to'):
        pathlib.Path._original_symlink_to = pathlib.Path.symlink_to
        def safe_symlink_to(self, target, target_is_directory=False):
            try:
                return pathlib.Path._original_symlink_to(self, target, target_is_directory=target_is_directory)
            except OSError as e:
                if e.winerror == 1314:
                    warnings.warn(f"Symlink creation failed due to permissions, copying file instead: {target} -> {self}")
                    if target.is_file():
                        shutil.copy2(target, self)
                    elif target.is_dir():
                        shutil.copytree(target, self, dirs_exist_ok=True)
                    else:
                        raise e
                else:
                    raise e
        pathlib.Path.symlink_to = safe_symlink_to
    # Patch os.symlink only if not already patched
    if not hasattr(os, '_original_symlink'):
        os._original_symlink = os.symlink
        def safe_os_symlink(src, dst, target_is_directory=False):
            try:
                return os._original_symlink(src, dst, target_is_directory=target_is_directory)
            except OSError as e:
                if e.winerror == 1314:
                    warnings.warn(f"Symlink creation failed due to permissions, copying file instead: {src} -> {dst}")
                    src_path = pathlib.Path(src)
                    dst_path = pathlib.Path(dst)
                    if src_path.is_file():
                        shutil.copy2(src_path, dst_path)
                    elif src_path.is_dir():
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        raise e
                else:
                    raise e
        os.symlink = safe_os_symlink

print("Applied robust symlink bypass patches for Windows") 