"""This Module contains different constants and standard values, to
easily access and modify them all in one place

"""
import sys
import json
import logging
import threading
from pathlib import Path

DEFAULTS = {
    "LOGGING": {
        "LOG_LEVEL": "INFO",
        "LOG_OUTPUT": "console"
    },
    "OCR": {
        "MODE": "auto"
    },
    "DATA": {
        "NAMES": ["Sarah Kellen", "Steven Bannon", "Leslie Groff"]
    }
}

_lock = threading.Lock()
_initialized = False
config = {}
_logger = logging.getLogger(__name__)
_logging_configured = False

# ================================ Paths ================================
# =======================================================================

def get_base_path() -> Path:
    """Gets the Basepath on which the Programm is being run

    Returns:
        Path: the Basepath
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.parent.parent
    else:
        return Path(__file__).resolve().parent.parent

def get_programs_path(base_dir) -> Path:
    """This function checks the base_path and the parent path for the
    coords dir

    Args:
        base_dir (_type_): The path from which the coords dir should
                           get searched from

    Returns:
        Path: The Path to the coords directory that was either found
              or substituted
    """
    coord_dir = base_dir.joinpath("coords")
    fallback_pool_dir = Path(r"\\mtmfp805\tf\_Fertigungsentwicklung"\
                             r"\Pool\Matas\01_Collab\coords")
    if coord_dir.is_dir():
        return coord_dir
    elif coord_dir.parent.parent.joinpath("programs").is_dir:
        coord_dir = coord_dir.parent.parent
        return coord_dir.joinpath("coords")
    elif fallback_pool_dir.is_dir():
        return fallback_pool_dir

BASE_DIR_PATH = get_base_path()
PICKLED_PROGRAMM_COORDS_PATH = get_programs_path(BASE_DIR_PATH)
ICON_PATH = BASE_DIR_PATH.joinpath(r"assets\icon.ico")
CONFIG_FILE = BASE_DIR_PATH.joinpath("config.json")
LOG_FILE = BASE_DIR_PATH.joinpath("app.log")

# =========================== Helper Methods ===========================

def _cast_value(value, reference):
    """Automatically casts values to the type of the corresponding DEFAULT value.
    JSON already handles most types natively, but we may need to cast from config."""
    
    # If value is already the same type as reference, return it
    if isinstance(value, type(reference)):
        return value
    
    # Handle list types - JSON preserves these natively
    if isinstance(reference, list):
        if isinstance(value, list):
            return value
        # If reference is a list but value is string, convert to single-item list
        return [value]
    
    # Handle dict types - JSON preserves these natively
    if isinstance(reference, dict):
        return value if isinstance(value, dict) else {}
    
    # Handle string values
    if isinstance(reference, str):
        return str(value)
    
    # Handle bool values
    if isinstance(reference, bool):
        if isinstance(value, bool):
            return value
        val_str = str(value).lower().strip()
        return val_str in ("true", "yes", "1", "on")
    
    # Handle int values
    if isinstance(reference, int):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    # Handle float values
    if isinstance(reference, float):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    return value

def _sync_to_file():
    """Helper Method to write the current config state to disk"""
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
        _logger.debug(f"Configuration synced to {CONFIG_FILE}")
    except Exception as e:
        _logger.error(f"Failed to sync configuration to file: {e}")

# ============================= Public API =============================

def initialize():
    """Initializes the config once during runtime in a thread-safe manner"""
    global _initialized
    with _lock:
        if _initialized:
            return

        _logger.debug("Starting config initialization")
        
        if not CONFIG_FILE.exists():
            _logger.info(f"Config file not found at {CONFIG_FILE}, creating with defaults")
            config.update(DEFAULTS)
            _sync_to_file()
        else:
            _logger.info(f"Loading config from {CONFIG_FILE}")
            try:
                with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
                    config.update(json.load(config_file))
                _logger.debug("Config file loaded successfully")
            except Exception as e:
                _logger.error(f"Failed to load config file: {e}")
                config.update(DEFAULTS)

        # First ensure all default sections and keys exist
        modified = False
   
        for section, params in DEFAULTS.items():
            if section not in config:
                _logger.debug(f"Adding missing section: {section}")
                config[section] = params.copy()
                modified = True
            else:
                for key, default_value in params.items():
                    if key not in config[section]:
                        _logger.debug(f"Adding missing key: {section}.{key}")
                        config[section][key] = default_value
                        modified = True

        # Next the values from the file get read and cast
        for section in config:
            if not isinstance(config[section], dict):
                continue
                
            for key, raw_val in config[section].items():
                # Try to find a reference type in DEFAULTS for casting
                ref_val = DEFAULTS.get(section, {}).get(key)

                try:
                    val = _cast_value(raw_val, ref_val) if ref_val is not None else raw_val
                except (ValueError, TypeError):
                    _logger.warning(f"Failed to cast {section}.{key}, using raw value")
                    val = raw_val

                globals()[key.upper()] = val
                _logger.debug(f"Set {key.upper()} = {val}")
        
        # Configure logging level
        log_level_str = config.get("LOGGING", {}).get("LOG_LEVEL", "INFO")
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        _logger.setLevel(log_level)
        _logger.info(f"Logging level set to {log_level_str}")

        if modified:
            _logger.info("Configuration was modified, syncing to file")
            _sync_to_file()
        
        _logger.info("Config initialization completed")
        _initialized = True

def setup_logging():
    """Configure logging handlers (console and/or file) based on config"""
    global _logging_configured
    
    if _logging_configured:
        return
    
    with _lock:
        log_level_str = config.get("LOGGING", {}).get("LOG_LEVEL", "INFO")
        log_output = config.get("LOGGING", {}).get("LOG_OUTPUT", "console").lower()
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Get root logger and set level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler if requested
        if log_output in ("console", "both"):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            _logger.info(f"Console logging enabled at {log_level_str} level")
        
        # Add file handler if requested
        if log_output in ("file", "both"):
            try:
                file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                _logger.info(f"File logging enabled at {LOG_FILE}")
            except Exception as e:
                _logger.error(f"Failed to set up file logging: {e}")
        
        _logging_configured = True
        _logger.info(f"Logging configured with output to: {log_output}")

def add_setting(section: str, key: str, value):
    """Adds a new setting (and section if needed) at runtime"""
    with _lock:
        if section not in config:
            _logger.debug(f"Creating new section: {section}")
            config[section] = {}

        config[section][key.lower()] = str(value)
        globals()[key.upper()] = value
        _logger.info(f"Added setting: {section}.{key} = {value}")
        _sync_to_file()

def remove_setting(section: str, key: str):
    with _lock:
        if section in config and key.lower() in config[section]:
            del config[section][key.lower()]
            if key.upper() in globals():
                del globals()[key.upper()]
            _logger.info(f"Removed setting: {section}.{key}")
            _sync_to_file()
        else:
            _logger.warning(f"Attempted to remove non-existent setting: {section}.{key}")

def remove_section(section: str):
    with _lock:
        if section in config:
            for key in config[section]:
                if key.upper() in globals():
                    del globals()[key.upper()]
           
            del config[section]
            _logger.info(f"Removed section: {section}")
            _sync_to_file()
        else:
            _logger.warning(f"Attempted to remove non-existent section: {section}")

def save_settings(**kwargs):
    """Ermöglicht es, Werte zur Laufzeit zu ändern und zu speichern."""
    with _lock:
        for key, value in kwargs.items():
            # Update im Config-Objekt (Sektion suchen)
            for section in DEFAULTS:
                if key.lower() in DEFAULTS[section]:
                    if section not in config:
                        config[section] = {}
                    config[section][key.lower()] = str(value)
                    globals()[key.upper()] = value # Auch im Speicher aktualisieren
                    _logger.info(f"Updated setting: {section}.{key} = {value}")

        _sync_to_file()

initialize()
setup_logging()

# --- Type hinting (only for IDE never runs) ---
if False:
    PARTNAMES: list[str]
    NAMES: list[str]
