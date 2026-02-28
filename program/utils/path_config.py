"""
Unified path configuration module
Supports automatic detection of local and Colab environments
"""
import os
import sys


def is_colab():
    """Detect if running in Colab environment"""
    try:
        import google.colab
        return True
    except ImportError:
        return os.environ.get('COLAB_GPU') is not None or 'google.colab' in sys.modules


def get_current_dir():
    """Get the utils directory (parent of path_config.py). Used as base for memory_db, replay_db."""
    return os.path.dirname(os.path.abspath(__file__))


def get_memory_db_path() -> str:
    """Get RAG memory_db path. Same as main.py and tools use - single source of truth."""
    return os.path.join(get_current_dir(), "memory_db")


def get_main_dir():
    """Get main directory (parent of Reorganize)"""
    current_dir = get_current_dir()
    return os.path.dirname(current_dir)


def get_project_root():
    """Get project root directory (LiveRobot-local root). Live2D folder is inside this directory."""
    current_dir = get_current_dir()  # .../LiveRobot-local/program/utils
    # Project root = directory containing "program" and "Live2D" (LiveRobot-local)
    # From program/utils -> go up to program -> go up to project root = 2 levels
    if is_colab():
        search_dir = current_dir
        max_levels = 5
        for _ in range(max_levels):
            if os.path.exists(os.path.join(search_dir, "main")):
                return os.path.abspath(search_dir)
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent
        return os.path.abspath(os.path.dirname(os.path.dirname(current_dir)))
    else:
        # LiveRobot-local: program/utils -> program -> LiveRobot-local (2 levels up)
        project_root = os.path.dirname(get_main_dir())
        return os.path.abspath(project_root)


def get_model_paths():
    """Get possible Live2D model path list. All paths are absolute (Live2D folder is inside LiveRobot-local)."""
    project_root = get_project_root()  # already absolute
    
    paths = [
        # Standard Live2D SDK samples (Live2D folder inside project root)
        os.path.join(project_root, "Live2D", "Samples", "Resources", "Hiyori", "Hiyori.model3.json"),
        os.path.join(project_root, "Live2D", "Samples", "Resources", "Haru", "Haru.model3.json"),
        os.path.join(project_root, "Live2D", "Samples", "Resources", "Mao", "Mao.model3.json"),
        # live2d-py examples
        os.path.join(project_root, "Encapsulation_Live2D", "live2d-py", "Resources", "v3", "Haru", "Haru.model3.json"),
        os.path.join(project_root, "Encapsulation_Live2D", "live2d-py", "Resources", "v3", "Mao", "Mao.model3.json"),
    ]
    # Normalize to absolute paths
    paths = [os.path.abspath(p) for p in paths]

    if is_colab():
        colab_paths = [
            os.path.join("/content", "LiveRobot", "Live2D", "Samples", "Resources", "Hiyori", "Hiyori.model3.json"),
            os.path.join("/content", "LiveRobot", "Live2D", "Samples", "Resources", "Haru", "Haru.model3.json"),
            os.path.join("/content", "LiveRobot", "Live2D", "Samples", "Resources", "Mao", "Mao.model3.json"),
            os.path.join("/content", "LiveRobot", "Encapsulation_Live2D", "live2d-py", "examples", "resources", "v3", "Haru", "Haru.model3.json"),
            os.path.join("/content", "drive", "MyDrive", "LiveRobot", "Live2D", "Samples", "Resources", "Hiyori", "Hiyori.model3.json"),
        ]
        paths.extend(colab_paths)

    return paths
