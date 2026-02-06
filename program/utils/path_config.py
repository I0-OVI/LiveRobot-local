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
    """Get current file directory (Reorganize directory)"""
    return os.path.dirname(os.path.abspath(__file__))


def get_main_dir():
    """Get main directory (parent of Reorganize)"""
    current_dir = get_current_dir()
    return os.path.dirname(current_dir)


def get_project_root():
    """Get project root directory (LiveRobot root)"""
    current_dir = get_current_dir()
    # current_dir is Reorganize/utils, so we need to go up 3 levels to reach project root
    # Reorganize/utils -> Reorganize -> main_live2d -> main -> project_root
    
    if is_colab():
        search_dir = current_dir
        max_levels = 5
        
        for _ in range(max_levels):
            if os.path.exists(os.path.join(search_dir, "main")):
                return search_dir
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent
        
        # Fallback: go up 3 levels from Reorganize/utils
        return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    else:
        # Go up 3 levels: Reorganize/utils -> Reorganize -> main_live2d -> main -> project_root
        return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


def get_model_paths():
    """Get possible model path list"""
    project_root = get_project_root()
    
    paths = [
        # Standard Live2D SDK samples
        os.path.join(project_root, "Live2D", "Samples", "Resources", "Hiyori", "Hiyori.model3.json"),
        os.path.join(project_root, "Live2D", "Samples", "Resources", "Haru", "Haru.model3.json"),
        os.path.join(project_root, "Live2D", "Samples", "Resources", "Mao", "Mao.model3.json"),
        # live2d-py examples
        os.path.join(project_root, "Encapsulation_Live2D", "live2d-py", "Resources", "v3", "Haru", "Haru.model3.json"),
        os.path.join(project_root, "Encapsulation_Live2D", "live2d-py", "Resources", "v3", "Mao", "Mao.model3.json"),
    ]
    
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
