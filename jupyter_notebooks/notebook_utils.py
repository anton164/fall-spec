def resolve_paths_from_parent_directory():
    # Resolve paths from root project directory
    import os
    import sys

    module_path = os.path.abspath(os.path.join(".."))
    if module_path not in sys.path:
        sys.path.append(module_path)