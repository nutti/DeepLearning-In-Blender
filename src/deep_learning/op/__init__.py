if "bpy" in locals():
    import importlib
    importlib.reload(train)
else:
    from . import train

import bpy