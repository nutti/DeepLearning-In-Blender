if "bpy" in locals():
    import importlib
    importlib.reload(tensor)
else:
    from . import tensor

import bpy