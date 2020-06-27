if "bpy" in locals():
    import importlib
    importlib.reload(activation)
    importlib.reload(linear)
else:
    from . import activation
    from . import linear

import bpy