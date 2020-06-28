bl_info = {
    "name": "Deep Learning in Blender",
    "author": "Nutti",
    "version": (0, 1, 0),
    "blender": (2, 80, 0),
    "location": "Node Editor",
    "description": "Deep Learning with Node System",
    "warning": "",
    "support": "COMMUNITY",
    "wiki_url": "https://github.com/nutti/DeepLearning-In-Blender",
    "doc_url": "https://github.com/nutti/DeepLearning-In-Blender",
    "tracker_url": "https://github.com/nutti/DeepLearning-In-Blender",
    "category": "Node",
}

if "bpy" in locals():
    import importlib
    importlib.reload(node)
    importlib.reload(op)
    importlib.reload(socket)
else:
    import bpy
    from . import node
    from . import op
    from . import socket

import bpy

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem


node_categories = [
    NodeCategory('DEEP_LEARNING', "Deep Learning", items=[
        NodeItem(node.activation.TanhNode.bl_idname),
        NodeItem(node.activation.SoftmaxNode.bl_idname),
        NodeItem(node.linear.LinearNode.bl_idname),
        NodeItem(node.system.BeginNode.bl_idname),
        NodeItem(node.system.EndNode.bl_idname),
    ])
]


classes = [
    socket.tensor.TensorSocket,
    op.train.DL_OT_Train,
    node.activation.TanhNode,
    node.activation.SoftmaxNode,
    node.linear.LinearNode,
    node.system.BeginNode,
    node.system.EndNode,
]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    nodeitems_utils.register_node_categories('DEEP_LEARNING_NODES', node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories('DEEP_LEARNING_NODES')
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
