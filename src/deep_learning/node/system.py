import bpy
from bpy.props import IntProperty


class BeginNode(bpy.types.Node):

    bl_idname = "BeginNode"
    bl_label = "Begin"

    def init(self, context):
        self.outputs.new("TensorSocket", "Output")

    def draw_buttons(self, context, layout):
        pass
    
    def draw_buttons_ext(self, context, layout):
        pass


class EndNode(bpy.types.Node):

    bl_idname = "EndNode"
    bl_label = "End"

    def init(self, context):
        self.inputs.new("TensorSocket", "Inputs")

    def draw_buttons(self, context, layout):
        pass
    
    def draw_buttons_ext(self, context, layout):
        pass
