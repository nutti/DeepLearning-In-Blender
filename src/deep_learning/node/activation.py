import bpy
from bpy.props import IntProperty


class TanhNode(bpy.types.Node):

    bl_idname = "TanhNode"
    bl_label = "Tanh"

    def init(self, context):
        self.inputs.new("TensorSocket", "Input")

        self.outputs.new("TensorSocket", "Output")

    def draw_buttons(self, context, layout):
        pass
    
    def draw_buttons_ext(self, context, layout):
        pass


class SoftmaxNode(bpy.types.Node):

    bl_idname = "SoftmaxNode"
    bl_label = "Softmax"

    def init(self, context):
        self.inputs.new("TensorSocket", "Input")

        self.outputs.new("TensorSocket", "Output")

    def draw_buttons(self, context, layout):
        pass
    
    def draw_buttons_ext(self, context, layout):
        pass
