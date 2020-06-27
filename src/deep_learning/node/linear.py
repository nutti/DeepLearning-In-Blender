import bpy
from bpy.props import IntProperty


class LinearNode(bpy.types.Node):

    bl_idname = "LinearNode"
    bl_label = "Linear"

    in_features = IntProperty(
        name="In Features",
        description="Number of input features",
        default=1,
        min=1,
    )

    out_features = IntProperty(
        name="Out Featrues",
        description="Number of output features",
        default=1,
        min=1,
    )

    def init(self, context):
        self.inputs.new("TensorSocket", "Input")

        self.outputs.new("TensorSocket", "Output")

    def draw_buttons(self, context, layout):
        col = layout.column(align=True)

        col.prop(self, "in_features")
        col.prop(self, "out_features")
    
    def draw_buttons_ext(self, context, layout):
        col = layout.column(align=True)

        col.prop(self, "in_features")
        col.prop(self, "out_features")
