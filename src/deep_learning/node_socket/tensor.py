import bpy


class TensorSocket(bpy.types.NodeSocket):
    bl_idname = "TensorSocket"
    bl_label = "Tensor Socket"

    def draw(self, context, layout, node, text):
        pass

    def draw_color(self, context, node):
        return (1.0, 0.4, 0.2, 0.5)