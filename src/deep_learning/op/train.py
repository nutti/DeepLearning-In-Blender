import bpy


class DL_OT_Train(bpy.types.Operator):

    bl_idname = "node.dl_train"
    bl_label = "Train"
    bl_description = "Train defined network"
    bl_options = {'REGISTER'}

    def execute(self, context):
        node_tree = bpy.data.materials["Material"].node_tree

        if "Begin" not in node_tree.nodes.keys():
            self.report({'WARNING'}, "Begin node is needed to train")
            return {'CANCELLED'}
        if "End" not in node_tree.nodes.keys():
            self.report({'WARNING'}, "End node is needed to train")
            return {'CANCELLED'}
        
        begin_node = node_tree.nodes["Begin"]
        end_node = node_tree.nodes["End"]

        node_list_to_run = []
        current_node = begin_node
        while True:
            if len(current_node.outputs) == 0:
                self.report({'WARNING'}, "Could not reach End node")
                return {'CANCELLED'}
            current_node = current_node.outputs[0].links[0].to_node
            if current_node == end_node:
                break
            node_list_to_run.append(current_node)
        
        node_to_layer = {
            "Linear": "LinearLayer",
            "Tanh": "TanhLayer",
            "Softmax": "SoftmaxLayer",
        }

        for n in node_list_to_run:
            node_id = n.name.split(".")[0]
            print(node_to_layer[node_id])

        return {'FINISHED'}
