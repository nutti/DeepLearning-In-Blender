import numpy as np

import bpy

from ..dl.model import Model
from ..dl.layers.linear import LinearLayer
from ..dl.layers.activation import TanhLayer, SoftmaxLayer
from ..dl.datasets import mnist


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
        
        layers = []
        for n in node_list_to_run:
            node_id = n.name.split(".")[0]
            if node_id == "Linear":
                layers.append(LinearLayer(n.in_features, n.out_features))
            elif node_id == "Tanh":
                layers.append(TanhLayer())
            elif node_id == "Softmax":
                layers.append(SoftmaxLayer())
        
        model = Model()
        model.add_layers(layers)
        model.initialize_params()

        # Load data
        data = mnist.load_mnist_data()
        images = mnist.normalize_data(data["train_images"]) 
        labels = mnist.make_one_hot_label(data["train_labels"])
        test_images = mnist.normalize_data(data["test_images"])
        test_labels = mnist.make_one_hot_label(data["test_labels"])

        # Train loop.
        epochs = 2
        batch_size = 100
        learning_rate = 0.1
        num_batches = int(labels.shape[-1] / batch_size)
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            for batch in range(num_batches):
                batch_mask = np.random.choice(labels.shape[-1], batch_size)
                X_train = images[:, batch_mask]
                y_train = labels[:, batch_mask]

                # gradient
                grads = model.gradient(X_train, y_train)

                # calculate loss
                if batch % 100 == 0:
                    loss = model.loss(X_train, y_train)
                    print("Batch {}: Loss = {}".format(batch, loss))

                # update
                model.update_paramters(grads, learning_rate)

            # predict
            a2 = model.predict(test_images)
            print(np.sum(np.argmax(np.log(a2), axis=0) == np.argmax(test_labels, axis=0)) / test_labels.shape[-1])


        return {'FINISHED'}
