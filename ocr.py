import tensorflow as tf
import cv2
import numpy as np

class OCR:

    def __init__(self, model_path: str, lable_path: str) -> None:
        self.graph = self.load_graph(model_path)
        self.lable = self.load_lable(lable_path)
        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto())
    
    def load_graph(self, model_path: str) -> tf.Graph:
        graph = tf.Graph()

        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        
        with graph.as_default() as graph:
            tf.import_graph_def(graph_def)
        
        return graph

    def load_lable(self, lable_path: str) -> list[str]:
        lable = []
        lines = tf.io.gfile.GFile(lable_path).readlines()

        for line in lines:
            lable.append(line.rstrip())
        
        return lable
    
    def tensor_from_image(self, image: cv2.typing.MatLike, size: int):
        image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

        image_data = np.asarray(image, dtype=np.uint8)
        image_data = cv2.normalize(image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)

        return np.expand_dims(image_data, 0)


    def lable_image(self, tensor: tf.Tensor) -> str:
        input_layer = "import/input"
        output_layer = "import/final_result"


        input_op = self.graph.get_operation_by_name(input_layer)
        output_op = self.graph.get_operation_by_name(output_layer)

        results = self.sess.run(output_op.outputs[0], {input_op.outputs[0]: tensor})

        results = np.squeeze(results)

        top = results.argsort()[-1:][::-1]

        return self.lable[top[0]]
    
    def lable_image_list(self, images: list[cv2.typing.MatLike], size: int) -> str:
        return "".join([self.lable_image(self.tensor_from_image(image, size)) for image in images])
