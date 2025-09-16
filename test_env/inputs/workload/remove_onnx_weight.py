import onnx
from onnx import helper
model = onnx.load("resnet50.onnx")

new_graph = helper.make_graph(
    nodes=model.graph.node,
    name=model.graph.name,
    inputs=model.graph.input,
    outputs=model.graph.output,
    value_info=model.graph.value_info
)

new_model = helper.make_model(new_graph)

# 保存新模型
onnx.save(new_model, "resnet50_clean.onnx")