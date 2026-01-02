import torch
import torch.nn as nn

class FlashAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function to export a specific 'Flash_Attention' node to ONNX.
    """
    @staticmethod
    def forward(ctx, q, k, v):
        # During PyTorch execution (e.g. training or inference in Python), 
        # we use the standard optimized implementation for correctness.
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @staticmethod
    def symbolic(g, q, k, v):
        # This method tells the ONNX exporter how to represent this function.
        # g.op() creates a new node in the ONNX graph.
        # "Flash_Attention" will be the 'op_type' of the node.
        output = g.op("Flash_Attention", q, k, v)
        # Infer the shape of the output
        # We assume the output shape is the same as the query shape
        output.setType(q.type())
        return output
# For single head attention
# dim_k = dim_v = hidden_dim/n_heads = hidden_dim
class FlashAttentionModel(nn.Module):
    def __init__(self, seq_len, hidden_dim, dim_k, dim_v):
        super().__init__()
        # inpu : seq_len * hidden_dim
        # q : hidden_dim * dim_k => Q=in
        # k : hidden_dim * dim_k
        # v : hidden_dim * dim_v
        self.q_proj = nn.Linear(hidden_dim, dim_k, bias=False)
        self.k_proj = nn.Linear(hidden_dim, dim_k, bias=False)
        self.v_proj = nn.Linear(hidden_dim, dim_v, bias=False)

    def forward(self, x):
        # 1. Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Use the custom function. 
        # When exporting to ONNX, this will be replaced by a single "Flash_Attention" node
        # taking q, k, v as inputs.
        output = FlashAttentionFunction.apply(q, k, v)
        
        return output

# --- Example of how to export ---
if __name__ == "__main__":
    # 1. Instantiate the model
    input_dim = 64
    dim_k = 64
    dim_v = 64
    model = FlashAttentionModel(input_dim, dim_k, dim_v)
    model.eval()

    # 2. Create dummy input
    batch_size = 1
    seq_len = 128
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # 3. Export to ONNX
    output_path = "flash_attention_custom.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        input_names=["input"], 
        output_names=["output"],
        verbose=False,
        do_constant_folding=True,
        export_params=False,
        opset_version=16  # Use a recent opset
    )
    
    print(f"Successfully exported model to {output_path}")
    print("The graph will look like: Input -> [Gemm, Gemm, Gemm] -> Flash_Attention -> Output")
