import torch
import torch.nn as nn

class FlashAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function to export a specific 'FlashAttention' node to ONNX.
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
        # Use custom domain to avoid ONNX checker failing
        output = g.op("custom::FlashAttention", q, k, v)
        # Infer the shape of the output
        # We assume the output shape is the same as the query shape
        output.setType(q.type())
        return output
# For single head attention
# dim_k = dim_v = hidden_dim/n_heads = hidden_dim
class FlashAttentionModel(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v, include_linear_layers=True):
        super().__init__()
        self.include_linear_layers = include_linear_layers
        # input : seq_len * input_dim
        # q : input_dim * dim_k => Q=in
        # k : input_dim * dim_k
        # v : input_dim * dim_v
        if self.include_linear_layers:
            self.q_proj = nn.Linear(input_dim, dim_k, bias=False)
            self.k_proj = nn.Linear(input_dim, dim_k, bias=False)
            self.v_proj = nn.Linear(input_dim, dim_v, bias=False)
            self.o_proj = nn.Linear(dim_v, input_dim, bias=False)
        else:
            # Add small dummy parameters to force the exporter to generate Add nodes.
            # This makes Stream framework transfer the Q/K/V tensors from off-chip 
            # as a bulk operation at the start.
            self.dummy_q = nn.Parameter(torch.zeros(dim_k))
            self.dummy_k = nn.Parameter(torch.zeros(dim_k))
            self.dummy_v = nn.Parameter(torch.zeros(dim_v))

    def forward(self, q, k=None, v=None):
        if self.include_linear_layers:
            # Input is (batch, seq, input_dim) in 'q'
            x = q
            q_out = self.q_proj(x)
            k_out = self.k_proj(x)
            v_out = self.v_proj(x)
            
            output = FlashAttentionFunction.apply(q_out, k_out, v_out)
            output = self.o_proj(output)
            return output
        else:
            # Inputs are q, k, v directly
            # To force the memory manager to load the entire Q, K, V tensors from off-chip 
            # before the FlashAttention kernel tiling execution starts, 
            # we insert dummy Add operations (which act like memory-load buffers).
            q_out = q + self.dummy_q
            k_out = k + self.dummy_k
            v_out = v + self.dummy_v
            # Just run the attention kernel
            return FlashAttentionFunction.apply(q_out, k_out, v_out)


# --- Example of how to export ---
if __name__ == "__main__":
    # 1. Instantiate the model
    input_dim = 64
    dim_k = 64
    dim_v = 64
    
    # MODE A: Full Model
    print("Exporting Full Model...")
    model = FlashAttentionModel(input_dim, dim_k, dim_v, include_linear_layers=True)
    model.eval()
    batch_size = 1
    seq_len = 128
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output_path = "flash_attention_full.onnx"
    torch.onnx.export(
        model, 
        (dummy_input), 
        output_path,
        input_names=["input"], 
        output_names=["output"],
        verbose=False, opset_version=16
    )
    
    # MODE B: Kernel Only
    print("Exporting Kernel Only...")
    model_kernel = FlashAttentionModel(input_dim, dim_k, dim_v, include_linear_layers=False)
    model_kernel.eval()
    q = torch.randn(batch_size, seq_len, dim_k)
    k = torch.randn(batch_size, seq_len, dim_k)
    v = torch.randn(batch_size, seq_len, dim_v)
    output_path_kernel = "flash_attention_kernel.onnx"
    torch.onnx.export(
        model_kernel, 
        (q, k, v), 
        output_path_kernel,
        input_names=["Q", "K", "V"], 
        output_names=["Output"],
        verbose=False, opset_version=16
    )
    print("Done.")
