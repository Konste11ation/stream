import torch
import torch.nn as nn
import math

class FlashAttentionV2(nn.Module):
    """
    A pure Python implementation of FlashAttention v2 using generic GEMM and SIMD operations.
    This implementation mimics the tiling and online softmax recomputation logic of the 
    FlashAttention algorithm, but runs using standard PyTorch operators.
    """
    def __init__(self, input_dim, dim_k, dim_v, num_heads=1, causal=False, block_size_q=128, block_size_kv=128):
        super().__init__()
        self.input_dim = input_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.causal = causal
        
        # Tiling block sizes
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        
        if dim_k % num_heads != 0:
            raise ValueError(f"dim_k ({dim_k}) must be divisible by num_heads ({num_heads})")
        if dim_v % num_heads != 0:
            raise ValueError(f"dim_v ({dim_v}) must be divisible by num_heads ({num_heads})")
            
        self.head_dim_k = dim_k // num_heads
        self.head_dim_v = dim_v // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(input_dim, dim_k, bias=False)
        self.k_proj = nn.Linear(input_dim, dim_k, bias=False)
        self.v_proj = nn.Linear(input_dim, dim_v, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim_k)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: Tensor of shape (batch_size, seq_len, dim_v)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Project inputs and reshape to (Batch, Heads, Seq, Head_Dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim_v).transpose(1, 2)
        
        # Initialize output tensor
        output = torch.zeros_like(v)
        
        # Statistics for online softmax (running max and running sum)
        # m: max value seen so far for each query row. Init to -inf.
        # l: sum of exponentials seen so far (normalized by max). Init to 0.
        m = torch.full((batch_size, self.num_heads, seq_len), float('-inf'), device=x.device)
        l = torch.zeros((batch_size, self.num_heads, seq_len), device=x.device)
        
        # --- Outer Loop: Iterate over blocks of Queries (Rows of Attention Matrix) ---
        for i in range(0, seq_len, self.block_size_q):
            i_end = min(i + self.block_size_q, seq_len)
            
            # Slice Q block: (B, H, Br, Dk)
            q_block = q[:, :, i:i_end, :]
            
            # Initialize accumulator for this block's output (unnormalized)
            o_block = torch.zeros_like(output[:, :, i:i_end, :])
            
            # --- Inner Loop: Iterate over blocks of Keys/Values (Columns of Attention Matrix) ---
            for j in range(0, seq_len, self.block_size_kv):
                j_end = min(j + self.block_size_kv, seq_len)
                
                # Slice K, V blocks: (B, H, Bc, Dk/Dv)
                k_block = k[:, :, j:j_end, :]
                v_block = v[:, :, j:j_end, :]
                
                # 1. GEMM: Compute Attention Scores S_ij = Q_i * K_j^T
                # Shape: (B, H, Br, Bc)
                s_block = torch.matmul(q_block, k_block.transpose(-2, -1))
                s_block = s_block * self.scale
                
                # Causal Masking (if enabled)
                if self.causal:
                    # Create mask for this specific block
                    r_idx = torch.arange(i, i_end, device=x.device).view(-1, 1)
                    c_idx = torch.arange(j, j_end, device=x.device).view(1, -1)
                    mask = r_idx >= c_idx
                    s_block = s_block.masked_fill(~mask, float('-inf'))
                
                # 2. SIMD: Online Softmax Updates
                
                # Compute max of current block: m_ij
                m_block_curr = torch.max(s_block, dim=-1).values # (B, H, Br)
                
                # Retrieve running max for these query rows
                m_prev = m[:, :, i:i_end]
                
                # Update running max: m_new = max(m_prev, m_curr)
                m_new = torch.maximum(m_prev, m_block_curr)
                
                # Compute P_ij = exp(S_ij - m_new)
                # We subtract m_new for numerical stability
                p_block = torch.exp(s_block - m_new.unsqueeze(-1))
                
                # Compute correction factor for previous partial sums
                # alpha = exp(m_prev - m_new)
                # If m_new > m_prev, we need to scale down previous sums
                alpha = torch.exp(m_prev - m_new)
                
                # Update running sum of exps: l_new = alpha * l_prev + rowsum(P_ij)
                l_prev = l[:, :, i:i_end]
                row_sum_p = torch.sum(p_block, dim=-1)
                l_new = alpha * l_prev + row_sum_p
                
                # 3. GEMM: Compute Partial Output P_ij * V_j
                pv_block = torch.matmul(p_block, v_block) # (B, H, Br, Dv)
                
                # 4. SIMD: Update Output Accumulator
                # O_new = alpha * O_prev + P_ij * V_j
                o_block = o_block * alpha.unsqueeze(-1) + pv_block
                
                # Save updated stats back to global state
                m[:, :, i:i_end] = m_new
                l[:, :, i:i_end] = l_new
            
            # End of Inner Loop: We have processed all K, V blocks for this Q block.
            # Finalize Output: O_i = O_i / l_i
            # We divide by the total sum of exponentials to complete the softmax
            o_block = o_block / l[:, :, i:i_end].unsqueeze(-1)
            
            # Store result
            output[:, :, i:i_end, :] = o_block

        # 5. Reshape back to (Batch, Seq, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_v)
        
        return output

def export_to_onnx(model_path="flash_attention_v2_manual.onnx"):
    """
    Exports the FlashAttentionV2 model to ONNX.
    Note: Since this model uses Python loops for tiling, torch.onnx.export (tracing) 
    will UNROLL these loops based on the sequence length of the dummy input.
    This results in a large graph but accurately represents the tiled operations 
    for that specific sequence length.
    """
    # Model parameters
    input_dim = 64
    dim_k = 64
    dim_v = 64
    num_heads = 4
    seq_len = 256 
    
    # Instantiate model
    # We use smaller block sizes here to ensure we generate multiple tiles 
    # and verify the tiling logic in the exported graph.
    model = FlashAttentionV2(
        input_dim=input_dim, 
        dim_k=dim_k, 
        dim_v=dim_v, 
        num_heads=num_heads, 
        causal=True,
        block_size_q=64, 
        block_size_kv=64
    )
    model.eval()

    # Dummy input
    x = torch.randn(1, seq_len, input_dim)

    # Export
    print(f"Exporting model to {model_path}...")
    torch.onnx.export(
        model,
        x,
        model_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        do_constant_folding=True,
        # We do NOT use dynamic axes for seq_len here because the Python loops 
        # range(0, seq_len, block) depend on the specific integer value of seq_len.
        # Tracing requires this to be fixed.
    )
    print("Export complete.")

if __name__ == "__main__":
    export_to_onnx()
