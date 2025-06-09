from torch import nn

class Model(nn.Module):
    def __init__(self,configs,patch_len=16,stride=18):
        super(Model,self).__init__()
        self.pred_len = configs.pred_len
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # Setup LLM
        
        # Patching Embedding
        
        # Reprogramming
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forcast()
        return dec_out[:,-self.pred_len:,:]