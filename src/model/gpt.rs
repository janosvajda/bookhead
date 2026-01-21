use burn::{
    nn,
    nn::{Embedding, Linear, LayerNorm},
    prelude::*,
    tensor::{Data, ElementConversion, Shape, Tensor, Int},
};
use burn::tensor::activation::{gelu, softmax};

use crate::config::ModelConfig;

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    // Build a transformer block from config.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self {
            ln1: nn::LayerNormConfig::new(cfg.d_model).init(device),
            attn: CausalSelfAttention::new(cfg, device),
            ln2: nn::LayerNormConfig::new(cfg.d_model).init(device),
            mlp: Mlp::new(cfg, device),
        }
    }

    // Run attention + MLP with residual connections.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.ln1.forward(x.clone());
        let h = self.attn.forward(h);
        let x = x + h;
        let h2 = self.ln2.forward(x.clone());
        let h2 = self.mlp.forward(h2);
        x + h2
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    // Initialize the feed-forward sub-layer.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(cfg.d_model, cfg.d_ff).init(device),
            fc2: nn::LinearConfig::new(cfg.d_ff, cfg.d_model).init(device),
        }
    }

    // Apply GELU-activated projection and return to model dimension.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = gelu(self.fc1.forward(x));
        self.fc2.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    // Initialize attention projection weights.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        assert!(cfg.d_model % cfg.n_heads == 0);
        let head_dim = cfg.d_model / cfg.n_heads;
        Self {
            qkv: nn::LinearConfig::new(cfg.d_model, 3 * cfg.d_model).init(device),
            proj: nn::LinearConfig::new(cfg.d_model, cfg.d_model).init(device),
            n_heads: cfg.n_heads,
            head_dim,
        }
    }

    // Compute causal self-attention for a sequence.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let qkv = self.qkv.forward(x); // [b, t, 3c]

        let q = qkv.clone().slice([0..b, 0..t, 0..c]);
        let k = qkv.clone().slice([0..b, 0..t, c..2*c]);
        let v = qkv.slice([0..b, 0..t, 2*c..3*c]);

        let q = q.reshape([b, t, self.n_heads, self.head_dim]).swap_dims(1, 2); // [b,h,t,hd]
        let k = k.reshape([b, t, self.n_heads, self.head_dim]).swap_dims(1, 2);
        let v = v.reshape([b, t, self.n_heads, self.head_dim]).swap_dims(1, 2);

        let scale = (self.head_dim as f32).sqrt();
        let att = q.matmul(k.swap_dims(2, 3)) / scale; // [b,h,t,t]

        let device = att.device();
        let mask = causal_mask::<B>(t, &device);
        let att = att + mask;

        let att = softmax(att, 3);
        let y = att.matmul(v); // [b,h,t,hd]
        let y = y.swap_dims(1, 2).reshape([b, t, c]);
        self.proj.forward(y)
    }
}

// Build a causal attention mask for sequence length t.
fn causal_mask<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(t * t);
    for i in 0..t {
        for j in 0..t {
            let value = if j > i { -1e9f32 } else { 0.0f32 };
            data.push(value.elem::<B::FloatElem>());
        }
    }
    Tensor::<B, 2>::from_data(Data::new(data, Shape::new([t, t])), device)
        .reshape([1, 1, t, t])
}

#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> Gpt<B> {
    // Construct the full GPT model from config.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let tok_emb = nn::EmbeddingConfig::new(cfg.vocab, cfg.d_model).init(device);
        let pos_emb = nn::EmbeddingConfig::new(cfg.seq_len, cfg.d_model).init(device);
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            blocks.push(Block::new(&cfg, device));
        }
        let ln_f = nn::LayerNormConfig::new(cfg.d_model).init(device);
        let lm_head = nn::LinearConfig::new(cfg.d_model, cfg.vocab).init(device);

        Self { tok_emb, pos_emb, blocks, ln_f, lm_head }
    }

    // Forward pass returning logits over the vocabulary.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = input_ids.dims();
        let device = input_ids.device();

        let tok = self.tok_emb.forward(input_ids);

        // positions [b,t]
        let pos_ids = Tensor::<B, 1, Int>::arange(0..t as i64, &device)
            .reshape([1, t])
            .repeat(0, b);
        let pos = self.pos_emb.forward(pos_ids);

        let mut x = tok + pos;
        for blk in self.blocks.iter() {
            x = blk.forward(x);
        }
        let x = self.ln_f.forward(x);
        self.lm_head.forward(x)
    }

}
