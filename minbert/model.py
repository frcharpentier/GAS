"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from minbert.utils import CfgNode as CN

from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification



def chaine_params(state_dict):
    return "\n".join(["%s : %s"%(X, param.size()) for X, param in state_dict.items()])
       


# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        
        
class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_pos_emb, config.n_embd)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.n_embd)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=config.LN_eps)
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.padding_idx = config.pad_token_id
        
        if config.typeBERT == "camembert":
            self.roberta = True
        elif config.typeBERT == "roberta":
            self.roberta = True
        else:
            self.roberta = False
          
        self.register_buffer("position_ids", torch.arange(config.max_pos_emb).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        
    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.
        Args:
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx
    
    def forward(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if self.roberta:
            position_ids = BERTEmbeddings.create_position_ids_from_input_ids(input_ids, self.padding_idx)
        else:
            position_ids = self.position_ids[:, 0 : seq_length]
         
        buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        token_type_ids = buffered_token_type_ids_expanded
        
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #3n lignes, n colonnes
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) #n lignes, n colonnes
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size)) # Inutile (seulement pour GPT)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0
        self.dimK = self.n_embd // self.n_head
        #self.block_size = config.block_size

    def forward(self, x, mask):
        y, _, _ = self.forward_att_QK(x, mask)
        return y

    def forward_QK(self, x, mask):
        y, _, QscalK = self.forward_att_QK(x, mask)
        return y, QscalK

    def forward_att(self, x, mask):
        y, att, _ = self.forward_att_QK(x, mask)
        return y, att

    def forward_att_QK(self, x, mask):
        # mask (B (batch_size) x T (seq_len))

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        #on multiplie (matriciellement) x par c_attn, la dimension 2 du résultat est la dimension
        # des plongements (la dimension 0 est la dimension du batch, la dimension 1, la dimension
        # de la phrase.) On divise le résultat en trois (batchs de phrases de) vecteurs q,k,v.
        # La matrice originale de la couche linéaire avait donc les n premières lignes consacrées à Q,
        # les n suivantes à K, et les n dernières à V.
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) Opération transpose_for_scores de la lib hugfc
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #QK = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        QscalK = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dimK))

        if mask is not None:
            QscalK = QscalK.masked_fill(mask[:, None, None, :T] == 0, -torch.inf) # BERT-style masking

        att = F.softmax(QscalK, dim=-1)
        attdo = self.attn_dropout(att)
        y = attdo @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, att, QscalK
    

class CausalSelfAttention_a_la_HF(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) #n lignes, n colonnes
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size)) # Inutile (seulement pour GPT)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward_att_QK(self, x, mask):
        B, T, C = x.size()
        q, k, v = self.query(x), self.key(x), self.value(x)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) Opération transpose_for_scores de la lib hugfc
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        QscalK = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            QscalK = QscalK.masked_fill(mask[:, None, None, :T] == 0, -torch.inf) # BERT-style masking

        att = F.softmax(QscalK, dim=-1)
        attdo = self.attn_dropout(att)
        y = attdo @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, att, QscalK
    
    def forward(self, x, mask):
        y, _, _ = self.forward_att_QK(x, mask)
        return y
    
    def forward_QK(self, x, mask):
        y, _, QscalK = self.forward_att_QK(x, mask)
        return y, QscalK
    
    def forward_att(self, x, mask):
        y, att, _ = self.forward_att_QK(x, mask)
        return y, att



class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        if not "LN_eps" in config.__dict__:
            LN_eps = 1e-05
        else:
            LN_eps = config.LN_eps
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=LN_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=LN_eps)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            #act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, mask):
        # Schéma pré-LN
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlpf(self.ln_2(x))
        return x
        
class postLN_Block(nn.Module):
    def __init__(self, config, a_la_HF=False):
        super().__init__()
        self.a_la_HF = a_la_HF
        if not "LN_eps" in config.__dict__:
            LN_eps = 1e-05
        else:
            LN_eps = config.LN_eps
        if a_la_HF:
            self.attn = CausalSelfAttention_a_la_HF(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_med = nn.LayerNorm(config.n_embd, eps=LN_eps)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.ln_fin = nn.LayerNorm(config.n_embd, eps=LN_eps)

    def forward(self, x, mask):
        # Schéma post-LN
        #x = self.ln_med(x + self.attn(x, mask))
        #x = self.ln_fin(x + self.mlpf(x))
        
        xa = self.attn(x, mask)
        x = x + xa
        x = self.ln_med(x)
        xp = self.c_fc(x)
        xp = F.gelu(xp)
        xp = self.c_proj(xp)
        xp = self.dropout(xp)
        x = x + xp
        x = self.ln_fin(x)
        
        return x
    
    def forward_QK(self, x, mask):
        # Schéma post-LN
        #x = self.ln_med(x + self.attn(x, mask))
        #x = self.ln_fin(x + self.mlpf(x))
        
        xa, QscalK = self.attn.forward_QK(x, mask)
        x = x + xa
        x = self.ln_med(x)
        xp = self.c_fc(x)
        xp = F.gelu(xp)
        xp = self.c_proj(xp)
        xp = self.dropout(xp)
        x = x + xp
        x = self.ln_fin(x)
        
        return x, QscalK

    def forward_att(self, x, mask):
        # Schéma post-LN
        #x = self.ln_med(x + self.attn(x, mask))
        #x = self.ln_fin(x + self.mlpf(x))
        
        xa, att = self.attn.forward_att(x, mask)
        x = x + xa
        x = self.ln_med(x)
        xp = self.c_fc(x)
        xp = F.gelu(xp)
        xp = self.c_proj(xp)
        xp = self.dropout(xp)
        x = x + xp
        x = self.ln_fin(x)
        
        return x, att

    def forward_att_QK(self, x, mask):
        # Schéma post-LN
        #x = self.ln_med(x + self.attn(x, mask))
        #x = self.ln_fin(x + self.mlpf(x))
        
        xa, att, QscalK = self.attn.forward_att_QK(x, mask)
        x = x + xa
        x = self.ln_med(x)
        xp = self.c_fc(x)
        xp = F.gelu(xp)
        xp = self.c_proj(xp)
        xp = self.dropout(xp)
        x = x + xp
        x = self.ln_fin(x)
        
        return x, att, QscalK
    

class BERTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        #self.activation = nn.Tanh()

    def forward(self, x):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        x0 = x[:, 0]
        y = F.tanh(self.dense(x0))
        return y
    

class CamembertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        classif_pdrop = (
            config.classif_pdrop 
                if ("classif_pdrop" in config.__dict__ and config.classif_pdrop is not None)
                else config.resid_pdrop
        )
        self.dropout = nn.Dropout(classif_pdrop)
        self.out_proj = nn.Linear(config.n_embd, config.n_labels)

    def forward(self, x):
        x = x[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    


def param_translation(input, a_la_HF=False):
    decalages = {"query":0, "key":1, "value":2}
    mots = input.split(".")
    if mots[0] == "roberta":
        mots = mots[1:]
        input = ".".join(mots)
    if mots[0] in ["embeddings", "pooler", "classifier"]:
        return input, False
    elif mots[0] == "encoder":
        assert mots[1] == "layer"
        ks = "encoder."+mots[2] + "."
        if mots[3] == "attention":
            if (not a_la_HF) and mots[4] == "self":
                ks += "attn.c_attn."+mots[-1]
                i = decalages[mots[5]]
                j = i + 1
                if mots[-1] == "weight":
                    return ks, (i, j, True)
                elif mots[-1] == "bias":
                    return ks, (i, j, False)
            elif a_la_HF and mots[4] == "self":
                ks += "attn." + mots[5] + "." + mots[-1]
                return ks, False
            elif mots[4] == "output":
                if mots[5] == "dense":
                    ks += "attn.c_proj." + mots[-1]
                    return ks, False
                elif mots[5] == "LayerNorm":
                    ks += "ln_med." + mots[-1]
                    return ks, False
        elif mots[3] == "intermediate":
            assert mots[4] == "dense"
            ks += "c_fc." + mots[-1]
            return ks, False
        elif mots[3] == "output":
            if mots[4] == "dense":
                ks += "c_proj." + mots[-1]
                return ks, False
            elif mots[4] == "LayerNorm":
                ks += "ln_fin." + mots[-1]
                return ks, False
    
    return "???", False

def config_translation(chf):
    C = CN()
    C.model_type = chf._name_or_path
    C.n_layer = chf.num_hidden_layers
    C.n_head = chf.num_attention_heads
    C.n_embd = chf.hidden_size
    C.vocab_size = chf.vocab_size
    C.max_pos_emb = chf.max_position_embeddings
    C.LN_eps = chf.layer_norm_eps
    C.attn_pdrop = chf.attention_probs_dropout_prob
    C.resid_pdrop = chf.hidden_dropout_prob
    C.embd_pdrop = chf.hidden_dropout_prob
    C.classif_pdrop = chf.classifier_dropout
    C.type_vocab_size = chf.type_vocab_size
    C.pad_token_id = chf.pad_token_id
    if chf.model_type == "camembert":
        C.typeBERT = "camembert"
    elif chf.model_type.startswith("roberta"):
        C.typeBERT = "roberta"
    else:
        C.typeBERT = "bert"
    C.n_labels = len(chf.id2label)
    return C


class BERT(nn.Module):
    """ BERT Language Model """

    @staticmethod
    def get_default_config(model_name=None):
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        if not model_name:
            C.model_type = 'bert'
            C.n_layer = None
            C.n_head = None
            C.n_embd =  None
            # these options must be filled in externally
            C.vocab_size = None
            C.block_size = None
            # dropout hyperparameters
            C.embd_pdrop = 0.1
            C.resid_pdrop = 0.1
            C.attn_pdrop = 0.1
            C.type_vocab_size=2
            C.pad_token_id = 0
            C.typeBERT == "bert"
        elif model_name == "bert-base-uncased":
            C.model_type = 'bert-base-uncased'
            C.n_layer = 12
            C.n_head = 12
            C.n_embd =768
            C.vocab_size = 30522
            C.max_pos_emb = 512
            C.LN_eps = 1e-12
            C.embd_pdrop = 0.1
            C.resid_pdrop = 0.1
            C.attn_pdrop = 0.1
            C.type_vocab_size = 2
            C.pad_token_id = 0
            C.typeBERT == "bert"
        elif model_name == "camembert-base":
            C.model_type = 'camembert-base'
            C.n_layer = 12
            C.n_head = 12
            C.n_embd =768
            C.vocab_size = 32005
            C.max_pos_emb = 514
            C.LN_eps = 1e-05
            C.embd_pdrop = 0.1
            C.resid_pdrop = 0.1
            C.attn_pdrop = 0.1
            C.type_vocab_size = 1
            C.pad_token_id = 1
            C.typeBERT = "camembert"
        return C
    
    def __init__(self, config, a_la_HF=False):
        super().__init__()
        self.a_la_HF = a_la_HF
        assert config.vocab_size is not None
        #assert config.block_size is not None
        #self.block_size = config.block_size
        self.block_size = config.max_pos_emb
        self.vocab_size = config.vocab_size
        
        #if not "LN_eps" in config.__dict__:
        #    LN_eps = 1e-05
        #else:
        #    LN_eps = config.LN_eps
        
        self.embeddings = BERTEmbeddings(config)
        self.encoder = nn.ModuleList([postLN_Block(config, a_la_HF) for _ in range(config.n_layer)])
        #self.pooler = BERTPooler(config)
        #self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.classifier = CamembertClassificationHead(config)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.embeddings.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @staticmethod
    def from_huggingface_model_name(modele_hf_name, a_la_HF=False):
        modele_hf = AutoModel.from_pretrained(modele_hf_name)
        return BERT.from_huggingface(modele_hf, a_la_HF)

    @staticmethod
    def from_huggingface(modele_hf, a_la_HF=False):
        config_hf = modele_hf.config
        model_type = config_hf._name_or_path
        #assert model_type in {'bert-base', 'camembert-base'}
        config = config_translation(config_hf)
        model = BERT(config, a_la_HF)
        sd = model.state_dict()
        sd_hf = modele_hf.state_dict()
        for k, T in sd_hf.items():
            ks, suite = param_translation(k, a_la_HF)
            if suite:
                i, j, dim2 = suite
                i = i*config.n_embd
                j = j*config.n_embd
                if dim2:
                    print("Copie de %s dans %s[%d:%d,:]"%(k, ks, i, j))
                    with torch.no_grad():
                        sd[ks][i:j,:].copy_(T)
                else:
                    print("Copie de %s dans %s[%d:%d]"%(k, ks, i, j))
                    with torch.no_grad():
                        sd[ks][i:j].copy_(T)
            else:
                print("Copie de %s dans %s"%(k,ks))
                with torch.no_grad():
                    if ks in sd:
                        sd[ks].copy_(T)
        return model


    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config["learning_rate"], betas=train_config["betas"])
        return optimizer

    def forward(self, idx, mask, labels=None, output_att = False, output_QK = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        #debug_out = []
        att_out = []
        QK_out = []

        # forward the BERT model itself
        x = self.embeddings(idx)
        #debug_out.append(x.detach().numpy())
        if output_QK:
            if output_att:
                for block in self.encoder:
                    x, atto, QKo = block.forward_att_QK(x, mask)
                    att_out.append(atto)
                    QK_out.append(QKo)
            else:
                for block in self.encoder:
                    x, QKo = block.forward_QK(x, mask)
                    QK_out.append(QKo)
        elif output_att:
            for block in self.encoder:
                x, atto = block.forward_att(x, mask)
                att_out.append(atto)
        else:
            for block in self.encoder:
                x = block(x, mask)
                #debug_out.append(x.detach().numpy())
        
        #x = self.pooler(x)
        
        logits = self.classifier(x)
        #debug_out.append(logits.detach().numpy())

        # Calcul de la perte :
        if labels is not None:
            #labels = labels.to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = None

        att_out = tuple(att_out)
        QK_out = tuple(QK_out)
        if output_QK:
            if output_att:
                return logits, loss, att_out, QK_out#, debug_out
            else:
                return logits, loss, QK_out#, debug_out
        elif output_att:
            return logits, loss, att_out#, debug_out
        else:
            return logits, loss#, debug_out


