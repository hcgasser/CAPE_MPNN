# ----------------------------------------------------------------
# Portions of this file were adapted from https://github.com/dauparas/ProteinMPNN
# Copyright (c) 2022 Justas Dauparas
# Licensed under the MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------


import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from kit.data import DD, str_to_file, file_to_str

from CAPE.MPNN.ProteinMPNN.protein_mpnn_utils import gather_nodes, ProteinMPNN, cat_neighbors_nodes


class CapeMPNN(nn.Module):
    base_model_pt_dir_path = None
    base_model_yaml_dir_path = None

    def __init__(self, base_model_name):
        super(CapeMPNN, self).__init__()

        if CapeMPNN.base_model_yaml_dir_path is None:
            raise Exception("CapeMPNN.base_model_yaml_dir_path must be set before instantiating CapeMPNN")

        self.base_model_name = base_model_name

        base_hparams_file_path=os.path.join(CapeMPNN.base_model_yaml_dir_path, f"{base_model_name}.yaml")

        self.hparams = DD.from_yaml(base_hparams_file_path)

        self.num_edges = None
        self.noise_level = None
        self.actor = ProteinMPNN(
            self.hparams.num_letters, 
            self.hparams.node_features, 
            self.hparams.edge_features,
            self.hparams.hidden_dim, 
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.vocab, 
            self.hparams.k_neighbors, 
            0.0, 
            self.hparams.dropout)
        self.device = None


    @staticmethod
    def from_file(pt_file_path_or_base_model_name):
        if os.path.exists(pt_file_path_or_base_model_name):  # this is a CAPE model
            pt_file_path = pt_file_path_or_base_model_name
            arguments = file_to_str(
                os.path.join(os.path.dirname(pt_file_path), 'CapeMPNN.args')
            ).split('\n')
        else:  # this is a base model
            base_model_name = pt_file_path_or_base_model_name
            pt_file_path = os.path.join(CapeMPNN.base_model_pt_dir_path, f"{base_model_name}.pt")
            arguments = [base_model_name]
            
        model = CapeMPNN(*arguments)
        model.load_from_pt(pt_file_path)
        return model

    def load_from_pt(self, pt_file_path):
        checkpoint = torch.load(pt_file_path)
        self.num_edges = checkpoint['num_edges']
        self.noise_level = checkpoint['noise_level']
        self.actor.load_state_dict(checkpoint['model_state_dict'])

    def save_to_file(self, pt_file_path):
        torch.save({
            'num_edges': self.num_edges,
            'noise_level': self.noise_level,
            'model_state_dict': self.actor.state_dict(),
        }, pt_file_path)
        args_file_path = os.path.join(os.path.dirname(pt_file_path), 'CapeMPNN.args')
        if not os.path.exists(args_file_path):
            str_to_file(f"{self.base_model_name}", args_file_path)

    def to(self, device):
        if self.device != device:
            super(CapeMPNN, self).to(device)
            self.device = device

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, decoding_order,
                omit_AA_mask=None, temperature=0.1):
        """ Graph-conditioned sequence model """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.actor.features(X, mask, residue_idx, chain_encoding_all)
        B, N = X.shape[0], X.shape[1]
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.actor.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.actor.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.actor.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                           (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                           permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.actor.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            # pdb.set_trace()
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.actor.W_out(h_V) / temperature
        # log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        omit_AA_mask_flag = omit_AA_mask != None
        if omit_AA_mask_flag:
            probs_masked = probs * (1.0 - omit_AA_mask)
            probs = probs_masked / torch.sum(probs_masked, dim=-1, keepdim=True)  # [B, 21]

        log_probs = torch.log(probs + 1e-20)
        log_probs = log_probs * mask[:, :, None]
        return log_probs

    #
    # adapted from ./training/model_utils.py ProteinMPNN
    #
    def forward_train(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.actor.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.actor.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.actor.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.actor.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.actor.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.actor.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all,
               residue_idx, mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None,
               chain_M_pos=None, omit_AA_mask=None, bias_by_res=None):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.actor.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)  # seem to be the node embeddings
        h_E = self.actor.W_e(E)  # seem to be the edge embeddings

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.actor.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask*chain_M_pos*mask #update chain_M to include missing regions

        # e.g. decoding_order[0, :] = tensor([159, 175, 132, 181, 254, 246, 221, 219, ...])
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]


        # permutation_matrix_reverse[0, 0, 159] = 1 because 159 is the first element in the decoding order - all others are zero
        # permutation_matrix_reverse[0, 1, 175] = 1 because 175 is the second element in the decoding order - all others are zero
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()

        # order_mask_backward for the Xth sequence element is 1 at every position that comes before
        # e.g. order_mask_backward[0, 132, 175] = 1 because 132 comes before 175 in the decoding order
        # but  order_mask_backward[0, 175, 132] = 0 because 175 comes after 132 in the decoding order
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

        # mask_attend.shape = [N, L_max, neighbors, 1]
        # dim 0 is the batchdimension
        # dim 1 is the sequence element ordered in N to C direction
        # dim 2 is are the neighors of the node in dim 1 in N to C direction
        # dim 3 is of size 1
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)

        # mask.shape = [N, L_max]... it is 1 if there are xyz coordinates available for the sequence element and 0 otherwise (either because missing or padding)
        # mask_1D.shape = [N, L_max, 1, 1]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        all_log_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
        all_values = torch.zeros((N_batch, N_nodes), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)  # holds the input sequence embedding for the decoder
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)  # the predicted sequence

        # h_V_stack is a list of hidden node tensors of shape [N, L_max, 128] - one for each layer
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.actor.decoder_layers))]

        # contant is used to adjust the logits of the output layer before feeding them into the softmax
        constant = torch.tensor(omit_AAs_np, device=device)  # shape = (21, ) - one for each AA. 1 if the AA should not be sampled
        constant_bias = torch.tensor(bias_AAs_np, device=device)

        #chain_mask_combined = chain_mask*chain_M_pos
        omit_AA_mask_flag = omit_AA_mask != None

        # E_idx holds the indices of the neighbors of each node
        # h_nodes.shape = [N, L_max, 128]
        # cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)
        # 1. adds an additional dimension in h_nodes (dim=-2). This dimension specifies the neighbor or the node in dimension 1
        # 2. then it concatenates the resulting tensor from 1. to the h_neighbors tensor in dimension -1

        # h_E.shape = [N, L_max, neighbors, 128]
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)  # adds zeros to the last dimension of the final Edge encodings of the Encoder
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)  # adds the node encodings to the last dimension
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder  # masks all nodes that come AFTER the current node - BUT from the ENCODER

        for t_ in range(N_nodes):
            t = decoding_order[:,t_] #[B]

            # gathers the various masks for the current position t
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:,None]) #[B,1]
            mask_gathered = torch.gather(mask, 1, t[:,None]) #[B,1]
            bias_by_res_gathered = torch.gather(bias_by_res, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B, 21]

            if (mask_gathered==0).all(): #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:,None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))  # the neighbors of the current node
                # h_E.shape = [N, L_max, neighbors, 128]
                # h_V.shape = [N, L_max, 128]
                h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))  # the edge encodings (output of the Encoder) of the current node
                # h_E_t.shape = [N, 1, neighbors, 128]
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t) # appends the edge encodings to the last dimension of the hidden state of the current node and its neighbors
                # h_ES_t.shape = [N, 1, neighbors, 256]
                # !!! so essentially h_ES_t holds the sequence and edge information of the currently decoded node and its neighbors

                # only attend over the not yet decoded nodes from the encoder (so do not consider information about nodes that have already been decoded)
                h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))

                # is XYZ information available for the current node? If not the layers will return 0 for this node
                mask_t = torch.gather(mask, 1, t[:,None])  # shape = [N, 1]
                for l, layer in enumerate(self.actor.decoder_layers):
                    # Updated relational features for future states

                    # h_V_stack[0] holds the h_V output of the encoder
                    # h_V_stack[j], for j > 0 holds the output of the j-1 decoder layer
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)  # appends the edge
                    h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))

                    # h_ESV_t consists of two components
                    # 1. the sequence and edge information of all the already decoded nodes in its neighborhood
                    # 2. the forward-looking node and edge information from the encoder
                    h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t

                    # calls the decoder layer with the previous layers output for the currently decoded node
                    # and the h_ESV_t information from above
                    h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
                logits = self.actor.W_out(h_V_t) / temperature

                probs = F.softmax(logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature, dim=-1)

                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(omit_AA_mask, 1, t[:,None, None].repeat(1,1,omit_AA_mask.shape[-1]))[:,0] #[B, 21]
                    probs_masked = probs*(1.0-omit_AA_mask_gathered)
                    probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]

                log_probs = torch.log(probs + 1e-20)

                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21),
                                   (chain_mask_gathered[:, :, None, ] * probs[:, None, :]).float())
                all_log_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21),
                                   (chain_mask_gathered[:, :, None, ] * log_probs[:, None, :]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:,None])
            S_t = (S_t*chain_mask_gathered+S_true_gathered*(1.0-chain_mask_gathered)).long()

            # feed the current output S_t back into the decoder
            temp1 = self.actor.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)

            # store the current output S_t into the result
            S.scatter_(1, t[:,None], S_t)
        output_dict = {"S": S,
                       "probs": all_probs,
                       "log_probs": all_log_probs,
                       "decoding_order": decoding_order}
        return output_dict
