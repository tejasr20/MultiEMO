from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP
import torch
import torch.nn as nn




'''
MultiEMO consists of three key components: unimodal context modeling, multimodal fusion, and emotion classification. 
'''
class MultiEMO(nn.Module):

    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers, 
                 model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device, modalities):
        super().__init__()

        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag

        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)

        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        
        self.visual_fc = nn.Linear(D_m_visual, model_dim)
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        
        self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout, modalities)
        self.fc = nn.Linear(model_dim * len(modalities), model_dim)
        # self.fc = nn.Linear(model_dim * 3, model_dim)
        

        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)
        self.modalities= modalities

 
    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):
        text_features=audio_features= visual_features= None
        if("t" in self.modalities):
            text_features = self.text_fc(texts)
            if self.dataset == 'IEMOCAP':# We empirically find that additional context modeling leads to improved model performances on IEMOCAP
                text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)
            text_features = text_features.transpose(0, 1)
        if("a" in self.modalities):
            audio_features = self.audio_fc(audios)
            audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)
            audio_features = audio_features.transpose(0, 1)
        if("v" in self.modalities):
            visual_features = self.visual_fc(visuals)
            visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)
            visual_features = visual_features.transpose(0, 1)
		#dialoguernn acts at an individual modality level so no change needed for modality support. 
        if self.multi_attn_flag == True and len(self.modalities)>=2: 
            #Change: If we have atleast two modalities, we can do fusion of features. Else use preexisitng features. 
                fused_text_features, fused_audio_features, fused_visual_features = self.multiattn(text_features, audio_features, visual_features)
        else: # this will simply concatenate without using the fusion module. I believe this will be useful for 
            # ablation studies to show the importance of the attn module as well as when we have only a single modality. 
            fused_text_features, fused_audio_features, fused_visual_features = text_features, audio_features, visual_features
        
        if("t" in self.modalities):
            fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
            fused_text_features = fused_text_features[padded_labels != -1]
        if("a" in self.modalities):
             fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
             fused_audio_features = fused_audio_features[padded_labels != -1]
        if("v" in self.modalities):
            fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])
            fused_visual_features = fused_visual_features[padded_labels != -1]
		# simple concatenation followed by an MLP- will need to concat only the features avaliable here. 
        fused_features = torch.cat([v for v in (fused_text_features, fused_audio_features, fused_visual_features) if v is not None], dim = -1)
        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs)
        return fused_text_features, fused_audio_features, fused_visual_features, fc_outputs, mlp_outputs



