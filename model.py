import torch.nn as nn
import torch
import random
import torchvision.models as models
from torchsummary import summary


path = '../pytorch_model_weights/'

# encoder
class Encoder(nn.Module):
    def __init__(self , encoder_dim , decoder_dim):
        super().__init__()
        
        # loading the pretrain resnet50 model
        resnet = models.resnet50(weights=None)
        weights = torch.load(path + 'resnet50-IMAGENET1K_V1.pth')
        resnet.load_state_dict(weights)
        
        # freeze layers
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2] # get the resnet convolution layers
        self.resnet = nn.Sequential(*modules)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        

    def forward(self , images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        outputs = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048) #(batch_size , num_layers , encoder hidden dim)
        
        mean_encoder_out = outputs.mean(dim=1)
        hidden  = self.init_h(mean_encoder_out)  # (batch_size, decoder hidden dim)
        return outputs , hidden 
    
    
#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self,encoder_outputs,hidden):
        u_hs = self.U(encoder_outputs)     #(batch_size,num_layers,attention_dim)
        
        w_ah = self.W(hidden) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        
        return torch.softmax(attention_scores, dim=1)   #(batch_size,num_layers)
    
    
# decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_dim, decoder_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(encoder_dim  + embedding_dim, decoder_dim)
        self.fc_out = nn.Linear(
            encoder_dim  + decoder_dim + embedding_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self,encoder_outputs, encoder_hidden, target_tensor=None):
        # input = [batch size]
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = (batch_size , num_layers , encoder hidden dim)

        
        target_tensor = target_tensor.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(target_tensor))
        
        # embedded = [1, batch size, embedding dim]
        a = self.attention( encoder_outputs , encoder_hidden)
        
        # a = [batch size, num_layers]
        a = a.unsqueeze(1)
        # a = [batch size, 1, num_layers]
        # encoder_outputs = [batch size, num_layers, encoder hidden dim ]

        
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, encoder dim ]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, encoder dim ]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, encoder hidden dim  + embedding dim]
        output, encoder_hidden = self.rnn(rnn_input, encoder_hidden.unsqueeze(0))
        # output = [seq length, batch size, decoder dim ]
        # hidden = [1, batch size, decoder hid dim]
        # output = [1, batch size, decoder dim]
        # hidden = [1, batch size, decoder dim]
        # this also means that output == hidden
        assert (output == encoder_hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]
        return prediction, encoder_hidden.squeeze(0), a.squeeze(1)
    
    
class Seq2SeqAttentionCNN(nn.Module):
    def __init__(self, output_dim, sos_token, eos_token, embedding_dim=256, encoder_dim=2048, 
                 decoder_dim=512, attention_dim=512, dropout=0.25):
        super().__init__()
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.encoder = Encoder(encoder_dim , decoder_dim)
        attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.decoder = Decoder(output_dim, embedding_dim, encoder_dim, decoder_dim, dropout, attention)
        
    
    def inference(self, img, max_output_length):
        img = img.unsqueeze(0)
        encoder_outputs , hidden = self.encoder(img)
        inputs = [self.sos_token]
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(img.device)
            output, hidden , _ = self.decoder(encoder_outputs, hidden, inputs_tensor)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == self.eos_token:
                break
        return inputs[1:]
        
        
    def forward(self, src, trg, teacher_forcing=0.7):
        # for torchsummary (only inputs float data type)
        trg = trg.to(dtype=torch.long)
        
        batch_size, trg_length = trg.shape[0], trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_length, trg_vocab_size).to(trg.device)
        
        # output = [trg_length, batch size, output dim]
        encoder_outputs , hidden  = self.encoder(src)
        inputs = trg[:,0]
        for t in range(1, trg_length):
            output, hidden , _ = self.decoder(encoder_outputs , hidden, inputs)
            # output = [batch size, output dim]
            # print(f'output : {output.shape}')
            outputs[:,t,:] = output
            teacher_force = random.random() < teacher_forcing
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            if teacher_force:
                inputs = trg[:,t] 
            else:
                inputs = top1
            # inputs = [batch size]
        
        return outputs

    
    
def main():
    output_dim = 1000 # length of vocab
    img_size = 224
    seq_len = 64
    sos_token = 1
    eos_token = 2
    model = Seq2SeqAttentionCNN(output_dim, sos_token, eos_token)
    summary(model, [(3,img_size,img_size),(seq_len,)])
    
    
if __name__ == "__main__":
    main()
    
    