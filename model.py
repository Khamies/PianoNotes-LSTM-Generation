import torch




class LSTM_Music(torch.nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers=1):
    super(LSTM_Music, self).__init__()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Variables
    self.num_layers = num_layers
    self.lstm_factor = num_layers
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size

  
    # X: bsz * seq_len * vocab_size 
    # Embedding
    self.embed = torch.nn.Linear(in_features= self.vocab_size , out_features=self.embed_size)

    #    X: bsz * seq_len * vocab_size 
    #    X: bsz * seq_len * embed_size

    # Encoder Part
    self.lstm = torch.nn.LSTM(input_size= self.embed_size,hidden_size= self.hidden_size, batch_first=True, num_layers= self.num_layers)
    self.output = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.vocab_size)
    self.log_softmax = torch.nn.LogSoftmax(dim=1) # we use binary cross entropy. logits: (batch_size*seq_len*notes_size, 2)

  def init_hidden(self, batch_size):
    hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    return (hidden_cell, state_cell)

  def get_embedding(self, x):
    x_embed = self.embed(x)
    
    # Total length for pad_packed_sequence method = maximum sequence length
    maximum_sequence_length = x_embed.size(1)

    return x_embed, maximum_sequence_length
  

  def forward(self, x,sentences_length,states):
    
    """
      x : bsz * seq_len
    
      hidden_encoder: ( num_lstm_layers * bsz * hidden_size, num_lstm_layers * bsz * hidden_size)

    """
    # Get Embeddings
    x_embed, maximum_padding_length = self.get_embedding(x)

    # Packing the input
    # print("&&&&&&&&&&&&&", x_embed.size(), x_embed.dtype, type(sentences_length), sentences_length.size())
    packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed, lengths= sentences_length, batch_first=True, enforce_sorted=False)


    packed_x_embed, states = self.lstm(packed_x_embed, states)

    x,  sentences_length = torch.nn.utils.rnn.pad_packed_sequence(packed_x_embed, batch_first=True, total_length=maximum_padding_length) # maximum_padding_length: to explicitly enforce the pad_packed_sequence layer to pad the sentences with the tallest sequence length.

    logits = self.output(x)

    # A trick to apply binary cross entropy by using cross entropy loss. 
    neg_logits = (1 - logits)
        
    binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
    # print(binary_logits.size())
    binary_logits = binary_logits.view(-1, 2)

    binary_logits = self.log_softmax(binary_logits)
    return (binary_logits, states)


  def inference(self, n_samples, sos=None):

    # generate random z 
    batch_size = 1
    length = torch.tensor([1])
    idx_sample = []

    if sos is None:
      x = torch.zeros(1,1,self.vocab_size).to(self.device)
      x[:,:,30] = 1


    hidden = self.init_hidden(batch_size)

    with torch.no_grad():
    
      for i in range(n_samples):
        
        pred, hidden = self.forward(x, length, hidden)
        pred = pred.exp()
        # print(pred.size())
        sample = torch.multinomial(pred,1)
        sample = sample.squeeze().unsqueeze(0).unsqueeze(1) #(88,1) -> (1,1,88)
        idx_sample.append(sample)

        x = sample.float()


      note_samples = idx_sample

    return note_samples
