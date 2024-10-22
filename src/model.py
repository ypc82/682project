import sys
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class HR_BiLSTM(nn.Module):
    def __init__(self, dropout_rate, hidden_size, word_emb, rela_emb, model_arch, attention_flag=False, cnn_flag=False):
        super(HR_BiLSTM, self).__init__()
        self.attention_flag = attention_flag
        self.cnn_flag = cnn_flag
        self.cnn_filters = 100
        self.arch_type = model_arch
        self.hidden_size = hidden_size
        self.emb_dim = word_emb.shape[1]
        self.nb_layers = 1
        self.dropout = nn.Dropout(dropout_rate)
        self.cos = nn.CosineSimilarity(1)

        # Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], self.emb_dim)
        self.word_embedding.weight = nn.Parameter(th.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = True
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], self.emb_dim)
        self.rela_embedding.weight = nn.Parameter(th.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = True

        # LSTM layer
        self.bilstm_1 = nn.LSTM(self.emb_dim, hidden_size, num_layers=self.nb_layers, bidirectional=True, batch_first=True)
        self.bilstm_2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=self.nb_layers, bidirectional=True, batch_first=True)
        
        # CNN layer
        self.cnn_1 = nn.Conv1d(hidden_size*4, self.cnn_filters, 1)
        self.cnn_2 = nn.Conv1d(hidden_size*4, self.cnn_filters, 3)
        self.cnn_3 = nn.Conv1d(hidden_size*4, self.cnn_filters, 5)

        # Attention Layer
        if self.attention_flag == True:
            self.attn = Attention(hidden_size*2)

    def revert_order(self, sorted_seq, sorted_seqidx):
        ori_seq = sorted_seq.clone()
        for new_idx, ori_idx in enumerate(sorted_seqidx):
            ori_seq[ori_idx] = sorted_seq[new_idx]
        return ori_seq

    def pack_seq(self, ori_seq, seqlen):
        sorted_seqlen, sorted_seqidx = th.sort(seqlen, descending=True)
        sorted_seq = ori_seq.clone()
        for new_idx, ori_idx in enumerate(sorted_seqidx):
            sorted_seq[new_idx] = ori_seq[ori_idx]
        packed_seq  = nn.utils.rnn.pack_padded_sequence(sorted_seq, sorted_seqlen, batch_first=True)
        return packed_seq, sorted_seqidx

    def encode_question(self, question, relation):
        if self.arch_type == 'B':
            return question

        elif self.arch_type == 'BB':
            question2, _ = self.bilstm_2(question)
            question2 = self.dropout(question2)
            return question2

        elif self.arch_type == 'BA':
            atten_output, atten_weight = self.attn(question, relation)
            return atten_output

        elif self.arch_type == 'BBR':
            # 1st way of Hierarchical Residual Matching
            question2, _ = self.bilstm_2(question)
            question2 = self.dropout(question2)
            return question + question2
            # 2nd way of Hierarchical Residual Matching
            #q1_max = nn.MaxPool1d(question_out_1.shape[2])(question_out_1)
            #q2_max = nn.MaxPool1d(question_out_2.shape[2])(question_out_2)
            #question_representation = q1_max + q2_max

        elif self.arch_type == 'BBA':
            question2, _ = self.bilstm_2(question)
            question2 = self.dropout(question2)
            atten_output, atten_weight = self.attn(question2, relation)
            return atten_output

        elif self.arch_type == 'BAB':
            atten_output, atten_weight = self.attn(question, relation)
            question2, _ = self.bilstm_2(atten_output)
            question2 = self.dropout(question2)
            return question2

        elif self.arch_type == 'BABR':
            atten_output, atten_weight = self.attn(question, relation)
            question2, _ = self.bilstm_2(atten_output)
            question2 = self.dropout(question2)
            return atten_output + question2
        
        elif self.arch_type == 'BABA':
            atten_output, atten_weight = self.attn(question, relation)
            question2, _ = self.bilstm_2(atten_output)
            question2 = self.dropout(question2)
            atten_output2, atten_weight2 = self.attn(question2, relation)
            return atten_output2

        elif self.arch_type == 'BABAR':
            atten_output, atten_weight = self.attn(question, relation)
            question2, _ = self.bilstm_2(atten_output)
            question2 = self.dropout(question2)
            atten_output2, atten_weight2 = self.attn(question2, relation)
            return atten_output + atten_output2

        else:
            print("Error: This model architecture does not exist!")
            sys.exit()
        
    def forward(self, question, rela_relation, word_relation):
        # Embedding Layer
        question = self.dropout(self.word_embedding(question))
        rela_relation = self.dropout(self.rela_embedding(rela_relation))
        word_relation = self.dropout(self.word_embedding(word_relation))

        # Bi-LSTM_1
        # question_out_1.shape = (batch_size, q_len, hidden_size)
        question_out_1, question_hidden = self.bilstm_1(question)
        question_out_1 = self.dropout(question_out_1)

        word_relation_out, word_relation_hidden = self.bilstm_1(word_relation)
        word_relation_out = self.dropout(word_relation_out)

        rela_relation_out, rela_relation_hidden = self.bilstm_1(rela_relation, word_relation_hidden)
        rela_relation_out = self.dropout(rela_relation_out)

        # r.shape = (batch_size, (rela_len+word_len), hidden_size)
        r = th.cat([rela_relation_out, word_relation_out], 1)
        
        # Question Representation
        q = self.encode_question(question_out_1, r)
        #q = self.encode_question(question_out_1, word_relation_out)
        q = q.permute(0, 2, 1)
        question_representation = nn.MaxPool1d(q.shape[2])(q).squeeze(dim=2)
        #print('q', question_representation.shape)

        # Relation Representation
        r = r.permute(0, 2, 1)
        relation_representation = nn.MaxPool1d(r.shape[2])(r).squeeze(dim=2)
        #print('r', relation_representation.shape)

        score = self.cos(question_representation, relation_representation)
        return score

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.ques_embedding = nn.Embedding(args.ques_embedding.shape[0], args.ques_embedding.shape[1])
        self.ques_embedding.weight.requires_grad = False
        self.ques_embedding.weight = nn.Parameter(th.from_numpy(args.ques_embedding).float())
        self.rela_text_embedding = nn.Embedding(args.rela_text_embedding.shape[0], args.rela_text_embedding.shape[1])
        self.rela_text_embedding.weight.requires_grad = False
        self.rela_text_embedding.weight = nn.Parameter(th.from_numpy(args.rela_text_embedding).float())
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.rela_text_embedding.shape[1])
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size,
                           num_layers=args.num_layers, batch_first=False,
                           dropout=args.dropout_rate, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=args.hidden_size*2, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=False,
                            dropout=args.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1)
        self.tanh = nn.Tanh()
        return

    def forward(self, ques_x, rela_text_x, rela_x):
        ques_x = th.transpose(ques_x, 0, 1)
        rela_text_x = th.transpose(rela_text_x, 0, 1)
        rela_x = th.transpose(rela_x, 0, 1)

        ques_x = self.ques_embedding(ques_x)
        rela_text_x = self.rela_text_embedding(rela_text_x)
        rela_x = self.rela_embedding(rela_x)

        ques_hs1, hidden_state = self.encode(ques_x)
        rela_hs, hidden_state = self.encode(rela_x, hidden_state)
        rela_text_hs, hidden_state = self.encode(rela_text_x, hidden_state)

        #h_0 = Variable(th.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).cuda()
        #c_0 = Variable(th.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).cuda()
        h_0 = Variable(th.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).to(device)
        c_0 = Variable(th.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).to(device)
        ques_hs2, _ = self.rnn2(ques_hs1, (h_0, c_0))

        ques_hs = ques_hs1 + ques_hs2
        ques_hs = ques_hs.permute(1, 2, 0)
        ques_h = F.max_pool1d(ques_hs, kernel_size=ques_hs.shape[2], stride=None)
        rela_hs = th.cat([rela_hs, rela_text_hs], 0)
        rela_hs = rela_hs.permute(1, 2, 0)
        rela_h = F.max_pool1d(rela_hs, kernel_size=rela_hs.shape[2], stride=None)

        ques_h = ques_h.squeeze(2)
        rela_h = rela_h.squeeze(2)

        output = self.cos(ques_h, rela_h)
        return output

    def encode(self, input, hidden_state=None, return_sequence=True):
        if hidden_state==None:
            h_0 = Variable(th.zeros([self.args.num_layers*2, len(input[0]), self.args.hidden_size])).to(device)
            c_0 = Variable(th.zeros([self.args.num_layers*2, len(input[0]), self.args.hidden_size])).to(device)
        else:
            h_0, c_0 = hidden_state
        h_input = h_0
        c_input = c_0
        outputs, (h_output, c_output) = self.rnn(input, (h_0, c_0))
        if return_sequence == False:
            return outputs[-1], (h_output, c_output)
        else:
            return outputs, (h_output, c_output)

def new_parameter(*size):
    out = nn.Parameter(th.FloatTensor(*size))
    th.nn.init.xavier_normal(out)
    return out


class ABWIM(nn.Module):
    def __init__(self, dropout_rate, hidden_size, word_emb, rela_emb, q_len, r_len):
        super(ABWIM, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layers = 1
        self.nb_filters = 100
        self.dropout = nn.Dropout(dropout_rate)
        # Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
        self.rela_embedding.weight = nn.Parameter(th.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = False # fix the embedding matrix
        # LSTM layer
        self.bilstm = nn.LSTM(word_emb.shape[1],
                              hidden_size,
                              num_layers=self.nb_layers,
                              bidirectional=True)
        # Attention
        self.W = nn.Parameter(th.rand((hidden_size*2, hidden_size*2))).cuda()
        # CNN layer
        self.cnn_1 = nn.Conv1d(hidden_size*4, self.nb_filters, 1)
        self.cnn_2 = nn.Conv1d(hidden_size*4, self.nb_filters, 3)
        self.cnn_3 = nn.Conv1d(hidden_size*4, self.nb_filters, 5)
        self.activation = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(q_len)
        self.maxpool_2 = nn.MaxPool1d(q_len-2)
        self.maxpool_3 = nn.MaxPool1d(q_len-4)
        self.linear = nn.Linear(self.nb_filters, 1, bias=False)

    def init_hidden(self, batch_size):
        return (Variable(th.zeros(2, batch_size, self.hidden_size)).cuda(),
                Variable(th.zeros(2, batch_size, self.hidden_size)).cuda())
    def ret_alpha(self, question, rela_relation, word_relation):
        question = th.transpose(question, 0, 1)
        rela_relation = th.transpose(rela_relation, 0, 1)
        word_relation = th.transpose(word_relation, 0, 1)
        question = self.word_embedding(question)
        question = self.dropout(question)
        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)
        #self.bilstm.flatten_parameters()
        question_out, _ = self.bilstm(question)
        question_out = question_out.permute(1,2,0)
        question_out = self.dropout(question_out)
        word_relation_out, word_relation_hidden = self.bilstm(word_relation)
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        word_relation_out = self.dropout(word_relation_out)
        rela_relation_out = self.dropout(rela_relation_out)
        relation = th.cat([rela_relation_out, word_relation_out], 0)
        relation = relation.permute(1,0,2)

        # attention layer
        energy = th.matmul(relation, self.W)
        energy = th.matmul(energy, question_out)
        alpha = F.softmax(energy, dim=-1)
        return alpha
    def forward(self, question, rela_relation, word_relation):
        question = th.transpose(question, 0, 1)
        rela_relation = th.transpose(rela_relation, 0, 1)
        word_relation = th.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        question = self.dropout(question)
        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)

        #        self.bilstm.flatten_parameters()
        question_out, _ = self.bilstm(question)
        question_out = question_out.permute(1,2,0)
        question_out = self.dropout(question_out)
        word_relation_out, word_relation_hidden = self.bilstm(word_relation)
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        word_relation_out = self.dropout(word_relation_out)
        rela_relation_out = self.dropout(rela_relation_out)
        relation = th.cat([rela_relation_out, word_relation_out], 0)
        relation = relation.permute(1,0,2)

        # attention layer
        #energy_tmp = energy.view(energy.shape[0], energy.shape[1]*energy.shape[2])
        #alpha = F.softmax(energy_tmp, dim=-1)
        #alpha = alpha.view(energy.shape[0], energy.shape[1], energy.shape[2])

        energy = th.matmul(relation, self.W)
        energy = th.matmul(energy, question_out)
        alpha = F.softmax(energy, dim=-1)
        alpha = alpha.unsqueeze(3)
        relation = relation.unsqueeze(2)
        atten_relation = alpha * relation
        atten_relation = th.sum(atten_relation, 1)
        atten_relation = atten_relation.permute(0, 2, 1)
        M = th.cat((question_out, atten_relation), 1)
        h1 = self.maxpool_1(self.activation(self.cnn_1(M)))
        h1 = self.dropout(h1)
        h2 = self.maxpool_2(self.activation(self.cnn_2(M)))
        h2 = self.dropout(h2)
        h3 = self.maxpool_3(self.activation(self.cnn_3(M)))
        h3 = self.dropout(h3)
        h = th.cat((h1, h2, h3),2)
        h = th.max(h, 2)[0]
        score = self.linear(h).squeeze()
        return score

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.contiguous().view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = th.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = th.bmm(attention_weights, context)

#        # concat -> (batch_size * output_len, 2*dimensions)
#        combined = th.cat((mix, query), dim=2)
#        combined = combined.view(batch_size * output_len, 2 * dimensions)
#
#        # Apply linear_out on every 2nd dimension of concat
#        # output -> (batch_size, output_len, dimensions)
#        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
#        output = self.tanh(output)
#
#        return output, attention_weights
        return mix, attention_weights

if __name__ == '__main__':
    import numpy as np
    dropout = 0.35
    hidden_size = 100
    learning_rate = 1

    word_embedding_path = 'SQ_word_emb_300d.txt'
    rela_embedding_path = 'SQ_rela_emb_300d.txt'
    word_embedding = []
    rela_embedding = []
    with open(word_embedding_path) as infile:
        for line in infile:
            tokens = line.strip().split()
            word_embedding.append([float(x) for x in tokens[1:]])
    with open(rela_embedding_path) as infile:
        for line in infile:
            tokens = line.strip().split()
            rela_embedding.append([float(x) for x in tokens[1:]])
    word_embedding = np.array(word_embedding)
    rela_embedding = np.array(rela_embedding)

    model = HR_BiLSTM(dropout, hidden_size, word_embedding, rela_embedding).cuda()
    optimizer = th.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 1 batch, batch_size = 4
    # question : maxlen = 6
    # pos_relas: maxlen = 3
    # pos_words: maxlen = 5
    question_data = [[[1,2,3,0,0,0],[1,2,3,4,0,0],[1,2,3,4,5,6],[1,2,0,0,0,0]]]
    pos_relas = [[[1,2,3],[1,2,0],[1,2,3],[1,0,0]]]
    pos_words = [[[1,2,0,0,0],[1,2,3,4,5],[1,0,0,0,0],[1,2,3,0,0]]]
    q_len = [[3,4,6,2]]
    pos_r_len = [[3,2,3,1]]
    pos_w_len = [[2,5,1,3]]
    for batch_count, question in enumerate(question_data, 1):
        q_length = Variable(th.LongTensor(q_len[batch_count-1])).cuda()
        pos_r_length = Variable(th.LongTensor(pos_r_len[batch_count-1])).cuda()
        pos_w_length = Variable(th.LongTensor(pos_w_len[batch_count-1])).cuda()

        q = Variable(th.LongTensor(question)).cuda()
        p_relas = Variable(th.LongTensor(pos_relas[batch_count-1])).cuda()
        p_words = Variable(th.LongTensor(pos_words[batch_count-1])).cuda()

        optimizer.zero_grad()
        all_pos_score = model(q, p_relas, p_words)
        #all_pos_score = model(q, p_relas, p_words, q_length, pos_r_length, pos_w_length)
        sys.exit()

