import sys
import os
import time
import datetime
import argparse
import math
#import random
from sklearn.utils import shuffle
import numpy as np
#from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable

from data_preprocess import DataManager
from model import HR_BiLSTM
from model import ABWIM

def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx*batch_size:(idx+1)*batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data

def cal_acc(sorted_score_label):
    if sorted_score_label[0][1] == 1:
        return 1
    else:
        return 0

def cal_mrr(sorted_score_label):
    for i in range(len(sroted_score_label)):
        if sorted_score_label[i][1] == 1:
            return 1/i

def extract_error(sorted_score_label):
    error_case = []
    for score, label, idx in sorted_score_label:
        if label == 1:
            break
        else:
            error_case.append(idx)
    return error_case

def save_best_model(model):
    now = datetime.datetime.now()

    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    if args.save_model_path == '':
        args.save_model_path = f'save_model/{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}.pt'
        with open('log.txt', 'a') as outfile:
            outfile.write(str(args)+'\n')
    
    print('save model at {}'.format(args.save_model_path))
    with open(args.save_model_path, 'wb') as outfile:
        torch.save(model, outfile)

def train(args):
    # Build model
    print('Build model')
    if args.model == 'ABWIM':
        q_len = corpus.maxlen_q
        r_len = corpus.maxlen_w + corpus.maxlen_r
        #print('q_len', q_len, 'r_len', r_len)
        model = ABWIM(args.dropout, args.hidden_size, corpus.word_embedding, corpus.rela_embedding, q_len, r_len).to(device)
    elif args.model == 'HR-BiLSTM':
        model = HR_BiLSTM(args.dropout, args.hidden_size, corpus.word_embedding, corpus.rela_embedding).to(device)
    #print(model)

    if args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    best_model = None
    best_val_loss = None
    train_start_time = time.time()
    print_str = ''

    earlystop_counter = 0
    global_step = 0

    for epoch_count in range(0, args.epoch_num):
        model.train()

        total_loss, total_acc = 0.0, 0.0
        nb_question = 0
        epoch_start_time = time.time()

        for batch_count, batch_data in enumerate(train_data, 1):
            variable_start_time = time.time()
            if args.batch_type == 'batch_question':
                training_objs = [obj for q_obj in batch_data for obj in q_obj]
                question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
                nb_question += len(batch_data)
            elif args.batch_type == 'batch_obj':
                question, pos_relas, pos_words, neg_relas, neg_words = zip(*batch_data)

            q = Variable(torch.LongTensor(question)).to(device)
            p_relas = Variable(torch.LongTensor(pos_relas)).to(device)
            p_words = Variable(torch.LongTensor(pos_words)).to(device)
            n_relas = Variable(torch.LongTensor(neg_relas)).to(device)
            n_words = Variable(torch.LongTensor(neg_words)).to(device)
            ones = Variable(torch.ones(len(question))).to(device)
            variable_end_time = time.time()
            
            model.zero_grad()
            all_pos_score = model(q, p_relas, p_words)
            all_neg_score = model(q, n_relas, n_words)

            loss = loss_function(all_pos_score, all_neg_score, ones)
            loss.backward()
            optimizer.step()
            #writer.add_scalar('data/pre_gen_loss', loss.item(), global_step)
            global_step += 1
            if torch.__version__ == '0.3.0.post4':
                total_loss += loss.data.cpu().numpy()[0]
            else:
                total_loss += loss.data.cpu().numpy()
            average_loss = total_loss / batch_count

            # Calculate accuracy and f1
            if batch_count % args.print_every == 0:
#                if args.batch_type == 'batch_question':
#                    all_pos = all_pos_score.data.cpu().numpy()
#                    all_neg = all_neg_score.data.cpu().numpy()
#                    start, end = 0, 0
#                    for idx, q_obj in enumerate(batch_data):
#                        end += len(q_obj)
#                        score_list = [all_pos[start]]
#                        batch_neg_score = all_neg[start:end]
#                        start = end
#                        label_list = [1]
#                        for ns in batch_neg_score:
#                            score_list.append(ns)
#                        label_list += [0] * len(batch_neg_score)
#                        #print('len(score_list), score_list')
#                        #print(len(score_list), score_list)
#                        #print('len(label_list), label_list')
#                        #print(len(label_list), label_list)
#                        score_label = [(x, y) for x, y in zip(score_list, label_list)]
#                        sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
#                        total_acc += cal_acc(sorted_score_label)
#                    #average_acc = total_acc / (batch_count * args.batch_size)
#                    #average_acc = total_acc / nb_question
#                    average_acc = total_acc / batch_count # batch_type = question, batch_size = 1
#                    elapsed = time.time() - epoch_start_time
#                    print_str = f'Epoch {epoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} #_question:{nb_question}'
#                else:
                elapsed = time.time() - epoch_start_time
                print_str = f'Epoch {epoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f}'
                print('\r', print_str, end='')
        print('\r', print_str, end='#\n')
        val_print_str, val_loss, _, _ = evaluation(model, 'dev', global_step)
        print('\rVal', val_print_str, '#\n')

        # this section handle earlystopping
        if not best_val_loss or val_loss < best_val_loss:
            earlystop_counter = 0
            best_model = model
            save_best_model(best_model)
            best_val_loss = val_loss
        else:
            earlystop_counter += 1
        if earlystop_counter >= args.earlystop_tolerance:
            print('EarlyStopping!')
            print(f'Total training time {time.time()-train_start_time:.2f}')
            break
    return best_model

def evaluation(model, mode='dev', global_step=None):
    model_test = model.eval()
    start_time = time.time()
    total_loss, total_acc = 0.0, 0.0
    error_idx = []

    if mode == 'test':
        input_data = test_data
        #print(model_test)
        print(model_test)
    else:
        input_data = val_data
    nb_question = sum(len(batch_data) for batch_data in input_data)
    count = 0;
    #print('nb_question', nb_question)
    for batch_count, batch_data in enumerate(input_data, 1):
        count+=1
        training_objs = [obj for q_obj in batch_data for obj in q_obj]
        question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
        q = Variable(torch.LongTensor(question)).to(device)
        p_relas = Variable(torch.LongTensor(pos_relas)).to(device)
        p_words = Variable(torch.LongTensor(pos_words)).to(device)
        n_relas = Variable(torch.LongTensor(neg_relas)).to(device)
        n_words = Variable(torch.LongTensor(neg_words)).to(device)
        ones = Variable(torch.ones(len(question))).to(device)
        
        pos_score = model_test(q, p_relas, p_words)
        #print('\reval ', batch_count, pos_score, '')
        #pos_alpha = model.ret_alpha(q, p_relas, p_words)
        neg_score = model_test(q, n_relas, n_words)
        loss = loss_function(pos_score, neg_score, ones)
        if torch.__version__ == '0.3.0.post4':
            total_loss += loss.data.cpu().numpy()[0]
        else:
            total_loss += loss.data.cpu().numpy()
        average_loss = total_loss / batch_count

        # Calculate accuracy and f1
        all_pos = pos_score.data.cpu().numpy()
        #all_alpha = pos_alpha.data.cpu().numpy()
        all_neg = neg_score.data.cpu().numpy()
        start, end = 0, 0
        for idx, q_obj in enumerate(batch_data):
            end += len(q_obj)
            #print('start', start, 'end', end)
            #input('Enter')
            score_list = [all_pos[start]]
            #score_list_alpha = [all_alpha[start]]
            label_list = [1]
            batch_neg_score = all_neg[start:end]
            for ns in batch_neg_score:
                score_list.append(ns)
            label_list += [0] * len(batch_neg_score)
            start = end
            score_label = [(x, y, idx) for idx, (x, y) in enumerate(zip(score_list, label_list))]
            #alpha_label = [(x, y) for x, y in zip(score_list_alpha, label_list)]
            #print("\n")
            #print(score_label[:10])
            #print("\n")
            #print('len(score_list)', len(score_list), 'len(label_list)', len(label_list), 'len(score_label)', len(score_label))
            sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
            #sorted_alpha_label = sorted(alpha_label, key=lambda x:x[0], reverse=True)
            #print("\n")
            #print(sorted_score_label)
            #file = open("result.txt","w")
            #file.write(sorted_alpha_label)
            #file.close()
            total_acc += cal_acc(sorted_score_label)
            if mode == 'test':
                error_cases = extract_error(sorted_score_label)
                error_idx.append(error_cases)
            #print(total_acc)
#        acc1 = total_acc / (batch_count * args.batch_size)
        # modify: batch_count should be batch_count * question_per_batch
        acc2 = total_acc / batch_count
        time_elapsed = time.time()-start_time
        print_str = f'Batch {batch_count} Spend Time:{time_elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{acc2:.4f}'
        print('\rVal', print_str, end='')
   # print("\n")
    #print(count)
#    if mode == 'dev':
#        writer.add_scalar('val_loss', average_loss.item(), global_step)

    time_elapsed = time.time()-start_time
    average_acc = total_acc / nb_question
#    print('acc1', acc1)
#    print('acc2', acc2)
#    print('average_acc', average_acc)
#    print(question_counter, nb_question)
    print_str = f'Batch {batch_count} Spend Time:{time_elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} # question:{nb_question}'
    return print_str, average_loss, average_acc, error_idx

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    #random.seed(1234)
    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument('-train', default=False, action='store_true')
    parser.add_argument('-test', default=False, action='store_true')
    parser.add_argument('--model', type=str, required=True) # [ABWIM/HR-BiLSTM]
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=2.0) # [0.1/0.5/1.0/2.0]
    parser.add_argument('--hidden_size', type=int, default=100) # [50/100/200/400]
    parser.add_argument('--optimizer', type=str, default='Adadelta')
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--batch_type', type=str, default='batch_question') # [batch_question/batch_obj]
    parser.add_argument('--batch_question_size', type=int, default=1) #32
    parser.add_argument('--batch_obj_size', type=int, default=128)
    parser.add_argument('--earlystop_tolerance', type=int, default=5)
    parser.add_argument('--save_model_path', type=str, default='')
    parser.add_argument('--pretrain_model', type=str, default=None)
    parser.add_argument('--data_type', type=str, default='WQ') # [SQ/WQ]
    parser.add_argument('--print_every', type=int, default=10)
    args = parser.parse_args()
    
    if args.model == 'ABWIM':
        args.margin = 0.1
        args.optimizer = 'Adadelta'
    loss_function = torch.nn.MarginRankingLoss(margin=args.margin)

    # Load data
    corpus = DataManager(args.data_type)

    if args.train:
        if args.data_type == 'SQ':
            train_data = corpus.token_train_data
            train_data_len = corpus.train_data_len
            val_data = corpus.token_val_data
            print('training data length:', len(train_data))
            print('validation data length:', len(val_data))
        else:
            # shuffle training data
            corpus.token_train_data, corpus.train_data_len = shuffle(corpus.token_train_data, corpus.train_data_len, random_state=1234)
            # split training data to train and validation
            split_num = int(0.9*len(corpus.token_train_data))
            print('split_num=', split_num)
            train_data = corpus.token_train_data[:split_num]
            train_data_len = corpus.train_data_len[:split_num]
            val_data = corpus.token_train_data[split_num:]
            print('training data length:', len(train_data))
            print('validation data length:', len(val_data))

        if args.batch_type == 'batch_question':
            # batchify questions, uncomment Line 119, 120
            train_data = batchify(train_data, args.batch_question_size)
        elif args.batch_type == 'batch_obj':
            # batchify train_objs, uncomment Line 121
            flat_train_data = [obj for q_obj in train_data for obj in q_obj]
            flat_train_data_len = [obj for q_obj in train_data_len for obj in q_obj]
            print('len(flat_train_data)', len(flat_train_data))
            print('len(flat_train_data_len)', len(flat_train_data_len))
            flat_train_data, flat_train_data_len = shuffle(flat_train_data, flat_train_data_len, random_state=1234)
            train_data = batchify(flat_train_data, args.batch_obj_size)
            train_data_len = batchify(flat_train_data_len, args.batch_obj_size)
        val_data = batchify(val_data, args.batch_question_size)

        # Create SummaryWriter
        #writer = SummaryWriter(log_dir='save_model/tensorboard_log')
        train(args)

    if args.test:
        print('test data length:', len(corpus.token_test_data))
        test_data = batchify(corpus.token_test_data, args.batch_question_size)
        if args.pretrain_model == None:
            print('Load best model', args.save_model_path)
            with open(args.save_model_path, 'rb') as infile:
                model = torch.load(infile)
        else:
            print('Load pretrain model', args.pretrain_model)
            with open(args.pretrain_model, 'rb') as infile:
                model = torch.load(infile) 
        log_str, _, test_acc, error_list = evaluation(model, 'test')
        
        print(log_str)
        print(test_acc)

        test_data = [obj for batch in test_data for obj in batch]
        if(len(test_data) != len(error_list)):
            print('Error: length of error list does not match with test data.', len(test_data), len(error_list))
            sys.exit()
        
        for idx, error_cases in enumerate(error_list):
        #Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
            if len(error_cases) > 0:
                question = test_data[idx][0][0]
                question = corpus.idx2word(question, id_type='word')
                correct_relation = test_data[idx][0][1]
                correct_relation = corpus.idx2word(correct_relation, id_type='relation')
                for error_idx in error_cases:
                    error_relation = test_data[idx][error_idx-1][3]
                    error_relation = corpus.idx2word(error_relation, id_type='relation')
                    with open(f'error_analysis_{args.save_model_path[-11:-3]}.txt', 'a') as outfile:
                        outfile.write(question+'\t'+correct_relation+'\t'+error_relation+'\n')

        with open('log.txt', 'a') as outfile:
            if args.pretrain_model == None:
                outfile.write(str(test_acc)+'\t'+args.save_model_path+'\n')
            else:
                outfile.write(str(test_acc)+'\t'+args.pretrain_model+'\n')

    # Close writer
    #writer.close()

