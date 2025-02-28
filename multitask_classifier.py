import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from optimizer import AdamaxW
from tqdm import tqdm
from itertools import cycle

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_multitask, test_model_multitask
from pcgrad import PCGrad
from smart_pytorch import SMARTLoss, sym_kl_loss

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        # Pretrained weights from MLM tasks on training dataset
        # self.bert = BertModel.from_pretrained('pretrained_weights.pt', config='bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config='bert-base-uncased')
        # print(self.bert.bert_layers[0].attention_dense.weight)
        # print(self.bert)
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.pair_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_linear = nn.Linear(config.hidden_size, 5)
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.paraphrase_linear_one = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.paraphrase_linear_two = nn.Linear(config.hidden_size, 1)
        self.similarity_linear = nn.Linear(config.hidden_size * 2, 1)
        self.smart_weight = config.smart_weight

    def freeze_bert_layers(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        bert_out = self.bert(input_ids, attention_mask)['pooler_output']
        out = self.dropout(bert_out)
        return out
    
    def predict_sentiment_with_emb(self, out):
        out = self.sentiment_dropout(out)
        out = self.sentiment_linear(out)
        return out
    
    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        out = self.forward(input_ids, attention_mask)
        return self.predict_sentiment_with_emb(out)
    
    def predict_paraphrase_with_emb(self, out_1, out_2):
        pair_out = torch.cat((out_1, out_2), dim=1)

        # torch.diag(out_1 @ out_2.T)  # Use dot product for paraphrase detection
        linear_one_out = self.paraphrase_linear_one(pair_out)
        linear_two_out = torch.squeeze(self.paraphrase_linear_two(linear_one_out), dim=1)
        return linear_two_out + self.cos_similarity(out_1, out_2)
    
    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        out_1 = self.forward(input_ids_1, attention_mask_1)
        out_2 = self.forward(input_ids_2, attention_mask_2)
        return self.predict_paraphrase_with_emb(out_1, out_2)
    
    def predict_similarity_with_emb(self, out_1, out_2):
        pair_out = torch.cat((out_1, out_2), dim=1)
        # Added a linear layer, better performance on dev set, but more overfitting 
        return torch.squeeze(self.similarity_linear(pair_out), dim=1) + (self.cos_similarity(out_1, out_2) + 1) * 5/2  # Scale it to 0-5
    
    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        out_1 = self.forward(input_ids_1, attention_mask_1)
        out_2 = self.forward(input_ids_2, attention_mask_2)
        return self.predict_similarity_with_emb(out_1, out_2)




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

# Helper function for calculate pearson correlation
def pearson_corr(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args, isRegression=True)
    para_dev_data = SentencePairDataset(para_dev_data, args, isRegression=True)
    # Note: we only use args.batch_size here to prevent out of memory problem (not using the whole dataset)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                      collate_fn=para_dev_data.collate_fn)
    
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size= args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size= args.batch_size,
                                      collate_fn=sts_dev_data.collate_fn)
    
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'smart_weight': 0.5}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    # Use PCGrad to wrap optimizer for gradient surgery
    optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    best_dev_acc = 0

    # Helper function to calculate loss given a batch
    def forward_prop(batch, pair_data=False, regression=False, return_logits=False, return_emb=False):
        result = {}
        emb = None
        if pair_data:
            b_ids_1, b_mask_1, b_labels = (batch['token_ids_1'],
                                        batch['attention_mask_1'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_labels = b_labels.to(device)

            b_ids_2, b_mask_2 = (batch['token_ids_2'],
                                        batch['attention_mask_2'])

            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            
            if return_emb: emb = (model(b_ids_1, b_mask_1), model(b_ids_2, b_mask_2))
            if regression:
                # Similarty sts
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                # Adding correlation as loss
                loss = 1 - pearson_corr(logits, b_labels.view(-1))
                # loss = F.mse_loss(logits, b_labels.view(-1), reduction='mean') + \
                #         F.cosine_embedding_loss(emb[0], emb[1], b_labels.view(-1), reduction='mean')
            else:
                # Paraphrase
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.view(-1), reduction='mean')
        else:
            # Sentiment SST
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)
            if return_emb: emb = model(b_ids, b_mask)
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')

        result['loss'] = loss
        if return_logits:
            result['logits'] = logits
        if return_emb:
            result['emb'] = emb
        return result
    
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        sst_train_cycle_loader = cycle(sst_train_dataloader)
        sts_train_cycle_loader = cycle(sts_train_dataloader)
        # Train on the entire para dataset once, run the other two several times
        # Setup for SMART
        sst_eval_fn = model.predict_sentiment_with_emb
        para_eval_fn = lambda x: model.predict_paraphrase_with_emb(x[0], x[1])
        sts_eval_fn = lambda x: model.predict_similarity_with_emb(x[0], x[1])
        smart_loss_sst = SMARTLoss(eval_fn=sst_eval_fn, loss_fn = sym_kl_loss)
        smart_loss_para = SMARTLoss(eval_fn=para_eval_fn, loss_fn = sym_kl_loss)
        smart_loss_sts = SMARTLoss(eval_fn=sts_eval_fn, loss_fn = sym_kl_loss)
        for para_batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            sst_batch = next(sst_train_cycle_loader)
            sts_batch = next(sts_train_cycle_loader)
            sst_forward = forward_prop(sst_batch, return_emb=True, return_logits=True)
            para_forward = forward_prop(para_batch, pair_data=True, return_emb=True, return_logits=True)
            sts_forward = forward_prop(sts_batch, pair_data=True, regression=True, return_emb=True, return_logits=True)
            # Do not need much regularization on paraphrase
            # + model.smart_weight * smart_loss_para(torch.stack(para_forward['emb']), para_forward['logits'])
            losses = [sst_forward['loss'] + model.smart_weight * smart_loss_sst(sst_forward['emb'], sst_forward['logits']), 
                      para_forward['loss'],
                      sts_forward['loss'] + model.smart_weight * smart_loss_sts(torch.stack(sts_forward['emb']), sts_forward['logits'])]
            # losses = [sst_forward['loss'], para_forward['loss'], sts_forward['loss']]  # Without using SMART
            optimizer.pc_backward(losses)
            
            optimizer.step()
            train_loss += torch.mean(torch.tensor(losses))
            num_batches += 1
            # With batch_size 10, total 14150 batches
            if num_batches > 600:
                break

        train_loss = train_loss / (num_batches)
        if epoch % 4 == 0 and epoch > 0:
            para_acc, _, _, sst_acc, _, _, sts_corr, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
            train_acc = (para_acc + sst_acc + sts_corr) / 3
        else:
            train_acc = 0
        para_acc, _, _, sst_acc, _, _, sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        dev_acc = (para_acc + sst_acc + sts_corr) / 3
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    '''
    # Freeze the BERT embedding layers and train the linear layers for each task separately
    print('Freeze the BERT embedding layers and train the linear layers for each task separately')
    print('------Start training on SST------')
    model.freeze_bert_layers()
    # First train sst
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for sst_batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            loss = forward_prop(sst_batch)['loss']

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc = 0
        dev_acc, dev_f1, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device, eval_para=False)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


    print('------Start training on Paraphrase------')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for para_batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            loss = forward_prop(para_batch, pair_data=True)['loss']

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 600:
                break
        train_loss = train_loss / (num_batches)

        train_acc = 0
        dev_acc, dev_f1, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    
    print('------Start training on STS------')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for sts_batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            loss = forward_prop(sts_batch, pair_data=True, regression=True)['loss']

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            
        train_loss = train_loss / (num_batches)

        train_acc = 0
        dev_acc, dev_f1, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device, eval_para=False)
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
    '''

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
