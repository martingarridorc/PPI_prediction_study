import data.data as d
import models.baseline2d as baseline2d
import models.fc2_20_2_dense as richoux
import models.dscript_like as dscript_like
import models.attention as attention
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
import argparse



def metrics(y_true, y_pred):
    true_positive = torch.sum((y_true == 1) & (y_pred == 1)).item()
    false_positive = torch.sum((y_true == 0) & (y_pred == 1)).item()
    true_negative = torch.sum((y_true == 0) & (y_pred == 0)).item()
    false_negative = torch.sum((y_true == 1) & (y_pred == 0)).item()

    pred_positive = true_positive + false_positive
    real_positive = true_positive + false_negative

    if real_positive == 0:
        accuracy = precision = recall = f1_score = 0
    else:
        accuracy = (true_positive + true_negative) / len(y_true)
        precision = true_positive / pred_positive if pred_positive > 0 else 0
        recall = true_positive / real_positive if real_positive > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score, true_positive, false_positive, true_negative, false_negative



data_name = "gold_stand"
model_name = "Rich-ATT"
learning_rate = 0.001
num_epochs = 25
bs = 2
max = 10000
subset = True
subset_size = 0.5
use_embeddings = True
mean_embedding = False
embedding_dim = 1280
use_wandb = False
early_stopping = True
forward_expansion = 1
dropout = 0.3
num_heads = 4


# example call: python main.py -d gold_stand -lr 0.001 -epo 50 -bs 1024 --max 1166
#               -s True -ss 0.5 -e True -m True -ed 2560 -wb True

if True:   #set false for debug, True for training on server
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-data', '--data_name', type=str, default='gold_stand', help='name of dataset')
    parser.add_argument('-model', '--model_name', type=str, default='dscript_like', help='name of model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-epoch', '--num_epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('-batch', '--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('-max', '--max_seq_len', type=int, default=10000, help='max sequence length')
    parser.add_argument('-sub', '--subset', action='store_true', help='use subset')
    parser.add_argument('-subsize', '--subset_size', type=float, default=0.5, help='subset size')
    parser.add_argument('-emb', '--use_embeddings', action='store_true', help='use embeddings')
    parser.add_argument('-mean', '--mean_embedding', action='store_true', help='use mean embedding')
    parser.add_argument('-emb_dim', '--embedding_dim', type=int, default=2560, help='embedding dimension')
    parser.add_argument('-wandb', '--use_wandb', action='store_true', help='use wandb')
    parser.add_argument('-es', '--early_stopping', type=int, default=0, help='early stopping patience')
    parser.add_argument('-fe', '--forward_expansion', type=int, default=1, help='forward expansion for attention module')
    parser.add_argument('-dropout', '--dropout', type=float, default=0.3, help='dropout value between 0 and 1')
    parser.add_argument('-heads', '--num_heads', type=int, default=4, help='number of heads for attention')

    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    bs = args.batch_size
    max = args.max_seq_len
    subset = args.subset
    subset_size = args.subset_size
    use_embeddings = args.use_embeddings
    mean_embedding = args.mean_embedding
    embedding_dim = args.embedding_dim
    use_wandb = args.use_wandb
    early_stopping = args.early_stopping
    forward_expansion = args.forward_expansion
    dropout = args.dropout
    num_heads = args.num_heads

per_tok_models = ["baseline2d", "dscript_like", "selfattention", "crossattention",
                   "ICAN_cross", "D-ATT_script", "Rich-ATT"]
mean_models = ["richoux"]

# some cases, could be done better or more elegant
if model_name in per_tok_models:
    mean_embedding = False
    use_embeddings = True
    use_2d_data = True
elif model_name in mean_models:
    use_2d_data = False
else:
    raise ValueError("The model name does not exist. Please check for spelling errors.")

train_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_train_all_seq.csv"
test_data = "/nfs/home/students/t.reim/bachelor/pytorchtest/data/" + data_name + "/" + data_name + "_test_all_seq.csv"

if embedding_dim == 2560:
    emb_name = 'esm2_t36_3B'
    layer = 36
elif embedding_dim == 1280: 
    emb_name = 'esm2_t33_650'
    layer = 33
elif embedding_dim == 5120:
    emb_name = 'esm2_t48_15B'
    layer = 48

if mean_embedding:
    emb_type = 'mean'
else:
    emb_type = 'per_tok'


if use_embeddings:
    embedding_dir = "/nfs/scratch/t.reim/embeddings/" + emb_name + "/" + emb_type + "/"

config={"learning_rate": learning_rate,
        "epochs": num_epochs,
        "max_size": max,
        "subset": subset,
        "subset_size": subset_size,
        "batchsize": bs,
        "use_embeddings": use_embeddings,
        "embedding_name": emb_name,
        "mean_embedding": mean_embedding,
        "embedding_dim": embedding_dim,
        "dataset": data_name,
        "model": model_name,
        "use_wandb": use_wandb,
        "early_stopping": early_stopping,
        "forward_expansion": forward_expansion,
        "dropout": dropout,
        "num_heads": num_heads}

if use_wandb == True:
    wandb.init(project="bachelor",
               name=f"{model_name}_{emb_name}_{subset_size}",
               config=config)    
    
print("Parameters:")
for key, value in config.items():
    print(f"{key}: {value}")


# create Datasets, dscript and baseline2d need dataset where embeddings are created later on in the model call

if use_2d_data:
    dataset = d.dataset2d(train_data, layer, max, embedding_dir)
else:
    dataset = d.MyDataset(train_data, layer, max, use_embeddings, mean_embedding, embedding_dir)

if subset:
    sampler = data.RandomSampler(dataset, num_samples=int(len(dataset)*subset_size), replacement=True)
    dataloader = data.DataLoader(dataset, batch_size=bs, sampler=sampler)
    print("Using Subset")
    print("Subset Size: ", int(len(dataset)*subset_size))
else:
    dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=True)


if use_2d_data:
    vdataset = d.dataset2d(test_data, layer, max, embedding_dir)
else:
    vdataset = d.MyDataset(test_data, layer, max, use_embeddings, mean_embedding, embedding_dir)

if subset:
    sampler = data.RandomSampler(vdataset, num_samples=int(len(vdataset)*subset_size), replacement=True)
    vdataloader = data.DataLoader(vdataset, batch_size=bs, sampler=sampler)
    print("Val Subset Size: ", int(len(vdataset)*subset_size))
else:
    vdataloader = data.DataLoader(vdataset, batch_size=bs, shuffle=True)

if use_embeddings:
    print("Using Embeddings: ", emb_name, " Mean: ", mean_embedding)
    insize = embedding_dim
else:     
    print("Not Using Embeddings")  
    print("Max Sequence Length: ", dataset.__max__())
    insize = dataset.__max__()*24

model_mapping = {
    "dscript_like": dscript_like.DScriptLike(embed_dim=insize, d=100, w=7, h=50, x0=0.5, k=20),
    "richoux": richoux.FC2_20_2Dense(embed_dim=insize),
    "baseline2d": baseline2d.baseline2d(embed_dim=insize, h=0),
    "selfattention": attention.SelfAttInteraction(embed_dim=insize, num_heads=num_heads, dropout=dropout, forward_expansion=forward_expansion),
    "crossattention": attention.CrossAttInteraction(embed_dim=insize, num_heads=num_heads, dropout=dropout, forward_expansion=forward_expansion),
    "ICAN_cross": attention.ICAN_cross(embed_dim=insize, num_heads=num_heads, forward_expansion=forward_expansion),
    "D-ATT_script": attention.AttentionDscript(embed_dim=insize, num_heads=num_heads, forward_expansion=forward_expansion),
    "Rich-ATT": attention.AttentionRichoux(embed_dim=insize, num_heads=num_heads, forward_expansion=forward_expansion)
}

model = model_mapping.get(model_name)
if model is None:
    raise ValueError(f"Invalid model_name: {model_name}")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)  

if torch.cuda.is_available():
    print("Using CUDA")
    model = model.cuda()
    criterion = criterion.cuda()
    device = torch.device("cuda")
else:    
    print("Using CPU")
    device = torch.device("cpu")    


best_val_acc = 0.0
es = early_stopping != 0
patience = early_stopping
counter = 0

for epoch in range(num_epochs):
    epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        if use_2d_data:
            outputs = model.batch_iterate(batch, device, layer, embedding_dir)
        else:
            tensor = batch['tensor'].to(device)
            outputs = model(tensor) 
        labels = batch['interaction']
        labels = labels.unsqueeze(1).float()
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        predicted_labels = torch.round(outputs.float())
        met = metrics(labels.to(device), predicted_labels)
        epoch_acc += met[0]
        epoch_prec += met[1]
        epoch_rec += met[2]
        epoch_f1 += met[3]
        tp, fp, tn, fn = tp + met[4], fp + met[5], tn + met[6], fn + met[7]
    avg_acc = epoch_acc / len(dataloader)
    avg_prec = epoch_prec / len(dataloader)
    avg_rec = epoch_rec / len(dataloader)
    avg_f1 = epoch_f1 / len(dataloader)
    avg_loss = epoch_loss / len(dataloader)
    if use_wandb == True:
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch_accuracy": avg_acc,
            "epoch_precision": avg_prec,
            "epoch_recall": avg_rec,
            "epoch_f1_score": avg_f1
        })
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {avg_acc}, Precision: {avg_prec}, Recall: {avg_rec}, F1 Score: {avg_f1}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    model.eval()
    val_loss, val_acc, val_prec, val_rec, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for val_batch in vdataloader:
            if use_2d_data:
                val_outputs = model.batch_iterate(val_batch, device, layer, embedding_dir)
            else:
                val_inputs = val_batch['tensor'].to(device)
                val_outputs = model(val_inputs)

            val_labels = val_batch['interaction']
            val_labels = val_labels.unsqueeze(1).float()
            predicted_labels = torch.round(val_outputs.float())
            met = metrics(val_labels.to(device), predicted_labels)
            val_acc += met[0]
            val_prec += met[1]
            val_rec  += met[2]
            val_f1  += met[3]
            val_loss += criterion(val_outputs, val_labels.to(device))

            # Compute TP, FP, TN, FN
            tp, fp, tn, fn = tp + met[4], fp + met[5], tn + met[6], fn + met[7]


    avg_loss = val_loss / len(vdataloader)
    avg_acc = val_acc / len(vdataloader)
    avg_prec = val_prec / len(vdataloader)
    avg_rec = val_rec / len(vdataloader)
    avg_f1 = val_f1 / len(vdataloader)
    if use_wandb == True:
        wandb.log({
            "val_loss": avg_loss,
            "val_accuracy": avg_acc,
            "val_precision": avg_prec,
            "val_recall": avg_rec,
            "val_f1_score": avg_f1
        })  
    print(f"Epoch {epoch+1}/{num_epochs}, Average Val Loss: {avg_loss}, Val Accuracy: {avg_acc}, Val Precision: {avg_prec}, Val Recall: {avg_rec}, Val F1 Score: {avg_f1}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    if es:
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered. No improvement in validation accuracy. Best epoch: " + str(epoch+1 - patience))
                break

if use_embeddings:
    torch.save(model.state_dict(), '/nfs/home/students/t.reim/bachelor/pytorchtest/models/pretrained/'+model_name+ '_'+ data_name +'_'+ emb_name+'_'+ emb_type+'.pt')
else:
    torch.save(model.state_dict(), '/nfs/home/students/t.reim/bachelor/pytorchtest/models/pretrained/'+model_name+'_'+ data_name +'_no_emb.pt')

