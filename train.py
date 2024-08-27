import torch
import torch.nn as nn
import os
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision.transforms as transforms
import torchvision
import tqdm
import argparse
import gc
from vit.model import ViT, ViTBase, ViTHuge, ViTLarge
from vit.dataset import ImageDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "base", 
                        type=str, help="Type of ViT model")
    
    parser.add_argument("--num-classes", default=10, 
                        type=int, help="Number of classes")
    
    parser.add_argument("--patch-size", default=16,
                        type=int, help="Size of image patch")
    
    parser.add_argument("--num-heads", default=12,
                        type=int, help="Number of attention heads")
    
    parser.add_argument("--embed-dim", default=64,
                        type=int, help="Size of each attention head for value")
    
    parser.add_argument("--depth", default=12,
                        type=int, help="number of attention layers")
    
    parser.add_argument("--mlp-dim", default=3072,
                        type=int, help="Demension of hidden layer in MLP Block")
    
    parser.add_argument("--lr", default=0.001,
                        type=float, help="Learning rate")
    
    parser.add_argument("--weight-decay", default=1e-4,
                        type=float)
    
    parser.add_argument("--batch-size", default=32,
                        type=int)
    
    parser.add_argument("--epochs", default=10,
                        type=int)
    
    parser.add_argument("--image-size", default=224,
                        type=int)
    
    parser.add_argument("--image-channels", default=3,
                        type=int)
    
    parser.add_argument("--adam-beta1", default=0.9,
                        type=float)
    
    parser.add_argument("--adam-beta2", default=0.999,
                        type=float)
    
    parser.add_argument("--adam-eps", default=1e-8,
                        type=float)
    
    parser.add_argument("--dropout", default=0.1,
                        type=float)
    
    parser.add_argument("--norm-eps", default=1e-12,
                        type=float)

    parser.add_argument("--dataset-dir", default="",
                        type=str)
    
    parser.add_argument("--annotated-file", default="",
                        type=str)
    
    parser.add_argument("--outputs-dir", default="./outputs",
                        type=str)
    
    parser.add_argument("--wandb-logger", default=False,
                        type=bool)

    parser.add_argument("--devices", default="cpu",
                        type=str)
    
    args = parser.parse_args()

    return args

def train():

    args = parse_args()

    if args.wandb_logger:
        import wandb
        wandb.init(project="Transformers training!!!")

    if args.model == "base":
        model = ViTBase()
    elif args.model == "larger":
        model = ViTLarge()
    elif args.model == "huge":
        model = ViTHuge()
    else:
        model = ViT(depth=args.depth,
                    num_heads=args.num_heads,
                    embed_dim=args.embed_dim,
                    mlp_dim=args.mp_dim,
                    num_classes=args.num_classes,
                    patch_size=args.patch_size,
                    image_size=args.image_size,
                    in_chans=args.image_channels,
                    dropout=args.dropout,
                    norm_eps=args.norm_eps)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.lr,
                                 betas = [args.adam_beta1, args.adam_beta2],
                                 eps=args.adam_eps,
                                 weight_decay=args.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
    ])
   
    if args.dataset_dir is not None:
        try:
            labels_path = os.path.join(args.dataset_dir, args.annotated_file)
            train_set = ImageDataset(annotation_file=labels_path,
                                     img_dir=args.dataset_dir,
                                     transform=transform,
                                     train=True)
            
            valid_set = ImageDataset(annotation_file=labels_path,
                                     img_dir=args.dataset_dir,
                                     transform=transform,
                                     train=False)
        
        except IsADirectoryError:
            print("Train set or Valid set is not exist")
        
    else:
        print("Data folder is not set.Use CIFAR10 dataset")

        args.image_channels = 3
        args.num_classess = 10

        train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                                 download=True, transform=transform)

        valid_set = torchvision.datasets.CIFAR10(root="./data", train=False,
                                                download=True, transform=transform)
        
    # train_ds = Dataset()

    train_ds = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    
    valid_ds = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    
    #torch.cuda.empty_cache()
    #gc.collect()

    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        #rain_acc = 0

        val_loss, val_ac = 0,0

        for batch_idx, (X, y) in enumerate(tqdm(train_ds), start = 1):
            X, y = X.to("cuda"), y.to("cuda")

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losss /= len(train_ds)
        #train_acc /= len(train_dataloader)
 
        model.eval()

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(valid_ds):
                X, y = X.to("cuda"), y.to("cuda")
                output = model(X)

                vloss = loss_fn(output, y)

                val_loss += vloss

        val_loss /= len(valid_ds)

        wandb.log({
            "training loss:": train_loss,
            "validation loss:": val_loss,
            "Epoch:": epoch,
        })

    
    save_path = os.path.join(args.outputs_dir, "ViT_Classifier.pt")
    print(f"Saving model to: {save_path}")
    torch.save(model.state_dict(),
               f=save_path)
    


if __name__ == "__main__":
    train()               

    