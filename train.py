from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import os
from process_image import Animal  # Sửa tên file import
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import MyResnet

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./animals")
    parser.add_argument('--epoch', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', '-b', type=int, default=128, help='Batch size')
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    return parser.parse_args()

def plot_confusion_matrix(true_labels, pred_labels, classes, epoch, writer):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes[:cm.shape[1]], yticklabels=classes[:cm.shape[0]],
           title=f'Confusion Matrix - Epoch {epoch}',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    
    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    writer.add_image('Confusion Matrix', img, epoch)
    plt.close(fig)




if __name__ == '__main__':
    writer = SummaryWriter()
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_point_dir = "checkpoint" if args.checkpoint is None else os.path.dirname(args.checkpoint)
    os.makedirs(check_point_dir, exist_ok=True)

    dataset = Animal(args.root, train=True)
    test_data = Animal(args.root, train=False)

    dataload = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    test_dataload = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=4, drop_last=False)

    model = MyResnet().to(device)
    criterion = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False
    for param in model.res.fc.parameters():
        param.requires_grad = True

    # Khởi tạo optimizer
    optimizer = optim.SGD(model.res.fc.parameters(), lr=0.001, momentum=0.9)

    categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirel']

    if args.checkpoint:
        check_point = torch.load(args.checkpoint, map_location=device)
        start = check_point['epoch']
        model.load_state_dict(check_point['model'])
        optimizer.load_state_dict(check_point['optimizer'])
        best_acc = check_point['best_acc']
        print(f"Loaded checkpoint from epoch {start}")
        if start >= args.epoch:
            args.epoch = start + 1
    else:
        best_acc = 0
        start = 0

    for epoch in range(start, args.epoch):
        model.train()
        process_bar = tqdm(dataload)
        for i, (image, Label) in enumerate(process_bar):
            image = image.to(device)
            Label = Label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, Label)
            process_bar.set_description('Epoch {}/{}. Iteration {}/ {}. Loss {:.4f}'.format(
                epoch+1, args.epoch, i, len(dataload), loss.item()))
            writer.add_scalar('Train/Loss', loss.item(), epoch*len(dataload)+i)
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0.0
    
        for i, (image, Label) in enumerate(test_dataload):
            image = image.to(device)
            Label = Label.to(device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, Label)
                test_loss += loss.item()
                all_preds.extend(output.argmax(dim=1).tolist())
                all_labels.extend(Label.tolist())
        
        accuracy = accuracy_score(all_labels, all_preds)
        avg_test_loss = test_loss / len(test_dataload)
        print(f"Epoch {epoch+1}/{args.epoch} - Test Loss: {avg_test_loss:.4f} - Accuracy: {accuracy:.4f}")
        writer.add_scalar('Accuracy', accuracy, epoch)

        print(f"Test labels: {len(all_labels)}, Test predictions: {len(all_preds)}")
        try:
            plot_confusion_matrix(all_labels, all_preds, categories, epoch, writer)
            print(f"Confusion matrix recorded for epoch {epoch}")
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")

        check_point = {
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(check_point, os.path.join(check_point_dir, "last_cnn.pt"))
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(check_point, os.path.join(check_point_dir, "best_cnn.pt"))

    writer.close()