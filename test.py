from argparse import ArgumentParser
import torch
from model import MyResnet
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose
import os
import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--image", '-i', default=None)
    parser.add_argument('--checkpoint', default='checkpoint/best_cnn.pt')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.image is None:
        print("Error: Please provide an image file using --image or -i")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyResnet().to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Chuyển sang chế độ đánh giá

    path = os.path.join('image', args.image)
    try:
        origin_img = Image.open(path)
    except FileNotFoundError:
        print(f"Error: Image file '{path}' not found")
        exit(1)

    compose = Compose([
        Resize((224, 224)), ToTensor()
    ])
    image = origin_img.convert('RGB')
    image = compose(image)
    image = image[None, :, :, :]
    image = image.to(device)
    
    l = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    with torch.no_grad():
        output = model(image)
        idx = torch.argmax(output, dim=1).item()
    
    plt.imshow(origin_img)
    plt.title(l[idx])
    plt.axis('off')
    plt.show()