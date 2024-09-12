import sys
sys.dont_write_bytecode = True

import os
import argparse
import torch
from torch.utils.data import DataLoader
from utils.model_utils import load_model
from utils.metrics import metrics
from dataset.dataset import LeafDataset, preprocess_images
from utils.utils import tensor2image, createFolder
from matplotlib import pyplot as plt
import random
import numpy as np
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

dataset_path = 'dataset'
model_dir = 'results/models'
defect_dir = 'results/defects/'

class evaluation():
    def __init__(self, model_dir, test_loader, defect_threshold, name = "unet") -> None:
        self.model_name = name
        self.model = load_model(name, model_dir)
        self.test_loader = [(images.to(device), labels.to(device)) for images, labels in test_loader]
        self.p_result = [0, 0, 0, 0]#tp, tn, fp, fn
        self.result = [0, 0]
        self.defect_threshold = defect_threshold

    def evaluate(self, pixel_base= False):

        createFolder(defect_dir)
        self.model.eval() 

        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images, labels
                outputs = self.model(images)
                preds = (outputs > 0.5).float()
                pixel_results = self.pixel_results(preds, labels)
                result = self.im_results(pixel_results)
                if pixel_base:
                    continue
                else:
                    if result != True:
                        im1 = tensor2image(images[0, :, : ,:])
                        im2 = tensor2image(labels[0, :, : ,:])
                        im3 = tensor2image(preds[0, :, : ,:])
                        f, axarr = plt.subplots(1, 3, figsize=(15, 5))

                        axarr[0].imshow(im1) 
                        axarr[0].set_title('Original')
                        axarr[0].axis('off')

                        axarr[1].imshow(im2)
                        axarr[1].set_title('Label')
                        axarr[1].axis('off')

                        axarr[2].imshow(im3)
                        axarr[2].set_title('Predicted')
                        axarr[2].axis('off')

                        plt.savefig(defect_dir + str(idx) + '.png')
            return self.p_result, self.result
    

    def pixel_results(self, preds:torch.tensor, labels:torch.tensor) -> None:

        im_result = [0, 0, 0, 0]
        true = preds == labels
        false = preds != labels
        
        im_result[0] = torch.sum(true[labels == 1]).item()
        self.p_result[0] += im_result[0]

        im_result[1] = torch.sum(true[labels == 0]).item()
        self.p_result[1] += im_result[1]

        im_result[2] += torch.sum(false[labels == 1]).item()
        self.p_result[2] += im_result[2]

        im_result[3] += torch.sum(false[labels == 0]).item()
        self.p_result[3] += im_result[3]

        return im_result
    
    def im_results(self, results) -> bool:
        tp, _, fp, fn = results
        is_True = False

        threshold = (tp)/ (tp + fp + fn)# iou formula
        if threshold > self.defect_threshold:
            is_True = True
            self.result[0] += 1
        else: 
            self.result[1] += 1
        return is_True

    def write_report():
        return
    
    def visualize_predictions(self, test, num_images=4):

        self.model.eval()

        self.model.to(device)
        
        fig, axes = plt.subplots(num_images, 3, figsize=(12, 12))
        
        sampled_indices = random.sample(range(len(test)), num_images)
        
        for ax, idx in zip(axes, sampled_indices):
            image, label = test[idx]
            image = image.to(device).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(image)
            
                pred = (output > 0.5)


            image = image.cpu().squeeze(0).numpy()
            image = np.transpose(image,[1,2,0])
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)


            pred = pred.cpu().squeeze(0).numpy()
            pred = np.clip(pred, 0, 1)

            label = label.cpu().squeeze(0).numpy()
            label = np.clip(label, 0, 1)
            ax[0].imshow(image)
            ax[1].imshow(label)
            ax[2].imshow(pred[0])
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
        plt.tight_layout()
        plt.savefig('results/random.png')

    def inference_img(self, im:str):
        image = Image.open(im)
        image_tensor = preprocess_images(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        print(image_tensor.shape)
        self.model.eval() 
        with torch.no_grad():
            outputs = self.model(image_tensor)
            preds = (outputs > 0.5).float()
            im1 = tensor2image(image_tensor[0, :, : ,:])
            im2 = tensor2image(preds[0, :, : ,:])
            f, axarr = plt.subplots(1, 2, figsize=(15, 5))

            axarr[0].imshow(im1) 
            axarr[0].set_title('Original')
            axarr[0].axis('off')
            axarr[1].imshow(im2)
            axarr[1].set_title('Predicted')
            axarr[1].axis('off')
            name = os.path.basename(im) 
            plt.savefig('results/' + name)
        return
    
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='unet50', help='Select the model architecture.(unet, unet34, unet50) Default: unet50')
    parser.add_argument('--image',default='', help='Select the one or two loss.(dice, bce, tversky, iou) Default: Dice')
    parser.add_argument('--dt', type=float, default=0.4, help='Select the threshold for when model consider an image as defect. Default: 0.2')
    parser.add_argument('--jv', action='store_true', help='If this flag setted. There will just random test with no evaluation')
    parser.add_argument('--pb', action='store_true', help='If this flag setted. There wont be image based evaluation')
    parser.add_argument('--worker', type=int, default=0, help='Select the number of workers. Default:0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    model_name = args.model
    model_name = str(model_name[0]).capitalize() + str(model_name[1]).capitalize() + str(model_name[2:])

    test = LeafDataset(image_dir=dataset_path, sub='test')
    test_loader = DataLoader(test, batch_size=1, num_workers=args.worker)
    tester = evaluation(model_dir, test_loader, args.dt, name=model_name)
    image = args.image
    if image == '':
        if args.jv:
            tester.visualize_predictions(test, num_images=4)
        else:
            results, im_results = tester.evaluate(pixel_base=args.pb)
            tester.visualize_predictions(test, num_images=4)
            metric = metrics(results)
            metric.quick()
    else:
        tester.inference_img(image)

