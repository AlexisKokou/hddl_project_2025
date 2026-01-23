from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
from  .dataset import Annotation
import torch
import  torch.nn.functional as F
from typing import Literal
import numpy as np
import tqdm
import plotly.graph_objects as go
def show_image(image, label, anno, mask):
    """Display an image with its label, annotations, and mask."""
    num_plots = 1  
    if mask is not None:
        num_plots += 2  # mask + overlap
    
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3.5, 5))
    if num_plots == 1:
        axes = [axes]
    
    idx = 0
    
    axes[idx].imshow(image)
    axes[idx].axis('off')
    axes[idx].set_title('Image')
    idx += 1
    
    if mask is not None:
        axes[idx].imshow(mask, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title('Mask')
        idx += 1
        
        axes[idx].imshow(image)
        axes[idx].imshow(mask, cmap='jet', alpha=0.4)
        axes[idx].axis('off')
        axes[idx].set_title('Image + Mask Overlay')
    
    fig.suptitle(f'Breed: {label[0]}, Family: {label[1]}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize a tensor image."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  
    return torch.clamp(tensor, 0, 1)


def display_classification(model, test_loader, num_breed=None, type="catdog", device="cuda"):
    """Display classification results for a batch of images."""
    batch = next(iter(test_loader))
    images, (breeds, families) = batch
    
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs)
        preds = torch.argmax(probs, dim=1).cpu()
    
    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(16):
        image = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        
        if type == "catdog":
            true_label = 'cat' if families[i] == 0 else 'dog'
            pred_label = 'cat' if preds[i] == 0 else 'dog'
        else: 
            true_label = breeds[i]
            pred_label = num_breed[preds[i].item()]
        
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Pred: {pred_label}\nTrue: {true_label}')
    
    plt.tight_layout()
    plt.show()

def display_segmentation(model, test_loader, device = 'cuda', threshold = 0.5):
    """Display segmentation results for a batch of images."""
    batch = next(iter(test_loader))
    images,_, masks = batch
    model.eval()
    with torch.no_grad():
            outputs = torch.sigmoid(model(images.to(device))).squeeze(1)
    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(4):
        iou, dice = calculate_metrics_seg(outputs[i].cpu(), masks[i], threshold=threshold)
        pred_binary = (outputs[i].cpu() > threshold).float().cpu()
        image = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        true_mask = masks[i].cpu().numpy()
        axes[i*4].imshow(image)
        axes[i*4].axis('off')
        axes[i*4].set_title(f'Original')
        axes[i*4+1].imshow(image)
        axes[i*4+1].imshow(true_mask, cmap='Reds', alpha=0.5)  
        axes[i*4+1].axis('off')
        axes[i*4+1].set_title("True Mask Overlay")
        axes[i*4+2].imshow(image)
        axes[i*4+2].imshow(pred_binary, cmap='Greens', alpha=0.5)
        axes[i*4+2].axis('off')    
        axes[i*4+2].set_title(f"Pred Overlay")
        axes[i*4+3].imshow(image)
        axes[i*4+3].imshow(true_mask, cmap='Reds', alpha=0.3)
        axes[i*4+3].imshow(pred_binary, cmap='Greens', alpha=0.3)
        axes[i*4+3].axis('off')
        axes[i*4+3].set_title(f'IoU: {iou:.3f}, Dice: {dice:.3f}')
    plt.tight_layout()
    plt.show()   


def display_heatmap_breed(all_trues,all_preds,breed_names ):
    """Display a heatmap of breed classification results."""
    accuracy = accuracy_score(all_trues, all_preds)
    print(f'Accuracy: {accuracy:.2f}')
    cm = confusion_matrix(all_trues, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=breed_names,yticklabels=breed_names)
    plt.xlabel('Predicted Breed')
    plt.ylabel('True Breed')
    plt.title('Confusion Matrix')
    plt.show()


def analyze_breed_classification(all_trues, all_preds, num_breed, top_k=5):
    """Analyze breed classification results."""
    cm = confusion_matrix(all_trues, all_preds)
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j: 
                confusions.append((cm[i, j], i, j))
    confusions.sort(reverse=True)
    
    print(f"=== TOP {top_k} MOST CONFUSED BREED PAIRS ===")
    for count, true_idx, pred_idx in confusions[:top_k]:
        print(f"{num_breed[true_idx]} -> {num_breed[pred_idx]}: {count} times")

    report = classification_report(all_trues, all_preds, output_dict=True, zero_division=0)
    breed_scores = []
    for breed_idx in range(len(num_breed)):
        if str(breed_idx) in report:
            breed_scores.append({
                'breed': num_breed[breed_idx],
                'f1_score': report[str(breed_idx)]['f1-score'],
                'precision': report[str(breed_idx)]['precision'],
                'recall': report[str(breed_idx)]['recall'],
                'support': report[str(breed_idx)]['support']
            })
    
    df = pd.DataFrame(breed_scores).sort_values('f1_score', ascending=False)
    print("\n=== EASILY DETECTED (Top 5) ===")
    print(df.head(5).to_string(index=False))
    print("\n=== HARDLY DETECTED (Bottom 5) ===")
    print(df.tail(5).to_string(index=False))
    return df


def calculate_metrics_seg(pred_mask, true_mask, threshold=0.5):
    """Calculate IoU and Dice coefficient for segmentation masks."""
    pred_mask = pred_mask.detach().cpu().numpy()
    true_mask = true_mask.detach().cpu().numpy()

    pred_binary = (pred_mask > threshold).astype(np.float32)
    true_binary = true_mask.astype(np.float32)
    intersection = np.sum(pred_binary * true_binary)
    union = np.sum(pred_binary) + np.sum(true_binary) - intersection

    iou = intersection / (union + 1e-7) 

    dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(true_binary) + 1e-7)
    
    return iou, dice




def show_table(df, title):
    """Display a DataFrame as a table using Plotly."""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    fig.update_layout(title=title, height=400)
    fig.show()

def catdog_evaluation(model, test_catdog_loader, device = "cuda"):
    model.eval()
    pred = []
    true = []
    tbar = tqdm.tqdm(test_catdog_loader)
    with torch.no_grad():
        for images, (_, family) in tbar:
            images = images.to(device)
            family = family.to(device)

            outputs = model(images)
            predict = torch.argmax(F.softmax(outputs, dim=1), 1)
            pred.append(predict)
            true.append(family)
    all_preds = torch.cat(pred).cpu().numpy()
    all_trues = torch.cat(true).cpu().numpy()
    precision_cat = precision_score(all_trues, all_preds, labels= [0], average='binary', pos_label=0)
    recall_cat = recall_score(all_trues, all_preds, labels=[0], average='binary', pos_label=0)
    precision_dog = precision_score(all_trues, all_preds, labels= [1], average='binary', pos_label=1)
    recall_dog = recall_score(all_trues, all_preds, labels=[1], average='binary', pos_label=1)
    accuracy = accuracy_score(all_trues, all_preds)
    f1_score_ = f1_score(all_trues, all_preds)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Cat Precision: {precision_cat:.2f}, Recall: {recall_cat:.2f}')
    print(f'Dog Precision: {precision_dog:.2f}, Recall: {recall_dog:.2f}')
    print(f"F1 score: {f1_score_}") 

def breed_evaluation(model, test_loader, breed_num,num_breed,top_k = 5, device = "cuda"):
    """Evaluate breed classification performance."""
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        tbar = tqdm.tqdm(test_loader)
        for images, (breeds, _) in tbar:
            images = images.to(device)
            breed = []
            for i in breeds:
                breed.append(breed_num[i])        
            outputs = model(images)
            predict = torch.argmax(F.softmax(outputs, dim = 1), 1)
            pred.append(predict)
            true.append(breed)
    all_preds = torch.cat(pred).cpu().numpy()
    all_trues = [x for sublist in true for x in sublist]
    display_heatmap_breed(all_preds=all_preds, all_trues= all_trues, breed_names= breed_num.keys())
    df  = analyze_breed_classification(all_trues, all_preds, num_breed, top_k=top_k)
    return df

def display_multitask(model, test_loader, inv_mapping,n_images = 4,type = "catdog", device = 'cuda',  threshold = 0.5):
    batch = next(iter(test_loader))
    images,labels, masks = batch
    if type == "catdog":
        cls_trues = labels[1]
    else:
        cls_trues = labels[0]
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        cls_preds_raw = torch.argmax(F.softmax(outputs[0], dim=1), dim=1)
        cls_preds = [inv_mapping[int(cls_pred_raw.cpu())] for cls_pred_raw in cls_preds_raw]
        seg_pred = outputs[1].squeeze(1)
    _, axes = plt.subplots(nrows=n_images, ncols=3, figsize=( 12,n_images*3))
    axes = axes.flatten()
    for i in range(n_images):
        cls_true = cls_trues[i]
        iou, dice = calculate_metrics_seg(seg_pred[i].cpu(), masks[i], threshold=threshold)
        pred_binary = (seg_pred[i].cpu() > threshold).float().cpu().numpy()
        image = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        true_mask = masks[i].cpu().numpy()
        axes[i*3].imshow(image)
        axes[i*3].axis('off')
        axes[i*3].set_title(f'Original')
        axes[i*3+1].imshow(image)
        axes[i*3+1].imshow(true_mask, cmap='Reds', alpha=0.5)  
        axes[i*3+1].axis('off')
        axes[i*3+1].set_title(f"True Mask of {cls_true}")
        axes[i*3+2].imshow(image)
        axes[i*3+2].imshow(pred_binary, cmap='Greens', alpha=0.5)
        axes[i*3+2].axis('off')    
        axes[i*3+2].set_title(f'Predict: {cls_preds[i]} ,IoU: {iou:.3f}, Dice: {dice:.3f}')
    plt.tight_layout()
    plt.show()   

def multi_evaluation(model, test_loader, mapping,type = "catdog",top_k = 5, device = "cuda"):
    """Evaluate multitask classification and segmentation performance."""
    model.eval()
    pred = []
    true = []
    results = []
    with torch.no_grad():
        tbar = tqdm.tqdm(test_loader)
        for images, (breeds, families),masks in tbar:
            if type == "catdog":
                cls_trues = families
            else:
                cls_trues = breeds
            images = images.to(device)
            true_batch = []
            for i in cls_trues:
                true_batch.append(mapping[i])        
            outputs = model(images)
            cls_outs = outputs[0]
            seg_outs = outputs[1].squeeze(1)
            predict = torch.argmax(F.softmax(cls_outs, dim = 1), 1).cpu()
            pred.append(predict)
            true.append(true_batch)
            for i in range(images.shape[0]):
                iou, dice = calculate_metrics_seg(seg_outs[i].cpu(), masks[i].cpu())
                results.append({
                    'True label': true_batch[i],
                    'Pred label': predict[i],
                    'iou': iou,
                    'dice': dice
                })
    df = pd.DataFrame(results)
    df['True label'] = df['True label'].apply(lambda x: int(x) if isinstance(x, torch.Tensor) else x)
    df['Pred label'] = df['Pred label'].apply(lambda x: int(x) if isinstance(x, torch.Tensor) else x)

    overall_stats = {
        'Total samples': len(df),
        'IoU mean': df['iou'].mean(),
        'IoU std': df['iou'].std(),
        'Dice mean': df['dice'].mean(),
        'Dice std': df['dice'].std(),
        'Classification accuracy': (df['True label'] == df['Pred label']).mean()
    }

    class_stats = df.groupby('True label').agg({
        'iou': ['mean', 'std', 'min', 'max'],
        'dice': ['mean', 'std', 'min', 'max']
    }).round(4)

    misclassified = df[df['True label'] != df['Pred label']]
    misclass_stats = {
        'Misclassified count': len(misclassified),
        'Misclassified IoU mean': misclassified['iou'].mean() if len(misclassified) > 0 else None,
        'Misclassified Dice mean': misclassified['dice'].mean() if len(misclassified) > 0 else None
    }

    print("OVERALL METRICS")
    print("*" * 50)
    for k, v in overall_stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    print("PER-CLASS METRICS")
    print("*" * 50)
    print(class_stats)

    print("MISCLASSIFICATION ANALYSIS")
    print("*" * 50)
    for k, v in misclass_stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and v is not None else f"{k}: {v}")
    return df
def evaluate_segmentation(model, test_loader, device="cuda"):
    """Evaluate segmentation performance on the test dataset."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, (labels, families), masks in tqdm.tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            for i in range(images.size(0)):
                pred_mask = outputs[i, 0]
                true_mask = masks[i]
                
                iou, dice = calculate_metrics_seg(pred_mask, true_mask)
                
                results.append({
                    'label': labels[i],
                    'family': families[i],
                    'iou': iou,
                    'dice': dice
                })
    
    return pd.DataFrame(results)