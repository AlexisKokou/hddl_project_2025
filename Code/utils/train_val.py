import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
#from typing import Literal
def train_catdog_classifier(model,train_loader, criterion,optimizer,scheduler, epochs = 10, lr= 1e-4, device = "cuda" ):
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        tbar = tqdm.tqdm(train_loader)
        for images, label in tbar:
            family = label[1]
            optimizer.zero_grad()
            images = images.to(device)
            family = family.to(device)
            outputs = model(images)
            loss = criterion(outputs, family)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
    
def train_breed_classifier(model,train_loader, criterion,optimizer,scheduler, get_breed_num, epochs = 10, lr= 1e-4, device = "cuda" ):
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        tbar = tqdm.tqdm(train_loader)
        for images, label in tbar:
            breed = label[0]
            optimizer.zero_grad()
            images = images.to(device)
            breed = get_breed_num(breed)
            breed = torch.tensor(breed).to(device)
            outputs = model(images)
            loss = criterion(outputs, breed)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)


def train_unet(model, train_loader, criterion, optimizer, scheduler, epochs=50, lr=1e-4, device='cuda'):
    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        tbar = tqdm.tqdm(train_loader)
        for images,_,masks in tbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
def train_multimodel(model, train_loader, criterion, optimizer, scheduler,mapping, type_cls = "catdog", epochs=50, lr=1e-4, device='cuda'):
    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        tbar = tqdm.tqdm(train_loader)
        for images,labels,masks in tbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if type_cls == "catdog":
                families = labels[1]
                fams = [mapping[f] for f in families]
                fams = torch.tensor(fams).to(device)
                pred_cls = outputs[0]
                cls_loss = criterion[0](pred_cls, fams)
            else:
                breeds = labels[0]
                bres = [mapping[b] for b in breeds]
                bres = torch.tensor(bres).to(device)
                pred_cls = outputs[0]
                cls_loss = criterion[0](pred_cls, bres)
            seg_loss = criterion[1](outputs[1].squeeze(1), masks)
            loss = seg_loss + cls_loss



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{epochs}  Classification loss: {cls_loss.item():.4f} Segmentation loss: {seg_loss.item():.4f} Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)