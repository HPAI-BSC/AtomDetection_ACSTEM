import time

import numpy as np
import torch
from torch.nn import functional as F


def train_epoch(train_loader, model, loss_function, optimizer, device, epoch):
    model.train()

    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (batch_images, batch_labels) in enumerate(train_loader):
        # Loading tensors in the used device
        step_images, step_labels = batch_images.to(device), batch_labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        step_output = model(step_images)
        loss = loss_function(step_output, step_labels)
        loss.backward()
        optimizer.step()

        step_total = step_labels.size(0)
        step_loss = loss.item()
        losses += step_loss*step_total
        total += step_total

        step_preds = torch.max(step_output.data, 1)[1]
        step_correct = (step_preds == step_labels).sum().item()
        correct += step_correct

    train_loss = losses / total
    train_acc = correct / total
    format_args = (epoch, train_acc, train_loss, time.time() - t0)
    print('EPOCH {} :: train accuracy: {:.4f} - train loss: {:.4f} at {:.1f}s'.format(*format_args))


def val_epoch(val_loader, model, loss_function, device, epoch):
    model.eval()

    y_true = []
    y_pred = []

    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            # Loading tensors in the used device
            step_images, step_labels = batch_images.to(device), batch_labels.to(device)

            step_output = model(step_images)
            loss = loss_function(step_output, step_labels)

            step_total = step_labels.size(0)
            step_loss = loss.item()
            losses += step_loss*step_total
            total += step_total

            step_preds = torch.max(step_output.data, 1)[1]
            y_pred.append(step_preds.cpu().detach().numpy())
            y_true.append(step_labels.cpu().detach().numpy())
            step_correct = (step_preds == step_labels).sum().item()
            correct += step_correct

    val_loss = losses / total
    val_acc = correct / total
    format_args = (epoch, val_acc, val_loss, time.time() - t0)
    print('EPOCH {} :: val accuracy: {:.4f} - val loss: {:.4f} at {:.1f}s'.format(*format_args))

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_true, y_pred


def test_epoch(test_loader, model, loss_function, device):
    model.eval()

    correct = 0
    total = 0
    losses = 0
    all_true = []
    all_pred = []
    t0 = time.time()
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            # Loading tensors in the used device
            step_images, step_labels = batch_images.to(device), batch_labels.to(device)

            step_output = model(step_images)
            loss = loss_function(step_output, step_labels)

            step_total = step_labels.size(0)
            step_loss = loss.item()
            losses += step_loss*step_total
            total += step_total

            step_preds = torch.max(step_output.data, 1)[1]
            step_correct = (step_preds == step_labels).sum().item()
            correct += step_correct

            all_true.append(step_labels.cpu().numpy())
            all_pred.append(step_preds.cpu().numpy())

    val_loss = losses / total
    val_acc = correct / total
    format_args = (val_acc, val_loss, time.time() - t0)
    print('EPOCH :: test accuracy: {:.4f} - test loss: {:.4f} at {:.1f}s'.format(*format_args))

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    return all_true, all_pred


def detection_epoch(detection_loader, model, device):
    model.eval()

    pred_probs = []
    coords_x = []
    coords_y = []
    t0 = time.time()
    with torch.no_grad():
        for batch_images, batch_x, batch_y in detection_loader:
            # Loading tensors in the used device
            step_images = batch_images.to(device)
            step_output = model(step_images)
            step_pred_probs = F.softmax(step_output, 1)

            step_pred_probs = step_pred_probs.cpu().numpy()
            step_x = batch_x.numpy()
            step_y = batch_y.numpy()

            coords_x.append(step_x)
            coords_y.append(step_y)
            pred_probs.append(step_pred_probs)

    return_pred_probs = np.concatenate(pred_probs, axis=0)
    return_coords_x = np.concatenate(coords_x, axis=0)
    return_coords_y = np.concatenate(coords_y, axis=0)
    return return_pred_probs, return_coords_x, return_coords_y
