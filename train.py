import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomHorizontalFlip,RandomResizedCrop, Resize, ToTensor)
import os
import torchvision.transforms.v2 as T
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_train_dir', type=str, default='./train/train')
parser.add_argument('--data_val_dir', type=str, default='./data/val')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0.005)

args = parser.parse_args()

train_ds = datasets.ImageFolder(root=args.data_train_dir)
val_ds = datasets.ImageFolder(root=args.data_val_dir)

label2id = train_ds.class_to_idx
id2label= {label:id for id,label in label2id.items()}


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
normalize = Normalize(mean=image_mean, std=image_std)

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
# Custom transform to add Speckle noise
class AddSpeckleNoise(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    def __call__(self, tensor):
        # Generate speckle noise
        noise = torch.randn_like(tensor) * self.noise_level
        # Add speckle noise to the image
        noisy_tensor = tensor * (1 + noise)
        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor
class AddPoissonNoise(object):
    def __init__(self, lam=1.0):
        self.lam = lam
    def __call__(self, tensor):
        # Generate Poisson noise
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        # Add Poisson noise to the image
        noisy_tensor = tensor + noise / 255.0  # Assuming the image is scaled between 0 and 1
        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor
# Custom transform to add Salt and Pepper noise
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor[(noise < self.salt_prob)] = 1  # Salt noise: setting some pixels to 1
        tensor[(noise > 1 - self.pepper_prob)] = 0  # Pepper noise: setting some pixels to 0
        return tensor
# Define the image augmentation transformations
transform = T.Compose([
    T.ToTensor(),  # Convert PIL image to tensor
    T.RandomApply([T.RandomHorizontalFlip()], p=0.2),
    T.RandomApply([T.RandomVerticalFlip()], p=0.2),
    T.RandomApply([T.RandomRotation(10)], p=0.2),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomInvert(p=0.3),
    T.RandomPosterize(bits=2, p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),
    T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
    T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),
    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),
    T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
    # T.ToPILImage()  # Convert tensor back to PIL image for saving
    normalize,
])

_val_transforms = Compose([ToTensor(),normalize,])

def train_transforms(examples):
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['img']]
    return examples

def collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # labels = torch.tensor([example["label"] for example in examples])
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

train_ds = datasets.ImageFolder(root='../data/train', transform=transform)
val_ds = datasets.ImageFolder(root='../data/val', transform=transform)

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=15)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',id2label=id2label,label2id=label2id)

metric_name = "accuracy"

args = TrainingArguments(
    f"simpsons",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)

# model = ViTForImageClassification.from_pretrained('/mnt/htchang/ML/machine-learning-2023nycu-classification/simpsons/checkpoint-58150',id2label=id2label,label2id=label2id)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# # train
trainer.train()

# evaluate
outputs = trainer.predict(val_ds)
print(outputs.metrics)

# confusion matrix
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)
labels = val_ds.classes
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(30, 30))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix.png')

#把val中 各類別分類錯誤的圖片 拿出來看
worng_img = []
for i in range(len(y_true)):
  if y_true[i] != y_pred[i]:
    worng_img.append(i)

# 印出前10張圖片，並標示預測錯誤的類別及正確的類別，並顯示機率
plt.figure(figsize=(20, 20))
for i in range(20):
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  
  # 將圖像轉換為 (224, 224, 3) 格式
  img = val_ds[worng_img[i]][0].numpy().transpose((1, 2, 0))
  plt.imshow(img)
  plt.xlabel(f"True: {labels[y_true[worng_img[i]]]} \n Predict: {labels[y_pred[worng_img[i]]]} \n prob: {outputs.predictions[worng_img[i]][y_pred[worng_img[i]]]/100}")
plt.savefig('wrong_img.png')
