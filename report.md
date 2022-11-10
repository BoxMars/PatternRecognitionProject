# CISC3024 Pattern Recognition Project

by Zhang Huakang D-B9-2760-6

> The notebook file of this project has been published on [kaggle](https://www.kaggle.com/code/boxzhang/cisc3024-pattern-recognition-project-by-db92760).

|Member Name|Contribution Percentage|
|-|-|
ZHANG HUAKANG| 100%
## 1. Data Loading
In this part, the image will be loaded from disk with its corresponding labels. After loading images and lables, a customized class `SatalliteDataset` inherited from `torch.utils.data.Dataset` is used to stored this data. When `__getitem__()` function is called, processed images and labels are returned.
```python
# Image processing for training data
train_tran = transforms.Compose([
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomCrop(64, padding=2),
                    transforms.ToTensor(),
                ])
# Image processing for testing and val. data
tran = transforms.Compose([
                    transforms.ToTensor(),
                ])
```
```python
# SatalliteDataset
class SatalliteDataset(Dataset):
    def __init__(self,images,labels, is_train):
        self.images_list=images
        self.labels_list=labels
        self.is_train=is_train
    
    def __len__(self):
        return len(self.labels_list)
    
    def __getitem__(self,idx):
        # Image Processing
        if self.is_train:
            return train_tran(self.images_list[idx]).to(device), self.labels_list[idx]
        return tran(self.images_list[idx]).to(device), self.labels_list[idx]
```
Then, `train`, `test` and `val` data will be encapsulated in `torch.utils.data.Dataloader` object to make it easier to get data when training the model.
```python
train_iterator = data.DataLoader(SatalliteDataset(X_train,y_train, True), batch_size=BATCH_SIZE)
```
## Example Image
Here is some example of dataset.

Unprocessed:
![](/slidev/img/plot.png)
Normalization:
![](/slidev/img/plotnor.png)
## Model Design
### AlexNet
![](/slidev/img/alexnet.png)
code version:
```python
class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  
            nn.MaxPool2d(2),  
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h
```
## Model Training
### Find the best learning rate
In this part, I build a class `LRFinder` to find the best learning rate. In the function `range_test()`, 

## Model Performance Evaluation