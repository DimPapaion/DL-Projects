import os
from Utils import *
from VOCDataset import *
from Models.ResNET50 import *
from VOCDataset import *
from fit import *
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix
import torch.optim as optim

def main():
    data_dir = './VOCdevkit/VOC2012'
    print(os.listdir(data_dir))
    Annot = os.listdir(data_dir + "/Annotations/")
    print(Annot)

    Images_list = os.listdir(data_dir + "/JPEGImages")
    label_dir = data_dir +"/ImageSets/Main/"
    print('No. of Images in VOC Dataset:', len(Images_list))
    print(Images_list[:5])
    print("List of label texts \n", os.listdir(label_dir))

    im_path = data_dir+"/JPEGImages/"

    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    # obtain the mean and std for normalization
    mean, std = Mean_std(data_dir=data_dir, image_dir= im_path, data_type="train",batch_size=16)
    print(mean)

    # Make transformations for train and validation set
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomChoice([
                                    transforms.ColorJitter(brightness=(0.80, 1.20)),
                                    transforms.RandomGrayscale(p = 0.35)]),
                                    transforms.RandomHorizontalFlip(p = 0.5),
                                    transforms.RandomVerticalFlip(p=0.05),
                                    transforms.RandomRotation(45),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = mean, std = std)])

    transformations_valid = transforms.Compose([transforms.Resize((230,230)),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean = mean, std = std)])


    # Create custom Datasets
    dataset = VOCData(data_dir=data_dir, image_dir=im_path, data_type="trainval",transform=transform)
    train_data = VOCData(data_dir=data_dir, image_dir=im_path, data_type="train",transform=transform)
    val_data = VOCData(data_dir=data_dir, image_dir=im_path, data_type="val",transform=transformations_valid)

    val_data, test_data = torch.utils.data.random_split(val_data, [4593,1230])

    print(" Number of train and validation dataset: ", len(dataset))
    print(" Number of train dataset: ", len(train_data))
    print(" Number of validation dataset: ", len(val_data))
    print(" Number of test dataset: ", len(test_data))

    #Plot images
    Plot_Image(*val_data[2228])  # Plot validation image
    Plot_Image(*train_data[5611])  #Plot train image
    Plot_Image(*test_data[154])  #Plot test image

    #Creating the DataLoaders.
    batch_size=16
    train_dl_un = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl_un = DataLoader(val_data, batch_size, num_workers=4, pin_memory=True)
    test_dl_un = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)

    # grab the device
    device = get_default_device()
    print(device)

    # specify model
    model_resnet50 = ResNet50(img_channel=3, num_classes=20)
    print(model_resnet50)
    to_device(model_resnet50, device);
    # retrieve the params number
    RESNET50_params = sum(p.numel() for p in model_resnet50.parameters())
    print(RESNET50_params)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.SGD(model_resnet50.parameters(), lr=0.0001, momentum=0.9)
    sheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)
    # train_loss, train_acc = train_(model, device=device, optimizer=optimizer, t,rain_loader=train_dl_b, epochs=30)
    [tr_loss, tr_acc], [val_loss, val_acc] = train_model(model_resnet50, device, criterion, optimizer, sheduler,
                                                         train_dl_un, val_dl_un, epochs=80)

    test_loss, test_acc, y_pred, y_true = test_model(model_resnet50, device, criterion, test_loader = test_dl_un)

    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    print(y_pred,y_true)
    CM(y_true,y_pred)  #confusion matrix

    resultsRESNET50 = make_df(tr_loss=tr_loss, tr_acc=tr_acc, valid_loss=val_loss, valid_acc= val_acc, epochs=81) #results

    plot_Accuracies(resultsRESNET50)
    plot_Losses(resultsRESNET50)

    finally_preds(model = model_resnet50,value="test", num=423)
    finally_preds(model=model_resnet50, value="train", num=837)

if __name__ == "__main__":
    main()