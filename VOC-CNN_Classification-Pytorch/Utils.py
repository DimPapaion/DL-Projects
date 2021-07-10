import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
import numpy as np


# Accuracy Evaluation

def Accuracy(y_true, y_scores):
  Acc_scores = 0.0
  for i in range(y_true.shape[0]):
    Acc_scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
  return Acc_scores


# Choose the available device.

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# Moving tensors or models in the selected device.

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Calculate Mean and Std
def Mean_std(data_dir, image_dir, data_type, batch_size):
    transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    train_data = VOCData(data_dir=data_dir, image_dir=im_path, data_type="train", transform=transform_train)
    train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(train_dl):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# Retrieve exact possition of the objects in the label list.
def get_class(label):
    names = []
    count_all = 0
    for i in label:
        if i == 1:
            count_all = count_all + 1
    count = 0
    while count_all > 0:
        sum = 0
        total = 0
        for i in label:
            if (i == 1 and count == 0):

                count_all = count_all - 1
                total = sum
                count = count + 1
                names.append(total)
            elif (i == 1 and count == 1):
                count_all = count_all - 1
                total = sum + 1
                count = count + 1
                names.append(total)
            elif (i == 1 and count == 2):
                count_all = count_all - 1
                total = sum + 2
                count = count + 1
                names.append(total)
            elif (i == 1 and count == 3):
                count_all = count_all - 1
                total = sum + 3
                count = count + 1
                names.append(total)
            elif (i == 1 and count == 4):
                count_all = count_all - 1
                total = sum + 4
                count = count + 1
                names.append(total)
            elif (i == 1 and count == 5):
                count_all = count_all - 1
                total = sum + 5
                count = count + 1
                names.append(total)
            elif (i == 1 and count == 6):
                count_all = count_all - 1
                total = sum + 6
                count = count + 1
                names.append(total)
            else:
                sum = sum + 1
    return names

#Function to visualize one single image from dataset

def Plot_Image( img, label):

  names = get_class(label)
  for i in names:
    print('True Labels: ', dataset.classes[i])
  plt.imshow(img.permute(1, 2, 0))

# Confusion matrix
def CM(y_true,y_pred):
  cf_matrix = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in dataset.classes],
                      columns = [i for i in dataset.classes])
  plt.figure(figsize = (12,7))
  return sns.heatmap(df_cm, annot=True)


# Retrieve predictions

def predict_image(img, model):
    image = to_device(img.unsqueeze(0), device)
    y_pred = model(image)

    _, preds = torch.max(y_pred, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

# Plot image, truth labels, pred labels.

def finally_preds(model,value, num):
  if value=="train":
    img, label = train_data[num]
    names = get_class(label)
    for i in names:
      print('True Labels: ', dataset.classes[i])

    plt.imshow(img.permute(1, 2, 0))
    print('Predicted Hightest probability for train set:' ,predict_image(img, model))
  elif value=="val":
    img, label = val_data[num]
    names = get_class(label)
    for i in names:
      print('True Labels: ', dataset.classes[i])

    plt.imshow(img.permute(1, 2, 0))
    print('Predicted Hightest probability for validation set:' ,predict_image(img, model))
  elif value=="test":
    img, label = test_data[num]
    names = get_class(label)
    for i in names:
      print('True Labels: ', dataset.classes[i])

    plt.imshow(img.permute(1, 2, 0))
    print('Predicted Hightest probability for test set:' ,predict_image(img, model))
  else:
    print("Invalid value!!! Please chose train,val,test.!")


# Dataframe with the infos about loss/acc
def make_df(tr_loss, tr_acc, valid_loss, valid_acc, epochs):
    epoch_list = list(range(1, epochs))
    results = pd.DataFrame({"Epochs": epoch_list,
                            "Train Loss": tr_loss,
                            "Train Accuracy": tr_acc,
                            "Validation Loss": valid_loss,
                            "Validation Accuracy": valid_acc},
                           columns=['Epochs', "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])

    return results

# Plotting loss

def plot_Accuracies(df):
    train_acc = df['Train Accuracy']
    valid_acc = df['Validation Accuracy']
    plt.plot(df['Epochs'], train_acc, 'b', label='Training Accuracy')
    plt.plot(df['Epochs'], valid_acc, 'r', label='validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return

#Plotting Acc

def plot_Losses(df):
    train_acc = df['Train Loss']
    valid_acc = df['Validation Loss']
    plt.plot(df['Epochs'], train_acc, 'b', label='Training Loss')
    plt.plot(df['Epochs'], valid_acc, 'r', label='validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return