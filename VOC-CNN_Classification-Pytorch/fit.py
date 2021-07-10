import tqdm
from Utils import *

# Function for the training set
def training(loader ,device, model ,optimizer ,criterion ,m ,scheduler):
    losst = 0.0
    acc = 0.0
    model.train(True)  # Set model to training mode
    for data, target in tqdm(loader):

        target = target.float()
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # Get metrics here
        losst += loss # sum up batch loss
        acc += Accuracy(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy())
        # acc += Accuracy(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(output).detach().numpy())
        # Backpropagate the system the determine the gradients
        loss.backward()
        # Update the paramteres of the model
        optimizer.step()
        # scheduler.step()
        # print("loss = ", losst)
    num_samples = float(len(loader.dataset))
    total_loss_ = losst.item( )/num_samples
    total_acc_ = acc/num_samples
    # Append the values to global arrays
    return total_loss_ ,total_acc_


# Make evaluation
def evaluation(loader, device, model, optimizer, criterion, m):
    model.eval()  # Set model to evaluate mode
    losst = 0.0
    acc = 0.0
    # torch.no_grad is for memory savings
    with torch.no_grad():
        for data, target in tqdm(loader):
            target = target.float()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            losst += loss  # sum up batch loss
            acc += Accuracy(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy())
        num_samples = float(len(loader.dataset))
        val_loss_ = losst.item() / num_samples
        val_acc_ = acc / num_samples
    return val_loss_, val_acc_


# fit the model for the training.
def train_model(model, device, criterion, optimizer, scheduler, train_loader, valid_loader, epochs):
    tr_loss, tr_acc = [], []
    val_loss, val_acc = [], []

    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        print("-------Epoch {}/{}----------".format(epoch + 1, epochs))
        scheduler.step()
        m = torch.nn.Sigmoid()
        tr_loss_, tr_acc_ = training(loader=train_loader, device=device, model=model, optimizer=optimizer,
                                     criterion=criterion, m=m, scheduler=scheduler)

        val_loss_, val_acc_ = evaluation(loader=valid_loader,device = device, model=model, optimizer=optimizer,
                                         criterion=criterion, m=m)
        # scheduler.step()
        tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
        val_loss.append(val_loss_), val_acc.append(val_acc_)

        print('train_loss: {:.4f}, train_Accuracy:{:.3f}, validation_Loss: {:.4f}, validation_Accuracy:{:.3f}'.format(
            tr_loss_, tr_acc_, val_loss_, val_acc_))
    return ([tr_loss, tr_acc], [val_loss, val_acc])


# evaluate the model in the test set and retrieve predictions and actuall labels

def test_model(model, device, criterion, test_loader):
    model.eval()
    losst = 0
    acc = 0
    m = torch.nn.Sigmoid()

    y_pred = np.empty((0, 20), float)
    y_true = np.empty((0, 20), float)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            target = target.float()
            data, target = data.to(device), target.to(device)

            bs, c, h, w = data.size()
            output = model(data.view(-1, c, h, w))
            loss = criterion(output, target)

            losst += loss  # sum  batch loss
            acc += Accuracy(torch.Tensor.cpu(target).detach().numpy(),
                            torch.Tensor.cpu(m(output)).detach().numpy())  # sum acc batch
            y_pred = np.append(y_pred, torch.Tensor.cpu(m(output)).detach().numpy(), axis=0)
            y_true = np.append(y_true, torch.Tensor.cpu(target).detach().numpy(), axis=0)

    num_samples = float(len(test_loader.dataset))
    test_loss = losst.item() / num_samples
    test_acc = acc / num_samples
    print('test_loss: {:.4f}, test_Accuracy:{:.3f}'.format(test_loss, test_acc))
    return test_loss, test_acc, y_pred, y_true