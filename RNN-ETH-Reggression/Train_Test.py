import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import torch
import numpy as np
from Utils import *




def train_step(x ,y, model, loss_fn, optimizer):
# Sets model to train mode
    model.train()

    # Makes predictions
    ypred = model(x)

    # Computes loss
    loss = loss_fn(y, ypred)

    # Computes gradients
    loss.backward()

    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()

    # Returns the loss
    return loss.item()


def fit_train(n_epochs, train_dl, valid_dl, model, loss_function, optimizer, batch_size, input_dim):
    train_losses = []
    val_losses = []
    device = "cuda" if torch.cuda.is_available() else "cpu" # check if cuda available.
    print("Training is starting...")
    for epoch in range(1, n_epochs + 1):

        batch_losses = []
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.view([batch_size, -1, input_dim]).to(device)
            y_batch = y_batch.to(device)
            # use train step to train the model
            loss = train_step(x_batch, y_batch, model=model, loss_fn=loss_function, optimizer=optimizer)
            batch_losses.append(loss)
        # append losses to global variables
        training_loss = np.mean(batch_losses)
        train_losses.append(training_loss)

        # check if validation set is available
        if valid_dl is not None:

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in valid_dl:
                    # model in evaluation mode
                    model.eval()
                    x_val = x_val.view([batch_size, -1, input_dim]).to(device)
                    y_val = y_val.to(device)
                    # make prediction
                    yhat = model(x_val)
                    val_loss = loss_function(y_val, yhat).item()
                    batch_val_losses.append(val_loss)

                # vallid global losses
                validation_loss = np.mean(batch_val_losses)
                val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                # if validation available print all
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")
        else:
            # print only train loss if no validation available
            print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}")
    return [train_losses, val_losses]


def test_fit(model, test_dl_one, input_dim):
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda available.
    with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in test_dl_one:
            # reshape
            x_test = x_test.view([batch_size, -1, input_dim]).to(device)
            y_test = y_test.to(device)

            model.eval()
            # make prediction in evaluation mode the model
            yhat = model(x_test)
            predictions.append(yhat.to(device).detach().numpy())
            values.append(y_test.to(device).detach().numpy())
    return predictions, values