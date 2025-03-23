import torch


def trainer(model, criterion, optimizer, trainloader, validloader, epochs=5):
    train_loss = []
    valid_loss = []
    
    for epoch in range(epochs):  # for each epoch
        train_batch_loss = 0
        valid_batch_loss = 0
        
        # Training
        for X, y in trainloader:

            optimizer.zero_grad()       # Zero all the gradients w.r.t. parameters

            y_hat = model(X).flatten()  # Forward pass to get output
            loss = criterion(y_hat, y)  # Calculate loss based on output
            loss.backward()             # Calculate gradients w.r.t. parameters
            optimizer.step()            # Update parameters

            train_batch_loss += loss.item()  # Add loss for this batch to running total

        train_loss.append(train_batch_loss / len(trainloader))
        
        # Validation
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood

            for X_valid, y_valid in validloader:

                y_hat = model(X_valid).flatten()  # Forward pass to get output
                loss = criterion(y_hat, y_valid)  # Calculate loss based on output

                valid_batch_loss += loss.item()
            
        valid_loss.append(valid_batch_loss / len(validloader))
        
    return train_loss, valid_loss