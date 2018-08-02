# machine-learning-from-scratch

#### Summary of Project

The project creates an artificial neural network and uses it to learn to identify articles of clothing. The articles of clothing are represented as 28*28 matrices, with each
value in the matrix representing a pixel color and is a number between 0 and 255. The code creates a training set and a validation set from the two files found in the directory
(“train_x” and “train_y”). The code then trains the neural network using the training set and checks the network using the validation set.

No machine learning library is used in this code. I only use numpy to help with dealing with data and matrices.

I’ve included a file report.pdf, which is the summary of my neural network that I handed in when I turned in this assignment.

#### How to Run

Very simple, assuming you have Python. Open a terminal, `cd` to the directory with all the files, and enter `python ml_code.py`.
The program will take several minutes to complete but it outputs the validation results after every epoch. The output is in the following format:
[number of epoch] [average loss from training set] [average loss  from validation set] [percentage correct on validation set]
