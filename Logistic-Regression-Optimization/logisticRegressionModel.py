import numpy as np

def sigmoid(z):
 
    return 1/(1+np.exp(-z))

class LogisticRegression():
   
    def __init__(self,lr=0.001,epochs=1000):
    
        self.lr=lr
        self.epochs=epochs
        self.weights = None
        self.bias = None
        print(f"LogisticRegression initialized with lr={self.lr}, epochs={self.epochs}")

    def fit(self,X,y):
   
        n_samples,n_features=X.shape
        print(f"Starting training on {n_samples} samples with {n_features} features...")

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.epochs):
            # Linear model calculation
            linear_predictions=np.dot(X,self.weights)+self.bias
            # Apply sigmoid function
            predictions=sigmoid(linear_predictions)

            # Calculate gradients
            dw=(1/n_samples)*np.dot(X.T,(predictions-y))
            db=(1/n_samples)*np.sum(predictions-y)

            # Update weights and bias
            self.weights-=self.lr*dw
            self.bias-=self.lr*db

             
            cost = - (1/n_samples) * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9)) # Added epsilon for log(0)
            print(f"Epoch {i+1}/{self.epochs}, Cost: {cost:.4f}")


    def predict(self,X):
      
        linear_predictions=np.dot(X,self.weights)+self.bias
        y_pred_proba = sigmoid(linear_predictions) # Probabilities
        # Class predictions based on threshold 0.5
        class_preds = [0 if y<=0.5 else 1 for y in y_pred_proba]
        return class_preds, y_pred_proba