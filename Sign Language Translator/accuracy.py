import pickle
from sklearn.metrics import accuracy_score
import numpy as np

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model']

def main():
    # Load the trained model
    model_path = 'model.p'
    model = load_model(r'model.p')

    # Load test data and labels
    data_dict = pickle.load(open(r'C:\Users\SHREYAS\Desktop\testing on epics - Copy - Copy\data.pickle', 'rb'))
    x_test = np.asarray(data_dict['data'])
    y_test = np.asarray(data_dict['labels'])

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Model accuracy: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    main()
