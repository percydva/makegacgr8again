from sympy import plot
from data_preprocessing import sns, pd, np, plt, sys, split_data
from model import GRU, torch, nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time 

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100
lookback = 20

sns.set_theme()
sns.set_style('darkgrid')

def read_file(file_name):
    return pd.read_csv(file_name, parse_dates=[0])

def train(model, X_train, y_train_gru):
    criterion = nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    return hist, y_train_pred

def pass_to_tensor(price, lookback):
    X_train, y_train, X_test, y_test = split_data(price, lookback)
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
    return X_train, y_train_gru, X_test, y_test_gru

def result(model, scaler, X_test, y_train_pred, y_train_gru, y_test_gru):
    y_test_pred = model(X_test)
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train_gru = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test_gru = scaler.inverse_transform(y_test_gru.detach().numpy())
    return y_train_gru, y_train_pred, y_test_gru, y_test_pred

def plot_training_loss(hist):
    plt.plot(hist, label='Training loss')
    plt.legend()
    plt.savefig('./output/training_loss.png')
    plt.show()

def plot_prediction(data, y_test_gru, y_test_pred):
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    axes.plot(data[len(data)-len(y_test_gru):].index, y_test_gru, color='red', label='Real ADS Stock Price')
    axes.plot(data[len(data)-len(y_test_gru):].index, y_test_pred, color='blue', label='Predicted ADS Stock Price')
    plt.title('Adidas Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Adidas Stock Price')
    plt.legend()
    plt.savefig('./output/ads_pred.png')
    plt.show()

def score(y_train_gru, y_train_pred, y_test_gru, y_test_pred):
    train_score = np.sqrt(mean_squared_error(y_train_gru[:, 0], y_train_pred[:, 0]))
    test_score = np.sqrt(mean_squared_error(y_test_gru[:, 0], y_test_pred[:, 0]))
    return train_score, test_score

def main(filename):
    data = read_file(filename)
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    price = data[['Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))
    X_train, y_train_gru, X_test, y_test_gru = pass_to_tensor(price, lookback)
    hist, y_train_pred = train(model, X_train, y_train_gru)
    plot_training_loss(hist)
    y_train_gru, y_train_pred, y_test_gru, y_test_pred = result(model, scaler, X_test, y_train_pred, y_train_gru, y_test_gru)
    train_score, test_score = score(y_train_gru, y_train_pred, y_test_gru, y_test_pred)
    print('Train Score: %.2f RMSE' % (train_score))
    print('Test Score: %.2f RMSE' % (test_score))
    plot_prediction(data, y_test_gru, y_test_pred)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please use the following command: python3 train.py ./data/ads_preprocessing.csv')
    else:
        main(sys.argv[1])
