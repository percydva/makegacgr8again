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

def read_file(filename1, filename2):
    ads_data = pd.read_csv(filename1, parse_dates=[0])
    nke_data = pd.read_csv(filename2, parse_dates=[0])
    return ads_data, nke_data

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

def plot_training_loss(hist, s):
    plt.plot(hist, label='Training loss')
    plt.legend()
    plt.savefig('./output/'+s+'_training_loss.png')
    plt.show()

def plot_prediction(data, y_test_gru, y_test_pred, s):
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    axes.plot(data[len(data)-len(y_test_gru):].index, y_test_gru, color='red', label='Real '+s+' Stock Price')
    axes.plot(data[len(data)-len(y_test_gru):].index, y_test_pred, color='blue', label='Predicted '+s+' Stock Price')
    plt.title(s+' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('./output/'+s+'_pred.png')
    plt.show()

def score(y_train_gru, y_train_pred, y_test_gru, y_test_pred):
    train_score = np.sqrt(mean_squared_error(y_train_gru[:, 0], y_train_pred[:, 0]))
    test_score = np.sqrt(mean_squared_error(y_test_gru[:, 0], y_test_pred[:, 0]))
    return train_score, test_score

def main(filename1, filename2):
    ads_data, nke_data = read_file(filename1, filename2)
    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    price1 = ads_data[['Close']]
    price2 = nke_data[['Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price1['Close'] = scaler.fit_transform(price1['Close'].values.reshape(-1, 1))
    price2['Close'] = scaler.fit_transform(price2['Close'].values.reshape(-1, 1))
    X1_train, y1_train_gru, X1_test, y1_test_gru = pass_to_tensor(price1, lookback)
    X2_train, y2_train_gru, X2_test, y2_test_gru = pass_to_tensor(price2, lookback)
    hist1, y1_train_pred = train(model, X1_train, y1_train_gru)
    hist2, y2_train_pred = train(model, X2_train, y2_train_gru)
    plot_training_loss(hist1, 'ads')
    plot_training_loss(hist2, 'nke')
    y1_train_gru, y1_train_pred, y1_test_gru, y1_test_pred = result(model, scaler, X1_test, y1_train_pred, y1_train_gru, y1_test_gru)
    y2_train_gru, y2_train_pred, y2_test_gru, y2_test_pred = result(model, scaler, X2_test, y2_train_pred, y2_train_gru, y2_test_gru)
    train1_score, test1_score = score(y1_train_gru, y1_train_pred, y1_test_gru, y1_test_pred)
    train2_score, test2_score = score(y2_train_gru, y2_train_pred, y2_test_gru, y2_test_pred)
    print('Train ADS Score: %.2f RMSE' % (train1_score))
    print('Test ADS Score: %.2f RMSE' % (test1_score))
    print('Train NKE Score: %.2f RMSE' % (train2_score))
    print('Test NKE Score: %.2f RMSE' % (test2_score))
    plot_prediction(ads_data, y1_test_gru, y1_test_pred, 'ads')
    plot_prediction(nke_data, y2_test_gru, y2_test_pred, 'nke')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Example command: python3 train.py ./data/<filename1>.csv ./data/<filename2>.csv')
    else:
        main(sys.argv[1], sys.argv[2])
