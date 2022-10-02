
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))


def plot_predictions(df_result, future_pred=None, model_name=None):
    data = []

    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)
    if future_pred is not None:
        prediction1 = go.Scatter(
            x=future_pred.index,
            y=future_pred.values,
            mode="lines",
            line={"dash": "dot"},
            name='Next 240 days',
            marker=dict(),
            text=df_result.index,
            opacity=0.8,
        )

        data.append(prediction1)
    if model_name == "lstm":
        layout = dict(
            title="Predictions vs Actual Values for the Ethereum Dataset using LSTM Model",
            xaxis=dict(title="Time", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )
    elif model_name == "gru":

        layout = dict(
            title="Predictions vs Actual Values for the Ethereum Dataset using GRU Model",
            xaxis=dict(title="Time", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )
    elif model_name == "rnn":

        layout = dict(
            title="Predictions vs Actual Values for the Ethereum Dataset using RNN Model",
            xaxis=dict(title="Time", ticklen=5, zeroline=False),
            yaxis=dict(title="Value", ticklen=5, zeroline=False),
        )
    else:
        print("Invalid value.!! Please chose lstm or gru or rnn.!")

    fig = dict(data=data, layout=layout)
    iplot(fig)

def create_seq(E_price,seq_len):
  data = []
  # create all possible sequences of length seq_len

  for index in range(len(E_price) - seq_len):
    data.append(E_price[index: index + seq_len])
  return data


def make_seq(data_, seq_len):
    data = create_seq(data_, seq_len)
    data = np.array(data);
    valid_set_size = int(np.round(valid_num / 100 * data.shape[0]));
    test_set_size = int(np.round(test_num / 100 * data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

# make dataset true/ predictions.
def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result