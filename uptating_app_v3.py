
#%%


import simpy

import numpy as np

import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go


def mm1_queue_simulation(time, lambda_val, mu_val, s):
    wait_times = []
    total_times = []
    total_numbers = []
    queue_lengths = []  # Initialize queue_lengths list
    customer_count = 0
    
    def customer(env, name, server, service_time, queue_lengths, server_utilization, service_times, total_times, total_numbers, customer_data):
        nonlocal customer_count
        arrival_time = env.now
        
        with server.request() as req:
            yield req

            waiting_time = env.now - arrival_time
            current_queue_length = len(server.queue)
            queue_lengths.append(current_queue_length)  # Append current queue length to the list

            current_server_utilization = (lambda_val / mu_val) / s
            total_number = current_queue_length + (s * current_server_utilization)

            total_time = 0
            start_service_time = env.now
            yield env.timeout(service_time)
            actual_service_time = env.now - start_service_time
            total_time = waiting_time + actual_service_time

            service_times.append(actual_service_time)
            server_utilization.append(current_server_utilization)

            total_times.append(total_time)
            total_numbers.append(total_number)

            if waiting_time > 0:
                wait_times.append(waiting_time)

            departure_time = arrival_time + total_time

            customer_data.append((customer_count, env.now, total_number, waiting_time, current_server_utilization, service_time, total_time, current_queue_length, departure_time))
            customer_count += 1
        
    def customer_generator(env, server, lambda_val, mu_val, s, queue_lengths, server_utilization, service_times, customer_data):
        while True:
            interarrival_time = np.random.exponential(1 / lambda_val)
            yield env.timeout(interarrival_time)

            service_time = np.random.exponential(1 / mu_val)

            customer_name = f"Customer {customer_count}"
            env.process(customer(env, customer_name, server, service_time, queue_lengths, server_utilization, service_times, total_times, total_numbers, customer_data))

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=s)
    server_utilization = []
    service_times = []
    customer_data = []

    env.process(customer_generator(env, server, lambda_val, mu_val, s, queue_lengths, server_utilization, service_times, customer_data))
    env.run(until=time)

    avg_queue_length = np.mean(queue_lengths)  # Calculate the average queue length
    
    return wait_times, total_times, total_numbers, queue_lengths, server_utilization, avg_queue_length  # Return avg_queue_length

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'color': 'white'}, children=[
    html.H1("M/M/1 Queue Simulation"),
    html.Div([
        html.Label("Arrival Rate (λ):"),
        dcc.Input(id="lambda", type="number", value=40, step=1),
    ]),
    html.Div([
        html.Label("Service Rate (μ):"),
        dcc.Input(id="mu", type="number", value=60, step=1),
    ]),
    html.Div([
        html.Label("Simulation Time:"),
        dcc.Input(id="time", type="number", value=150, step=1),
    ]),
    html.Div([
        html.Label("Servers:"),
        dcc.Input(id="s", type="number", value=1, step=1),
    ]),
    html.Div([
        html.Label("Time Range:"),
        dcc.RangeSlider(
            id='time-range',
            min=0,
            max=150,  # Initially set to 150
            step=1,
            marks={i: str(i) for i in range(0, 151, 10)},
            value=[0, 150]
        ),
    ]),
    html.Div(id="output-graph"),
    html.Div(id="simulation-results")
])

@app.callback(
    Output("time-range", "max"),  # Update max attribute of the range slider
    [Input("time", "value")]
)
def update_range_slider_max(simulation_time):
    return simulation_time

@app.callback(
    Output("output-graph", "children"),
    [Input("time-range", "value"),
     Input("lambda", "value"),
     Input("mu", "value"),
     Input("s", "value")]
)
def update_output(time_range, lambda_val, mu_val, s):
    if lambda_val is None or mu_val is None or s is None:
        return "Please enter valid values for lambda, mu, and s."

    wait_times, total_times, total_numbers, queue_lengths, server_utilization, avg_queue_length = mm1_queue_simulation(150, lambda_val, mu_val, s)

    # Filter data within the selected time range
    start_index = time_range[0]
    end_index = time_range[1] + 1  # Add 1 to include the end index
    wait_times = wait_times[start_index:end_index]
    total_times = total_times[start_index:end_index]
    total_numbers = total_numbers[start_index:end_index]
    queue_lengths = queue_lengths[start_index:end_index]
    server_utilization = server_utilization[start_index:end_index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(wait_times))), y=wait_times, mode='lines', name='Wait Times'))
    fig.add_trace(go.Scatter(x=list(range(len(total_times))), y=total_times, mode='lines', name='Total Times'))
    fig.add_trace(go.Scatter(x=list(range(len(total_numbers))), y=total_numbers, mode='lines', name='Total Numbers'))
    fig.add_trace(go.Scatter(x=list(range(len(queue_lengths))), y=queue_lengths, mode='lines', name='Queue Lengths'))
    fig.add_trace(go.Scatter(x=list(range(len(server_utilization))), y=server_utilization, mode='lines', name='Server Utilization'))
    
    # Add a line for average queue length
    fig.add_trace(go.Scatter(x=[0, len(queue_lengths)-1], y=[avg_queue_length, avg_queue_length], mode='lines', name='Avg Queue Length', line=dict(color='orange', dash='dash')))

    fig.update_layout(title='Simulation Results')
    
    graph = dcc.Graph(figure=fig)
    return graph

@app.callback(
    Output("simulation-results", "children"),
    [Input("time-range", "value"),
     Input("lambda", "value"),
     Input("mu", "value"),
     Input("s", "value")]
)
def update_simulation_results(time_range, lambda_val, mu_val, s):
    if lambda_val is None or mu_val is None or s is None:
        return None

    wait_times, total_times, total_numbers, queue_lengths, server_utilization, avg_queue_length = mm1_queue_simulation(150, lambda_val, mu_val, s)

    # Filter data within the selected time range
    start_index = time_range[0]
    end_index = time_range[1] + 1  # Add 1 to include the end index
    wait_times = wait_times[start_index:end_index]
    total_times = total_times[start_index:end_index]
    total_numbers = total_numbers[start_index:end_index]
    queue_lengths = queue_lengths[start_index:end_index]
    server_utilization = server_utilization[start_index:end_index]

    avg_queue_length = np.mean(queue_lengths)
    avg_wait_time = np.mean(wait_times)

    average_combined_time = np.mean(total_times)
    average_combined_time = np.array(average_combined_time)
    unique_values, counts = np.unique(average_combined_time, return_counts=True)
    mode_value = unique_values[np.argmax(counts)]  

    avg_wait_times = (sum(wait_times) / len(wait_times))
    avg_server_utilization = np.mean(server_utilization)

    return html.Div([
        html.H3("Simulation Results:"),
        html.Div([
            html.Div(f"Average Queue Length: {avg_queue_length:.5f}"),
            html.Div(f"Actual Average Time in System: {average_combined_time:.5f}"),
            html.Div(f"Mode Average Time in System: {mode_value:.5f}"),
            html.Div(f"Average Wait Time: {avg_wait_time:.5f}"),
            html.Div(f"Average Server Utilization: {avg_server_utilization:.5f}")
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8077)
# %%
