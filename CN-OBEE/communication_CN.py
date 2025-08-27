def send_information(node, information_list, ave_value, last_value, std_dev, current_time, G, U):
    if current_time not in U[node]:
        U[node][current_time] = {
            'information_list': [],
            'ave_value': {},
            'last_value': {},  # 新增：存储每个邻居的最后一个值
            'std_dev': {}  # 新增：存储每个邻居的标准差
        }
    U[node][current_time]['information_list'].extend(information_list)
    U[node][current_time]['ave_value'][node] = ave_value
    U[node][current_time]['last_value'][node] = last_value  # 存储最后一个值
    U[node][current_time]['std_dev'][node] = std_dev  # 存储标准差

    for neighbor in G.neighbors(node):
        message = {
            'time': current_time,
            'sender': node,
            'information_list': information_list,
            'ave_value': ave_value,
            'last_value': last_value,
            'std_dev': std_dev
        }
        receive_information(neighbor, message, U)


def receive_information(node, message, U):
    current_time = message['time']
    sender = message['sender']
    if current_time not in U[node]:
        U[node][current_time] = {
            'information_list': [],
            'ave_value': {},
            'last_value': {},  # 新增
            'std_dev': {}  # 新增
        }
    U[node][current_time]['information_list'].extend(message['information_list'])
    U[node][current_time]['ave_value'][sender] = message['ave_value']
    U[node][current_time]['last_value'][sender] = message['last_value']  # 存储最后一个值
    U[node][current_time]['std_dev'][sender] = message['std_dev']  # 存储标准差


def send_mean(G, node, Threshold, U):
    for neighbor in G.neighbors(node):
        if neighbor not in U:
            U[neighbor] = {}
        if 'mean_values' not in U[neighbor]:
            U[neighbor]['mean_values'] = []
        U[neighbor]['mean_values'].append({'node': node, 'mean_value': Threshold})


def send_influence(G, node, influence, U):
    for neighbor in G.neighbors(node):
        if 'influences' not in U[neighbor]:
            U[neighbor]['influences'] = []
        U[neighbor]['influences'].append({'node': node, 'influence': influence})


def send_threshold(G, node, threshold, U):
    for neighbor in G.neighbors(node):
        if neighbor not in U:
            U[neighbor] = {}
        if 'threshold' not in U[neighbor]:
            U[neighbor]['threshold'] = {}
        U[neighbor]['threshold'][node] = threshold


