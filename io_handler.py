import os
import json
import numpy as np

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(APP_ROOT, 'data')

GRAVITY = 9.8
SOURCE = 2
PUMP = 1


def read_inp(file):
    node_id = 0
    edge_id = 0
    pump_id = 0

    nodes = []
    edges = []
    pumps = []

    node_map = {}

    read_format = ''

    for line in file:

        if (line.isspace()):
            read_format = ''
            continue

        if ('[JUNCTIONS]' in line):
            read_format = 'node'
            next(file)
            continue

        if ('[RESERVOIRS]' in line):
            read_format = 'source'
            next(file)
            continue

        if ('[TANKS]' in line):
            read_format = 'tank'
            next(file)
            continue

        if ('[PIPES]' in line):
            read_format = 'pipe'
            next(file)
            continue

        if ('[PUMPS]' in line):
            read_format = 'pump'
            next(file)
            continue

        if ('[VALVES]' in line):
            read_format = 'valve'
            next(file)
            continue

        if (';PUMP: PUMP:' in line):
            pump_id += 1
            read_format = 'pump_curve'
            continue

        if ('[COORDINATES]' in line):
            next(file)
            read_format = 'coordinates'
            continue

        if (read_format is 'node'):
            data = line.split()
            node_id += 1
            name = data[0]
            head = float(data[1])
            demand = float(data[2])
            node_type = 1 if demand > 0 else 0

            node_map[name] = node_id

            node = {'node_id': node_id,
                    'node_name': name,
                    'demand': demand,
                    'head': head,
                    'node_type': node_type}
            nodes.append(node)

        if (read_format is 'source'):
            data = line.split()
            node_id += 1
            name = data[0]
            head = float(data[1])
            demand = -1
            node_type = 2

            node_map[name] = node_id

            node = {'node_id': node_id,
                    'node_name': name,
                    'demand': demand,
                    'head': head,
                    'node_type': node_type}
            nodes.append(node)

        if (read_format is 'tank'):
            node_id += 1
            data = line.split()
            name = data[0]
            head = float(data[1])
            demand = 0
            node_type = 3
            node_map[name] = node_id

            node = {'node_id': node_id,
                    'node_name': name,
                    'demand': demand,
                    'head': head,
                    'node_type': node_type}
            nodes.append(node)

        if (read_format is 'pipe'):
            data = line.split()
            edge_id += 1
            head_id = node_map[data[1]]
            tail_id = node_map[data[2]]
            length = float(data[3])
            diameter = float(data[4])
            roughness = float(data[5])
            edge_type = 0

            edge = {'edge_id': edge_id,
                    'head_id': head_id,
                    'tail_id': tail_id,
                    'length': length,
                    'diameter': diameter,
                    'roughness': roughness,
                    'edge_type': edge_type}
            edges.append(edge)

        if (read_format is 'pump'):
            edge_id += 1
            data = line.split()
            head_id = node_map[data[1]]
            tail_id = node_map[data[2]]

            # default for predirection
            length = 0.1
            diameter = 250.0
            roughness = 1.50
            edge_type = 1

            edge = {'edge_id': edge_id,
                    'head_id': head_id,
                    'tail_id': tail_id,
                    'length': length,
                    'diameter': diameter,
                    'roughness': roughness,
                    'edge_type': edge_type}
            edges.append(edge)

        if (read_format is 'valve'):
            edge_id += 1
            data = line.split()
            head_id = node_map[data[1]]
            tail_id = node_map[data[2]]
            length = 0.1
            diameter = float(data[3])
            roughness = float(data[6])
            edge_type = 2

            edge = {'edge_id': edge_id,
                    'head_id': head_id,
                    'tail_id': tail_id,
                    'length': length,
                    'diameter': diameter,
                    'roughness': roughness,
                    'edge_type': edge_type}
            edges.append(edge)

        if (read_format is 'pump_curve'):
            data = line.split()
            pump_name = data[0].split(".")
            if pump_name[0] in node_map:
                head_id = node_map[pump_name[0]]
                tail_id = node_map[pump_name[1]]
                x_value = float(data[1])
                y_value = float(data[2])

                curve = {'pump_id': pump_id,
                         'head_id': head_id,
                         'tail_id': tail_id,
                         'x_value': x_value,
                         'y_value': y_value}
                pumps.append(curve)
                pump_id_proxy = pump_id

        if (read_format is 'coordinates'):
            data = line.split()
            name = data[0]
            x = float(data[1])
            y = float(data[2])
            node_id = node_map[name]
            nodes[node_id - 1]['x'] = x
            nodes[node_id - 1]['y'] = y

        pump_curves = post_process_pumps(pumps, edges)

    return nodes, edges, pump_curves


def get_pump_edges(edges):
    return [edge for edge in edges if edge['edge_type'] == PUMP]


def post_process_pumps(pumps, edges):
    pump_curve = {}
    for pump in pumps:
        pump_id = int(pump['pump_id'])
        if pump_id not in pump_curve:
            pump_info = {}
            pump_info['pump_id'] = pump_id

            pump_info['head_id'] = int(pump['head_id'])
            pump_info['tail_id'] = int(pump['tail_id'])
            pump_info['x'] = []
            pump_info['y'] = []
            pump_info['coeff'] = []
            pump_curve[pump_id] = pump_info

        if pump_id <= 20:
            pump_curve[pump_id]['x'].append(float(pump['x_value']) / 100.0)
            pump_curve[pump_id]['y'].append(float(pump['y_value']) * 100.0)
        else:
            pump_curve[pump_id]['x'].append(float(pump['x_value']) / 1000.0)
            pump_curve[pump_id]['y'].append(float(pump['y_value']) * 100.0)

    pump_edges = get_pump_edges(edges)

    for pump_info in pump_curve.values():

        # Fit pump curve
        x = np.asarray(pump_info['x']) ** 2
        y = np.asarray(pump_info['y'])
        AM = np.vstack((x, np.ones_like(x))).T
        m, c = np.linalg.lstsq(AM, y)[0]
        pump_info['coeff'] = [m, c]

        # Match pump with edge
        for edge in pump_edges:
            if edge['head_id'] == pump_info['head_id'] and edge['tail_id'] == pump_info['tail_id']:
                #				pump_info['edge_id'] = edge['edge_id']
                pump_info['edge_id'] = edge['edge_id']  # modified

    return [pump_info for pump_info in pump_curve.values()]


def extract_var_edges(edges, num_v):
    num_e = len(edges)
    A = np.zeros((num_v, num_e))
    L = np.zeros(num_e)
    for i, edge in enumerate(edges):
        head_id = int(edge['head_id'])
        tail_id = int(edge['tail_id'])
        length = float(edge['length'])
        diameter = float(edge['diameter']) / 1000.0
        roughness = float(edge['roughness']) / 1000.0
        friction = (2 * np.log10(roughness / (3.71 * diameter))) ** 2
        L[i] = (8 * length * friction) / (np.pi ** 2 * GRAVITY * diameter ** 5)
        A[head_id - 1, i] = 1
        A[tail_id - 1, i] = -1
    return np.asmatrix(A), np.asmatrix(L).T


def extract_var_nodes(nodes):
    num_v = len(nodes)
    d = np.zeros(num_v)
    hc = np.zeros(num_v)
    for i, node in enumerate(nodes):
        demand = float(node['demand'])
        head = float(node['head'])
        if node['node_type'] == SOURCE:
            d[i] = 1000
        else:
            d[i] = -demand / 1000.0
        hc[i] = head
    return np.asmatrix(d).T, np.asmatrix(hc).T


def extract_var_pumps(pump_curves, L):
    num_e = L.shape[0]
    dh_max = np.zeros(num_e)
    L_pump = np.copy(L)
    pump_head_list = []
    for pump_info in pump_curves:
        dh_max[pump_info['edge_id'] - 1] = pump_info['coeff'][1]
        L_pump[pump_info['edge_id'] - 1] = -pump_info['coeff'][0]
        pump_head_list.append(pump_info['head_id'] - 1)
    return np.asmatrix(dh_max).T, np.asmatrix(L_pump), pump_head_list


def save_var(network, filename, variable):
    network_var_dir = get_var_dir(network)
    path = os.path.join(network_var_dir, filename)
    np.save(path, variable)


def load_var(network, filename):
    network_var_dir = get_var_dir(network)
    path = os.path.join(network_var_dir, filename)
    path += '.npy'
    return np.load(path)


def get_var_dir(network):
    network_var_dir = str(network.id) + '_' + network.name
    return os.path.join(DATA_FOLDER, network_var_dir)
