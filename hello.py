import os
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import Column, Integer, String, Date, Float, ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.types import PickleType

from werkzeug import secure_filename
import json
import time
import io
import sqlite3
import numpy as np
import shutil
from cvxpy import *

from io_handler import *
from solve_handler import *

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(APP_ROOT, 'data')

app = Flask(__name__)
CORS(app)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'

db = SQLAlchemy(app)
   
class Network(db.Model):
    __tablename__ = 'network'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    nodes = relationship('Node', backref='network')
    edges = relationship('Edge', backref='network')
    pumps = relationship('Pump', backref='network')

    def __init__(self, name):
        self.name = name

    @property
    def serialize(self):
        return {'id': self.id,
                'name': self.name}

class Node(db.Model):
    __tablename__ = 'node'
    id = Column(Integer, primary_key=True)
    node_id = Column(Integer)
    node_name = Column(String)
    demand = Column(Float)
    head = Column(Float)
    node_type = Column(Integer)
    x = Column(Float)
    y = Column(Float)
    network_id = Column(Integer, ForeignKey('network.id'))

    def __init__(self, node_id, node_name, demand, head, node_type, x, y, network):
        self.node_id = node_id
        self.node_name = node_name
        self.demand = demand
        self.head = head
        self.node_type = node_type
        self.x = x
        self.y = y
        self.network = network

    @property
    def serialize(self):
        return {'node_id': self.node_id,
                'node_name': self.node_name,
                'demand': self.demand,
                'head': self.head,
                'node_type': self.node_type,
                'x': self.x,
                'y': self.y,
                'network': self.network.serialize}

class Edge(db.Model):
    __tablename__ = 'edge'
    id = Column(Integer, primary_key=True)
    edge_id = Column(Integer)
    head_id = Column(Integer)
    tail_id = Column(Integer)
    length = Column(Float)
    diameter = Column(Float)
    roughness = Column(Float)
    edge_type = Column(Integer)
    network_id = Column(Integer, ForeignKey('network.id'))

    def __init__(self, edge_id, head_id, tail_id, length, diameter, roughness, edge_type, network):
        self.edge_id = edge_id
        self.head_id = head_id
        self.tail_id = tail_id
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.edge_type = edge_type
        self.network = network

    @property
    def serialize(self):
        return {'edge_id': self.edge_id,
                'head_id': self.head_id,
                'tail_id': self.tail_id,
                'length': self.length,
                'diameter': self.diameter,
                'roughness': self.roughness,
                'edge_type': self.edge_type,
                'network': self.network.serialize} 

class Pump(db.Model):
    __tablename__ = 'pump'
    id = Column(Integer, primary_key=True)
    pump_id = Column(Integer)
    edge_id = Column(Integer)                
    head_id = Column(Integer)
    tail_id = Column(Integer)
    x = Column(PickleType)
    y = Column(PickleType)
    coeff = Column(PickleType)
    network_id = Column(Integer, ForeignKey('network.id'))

    def __init__(self, pump_id, edge_id, head_id, tail_id, x, y, coeff, network):
        self.pump_id = pump_id
        self.edge_id = edge_id
        self.head_id = head_id
        self.tail_id = tail_id
        self.x = x
        self.y = y
        self.coeff = coeff
        self.network = network

    @property
    def serialize(self):
        return {'pump_id': self.pump_id,
                'edge_id': self.edge_id,
                'head_id': self.head_id,
                'tail_id': self.tail_id,
                'x': self.x,
                'y': self.y,
                'coeff': self.coeff,            
                'network': self.network.serialize} 

def create_pumps(network, pump_curves):
    for info in pump_curves:
        pump = Pump(info['pump_id'],
                    info['edge_id'], 
                    info['head_id'], 
                    info['tail_id'], 
                    info['x'], 
                    info['y'], 
                    info['coeff'],
                    network)
        db.session.add(pump)
    db.session.commit()

def create_nodes(network, nodes):
    for info in nodes:
        node = Node(info['node_id'], 
                    info['node_name'], 
                    info['demand'], 
                    info['head'], 
                    info['node_type'], 
                    info['x'], 
                    info['y'],
                    network)
        db.session.add(node)
    db.session.commit()

def create_edges(network, edges):
    for info in edges:
        edge = Edge(info['edge_id'], 
                    info['head_id'], 
                    info['tail_id'], 
                    info['length'], 
                    info['diameter'], 
                    info['roughness'], 
                    info['edge_type'],
                    network)
        db.session.add(edge)
    db.session.commit()

def create_network(name):
    print(name)
    
    network = Network(name)
    db.session.add(network) 
    db.session.commit()

    # Read info from .inp file
    file = open(name, 'r')
    nodes, edges, pump_curves = read_inp(file)

    # ======== PLEASE CHECK THESE VARIABLES ========
    # Get all variables from the network
    A, L = extract_var_edges(edges, len(nodes))    
    d, hc = extract_var_nodes(nodes)   
    dh_max, L_pump, pump_head_list = extract_var_pumps(pump_curves, L)    
    # ==============================================

    # PREDIRECTION 
    obj = predirection(A, L, d)
    print('pre=======127')
    print(obj)
    print('=======')

    # Save varibles locally
    network_var_dir = get_var_dir(network)
    create_folder(network_var_dir)    

    save_var(network, 'A', A)
    save_var(network, 'L', L)
    save_var(network, 'd', d)
    save_var(network, 'hc', hc)
    save_var(network, 'dh_max', dh_max)
    save_var(network, 'L_pump', L_pump)
    save_var(network, 'pump_head_list', pump_head_list)

    # print('dfghjklfghjkl')
    # A = load_var(network, 'A')  
    # L_pump = load_var(network, 'L_pump')
    # dh_max = load_var(network, 'dh_max')
    # d = load_var(network, 'd')    
    # obj, q = solve_imaginary_flow(A, L_pump, dh_max, d)

    # # # obj = get_imaginary_flow(network)
    # print('flow=======-32874')
    # print(obj)
    # print('=======')

    # A = load_var(network, 'A')    
    # L_pump = load_var(network, 'L_pump')
    # dh_max = load_var(network, 'dh_max')
    # hc = load_var(network, 'hc')
    # pump_head_list = load_var(network, 'pump_head_list')
 
    # obj, h = solve_imaginary_pressure(A, L_pump, dh_max, hc, pump_head_list, q)
    # # obj = get_imaginary_pressure(network)
    # print('press=======1312')
    # print(obj)
    # print('=======')

    # A = load_var(network, 'A')    
    # L_pump = load_var(network, 'L_pump')
    # dh_max = load_var(network, 'dh_max')
    # d = load_var(network, 'd')    
    # obj, q = solve_max_flow(A, L_pump, dh_max, d, h)

    # # obj = get_max_flow(network)
    # print('max=======-9')
    # print(obj)
    # print('=======')

    # Save nodes, edges to database
    create_nodes(network, nodes)
    create_edges(network, edges)
    create_pumps(network, pump_curves)

    print('xcvxcvbnxcvbncvbnm,')

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def create_database():
    db.create_all()
    create_folder(DATA_FOLDER)
    tic = time.clock()  
    create_network('Small.inp')
    toc = time.clock()
    print('Small use time: %f' % (toc-tic))

    # tic = time.clock()  
    # create_network('Big.inp')
    # toc = time.clock()
    # print('Big use time: %f' % (toc-tic))

    db.session.commit()

# shutil.rmtree(DATA_FOLDER)
db.drop_all()
create_database()

## HAVE NOT TEST
@app.route('/api/imaginary_flow/<network_id>')
def get_imaginary_flow(network_id):
    network = Network.query.filter_by(id=network_id).first()
    A = load_var(network, 'A')  
    L_pump = load_var(network, 'L_pump')
    dh_max = load_var(network, 'dh_max')
    d = load_var(network, 'd')
    obj_flow, q = solve_imaginary_flow(A, L_pump, dh_max, d)
    save_var(network, 'q', q)
    edges = []
    for i, item in enumerate(q):
        edge = {'edge_id': i + 1, 'flow': item.tolist()[0][0]}
        edges.append(edge)
    return jsonify(obj_flow=obj_flow, edges=edges)    

@app.route('/api/imaginary_pressure/<network_id>')
def get_imaginary_pressure(network_id):
    network = Network.query.filter_by(id=network_id).first()
    A = load_var(network, 'A')  
    L_pump = load_var(network, 'L_pump')
    dh_max = load_var(network, 'dh_max')
    hc = load_var(network, 'hc')
    pump_head_list = load_var(network, 'pump_head_list')    
    q = load_var(network, 'q')
    obj_pressure, h = solve_imaginary_pressure(A, L_pump, dh_max, hc, pump_head_list, q)
    save_var(network, 'h', h)
    nodes = []
    for i, item in enumerate(h):
        node = {'node_id': i + 1, 'pressure': item.tolist()[0][0]}
        nodes.append(node)
    return jsonify(obj_pressure=obj_pressure, nodes=nodes)   

def get_imaginary_flow_and_pressure(network_id):
    network = Network.query.filter_by(id=network_id).first()
    A = load_var(network, 'A')  
    L_pump = load_var(network, 'L_pump')
    dh_max = load_var(network, 'dh_max')
    d = load_var(network, 'd')
    hc = load_var(network, 'hc')
    pump_head_list = load_var(network, 'pump_head_list')    
    obj_flow, q = solve_imaginary_flow(A, L_pump, dh_max, d)
    obj_pressure, h = solve_imaginary_pressure(A, L_pump, dh_max, hc, pump_head_list, q)
    save_var(network, 'q', q)
    save_var(network, 'h', h)
    edges = []
    for i, item in enumerate(q):
        edge = {'edge_id': i + 1, 'flow': item.tolist()[0][0]}
        edges.append(edge)
    nodes = []
    for i, item in enumerate(h):
        node = {'node_id': i + 1, 'pressure': item.tolist()[0][0]}
        nodes.append(node)
    return jsonify(obj_flow=obj_flow, obj_pressure=obj_pressure, nodes=nodes, edges=edges) 

@app.route('/api/max_flow/<network>')
def get_max_flow(network):
    A = load_var(network, 'A')    
    L_pump = load_var(network, 'L_pump')
    dh_max = load_var(network, 'dh_max')
    d = load_var(network, 'd')  
    h = load_var(network, 'h')  
    obj, q = solve_max_flow(A, L_pump, dh_max, d, h)
    save_var(network, 'q', q)
    return jsonify(obj=obj, flow=q)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/pump_curve')
def pump_curve():
    return render_template("pump_curve.html")

@app.route('/api/hello')
def hello():
    return "hello"

@app.route('/api/hello2')
def hello2():
    print(jj)
    return "hello"

@app.route('/api/nodes/<network>')
def get_nodes(network):
    nodes = Node.query.filter(Node.network.has(id=network))
    return jsonify(json_list = [node.serialize for node in nodes])

@app.route('/api/edges/<network>')
def get_edges(network):
    edges = Edge.query.filter(Edge.network.has(id=network))  
    return jsonify(json_list = [edge.serialize for edge in edges])

@app.route('/api/networks')
def get_networks():
    networks = Network.query.all()
    return jsonify(json_list = [network.serialize for network in networks])

@app.route('/api/pumps/<network>')
def get_pumps(network):    
    pumps = Pump.query.filter(Pump.network.has(id=network))  
    return jsonify(json_list = [pump.serialize for pump in pumps])

@app.route('/api/pumps/<network>/<pump_id>')
def get_pump(network, pump_id):
    print(pump_id)
    pumps = Pump.query.filter(Pump.network.has(id=network) & (Pump.pump_id == pump_id))  
    return jsonify(json_list = [pump.serialize for pump in pumps])

@app.route('/api/cvx')
def getget():
    m = 20
    n = 10
    p = 4
    A = np.random.rand(m,n)
    b = np.random.rand(m,1)
    C = np.random.rand(p,n); 
    d = np.random.rand(p,1); 
    e = np.random.rand(1);
    x = Variable(n,1)
    constraints = [C*x == d, norm( x, float('Inf') ) <= e]
    obj = Minimize( norm( A * x - b, 2 ) )
    prob=Problem(obj, constraints)
    prob.solve(verbose=True, solver=MOSEK)
    print(obj.value)
    return str(prob.value)
       
    

