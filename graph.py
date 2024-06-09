import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import timeit
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import DisjointSet
import math
from numpy import linalg as LA
from numba import jit
import copy


class Network(object):
    def __init__(self , num_nodes):
        self.adj = {i:set() for i in range (num_nodes)}

    def add_edge(self , i , j):
        self.adj[i].add(j)
        self.adj[j].add(i)
    
    def add_random_edge(self):
        added = False
        while not added:
            i = random.randint(0 , len(self.adj)-1)
            j = random.randint(0 , len(self.adj)-1)
            if i != j:
                if j not in self.adj[i]:
                    self.add_edge(i , j)
                    added = True
        self.add_edge(i , j)

    def add_triangle(self):
        added = False
        while not added:
            i = random.randint(0 , len(self.adj)-1)
            j = random.randint(0 , len(self.adj)-1)
            k = random.randint(0 , len(self.adj)-1)
            if i != j and j != k and i != k:
                if j in self.adj[i] and k in self.adj[i]:
                    if j not in self.adj[k]:
                        self.add_edge(j , k)
                        added = True
        self.add_edge(j , k)

    def add_triangle_2(self):

        edges_added = 0

        i = random.randint(0 , len(self.adj)-1)
        j = random.randint(0 , len(self.adj)-1)
        k = random.randint(0 , len(self.adj)-1)
        if i != j and j != k and i != k:
            if j not in self.adj[i]:
                self.add_edge(i , j)
                edges_added += 1
            if k not in self.adj[i]:
                self.add_edge(i , k)
                edges_added += 1
            if j not in self.adj[k]:
                self.add_edge(j , k)
                edges_added += 1

        return edges_added
    
    def add_triangle_3(self):
        i = random.randint(0 , len(self.adj)-1)
        if len(self.adj[i]) < 1:
            return 0
        j = np.random.choice(list(self.adj[i]))
        if len(self.adj[j]) < 2:
            return 0
        k = np.random.choice(list(self.adj[j]))
        if k not in self.adj[i] and i != k:
            self.add_edge(i , k)
            return 1
        return 0
                

    def number_of_triangles(self):
        triangles = 0
        for i in self.adj:
            for j in self.adj[i]:
                for k in self.adj[j]:
                    if k in self.adj[i]:
                        triangles += 1
        return triangles//6
    
    def number_of_triangles_quick(self):
        adj_matrix = np.matrix([[1 if j in self.adj[i] else 0 for j in range(len(self.adj))] for i in range(len(self.adj))])
        print(adj_matrix.shape)
        adj_matrix_cubed = adj_matrix**3
        """diag, P = LA.eig(adj_matrix)
        DN = np.diag(diag**3)
        P1 = LA.inv(P)

        adj_matrix_cubed = P*DN*P1"""
        print("HERE 2")
        return np.trace(adj_matrix_cubed)//6
    
    def neighbors(self , i):
        return self.adj[i]
    
    def edge_list(self):
        return [(i,j)for i in self.adj for j in self.adj[i] if i < j]
    
    def num_edges(self):
        return sum([len(self.adj[i]) for i in self.adj])//2
    
    def find_component(self, i):
        c = set()
        stack = [i]
        while stack:
            node = stack.pop()
            c.add(node)
            stack.extend(self.neighbors(node) - c)
        return c
    
    def degree_distributions_individual(self):
        return np.array([len(self.adj[i]) for i in self.adj])
    
    def degree_distributions(self):
        degrees = [len(self.adj[i]) for i in self.adj]
        return pd.Series(degrees).value_counts().sort_index()
    
    def friends_degree_distribution(self):
        return np.array([np.mean([len(self.adj[j]) for j in self.adj[i]]) for i in self.adj])
    

class random_graph(Network):
    def __init__(self , num_nodes , p):
        super().__init__(num_nodes)
        for i in range(num_nodes):
            for j in range(i+1 , num_nodes):
                if random.random() > (1-p):
                    self.add_edge(i , j)

class lambda_graph(Network):
    def __init__(self , num_nodes , lambda_):
        super().__init__(num_nodes)
        num_edges = lambda_
        num_added_edges = self.num_edges()
        while num_added_edges < num_edges:
            i = random.randint(0 , num_nodes-1)
            j = random.randint(0 , num_nodes-1)
            if i != j:
                if j not in self.adj[i]:
                    num_added_edges += 1
                self.add_edge(i , j)

class configuration_graph(Network):
    def __init__(self , num_nodes , degree_sequence):
        super().__init__(num_nodes)
        S = np.array([ i for i in range (num_nodes) for _ in range (degree_sequence[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)

class poisson_configuration_graph(Network):
    def __init__(self , num_nodes , lambda_):
        super().__init__(num_nodes)
        S = np.random.poisson(lambda_ , num_nodes)
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)

class geometric_configuration_graph(Network):
    def __init__(self , num_nodes , p):
        super().__init__(num_nodes)
        S = np.random.geometric(p , num_nodes) - 1
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)

class popular_geometric_configuration_graph(Network):
    def __init__(self , num_nodes , p, popular_nodes_num):
        super().__init__(num_nodes)
        S = np.random.geometric(p , num_nodes) - 1
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)
        # print(self.adj[0])
        popular_nodes = []
        temp = self.adj.copy()
        for i in range(popular_nodes_num):
            popular_nodes.append(max(temp, key=lambda x: len(temp[x])))
            temp.pop(max(temp, key=lambda x: len(temp[x])))
            #print(popular_nodes)
            #print(len(self.adj[0]))
        for i in popular_nodes:
            for j in popular_nodes:
                if i != j:
                    #print(i,j)
                    #print(i,j)
                    #print(self.adj[i])
                    #print(self.adj[j])
                    self.add_edge(i, j)
        
class popular_poisson_configuration_graph(Network):
    def __init__(self , num_nodes , lambda_, popular_nodes_num):
        super().__init__(num_nodes)
        S = np.random.poisson(lambda_ , num_nodes)
        S = np.array([ i for i in range (num_nodes) for _ in range (S[i])])
        S = np.random.permutation(S)
        if len (S) % 2:
            S = S[:-1]
        S = S.reshape(-1 ,2)
        for i , j in S:
            self.add_edge(i , j)
        popular_nodes = []
        temp = self.adj.copy()
        for i in range(popular_nodes_num):
            popular_nodes.append(max(temp, key=lambda x: len(temp[x])))
            temp.pop(max(temp, key=lambda x: len(temp[x])))
        for i in popular_nodes:
            for j in popular_nodes:
                if i != j:
                    self.add_edge(i, j)

def SIR_simulation(graph, lamda, num_initial_infected, num_steps, vaccination_rate = 0):
    num_vaccinated = int(vaccination_rate * len(graph.adj))
    inital_vaccinated = random.sample(range(len(graph.adj)), num_vaccinated)
    vaccinated = set(inital_vaccinated)
    inital_infected = random.sample(range(len(graph.adj)), num_initial_infected)
    if num_initial_infected == 1:
        inital_infected = [1]
    infected = set(inital_infected)
    num_infected = [len(infected)]
    susceptible = set(range(len(graph.adj))) - infected - vaccinated
    num_susceptible = [len(susceptible)]
    recovered = set()
    num_recovered = [len(recovered)]
    for _ in range(num_steps):
        if infected == set():
            return num_infected, num_susceptible, num_recovered
        new_infected = set()
        for i in infected:
            for j in graph.neighbors(i):
                if j in susceptible:
                    if random.random() < lamda:
                        new_infected.add(j)
        susceptible -= new_infected
        recovered |= infected
        infected |= new_infected
        infected -= recovered
        num_infected.append(len(infected))
        num_susceptible.append(len(susceptible))
        num_recovered.append(len(recovered))
    return num_infected, num_susceptible, num_recovered

def SIR_simulation_2(graph, lamda, num_initial_infected, num_steps, vaccination_rate = 0):
    num_vaccinated = int(vaccination_rate * len(graph.adj))
    inital_vaccinated = random.sample(range(len(graph.adj)), num_vaccinated)
    vaccinated = set(inital_vaccinated)
    inital_infected = random.sample(range(len(graph.adj)), num_initial_infected)
    if num_initial_infected == 1:
        inital_infected = [1]
    infected = set(inital_infected)
    num_infected = [len(infected)]
    susceptible = set(range(len(graph.adj))) - infected - vaccinated
    num_susceptible = [len(susceptible)]
    recovered = set()
    num_recovered = [len(recovered)]
    for _ in range(num_steps):
        if infected == set():
            return infected, susceptible, recovered, vaccinated
        new_infected = set()
        for i in infected:
            for j in graph.neighbors(i):
                if j in susceptible:
                    if random.random() < lamda:
                        new_infected.add(j)
        susceptible -= new_infected
        recovered |= infected
        infected |= new_infected
        infected -= recovered
        num_infected.append(len(infected))
        num_susceptible.append(len(susceptible))
        num_recovered.append(len(recovered))
    return infected, susceptible, recovered, vaccinated

def time_SIR_simulation(graph, lamda, num_initial_infected, time_infected, num_steps):
    initial_infected = random.sample(range(len(graph.adj)), num_initial_infected)
    infected = set(initial_infected)
    num_infected = [len(infected)]
    susceptible = set(range(len(graph.adj))) - infected
    num_susceptible = [len(susceptible)]
    recovered = set()
    num_recovered = [len(recovered)]
    tracking_infected = {i: time_infected for i in infected}
    #print(tracking_infected)
    for _ in range(num_steps):
        if infected == set():
            #print(infected)
            return num_infected, num_susceptible, num_recovered
        new_infected = set()
        for i in infected:
            for j in graph.neighbors(i):
                if j in susceptible:
                    if random.random() < lamda:
                        new_infected.add(j)
        non_overlapping_new = new_infected - infected
        susceptible -= new_infected
        tracking_infected = {i: tracking_infected[i]-1 for i in infected}
        tracking_infected |= {i: time_infected for i in non_overlapping_new}
        #print(tracking_infected)
        infected |= new_infected
        for i in infected:
            if tracking_infected[i] == 0:
                recovered.add(i)
                tracking_infected.pop(i)
        #print(recovered)
        infected -= recovered
        num_infected.append(len(infected))
        num_susceptible.append(len(susceptible))
        num_recovered.append(len(recovered))
    return num_infected, num_susceptible, num_recovered

def one_loop_SIR_simulation(graph, lamda):
    C = DisjointSet(range(len(graph.adj)))
    for i in range(len(graph.adj)):
        for j in graph.neighbors(i):
            if random.random() < lamda:
                C.merge(i, j)
    return C

def infection_probability_equation(graph, lamda, num_inital_infected, steps):
    """inital_infected = random.sample(range(len(graph.adj)), num_inital_infected)
    inital_infected = np.array(inital_infected)
    s_i = np.zeros(len(graph.adj))
    s_i[inital_infected] = 1"""
    s_i = np.random.uniform(0, 0.1, len(graph.adj))
    for i in range(steps):
        temp_s_i = s_i.copy()
        for j in range(len(graph.adj)):
            s_i[j] = np.prod([1 - lamda + lamda * temp_s_i[k] for k in graph.neighbors(j)]) 
    return s_i

def quick_SIR(graph, lamda, node):
    C = DisjointSet(range(len(graph.adj)))
    num_edges = graph.num_edges()
    random_i = np.arange(len(graph.adj))
    random.shuffle(random_i)
    c_node_k = []
    for i in random_i:
        random_j = np.array(list(graph.neighbors(i)))
        random.shuffle(random_j)
        for j in random_j:
            if random.random() < lamda:
                C.merge(i, j)
                c_node_k.append(C.subset_size(node))
    max_index = min(len(c_node_k)-1, 10)
    c_node_lambda = np.sum([math.comb(num_edges, k) * lamda**k * (1-lamda)**(num_edges-k) * c_node_k[k] for k in range(max_index)])
    return c_node_lambda
    
       
def week_one_two():
    """for i in range(11):
        edges = [random_graph(100 , i*0.1).num_edges() for _ in range(1000)]
        plt.hist(edges, bins = 50)
        plt.xlabel('Number of edges')
        plt.ylabel('Number of graphs')
        avg = np.mean(edges)
        plt.title(f'p = {np.round(i*0.1,2)}, Average = {np.round(avg)}, Variance = {np.round(np.var(edges),2)}, Expected Value = {100*99*i*0.1/2}, Expected Variance = {np.round(100*99*i*0.1*(1-i*0.1)/2,2)}')
        plt.axvline(avg , color = 'red')
        plt.savefig(f'p={i*0.1}.png',bbox_inches='tight')
        plt.close()
        #plt.show()"""
    
    """random_grah_times = np.log([timeit.timeit(lambda: random_graph(2**ii, 10/ ((2**ii)-1)) , number=100) for ii in range(6, 11)])
    lambda_graph_times = np.log([timeit.timeit(lambda: lambda_graph(2**ii, 10), number=100) for ii in range(6, 11)])
    x_axis = np.log([2**i for i in range(6, 11)])
    random_gradient, intercept = np.polyfit(x_axis, random_grah_times, 1)
    random_gradient = round(random_gradient, 2)
    lambda_gradient, intercept = np.polyfit(x_axis, lambda_graph_times, 1)
    lambda_gradient = round(lambda_gradient, 2)
    plt.plot(x_axis, random_grah_times, label=f'Random Graph, Gradient = {random_gradient}', color='blue')
    print('Random Graph Done')
    plt.plot(x_axis, lambda_graph_times, label=f'Set Number of Edges, Gradient={lambda_gradient}', color='red')
    plt.xlabel('Log Number of nodes')
    print('Lambda graph Done')
    plt.ylabel('Log Time taken')
    plt.title('Time taken for Random Graph and Lambda Graph')
    plt.legend()
    plt.show()"""

    """num_components = []
    for i in range(21):
        num_components_one = [len(lambda_graph(4096, (4096*4095*0.001*i/20/2)).find_component(1)) for _ in range(40)]
        num_components.append(np.mean(num_components_one))
    plt.plot([0.001*i/20 for i in range(21)], num_components)
    plt.axvline(1/4095, color='red')
    plt.xlabel('p')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as p changes')
    plt.show()"""

    #print(configuration_graph(10, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).degree_distributions())

    """degrees = [poisson_configuration_graph(10000, 10).degree_distributions()]
    degrees = pd.DataFrame(degrees)
    combined_degrees = degrees
    values = combined_degrees.values.flatten()
    #print(values)
    coloumns = combined_degrees.columns
    if 0 in coloumns:
        index_0 = np.where(coloumns == 0)
        value_0 = values[index_0]
        coloumns = np.delete(coloumns, index_0)
        values = np.delete(values, index_0)
        coloumns = np.insert(coloumns, 0, 0)
        values = np.insert(values, 0, value_0)

    #combined_degrees = combined_degrees.to_numpy().flatten()
    plt.plot(coloumns,values)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    
    mean = np.sum([i*j for i, j in zip(coloumns, values)])/np.sum(values)
    print(mean)
    plt.title(f'Degree distribution of a Poisson Configuration Graph, mean = {mean}')
    plt.axvline(mean, color='red')
    plt.show()"""
   
    """degrees = [geometric_configuration_graph(10000, 1/11).degree_distributions()]
    degrees = pd.DataFrame(degrees)
    combined_degrees = degrees.groupby(degrees.index).sum()
    values = combined_degrees.values.flatten()
    coloumns = combined_degrees.columns
    if 0 in coloumns:
        index_0 = np.where(coloumns == 0)
        value_0 = values[index_0]
        coloumns = np.delete(coloumns, index_0)
        values = np.delete(values, index_0)
        coloumns = np.insert(coloumns, 0, 0)
        values = np.insert(values, 0, value_0)

    #combined_degrees = combined_degrees.to_numpy().flatten()
    plt.plot(coloumns,values)
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    
    mean = np.sum([i*j for i, j in zip(coloumns, values)])/np.sum(values)
    print(mean)
    plt.title(f'Degree distribution of a Geometric Configuration Graph, mean = {mean}')
    plt.axvline(mean, color='red')
    plt.show()"""

    geometric_config_graph = geometric_configuration_graph(10000, 1/11)
    poisson_config_graph = poisson_configuration_graph(10000, 10)
    """
    friend_degrees_geometric = []
    for i in range(10000):
        neighbours = np.array(list(geometric_config_graph.neighbors(np.random.randint(0, 10000))))
        if len(neighbours) > 0:
            friend_degrees_geometric.append(len(geometric_config_graph.neighbors(np.random.choice(neighbours))))
            i-=1
    friend_degrees_geometric = np.array(friend_degrees_geometric)
    plt.hist(friend_degrees_geometric, bins=20)
    mean = np.mean(friend_degrees_geometric)
    plt.axvline(mean, color='red')
    plt.title(f'Geometric friend degrees, mean = {mean}')
    plt.savefig('Geo_friend', bbox_inches='tight')
    plt.show()
    
    
    friend_degrees_poisson = []
    for i in range(10000):
        neighbours = np.array(list(poisson_config_graph.neighbors(np.random.randint(0, 10000))))
        if len(neighbours) > 0:
            friend_degrees_poisson.append(len(poisson_config_graph.neighbors(np.random.choice(neighbours))))
            i-=1
    friend_degrees_poisson = np.array(friend_degrees_poisson)
    plt.hist(friend_degrees_poisson, bins=20)
    mean = np.mean(friend_degrees_poisson)
    plt.axvline(mean, color='red')
    plt.title(f'Poisson friend degrees, mean = {mean}')
    plt.savefig('Poisson_friend', bbox_inches='tight')
    plt.show()
    """
    
    """delta_friends = geometric_config_graph.friends_degree_distribution() - geometric_config_graph.degree_distributions_individual()
    zero_index = np.where(geometric_config_graph.degree_distributions_individual() == 0)
    delta_friends = np.delete(delta_friends, zero_index)
    delta_friends[np.isnan(delta_friends)] = 0
    print(np.mean(delta_friends))
    mean = np.mean(delta_friends)
    plt.hist(delta_friends, bins=40)
    plt.axvline(mean, color='red')
    plt.title(f'Geometric Delta Configuration Graph, mean = {mean}')
    plt.savefig('Geo_delta', bbox_inches='tight')
    plt.show()

    delta_friends = poisson_config_graph.friends_degree_distribution() - poisson_config_graph.degree_distributions_individual()
    zero_index = np.where(poisson_config_graph.degree_distributions_individual() == 0)
    delta_friends = np.delete(delta_friends, zero_index)
    print(np.mean(delta_friends))
    mean = np.mean(delta_friends)
    plt.hist(delta_friends, bins=40)
    plt.axvline(mean, color='red')
    plt.title(f'Poisson Delta Configuration Graph, mean = {mean}')
    plt.savefig('Poisson_delta', bbox_inches='tight')
    plt.show()"""

    """components = []
    for i in range(11):
        num_components_one = [len(poisson_configuration_graph(10000, (2/10 * i)).find_component(1)) for _ in range(40)]
        components.append(np.mean(num_components_one))
    plt.plot([2/10 * i for i in range(11)], components)
    plt.xlabel('lambda')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as lambda changes')
    plt.savefig('Poisson_components', bbox_inches='tight')
    plt.show()

    components = []
    for i in range(11):
        num_components_one = [len(geometric_configuration_graph(10000, 1 - 2/3 * i/10).find_component(1)) for _ in range(40)]
        components.append(np.mean(num_components_one))
    
    p = np.array([1 - 2/3 * i/10 for i in range(11)])
    mean_degree = (1-p)/p
    plt.plot(mean_degree, components)
    plt.xlabel('Mean Degree')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as mean degree changes')
    plt.savefig('Geometric_components', bbox_inches='tight')
    plt.show()"""

    """components = []
    for ii in range(6):
        temp = []
        for i in range(11):
            num_components_one = [len(popular_geometric_configuration_graph(10000, 1 - 2/3 * i/10, ii*3).find_component(1)) for _ in range(10)]
            temp.append(np.mean(num_components_one))
        components.append(temp)
    
    p = np.array([1 - 2/3 * i/10 for i in range(11)])
    mean_degree = (1-p)/p
    for i in range(6):
        plt.plot(mean_degree, components[i], label=f'Popular Nodes = {i*3}')
    plt.xlabel('Mean Degree')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as mean degree changes')
    plt.legend()
    plt.savefig('Geo_Popular_components', bbox_inches='tight')
    plt.show()"""

    components = []
    for ii in range(6):
        temp = []
        for i in range(11):
            num_components_one = [len(popular_poisson_configuration_graph(10000, 2/10 * i, ii*3).find_component(1)) for _ in range(40)]
            temp.append(np.mean(num_components_one))
        components.append(temp)
    for i in range(6):
        plt.plot([2/10 * ii for ii in range(11)], components[i], label=f'Popular Nodes = {i*3}')
    plt.xlabel('lambda')
    plt.ylabel('Number of components for node 1')
    plt.title('Number of components for node 1 as lambda changes')
    plt.legend()
    plt.savefig('Poisson_Popular_components', bbox_inches='tight')
    plt.show()

def week_three_four():
    poission_graph = poisson_configuration_graph(10000, 10)
    geometric_graph = geometric_configuration_graph(10000, 1/11)
    #poission_graph = geometric_graph
    """num_infected, num_susceptible, num_recovered = SIR_simulation(poission_graph, 0.02, 5, 7, 400)
    plt.plot(num_infected, label='Infected')
    plt.plot(num_susceptible, label='Susceptible')
    plt.plot(num_recovered, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of nodes')
    plt.title('SIR Simulation')
    plt.legend()
    #plt.savefig('SIR', bbox_inches='tight')
    plt.show()"""
    total_recovered = []
    """for i in range(20):
        temp_recovered = [time_SIR_simulation(poission_graph, 0.0221*i/20, 5, 10, 410)[2][-1] for _ in range(80)]
        total_recovered.append(temp_recovered)
    total_recovered = np.array(total_recovered)
    total_recovered = np.mean(total_recovered, axis=1)
    plt.plot([1-(1-0.0221*i/20)**10 for i in range(20)], total_recovered, label='Time 10', color='purple')"""
    
    



    total_recovered = []
    temp_recovered = time_SIR_simulation(poission_graph, 0.07, 5, 3, 403)[2] # 0.014 = Supercritical
    plt.plot([i/3 for i in range(len(temp_recovered))], temp_recovered, label='Time 3', color='green')
    


    total_recovered = []
    temp_recovered = time_SIR_simulation(poission_graph, 0.05, 5, 2, 402)[2]# 0.014 = Supercritical
    plt.plot([i/2  for i in range(len(temp_recovered))], temp_recovered, label='Time', color='red') 
    plt.xlabel('Lambda')
    plt.ylabel('Number of nodes recovered')
    plt.title('Number of nodes recovered as lambda changes')

    #plt.savefig('SIR_Lambda', bbox_inches='tight')
    #plt.show()

    """total_recovered = []
    for i in range(20):
        temp_recovered = [SIR_simulation(poission_graph, 0.01*i, 5, 400)[2][-1] for _ in range(80)]
        total_recovered.append(np.mean(temp_recovered))
    plt.plot([0.01*i for i in range(20)], total_recovered, label = 'No Time', color = 'Blue') #0.1 = Supercritical"""
    plt.xlabel('Lambda')
    plt.ylabel('Number of nodes recovered')
    plt.title('Number of nodes recovered as lambda changes')
    plt.savefig('SIR_time_changing_3', bbox_inches='tight')
    plt.legend()
    plt.show()

    """infection_probability = infection_probability_equation(poission_graph, 0.3, 5, 100)
    plt.hist(infection_probability, bins=20)
    plt.xlabel('Non-Infection Probability')
    plt.ylabel('Nodes')
    plt.title('Non-Infection Probability for each node')
    plt.show()"""

    """cluster_sizes = []
    cluster_sizes_std = []
    coefficient_of_variation = []
    for i in range(50):
        temp = [one_loop_SIR_simulation(poission_graph, 0.01*i).subset_size(1) for _ in range(60)]
        cluster_sizes.append(np.mean(temp))
        cluster_sizes_std.append(np.std(temp))
        coefficient_of_variation.append(np.std(temp)/np.mean(temp))
    plt.plot([0.01*i for i in range(50)], cluster_sizes)
    
    plt.errorbar([0.01*i for i in range(50)], cluster_sizes, yerr=cluster_sizes_std)
    plt.xlabel('Lambda')
    plt.ylabel('Number of clusters')
    plt.title('Number of clusters as lambda changes')
    plt.savefig('Clusters', bbox_inches='tight')
    plt.show()

    plt.plot([0.01*i for i in range(50)], coefficient_of_variation)
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient of Variation')
    plt.title('Coefficient of Variation as lambda changes')
    plt.savefig('Coefficient', bbox_inches='tight')
    plt.show()"""

    """quick_recovery = []
    for i in range(50):
        temp = [quick_SIR(poission_graph, 0.01*i, 1) for _ in range(60)]
        quick_recovery.append(np.mean(temp))
    plt.plot([0.01*i for i in range(50)], quick_recovery)
    plt.xlabel('Lambda')
    plt.ylabel('Number of nodes recovered')
    plt.title('Number of nodes recovered as lambda changes')
    plt.savefig('Quick_SIR', bbox_inches='tight')
    plt.show()"""

    """nodes = 10000
    times_infected = {i: 0 for i in range(nodes)}
    for i in range(100):
        infected, susceptible, recovered = SIR_simulation_2(poission_graph, 0.16, 5, 400)
        for i in recovered:
            times_infected[i] += 1
        for i in infected:
            times_infected[i] += 1
    values = np.array(list(times_infected.values()))
    values = values/100
    plt.hist(values, bins=20)
    plt.xlabel('Rate of infection')
    plt.ylabel('Number of nodes')
    plt.title('Rate of infection for each node')
    plt.show()

    geometric_configuration_graph = geometric_configuration_graph(10000, 1/11)
    times_infected = {i: 0 for i in range(nodes)}
    for i in range(100):
        infected, susceptible, recovered = SIR_simulation_2(geometric_configuration_graph, 0.12, 5, 400)
        for i in recovered:
            times_infected[i] += 1
        for i in infected:
            times_infected[i] += 1
    values = np.array(list(times_infected.values()))
    values = values/100
    plt.hist(values, bins=20)
    plt.xlabel('Rate of infection')
    plt.ylabel('Number of nodes')
    plt.title('Rate of infection for each node')
    plt.show()"""

    """geometric_graph = geometric_configuration_graph(10000, 1/11)
    poisson_graph = poisson_configuration_graph(10000, 10)

    total_recovered_zero = []
    total_recovered_twenty = []
    total_recovered_forty = []
    for i in range(30):
        temp_recovered = [SIR_simulation(poission_graph, 0.01*i, 1, 400, 0)[2][-1] for _ in range(60)]
        temp_recovered_twenty = [SIR_simulation(poission_graph, 0.01*i, 1, 400, 0.2)[2][-1] for _ in range(60)]
        temp_recovered_forty = [SIR_simulation(poission_graph, 0.01*i, 1, 400, 0.4)[2][-1] for _ in range(60)]
        total_recovered_zero.append(np.mean(temp_recovered))
        total_recovered_twenty.append(np.mean(temp_recovered_twenty))
        total_recovered_forty.append(np.mean(temp_recovered_forty))
    plt.plot([0.01*i for i in range(30)], total_recovered_zero, label='zero') #0.1 = Supercritical
    plt.plot([0.01*i for i in range(30)], total_recovered_twenty, label='twenty') 
    plt.plot([0.01*i for i in range(30)], total_recovered_forty, label='forty')
    plt.xlabel('Lambda')
    plt.ylabel('Number of nodes recovered')
    plt.title('Number of nodes recovered as lambda changes')
    plt.legend()
    plt.savefig('SIR_Lambda_notime_vacc', bbox_inches='tight')
    plt.show()"""

if __name__ == '__main__':
    poission_graph = poisson_configuration_graph(10000, 10)
    #print(poission_graph.number_of_triangles_quick())
    triangle_poisson_graph = copy.deepcopy(poission_graph)
    for i in range(50):
        triangle_poisson_graph.add_triangle()
    print("HERE")
    standeredised_poisson_graph = copy.deepcopy(poission_graph)
    for i in range(50):
        standeredised_poisson_graph.add_random_edge()
    print("HERE")
    quick_recovery = []
    quick_recovery_tri = []
    for i in range(50):
        temp = [SIR_simulation(standeredised_poisson_graph, 0.01*i, 3, 400) for _ in range(60)]
        temp_tri = [SIR_simulation(triangle_poisson_graph, 0.01*i, 3, 400) for _ in range(60)]
        quick_recovery.append(np.mean(temp))
        quick_recovery_tri.append(np.mean(temp_tri))
    plt.plot([0.01*i for i in range(50)], quick_recovery)
    plt.plot([0.01*i for i in range(50)], quick_recovery_tri, label='Triangle', color='red')
    plt.xlabel('Lambda')
    plt.ylabel('Number of nodes recovered')
    plt.title('Number of nodes recovered as lambda changes')
    plt.savefig('Quick_SIR_tri', bbox_inches='tight')
    plt.show()

    geometric_graph = geometric_configuration_graph(10000, 1/11)

    
    #poission_graph = geometric_graph
    




## Question 2: It is a Binomial distribution. The average number of edges is n(n-1)p/2. The variance is n(n-1)p(1-p)/2.

## Question 3: k = number of edges, P(k) = (n-1 choose k) * p^k * (1-p)^(n-1-k), E[k] = (n-1)p, Var(k) = (n-1)p(1-p).

## Question 4: G(n, p) = G(n, \lambda / n-1). p = @lambda / (n-1). E[k] = (n-1)p = \lambda, Var(k) = (n-1)p(1-p) = \lambda (n-1-\lambda)/(n-1).
## As n goes to infinity, Var(k) goes to \lambda. The expected value of the number of edges is \lambda and doesn't depend on n.
## As n goes to infinity, P(k) = Poisson(\lambda). As (n-1 choose k)(1/n-1)^k -> 1/k! as n goes to infinity and (1-p)^(n-1-k) -> e^(-\lambda) as 1+1/x -> e as x -> infinity.
## So P(k) -> e^(-\lambda) * \lambda^k / k!.

## Questiion 5: For a probabilty configuration model, the expected value of the degree of a node is the same as the expected value of the degree of a node in a Poisson configuration model.
## The expected value of the degree of a node in a Poisson configuration model is \lambda. The expected value of the degree of a node in a probability configuration model is \sum_{i=1}^{n} i * P(i).
## For friends of a node, the expected value of the degree of a friend of a node in a Poisson configuration model is \lambda. The expected value of the degree of a friend of a node in a probability configuration model is \sum_{i=1}^{n} i * P(i) / n.