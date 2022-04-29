import netsquid as ns
import numpy as np
import os

import pypuf.simulation, pypuf.io
from Simulation_pypuf.challenge_test import arbitrary_challenges

import netsquid.qubits.ketstates as ks
import netsquid.components.instructions as instr
from netsquid.nodes import Node, Network, Connection
from netsquid.protocols import NodeProtocol, Signals
from netsquid.components import QuantumChannel, QuantumProgram, ClassicalChannel, FibreDelayModel, DephaseNoiseModel, T1T2NoiseModel, QSource, SourceStatus, FibreLossModel
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.qubits import StateSampler



'''
Parameter of CPUF:
n: input size
m: output size
N: # of CRPs
k: For XORPUF (k-APUF in parallel)
puf_noisiness: CPUF device noise
'''
n = 32 
m = 24
num_qubit_lock = m // 4
N = 1
k = 5
puf_noisiness = 0

'''
CPUF instance (n-bit challenge, m-bit response)
'''
def CPUF_gen(n, m, N, k, puf_noisiness):
	seed_puf_instances, puf_instances, seed_challenges_instances = [], [], []
	seed_challenges = int.from_bytes(os.urandom(4), "big")
	challenges = pypuf.io.random_inputs(n, N, seed_challenges)
	responses = np.zeros(m)
	for i in range(m):
		seed_puf_instances.append(int.from_bytes(os.urandom(4), "big")) 
		puf_instances.append(pypuf.simulation.XORArbiterPUF(n=n, noisiness=puf_noisiness, seed=seed_puf_instances[i], k=k)) 
		crps_instances = arbitrary_challenges.random_challenges_crps(puf_instances[i], n, N, challenges)
		responses[i] = crps_instances.responses
	
	challenges_pp = (1 - challenges) // 2
	responses_pp = (1 - responses) // 2
	return challenges_pp, responses_pp

challenges_pp, responses_pp = CPUF_gen(n, m, N, k, puf_noisiness)
responses_pp_atob, responses_pp_btoa = responses_pp[:m//2], responses_pp[m//2:]
print('Response:', responses_pp)
print('ResponseAtoB:', responses_pp_atob)
print('ResponseBtoA:', responses_pp_btoa)

"""
Factory to create a quantum processor for each end node.
Has three memory positions and the physical instructions necessary
for teleportation.
"""
def create_processor(dephase_rate, t_times, memory_size, add_qsource=False, q_source_probs=[1., 0.]):
	
	gate_noise_model = None
	memory_noise_model = None
	# gate_noise_model = DephaseNoiseModel(dephase_rate, time_independent=False)
	# memory_noise_model = T1T2NoiseModel(T1=t_times['T1'], T2=t_times['T2'])

	physical_instructions = [
		PhysicalInstruction(instr.INSTR_INIT,
							duration=1,
							parallel=False,
							quantum_noise_model =gate_noise_model),
		PhysicalInstruction(instr.INSTR_H,
							duration=1,
							parallel=False,
							quantum_noise_model =gate_noise_model),
		PhysicalInstruction(instr.INSTR_X,
							duration=1,
							parallel=False,
							quantum_noise_model =gate_noise_model),
		PhysicalInstruction(instr.INSTR_Z,
							duration=1,
							parallel=False,
							quantum_noise_model =gate_noise_model),
		PhysicalInstruction(instr.INSTR_MEASURE,
							duration=10,
							parallel=False,
							quantum_noise_model =gate_noise_model),
		PhysicalInstruction(instr.INSTR_MEASURE_X,
							duration=10,
							parallel=False,
							quantum_noise_model =gate_noise_model)
	]
	processor = QuantumProcessor("quantum_processor",
								 num_positions=memory_size,
								 mem_noise_models= memory_noise_model,
								 phys_instructions=physical_instructions)
	if add_qsource:
		qubit_source = QSource('qubit_source',
							   StateSampler([ks.s0, ks.s1], q_source_probs),
							   num_ports=1,
							   status=SourceStatus.OFF)
		processor.add_subcomponent(qubit_source,
								   name='qubit_source')
	return processor

def SendBehavior():

	return None

def RecvBehavior():

	return None
"""
Program to encode a bit according to a secret key and a basis.
"""
class EncodeQubitProgram(QuantumProgram):

	def __init__(self, base, bit):
		super().__init__()
		self.base = base
		self.bit = bit

	def program(self):
		q1, = self.get_qubit_indices(1)
		# Don't init since qubits come initalised from the qsource
		# self.apply(instr.INSTR_INIT, q1)
		if self.bit == 1:
			self.apply(instr.INSTR_X, q1)
		if self.base == 1:
			self.apply(instr.INSTR_H, q1)
		yield self.run()

class RandomMeasurement(QuantumProgram):
	def __init__(self, base):
		super().__init__()
		self.base = base

	def program(self):
		q1, = self.get_qubit_indices(1)
		if self.base == 0:
			self.apply(instr.INSTR_MEASURE, q1, output_key="M")
		elif self.base == 1:
			self.apply(instr.INSTR_MEASURE_X, q1, output_key="M")
		yield self.run()


class ServerProtocol(NodeProtocol):
	def __init__(self, node):
		super().__init__(node)
		self.node = node
		
	def run(self):
		print('Challenges(Server):', challenges_pp)
		self.node.ports['classicIO'].tx_output(challenges_pp)

		self.node.qmemory.subcomponents['qubit_source'].status = SourceStatus.INTERNAL
		# Transmit encoded qubits to Client
		for i in range(num_qubit_lock):
			# Await a qubit
			yield self.await_port_output(self.node.qmemory.subcomponents['qubit_source'].ports['qout0'])
			qubits = self.node.qmemory.subcomponents['qubit_source'].ports['qout0'].rx_output().items
			self.node.qmemory.put(qubits, positions=[0], replace=True)
			self.node.qmemory.execute_program(EncodeQubitProgram(responses_pp_atob[2*i+1], responses_pp_atob[2*i]))
			yield self.await_program(self.node.qmemory)
			self.node.qmemory.pop(0)
			self.node.ports['qubitIO'].tx_output(self.node.qmemory.ports['qout'].rx_output())
			
		self.node.qmemory.subcomponents['qubit_source'].status = SourceStatus.OFF

		print("Server: Qubits are sent!")

		results = []
		i = 0

		def record_measurement(measure_program):
			measurement_result = measure_program.output['M'][0]
			results.append(measurement_result)

		def measure_qubit(message):
			self.node.qmemory.put(message.items[0], positions=[i])
			measure_program = RandomMeasurement(responses_pp_btoa[2*i+1])
			self.node.qmemory.set_program_done_callback(record_measurement, measure_program=measure_program, once=False)
			self.node.qmemory.execute_program(measure_program, qubit_mapping=[i])

		# Not sure why this timer has to have a huge number...
		delay_timer = 100
		for i in range(num_qubit_lock):
			# Await a qubit from Client
			self.node.ports['qubitIO'].forward_input(self.node.qmemory.ports[f"qin{i}"])
			self.node.qmemory.ports[f"qin{i}"].bind_input_handler(measure_qubit)
			yield self.await_port_input(self.node.ports['qubitIO']) 

		yield self.await_program(self.node.qmemory) 
		print("Server: Qubits are received!")
		print(results)
		
		if (results == responses_pp_btoa[0::2]).all():
			print("Server: Client Authenticated!")
		

class ClientProtocol(NodeProtocol):
	def __init__(self, node):
		super().__init__(node)
		self.node = node

	 
	def run(self):
		yield self.await_port_input(self.node.ports['classicIO'])
		challenges_rec = self.node.ports['classicIO'].rx_input()	
		print('Challenges(Client):', np.array(challenges_rec.items[0]))

		results = []
		i = 0

		def record_measurement(measure_program):
			measurement_result = measure_program.output['M'][0]
			results.append(measurement_result)

		def measure_qubit(message):
			self.node.qmemory.put(message.items[0], positions=[i])
			measure_program = RandomMeasurement(responses_pp_atob[2*i+1])
			self.node.qmemory.set_program_done_callback(record_measurement, measure_program=measure_program, once=False)
			self.node.qmemory.execute_program(measure_program, qubit_mapping=[i])

		# Not sure why this timer has to have a huge number...
		delay_timer = 100
		for i in range(num_qubit_lock):
			# Await a qubit from Server
			self.node.ports['qubitIO'].forward_input(self.node.qmemory.ports[f"qin{i}"])
			self.node.qmemory.ports[f"qin{i}"].bind_input_handler(measure_qubit)
			yield self.await_port_input(self.node.ports['qubitIO']) 

		yield self.await_program(self.node.qmemory) 
		print("Client: Qubits are received!")
		print(results)

		if (results == responses_pp_atob[0::2]).all():
			print("Client: Server Authenticated!")


		self.node.qmemory.subcomponents['qubit_source'].status = SourceStatus.INTERNAL
		# Transmit encoded qubits to Client
		for i in range(num_qubit_lock):
			# Await a qubit
			yield self.await_port_output(self.node.qmemory.subcomponents['qubit_source'].ports['qout0'])
			qubits = self.node.qmemory.subcomponents['qubit_source'].ports['qout0'].rx_output().items
			self.node.qmemory.put(qubits, positions=[0], replace=True)
			self.node.qmemory.execute_program(EncodeQubitProgram(responses_pp_btoa[2*i+1], responses_pp_btoa[2*i]))
			yield self.await_program(self.node.qmemory)
			self.node.qmemory.pop(0)
			self.node.ports['qubitIO'].tx_output(self.node.qmemory.ports['qout'].rx_output())
			
		self.node.qmemory.subcomponents['qubit_source'].status = SourceStatus.OFF

		print("Client: Qubits are sent!")

class QubitConnection(Connection):
	def __init__(self, length, dephase_rate, loss=(0, 0), name='QubitConn'):
		super().__init__(name=name)
		error_models = {'quantum_noise_model': DephaseNoiseModel(dephase_rate=dephase_rate, time_independent=False),
						'delay_model': FibreDelayModel(length=length),
						'quantum_loss_model': FibreLossModel(p_loss_init=loss[0], p_loss_length=loss[1])
						}

		qchannel_1 = QuantumChannel(name="qchannelAtoB",
								   length=length,
								   models=error_models)
		qchannel_2 = QuantumChannel(name="qchannelBtoA",
								   length=length,
								   models=error_models)

		self.add_subcomponent(qchannel_1,
							  forward_output=[('B', 'recv')],
							  forward_input=[('A', 'send')])

		self.add_subcomponent(qchannel_2,
							  forward_output=[('A', 'recv')],
							  forward_input=[('B', 'send')])



def generate_network(node_distance=1e3, dephase_rate=0.2, key_size=num_qubit_lock, t_time=None,
					 q_source_probs=[1., 0.], loss=(0, 0)):
	"""
	Generate the network. For BB84, we need a quantum and classical channel.
	"""
	# if t_time is None:
	#     t_time = {'T1': 11, 'T2': 10}

	network = Network("HybridPUF Network")
	alice = Node("alice", qmemory=create_processor(dephase_rate, t_time, key_size, add_qsource=True, q_source_probs=q_source_probs))
	bob = Node("bob", qmemory=create_processor(dephase_rate, t_time, key_size, add_qsource=True, q_source_probs=q_source_probs))
	network.add_nodes([alice, bob])
	q_conn = QubitConnection(length=node_distance, dephase_rate=dephase_rate, loss=loss)
	network.add_connection(alice,
						   bob,
						   label='q_chan',
						   connection=q_conn,
						   port_name_node1='qubitIO',
						   port_name_node2='qubitIO')
	network.add_connection(alice,
						   bob,
						   label="c_chan",
						   channel_to=ClassicalChannel('AcB', delay=10),
						   channel_from=ClassicalChannel('BcA', delay=10),
						   port_name_node1="classicIO",
						   port_name_node2="classicIO")
	return network




if __name__ == '__main__':

	n = generate_network()
	node_a = n.get_node("alice")
	node_b = n.get_node("bob")



	p1 = ServerProtocol(node_a)
	p2 = ClientProtocol(node_b)

	p1.start()
	p2.start()
	run_stats = ns.sim_run(duration=200000000)
