from time import sleep

from mininetfed.sim.net import MininetFed
from mininetfed.sim.nodes import FedServerNode, FedClientNode, FedBrokerNode
from mininetfed.sim.util.docker_utils import build_fed_node_docker_image, build_fed_broker_docker_image

# See server/client_selection.py for the available client_selector models
# TODO: acertar as opcoes no mininetfed
server_args = {"min_trainers": 4, "num_rounds": 100, "stop_acc": 0.999,
               'client_selector': 'All', 'aggregator': "FedAvg"}


def topology():

    client_dimage = build_fed_node_docker_image("basic_client", "client_code/client_requirements.txt")["tag"]

    net = MininetFed()

    s1 = net.addSwitch(name="s1", failMode='standalone')

    broker = net.addHost(name="broker", cls=FedBrokerNode)
    net.addLink(s1, broker)

    server = net.addHost(name="server", cls=FedServerNode, server_args=server_args)
    net.addLink(s1, server)

    clients = []
    for i in range(4):
        c = net.addHost(name=f'client{i}', cls=FedClientNode, script="mnist_trainer.py", dimage=client_dimage, client_folder=f"clients/client{i}")
        net.addLink(s1, c)
        clients.append(c)

    print('*** Starting network...\n')
    net.build()
    net.addNAT(name='nat0', linkTo='s1', ip='192.168.210.254').configDefault()
    s1.start([])

    broker.run()
    broker_address = broker.IP(intf="brk-eth0")

    server.run(broker_address=broker_address)

    sleep(3)

    for client in clients:
        client.run(broker_address=broker_address)

if __name__ == '__main__':
    topology()
