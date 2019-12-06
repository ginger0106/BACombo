"""
Gas Station Refueling example

Covers:

- Resources: Resource
- Resources: Container
- Waiting for other processes

Scenario:
  A gas station has a limited number of gas pumps that share a common
  fuel reservoir. Cars randomly arrive at the gas station, request one
  of the fuel pumps and start refueling from that reservoir.

  A gas station control process observes the gas station's fuel level
  and calls a tank truck for refueling if the station's level drops
  below a threshold.

"""

TRAINING_TIME = 30
CAPACITY = 150
SEG_SIZE = 50

import itertools
import random
import numpy as np
import simpy


RANDOM_SEED = 42
GAS_STATION_SIZE = 200     # liters
THRESHOLD = 10             # Threshold for calling the tank truck (in %)
FUEL_TANK_SIZE = 50        # liters
FUEL_TANK_LEVEL = [5, 25]  # Min/max levels of fuel tanks (in liters)
REFUELING_SPEED = 2        # liters / second
TANK_TRUCK_TIME = 300      # Seconds it takes the tank truck to arrive
T_INTER = [30, 300]        # Create a car every [min, max] seconds
SIM_TIME = 1000            # Simulation time in seconds

class client_simulate:
    def __init__(self, env,idx, num_clients):
        self.exit_bw = simpy.Container(env, init=CAPACITY, capacity=CAPACITY)
        self.transfer_time = [env.now]
        self.training_time = [env.now]
        self.idx = idx
        self.seg_transfer_time = [[-1]*num_clients]
        self.send_que = simpy.Container(env, init=0, capacity=1000)
        self.sigal = env.event()

    def train_process(self, env, my_round):
        yield env.timeout(TRAINING_TIME)
        self.sigal.succeed()
        self.training_time.append(env.now)
        idx_list = self.get_idx_list()
        yield env.timeout(self.get_transfer_time(idx_list))
        self.transfer_time.append(env.now)

    def change_capacity(self, env):
        pass

    def get_transfer_time(self, idx_list):
        seg_transfer_time = np.max([self.seg_transfer_time(i) for i in idx_list])
        return seg_transfer_time

    def get_idx_list(self):
        return [1, 2, 3]

    def pull_seg(self, env, client_simulate,idx, bandwidth, my_round):
        # called by other client
        yield self.sigal
        yield client_simulate.send_que.put(1)
        exit_bw = client_simulate.exit_bw.level
        bottleneck_num = client_simulate.send_que.level
        link_bw = bandwidth[idx][self.idx]
        final_bw = np.min(exit_bw/bottleneck_num, link_bw)
        transfer_time_from_idx = SEG_SIZE / final_bw
        yield env.timeout(SEG_SIZE*bottleneck_num/exit_bw)
        yield client_simulate.send_que.get(1)
        last_time = transfer_time_from_idx - SEG_SIZE*bottleneck_num/exit_bw
        yield env.timeout(last_time)
        return env.now





def car(name, env, gas_station, fuel_pump):
    """A car arrives at the gas station for refueling.

    It requests one of the gas station's fuel pumps and tries to get the
    desired amount of gas from it. If the stations reservoir is
    depleted, the car has to wait for the tank truck to arrive.

    """
    fuel_tank_level = random.randint(*FUEL_TANK_LEVEL)
    print('%s arriving at gas station at %.1f' % (name, env.now))
    with gas_station.request() as req:
        start = env.now
        # Request one of the gas pumps
        yield req

        # Get the required amount of fuel
        liters_required = FUEL_TANK_SIZE - fuel_tank_level
        yield fuel_pump.get(liters_required)

        # The "actual" refueling process takes some time
        yield env.timeout(liters_required / REFUELING_SPEED)

        print('%s finished refueling in %.1f seconds.' % (name,
                                                          env.now - start))


def gas_station_control(env, fuel_pump):
    """Periodically check the level of the *fuel_pump* and call the tank
    truck if the level falls below a threshold."""
    while True:
        if fuel_pump.level / fuel_pump.capacity * 100 < THRESHOLD:
            # We need to call the tank truck now!
            print('Calling tank truck at %d' % env.now)
            # Wait for the tank truck to arrive and refuel the station
            yield env.process(tank_truck(env, fuel_pump))

        yield env.timeout(10)  # Check every 10 seconds


def tank_truck(env, fuel_pump):
    """Arrives at the gas station after a certain delay and refuels it."""
    yield env.timeout(TANK_TRUCK_TIME)
    print('Tank truck arriving at time %d' % env.now)
    ammount = fuel_pump.capacity - fuel_pump.level
    print('Tank truck refuelling %.1f liters.' % ammount)
    yield fuel_pump.put(ammount)


def car_generator(env, gas_station, fuel_pump):
    """Generate new cars that arrive at the gas station."""
    for i in itertools.count():
        yield env.timeout(random.randint(*T_INTER))
        env.process(car('Car %d' % i, env, gas_station, fuel_pump))


# Setup and start the simulation
print('Gas Station refuelling')
random.seed(RANDOM_SEED)

# Create environment and start processes
env = simpy.Environment()
gas_station = simpy.Resource(env, 2)
fuel_pump = simpy.Container(env, GAS_STATION_SIZE, init=GAS_STATION_SIZE)
env.process(gas_station_control(env, fuel_pump))
env.process(car_generator(env, gas_station, fuel_pump))

# Execute!
env.run(until=SIM_TIME)