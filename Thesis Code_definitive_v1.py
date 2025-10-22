import random

import simpy

from simpy.events import AnyOf, AllOf, Event

import csv

#import pulp

import math

#import gurobipy

import time

import numpy as np

from itertools import permutations

from collections import defaultdict  

import pandas as pd

# ORR PARAMETERS

RELEASE_RULE="HUMAN_CENTRIC"                           # can be IM, PR, WL, WL_MOD, WB, WB_MOD,WL_DIRECT,HUMAN_CENTRIC

STARVATION_AVOIDANCE = False                 # can be True or False 

# WORKERS PARAMETERS                        # flexible/output control works only with AL_MOD

WORKER_MODE = "static"

# can be 'static', 'reactive' or 'flexible' (=> output control)

WORKER_FLEXIBILITY  = "flat"          # can be 'triangular', 'chain', 'flat', 'chain upstream'

WORKER_EFFICIENCY_DECREMENT  = 0          # can be any float value between 0 and 1

TRANSFER_TIME = 0

PERMANENCE_TIME = 0

# SHOP PARAMETERS

SHOP_FLOW = "directed"                    # directed or undirected

SHOP_LENGTH = "variable"                    # a number or "variable"

# JOBS PARAMETERS

JOBS_MEAN = 30

JOBS_VARIANCE = 900

JOBS_MU = np.log((JOBS_MEAN * JOBS_MEAN)/math.sqrt((JOBS_MEAN * JOBS_MEAN) + JOBS_VARIANCE))

JOBS_SIGMA = math.sqrt(np.log(((JOBS_MEAN * JOBS_MEAN) + JOBS_VARIANCE)/(JOBS_MEAN * JOBS_MEAN)))

# JOBS GENERATOR PARAMETERS

INPUT_RATE = 0                              # keep equal to 0 

TARGET_UTILIZATION = 0.9375
#TARGET_UTILIZATION = 0.89
UNIFIED_RESULTS = []

DISPATCH_RULE = 'EDD'  # Can be 'FIFO' or 'EDD'

# JOBS RELEASE PARAMETERS

ABSENTEEISM_LEVEL = 0.0  # Will be set by parameter sweep

###########################################

PAR1=[

    # algorith, workforce, starv av. 

    #["IM","static", False],
    
    ["HUMAN_CENTRIC","static", False],

    ["WL_DIRECT","static", False],

    ]

# Integration verification - parameter structure
DISRUPTION_LEVELS = {
    "absenteeism": ["none", 
                    #"low", 
                    #"medium", 
                    #"high"
                    ],
    "downtime": ["none", 
                 #"low", 
                 #"medium", 
                 #"high"
                 ]
}

HUMAN_VARIABILITY_LEVELS = [1, 
                            1.025, 
                            1.05, 
                            1.075
                            ]

# Modify your main parameter loop
PAR2 = []
for config in PAR1:
    for shop_flow in ["directed"]:#,"undirected"]:
        for shop_length in ["variable"]:
            for absenteeism_level in DISRUPTION_LEVELS["absenteeism"]:
                for downtime_level in DISRUPTION_LEVELS["downtime"]:
                    for human_variability in HUMAN_VARIABILITY_LEVELS:
                        temp = config + [shop_flow, shop_length, absenteeism_level, downtime_level, human_variability]
                        PAR2.append(temp)

for config in PAR2:
    RELEASE_RULE = config[0]
    WORKER_MODE = config[1]
    STARVATION_AVOIDANCE = config[2] 
    SHOP_FLOW = config[3]
    SHOP_LENGTH = config[4]
    ABSENTEEISM_LEVEL = config[5]    
    DOWNTIME_LEVEL = config[6]
    HUMAN_RATIO = config[7]

    if ABSENTEEISM_LEVEL == "none":
        MACHINE_ABSENTEEISM_TYPE = ["none"]
    else:
        MACHINE_ABSENTEEISM_TYPE = [#"static", 
                                    "daily"
                                    ]

    if   RELEASE_RULE=="PR" or RELEASE_RULE=="IM":

        WORKLOAD_NORMS=[0]
  
    else:

        if SHOP_FLOW=="directed" and SHOP_LENGTH==5:
            WORKLOAD_NORMS = [1900, 2100,2300, 2500,2700, 0]
        elif SHOP_FLOW=="directed" and SHOP_LENGTH=="variable":
            #WORKLOAD_NORMS = [1400,1500,1600, 1800,2000, 0]
            WORKLOAD_NORMS = [1600,1700,1800,1900,2000,2100]
            #WORKLOAD_NORMS = [0]
        elif SHOP_FLOW=="undirected" and SHOP_LENGTH==5:
            WORKLOAD_NORMS = [6900,7100,7300,7500,7700,0]           
        elif SHOP_FLOW=="undirected" and SHOP_LENGTH=="variable":
            WORKLOAD_NORMS = [1450,1500,1600,1900,2000, 0]
            
        # HUMAN-MACHINE WORKSTATION CONFIGURATION
    
    HUMAN_MACHINE_PHASES = {
        0: False,  # Machine 1 - Pure machine station
        1: False,  # Machine 2 - Pure machine station  
        2: True,   # Machine 3 - Human-machine collaborative station
        3: True,   # Machine 4 - Human-machine collaborative station
        4: True    # Machine 5 - Human-machine collaborative station
    }

    # RATIOS FOR HUMAN-MACHINE COLLABORATION
    MACHINE_RATIO = 1  # Adjust as needed
      
    N_WORKERS = 5

    N_PHASES = 5                # Machines/Workers

    # SIMULATOR PARAMETERS

    SIMULATION_LENGTH = 500000

    FIRST_RUN = 0

    LAST_RUN = 1

    WARMUP = 200000

    # average number of stations per configuration 

    # "undirected-variable" -> 4.015384615384615

    # "directed-variable" ->   2.5806451612903225

    SCREEN_DEBUG = True

    TIME_BTW_SCREEN_DEBUGS = 480*500         # expressed in simulation time units

    CSV_OUTPUT_JOBS = True

    CSV_OUTPUT_SYSTEM = True

    RUN_DEBUG = True                       # used to compute the warmup period 

    # <if run debug>

    if RUN_DEBUG:

        JOBS_RELEASED_DEBUG=list()

        JOBS_ENTRY_DEBUG=list()

        JOBS_DELIVERED_DEBUG=list()

        results_DEBUG = list()

    # </>

    TIME_BTW_DEBUGS = 480*1

    MAX_COLS = 130

    start = time.time()

    JOBS_WARMUP = 0

    JOBS_ROUTINGS_AVG_MACHINES=0

    def getShopConfigurations():
        """Initializes shop configurations and calculates the input rate.

    This function sets up the manufacturing shop's parameters based on global settings
    such as shop flow and length. It calculates the average number of machines in job
    routings and determines the input rate required to meet the target utilization.
    The function also prints the calculated input rate and the corresponding number of
    jobs per day, providing immediate feedback on the simulation's parameters.

    The commented-out code block shows a more dynamic way to generate configurations,
    which has been replaced by a simplified approach using predefined averages for
    different shop setups.

    Args:
        None

    Returns:
        None
    """
        """

        #print("generating product flow")

        #temp=SHOP_CONFIGURATION.split("%20")

        global SHOP_FLOW

        global SHOP_LENGTH

        configurations=list()

        if(SHOP_LENGTH == "variable"):  

            for rr in range(1,N_PHASES+1):

                for i in permutations([j for j in range(N_PHASES)],r=rr):

                    configurations.append(i)

        else:

            configurations=list(permutations([i for i in range(N_PHASES)], r=int(SHOP_LENGTH)))

        if(SHOP_FLOW == "directed" ):

            def is_directed(configuration):

                for i in range(0,len(configuration)-1):

                    if(configuration[i] > configuration[i+1]):

                        return False

                return True

            i=0

            while(i<len(configurations)):

                if(not is_directed(configurations[i])):

                    del(configurations[i])

                else:

                    i+=1

        global TARGET_UTILIZATION

        global INPUT_RATE

        global JOBS_MEAN

        global JOBS_ROUTINGS_AVG_MACHINES

        JOBS_ROUTINGS_AVG_MACHINES = (sum(len(configuration) for configuration in configurations)/len(configurations))

        INPUT_RATE = (480*N_PHASES*TARGET_UTILIZATION) / (JOBS_MEAN*(JOBS_ROUTINGS_AVG_MACHINES))/480

        print("INPUT RATE", INPUT_RATE)

        print(INPUT_RATE*480, "jobs per day")

        time.sleep(2)

        return configurations

        """

        global TARGET_UTILIZATION

        global INPUT_RATE

        global JOBS_MEAN

        global JOBS_ROUTINGS_AVG_MACHINES

        if SHOP_LENGTH == 5:

            JOBS_ROUTINGS_AVG_MACHINES=5

        else:

            if (SHOP_FLOW=="directed"):

                JOBS_ROUTINGS_AVG_MACHINES=2.5806451612903225

            else: 

                JOBS_ROUTINGS_AVG_MACHINES=4.015384615384615

        INPUT_RATE = (480*N_PHASES*TARGET_UTILIZATION)/(JOBS_MEAN*(JOBS_ROUTINGS_AVG_MACHINES))/480

        print("INPUT RATE", INPUT_RATE)

        print(INPUT_RATE*480, "jobs per day")

        time.sleep(2)

    getShopConfigurations()

    if (SHOP_FLOW=="directed" and SHOP_LENGTH=="variable"):

        DUE_DATE_MAX = 4000

    elif (SHOP_FLOW=="directed" and SHOP_LENGTH==5):

        DUE_DATE_MAX = 12500

    elif (SHOP_FLOW=="undirected" and SHOP_LENGTH==5):

        DUE_DATE_MAX = 12500

    elif (SHOP_FLOW=="undirected" and SHOP_LENGTH=="variable"):

        DUE_DATE_MAX = 1500

#####CLASES#####

    class Job(object):
        """Represents a single job moving through the manufacturing system.

    This class encapsulates all the properties and behaviors of a job, including its
    unique identifier, arrival and completion dates, and routing information. It also
    tracks the processing times required at each machine and the job's current position
    in its routing sequence. The class provides methods to calculate various performance
    metrics, such as tardiness and flow time, and to determine the job's contribution
    to system-wide workload measures.

    Attributes:
        id (int): A unique identifier for the job.
        ArrivalDate (float): The simulation time when the job arrived in the system.
        ReleaseDate (float): The simulation time when the job was released to the shop floor.
        CompletationDate (float): The simulation time when the job was completed.
        ArrivalDateMachines (list): A list of arrival times at each machine.
        CompletationDateMachines (list): A list of completion times at each machine.
        ArrivalDateQueue (list): A list of arrival times in each machine's queue.
        Routing (list): The sequence of machines the job must visit.
        DueDate (float): The job's due date.
        ProcessingTime (list): The processing time required at each machine.
        RemainingTime (list): The remaining processing time at each machine.
        Position (int): The job's current position in its routing sequence.
        ShopLoad (float): The total processing time for the job.
    """

        def __init__(self,env, id): #
            """Initializes a new Job object with random routing and processing times.

        This constructor sets up a new job with a unique ID and captures its arrival
        time in the simulation. It generates a random routing sequence for the job,
        either directed or undirected, and with a fixed or variable number of machines,
        based on global settings. Processing times for each machine in the routing are
        drawn from a log-normal distribution, and a due date is assigned within a
        predefined range.

        Args:
            env (simpy.Environment): The simulation environment.
            id (int): The unique identifier for the job.
        """
            
            #self.env = env

            global DUE_DATE_MIN

            global DUE_DATE_MAX

            self.id = id    

            self.ArrivalDate = env.now 

            self.ReleaseDate = None            

            self.CompletationDate = None

            self.ArrivalDateMachines = [0] * N_PHASES
            self.CompletationDateMachines = [0] * N_PHASES
            self.ArrivalDateQueue = [0] * N_PHASES

            # Set ProcessingTime (i.e. time to spend at the stations known before the release) and 

            # RemainingTime variables are used to determine the position of jobs along the shop 

            # As soon as they are processed by stations, values are reduced or eventually set to 0 

            #self.ProcessingTime = list(np.random.lognormal(JOBS_MU, JOBS_SIGMA)  for i in range(N_PHASES))

            #print(self.ProcessingTime)

            n_machines=N_PHASES

            if SHOP_LENGTH=="variable":

                n_machines=np.random.randint(low=1, high=N_PHASES)

            self.Routing = list()

            while len(self.Routing)<n_machines:        

                x = np.random.randint(low=0, high=N_PHASES)

                if x not in self.Routing:

                    self.Routing.append(x)

            if SHOP_FLOW=="directed":

                self.Routing.sort()

            #print (self.Routing)

            #time.sleep(5)

            #self.Routing = JOBS_ROUTINGS[np.random.randint(low=0, high=len(JOBS_ROUTINGS))]     

            DUE_DATE_MIN = 83.686*len(self.Routing)
            
            self.DueDate = self.ArrivalDate + np.random.uniform(DUE_DATE_MIN,DUE_DATE_MAX)

            self.ProcessingTime = list(0 for i in range(N_PHASES))   

            for i in self.Routing:

                x = np.random.lognormal(JOBS_MU, JOBS_SIGMA)

                while x>250:

                    x = np.random.lognormal(JOBS_MU, JOBS_SIGMA)

                self.ProcessingTime[i] = x

            self.RemainingTime = list(self.ProcessingTime)

            self.Position = 0

            self.ShopLoad=sum(self.ProcessingTime)

            self.__GTT__= None

            self.__SFT__= None

            self.__tardy__= None

            self.__tardiness__ = None

            self.__lateness__=None

            #print(self.Routing)

            #print(self.ProcessingTime)

            #print(self.get_CAW())

            #die()

        def get_current_machine(self):
            """Gets the current machine for the job based on its routing position.

        This method returns the identifier of the machine where the job is currently
        located or headed. It includes a bounds check to prevent `IndexError` if the
        job's position exceeds its routing length, which can occur in certain edge cases.
        If an out-of-bounds position is detected, it is reset to the last valid position.

        Args:
            None

        Returns:
            int: The identifier of the current machine in the job's routing.
        """
            # Add bounds checking to prevent IndexError
            if self.Position >= len(self.Routing):
                #print(f"ERROR: Job {self.id} Position {self.Position} exceeds Routing length {len(self.Routing)}")
                #print(f"Job Routing: {self.Routing}")
                #print(f"Job arrived at: {self.ArrivalDate}, released at: {self.ReleaseDate}")
                # Reset position to last valid position as emergency fix
                self.Position = len(self.Routing) - 1
                
            return self.Routing[self.Position]

        def get_CAW(self):
            """Calculates the job's contribution to the corrected aggregated workload (CAW).

        CAW is a workload measure that accounts for the job's remaining processing time
        and its position in the routing sequence. The contribution of each remaining
        operation is weighted, with earlier operations in the sequence having a higher
        weight. This method is used in workload control mechanisms to prioritize jobs.

        Args:
            None

        Returns:
            list: A list representing the job's CAW contribution at each machine.
        """

            """ contribution to the corrected aggregated workload """

            load = list(0 for i in range(N_PHASES))

            weight = 1        

            for i in range(self.Position, len(self.Routing)):

                load[self.Routing[i]] = self.RemainingTime[self.Routing[i]]/weight

                weight += 1        

            #print()

            #print("Routing",self.Routing)

            #print("Position",self.Position)

            #print("ProcessingTime",self.ProcessingTime)

            #print(load)

            #time.sleep(5)

            return load

        def get_CSL(self):
            """Calculates the job's contribution to the corrected shop load (CSL).

        CSL is a workload measure that distributes the job's total processing time
        evenly across all machines, normalized by the length of the job's routing.
        This provides a smoothed measure of the job's impact on the overall shop load.

        Args:
            None

        Returns:
            list: A list representing the job's CSL contribution at each machine.
        """

            """ contribution to the corrected shop load """

            load = list(0 for i in range(N_PHASES))

            for i in range(N_PHASES):

                load[i]=self.ProcessingTime[i]/len(self.Routing)

            return load

        def is_collaborative(self):
            """Determines if the job requires processing at a human-machine station.

        This method checks if the job's routing includes any station designated as a
        human-machine collaborative station. This is primarily used in the human-centric
        release rule to differentiate between collaborative and independent jobs.

        Args:
            None

        Returns:
            bool: True if the job is collaborative, False otherwise.
        """
            """
            Determine if job is collaborative (uses at least one human-machine station)
            or independent (uses only pure machine stations)
            Only relevant for human centric rule
            """
            if RELEASE_RULE != "HUMAN_CENTRIC":
                return False
                
            for machine_id in self.Routing:
                if HUMAN_MACHINE_PHASES.get(machine_id, False):
                    return True
            return False
        
        def get_CSL_with_ratios(self):
            """Calculates the corrected shop load with human-machine ratios applied.

        This method extends the CSL calculation to account for the different processing
        capacities of humans and machines at collaborative workstations. It splits the
        workload contribution into two components, one for machines and one for humans,
        based on predefined ratios.

        Args:
            None

        Returns:
            tuple: A tuple containing two lists: (machine_load, human_load).
        """
            """
            Contribution to corrected shop load with machine/human ratios applied
            Returns tuple: (machine_load, human_load)
            """
            machine_load = list(0 for i in range(N_PHASES))
            human_load = list(0 for i in range(N_PHASES))
            
            for i in range(N_PHASES):
                base_load = self.ProcessingTime[i] / len(self.Routing)
                
                if HUMAN_MACHINE_PHASES.get(i, False):
                    # Human-machine station
                    machine_load[i] = base_load * MACHINE_RATIO
                    human_load[i] = base_load * HUMAN_RATIO
                else:
                    # Pure machine station
                    machine_load[i] = base_load
                    human_load[i] = 0
                    
            return machine_load, human_load

        def get_CAW_routing_only(self):
            """Calculates the corrected aggregated workload considering only the job's routing.

        This method is a LUMS-COR compliant version of the CAW calculation. It differs
        from the standard `get_CAW` by only considering the stations present in the
        job's actual routing, rather than all machines in the system. The workload is
        weighted by the job's position in the remaining routing sequence.

        Args:
            None

        Returns:
            list: A list representing the job's CAW contribution, with non-zero values
                  only for machines in the job's routing.
        """
            load = list(0 for i in range(N_PHASES))

            for i, station_id in enumerate(self.Routing[self.Position:]):
                # i starts from 0 for remaining stations
                position_correction = i + 1  # First remaining station gets weight 1, second gets 1/2, etc.
                load[station_id] = self.RemainingTime[station_id] / position_correction
            
            return load
                    
        def get_CSL_routing_only(self):
            """Calculates the corrected shop load, considering only stations in the job's routing.

        This is a LUMS-COR compliant version of the CSL calculation. It provides a more
        accurate measure of the job's workload contribution by only including the machines
        that the job will actually visit.

        Args:
            None

        Returns:
            list: A list of the job's CSL contributions, with non-zero values only for
                  machines in the job's routing.
        """
            """
            LUMS-COR compliant: Contribution to corrected shop load
            Only considers stations in the job's actual routing
            """
            load = list(0 for i in range(N_PHASES))
            
            for station_id in self.Routing:
                load[station_id] = self.ProcessingTime[station_id] / len(self.Routing)
            
            return load
            
        def get_CSL_with_ratios_routing_only(self):
            """Calculates CSL with human-machine ratios, considering only the job's routing.

        This LUMS-COR compliant method combines the logic of `get_CSL_with_ratios` and
        `get_CSL_routing_only`. It provides a split of the workload into human and machine
        components, but only for the stations included in the job's actual routing.

        Args:
            None

        Returns:
            tuple: A tuple containing two lists: (machine_load, human_load), with non-zero
                   values only for machines in the job's routing.
        """
            """
            LUMS-COR compliant: Corrected shop load with machine/human ratios
            Only considers stations in the job's actual routing
            Returns tuple: (machine_load, human_load)
            """
            machine_load = list(0 for i in range(N_PHASES))
            human_load = list(0 for i in range(N_PHASES))
            
            for station_id in self.Routing:
                base_load = self.ProcessingTime[station_id] / len(self.Routing)
                
                if HUMAN_MACHINE_PHASES.get(station_id, False):
                    # Human-machine station
                    machine_load[station_id] = base_load * MACHINE_RATIO
                    human_load[station_id] = base_load * HUMAN_RATIO
                else:
                    # Pure machine station
                    machine_load[station_id] = base_load
                    human_load[station_id] = 0
                    
            return machine_load, human_load

        def get_CAW_with_ratios_routing_only(self):
            """Calculates CAW with human-machine ratios, considering only the job's routing.

        This is a LUMS-COR compliant method that extends the `get_CAW_routing_only`
        calculation to include human-machine workload splitting. The workload contribution
        is weighted by the job's position and split between human and machine components
        at collaborative stations.

        Args:
            None

        Returns:
            tuple: A tuple containing two lists: (machine_load, human_load), with non-zero
                   values only for machines in the job's remaining routing.
        """
            """
            LUMS-COR compliant: Corrected aggregated workload with machine/human ratios
            Only considers stations in the job's actual routing with position correction
            Returns tuple: (machine_load, human_load)
            """
            machine_load = list(0 for i in range(N_PHASES))
            human_load = list(0 for i in range(N_PHASES))
            
            for pos_in_routing in range(self.Position, len(self.Routing)):
                station_id = self.Routing[pos_in_routing]
                # Position correction: divide by (position in remaining routing + 1)
                position_correction = pos_in_routing - self.Position + 1
                base_load = self.RemainingTime[station_id] / position_correction
                
                if HUMAN_MACHINE_PHASES.get(station_id, False):
                    # Human-machine station - apply ratios
                    machine_load[station_id] = base_load * MACHINE_RATIO
                    human_load[station_id] = base_load * HUMAN_RATIO
                else:
                    # Pure machine station
                    machine_load[station_id] = base_load
                    human_load[station_id] = 0
                    
            return machine_load, human_load

        def get_dual_CSL_routing_only(self):
            """Gets the corrected shop load split into human and machine components.

        This method calculates the CSL for both human and machine workloads, but only
        for the stations in the job's routing. This is used in release mechanisms that
        need to consider both human and machine capacities separately.

        Args:
            None

        Returns:
            tuple: A tuple of two lists: (machine_load, human_load), representing the
                   workload contributions at each machine.
        """
            """
            Get corrected shop load split into human and machine components
            Returns tuple: (machine_load, human_load)
            """
            machine_load = list(0 for i in range(N_PHASES))
            human_load = list(0 for i in range(N_PHASES))
            
            for station_id in self.Routing:
                base_load = self.ProcessingTime[station_id] / len(self.Routing)
                
                if HUMAN_MACHINE_PHASES.get(station_id, False):
                    # Human-machine station - split the load
                    machine_load[station_id] = base_load * MACHINE_RATIO
                    human_load[station_id] = base_load * HUMAN_RATIO
                else:
                    # Pure machine station - all load goes to machines
                    machine_load[station_id] = base_load
                    human_load[station_id] = 0
                    
            return machine_load, human_load

        # Consultive statistics

        def get_GTT(self):
            """Calculates the gross-total-time (GTT) of the job.

        GTT, also known as total delivery time, is the total time a job spends in the
        system, from its arrival to its completion. This is a key performance indicator
        for overall system throughput and efficiency.

        Args:
            None

        Returns:
            float: The total time the job spent in the system.
        """

            """ total_delivery_time """

            if self.__GTT__ == None:

                self.__GTT__ = self.CompletationDate - self.ArrivalDate

            return self.__GTT__

        def get_SFT(self):
            """Calculates the shop-floor-time (SFT) of the job.

        SFT, or total manufacturing lead time, is the time a job spends on the shop
        floor, from its release to its completion. This metric excludes the time spent
        in the pre-shop pool, providing a more focused measure of manufacturing efficiency.

        Args:
            None

        Returns:
            float: The time the job spent on the shop floor.
        """

            """total_manufacturing_lead_time"""

            if self.__SFT__ == None:

                self.__SFT__= self.CompletationDate - self.ReleaseDate  

            return self.__SFT__

        def get_SFT_Machines(self):
            """Calculates the shop-floor-time (SFT) for each machine.

        This method computes the time the job spent at each individual machine, from
        its arrival at the machine to its completion. This allows for a more granular
        analysis of where time is being spent in the manufacturing process.

        Args:
            None

        Returns:
            list: A list of SFT values for each machine.
        """

            '''array with the SFT on each machine'''

            SFT_Machines = list(0 for i in range(N_PHASES))

            for i in range(N_PHASES):

                SFT_Machines[i] = self.CompletationDateMachines[i] - self.ArrivalDateMachines[i]

            return SFT_Machines

        def get_LT_Machines(self):
            """Calculates the lead time for each machine.

        Lead time at a machine is the total time a job spends from its arrival in the
        machine's queue to its completion. This includes both waiting time and processing
        time, providing insight into queue lengths and machine utilization.

        Args:
            None

        Returns:
            list: A list of lead times for each machine.
        """

            '''array with the SFT on each machine'''

            LT_Machines = list(0 for i in range(N_PHASES))

            for i in range(N_PHASES):

                LT_Machines[i] = self.CompletationDateMachines[i] - self.ArrivalDateQueue[i]

            return LT_Machines

        def get_tardy(self):
            """Determines if the job was completed after its due date.

        This method checks if the job is tardy by comparing its completion date to its
        due date. It is a binary measure, returning 1 for tardy jobs and 0 for on-time jobs.

        Args:
            None

        Returns:
            int: 1 if the job is tardy, 0 otherwise.
        """

            if self.__tardy__ == None:

                if self.DueDate < self.CompletationDate: 

                    self.__tardy__=  1

                else:

                    self.__tardy__=  0  

            return self.__tardy__

        def get_tardiness(self):
            """Calculates the tardiness of the job.

        Tardiness is the amount of time by which a job's completion date exceeds its
        due date. If the job is completed on or before its due date, the tardiness is
        zero. This is a critical measure of on-time delivery performance.

        Args:
            None

        Returns:
            float: The amount of time the job is late, or 0 if on-time.
        """

            if self.__tardiness__ == None:

                self.__tardiness__= max(0, self.CompletationDate - self.DueDate)

            return self.__tardiness__

        def get_lateness(self):
            """Calculates the lateness of the job.

        Lateness is the difference between the job's completion date and its due date.
        Unlike tardiness, lateness can be negative if the job is completed early. This
        metric provides a measure of how close the completion was to the due date.

        Args:
            None

        Returns:
            float: The difference between the completion date and the due date.
        """

            if self.__lateness__ == None:

                self.__lateness__= self.CompletationDate-self.DueDate 

            return self.__lateness__

    class Jobs_generator(object):
        """Generates jobs and adds them to the pre-shop pool.

    This class is responsible for creating new jobs and introducing them into the
    simulation. It uses a continuous generation process, where the time between
    job arrivals follows an exponential distribution. This approach models a random
    arrival pattern, which is common in many real-world manufacturing systems. The
    generated jobs are placed in a downstream pool, where they await release to the
    shop floor.

    Attributes:
        env (simpy.Environment): The simulation environment.
        PoolDownstream (Pool): The pre-shop pool where newly generated jobs are stored.
        input_rate (float): The average rate at which jobs are generated.
        generated_orders (int): A counter for the number of jobs generated.
    """

        def __init__(self, env, PoolDownstream):
            """Initializes the Jobs_generator.

        This constructor sets up the job generator with references to the simulation
        environment and the downstream pool. It also initializes the input rate and
        the order counter. The main job generation process, `_continuous_generator`,
        is started as a SimPy process.

        Args:
            env (simpy.Environment): The simulation environment.
            PoolDownstream (Pool): The pool to which new jobs will be added.
        """

            self.env = env

            self.PoolDownstream = PoolDownstream

            self.input_rate = INPUT_RATE

            self.generated_orders = 0             

            #self.generated_processing_time = list(0 for i in range(N_PHASES))

            # external references

            #self.PoolDownstream = PoolDownstream

            #self.env.process(self._periodic_generator(480))

            self.env.process(self._continuous_generator())

        """

        def _periodic_generator(self, period):

            while True:

                for i in range(0, np.random.poisson(self.input_rate * period)):

                    self.PoolDownstream.append(Job(self.env, self.generated_orders))

                # <if run debug>    

                global JOBS_ENTRY_DEBUG

                JOBS_ENTRY_DEBUG.append(job)

                # </>

                self.generated_orders += 1

                yield env.timeout(period)

        """

        def _continuous_generator(self):
            """Continuously generates jobs with random inter-arrival times.

        This method runs as a SimPy process and forms the core of the job generator.
        In an infinite loop, it creates a new `Job` object, adds it to the downstream
        pool, and then waits for a randomly generated time before creating the next
        job. The inter-arrival times follow an exponential distribution, determined
        by the `input_rate`.

        Args:
            None

        Yields:
            simpy.events.Timeout: A timeout event that pauses the process for a
                                  random duration, simulating the time until the
                                  next job arrival.
        """

            while True:

                job = Job(self.env, self.generated_orders + 1)

                self.PoolDownstream.append(job)

                #for i in range(N_PHASES):

                #    self.generated_processing_time[i] += job.ProcessingTime[i]

                # <if run debug>

                if RUN_DEBUG:

                    global JOBS_ENTRY_DEBUG                

                    JOBS_ENTRY_DEBUG.append(job)

                # </>

                self.generated_orders += 1

                yield env.timeout(np.random.exponential(1/self.input_rate))        

 ###JOBS###
    
    def get_corrected_shop_load_with_ratios(pools):
        """Calculates the corrected shop load, split by human and machine contributions.

    This function aggregates the corrected shop load (CSL) from all jobs in the provided
    pools. It is LUMS-COR compliant, meaning it only considers the stations in each
    job's actual routing. The workload is split into human and machine components based
    on predefined ratios for collaborative workstations.

    Args:
        pools (list): A list of `Pool` objects containing the jobs to be included in
                      the calculation.

    Returns:
        tuple: A tuple of two lists (machine_shop_load, human_shop_load), representing
               the total machine and human workload at each station.
    """
        """
        Get shop load with machine ratios applied for human-machine stations
        Uses routing-only job contributions (LUMS-COR compliant)
        Returns tuple: (machine_load, human_load)
        """
        machine_shop_load = list(0 for i in range(N_PHASES))
        human_shop_load = list(0 for i in range(N_PHASES))
        
        for pool in pools:
            for job in pool:
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                for i in range(N_PHASES):
                    machine_shop_load[i] += job_machine_load[i]
                    human_shop_load[i] += job_human_load[i]
                    
        return machine_shop_load, human_shop_load
    
    def get_corrected_aggregated_workload_with_ratios(pools):
        """Calculates the corrected aggregated workload, split by human and machine contributions.

    This function computes the total corrected aggregated workload (CAW) for all jobs
    in the specified pools. It is LUMS-COR compliant and includes a position correction,
    giving more weight to imminent operations. The workload is also split into human and
    machine components, making it suitable for human-centric control systems.

    Args:
        pools (list): A list of `Pool` objects containing the jobs.

    Returns:
        tuple: A tuple of two lists (machine_aggregated_load, human_aggregated_load),
               representing the total machine and human CAW at each station.
    """
        """
        Get corrected aggregated workload with machine ratios applied for human-machine stations
        Uses routing-only job contributions with position correction (LUMS-COR compliant)
        Returns tuple: (machine_aggregated_load, human_aggregated_load)
        """
        machine_aggregated_load = list(0 for i in range(N_PHASES))
        human_aggregated_load = list(0 for i in range(N_PHASES))
        
        for pool in pools:
            for job in pool:
                job_machine_load, job_human_load = job.get_CAW_with_ratios_routing_only()
                
                for i in range(N_PHASES):
                    machine_aggregated_load[i] += job_machine_load[i]
                    human_aggregated_load[i] += job_human_load[i]
                    
        return machine_aggregated_load, human_aggregated_load

    def get_corrected_aggregated_workload_routing_only(pools):
        """Calculates the corrected aggregated workload, considering only job routings.

    This LUMS-COR compliant function aggregates the corrected aggregated workload (CAW)
    from all jobs in the given pools. It only accounts for the machines present in each
    job's specific routing, providing a more precise measure of the workload that each
    station can expect to receive.

    Args:
        pools (list): A list of `Pool` objects.

    Returns:
        list: A list representing the total CAW at each station.
    """
        """
        LUMS-COR compliant: Get corrected aggregated workload considering only job routing
        """
        aggregated_WL = list(0 for i in range(N_PHASES))
        
        for pool in pools:
            for job in pool:
                job_contribution = job.get_CAW_routing_only()
                
                for i in range(N_PHASES):
                    aggregated_WL[i] += job_contribution[i]

        return aggregated_WL

    def get_corrected_shop_load_routing_only(pools):
        """Calculates the corrected shop load, considering only job routings.

    This function provides a LUMS-COR compliant calculation of the corrected shop load
    (CSL). It sums the CSL contributions of all jobs in the provided pools, but only
    for the machines that are actually part of each job's routing. This results in a
    more accurate representation of the workload distribution.

    Args:
        pools (list): A list of `Pool` objects.

    Returns:
        list: A list representing the total CSL at each station.
    """
        """
        LUMS-COR compliant: Get corrected shop load considering only job routing
        """
        shop_load = list(0 for i in range(N_PHASES))
        
        for pool in pools:
            for job in pool:
                job_contribution = job.get_CSL_routing_only()
                
                for i in range(N_PHASES):
                    shop_load[i] += job_contribution[i]

        return shop_load

 ###JOBS###
   
    class Orders_release(object):
        """Manages the release of jobs from the pre-shop pool to the shop floor.

    This class implements various order release mechanisms, such as immediate release,
    human-centric release, and direct workload control. It acts as a gatekeeper,
    deciding when and which jobs should be moved from the upstream pool to the first
    machine in their routing. The choice of release rule is determined by global
    simulation parameters.

    Attributes:
        env (simpy.Environment): The simulation environment.
        PoolUpstream (Pool): The pre-shop pool of jobs waiting for release.
        Pools (list): A list of downstream pools, one for each machine.
        system (System): A reference to the main system object.
        released_workload (list): A list tracking the workload released to each machine.
        downtime_manager (MachineDowntimeManager): A manager for machine downtimes.
        absenteeism_manager (DailyPlanningAbsenteeismManager): A manager for worker absenteeism.
    """

        def __init__(self, env, rule, system, workload_norm =- 1):
            """Initializes the Orders_release mechanism.

        This constructor sets up the order release process based on the specified rule.
        It establishes references to the necessary system components, such as the job
        pools and disruption managers. Depending on the chosen release rule, it starts
        the corresponding SimPy process to handle job releases throughout the simulation.

        Args:
            env (simpy.Environment): The simulation environment.
            rule (str): The name of the release rule to be used (e.g., "IM", "HUMAN_CENTRIC").
            system (System): The main system object.
            workload_norm (float, optional): The workload norm used for workload control
                                             rules. Defaults to -1.
        """
            self.env = env
            self.PoolUpstream = system.PSP
            self.Pools = system.Pools
            self.system = system
            self.released_workload = list(0 for i in range(N_PHASES))
            self.downtime_manager = system.downtime_manager
            self.system.release_trigger = self.env.event()
            self.absenteeism_manager = system.absenteeism_manager

            if rule == "IM":      # immediate release
                self.env.process(self.Immediate_release())

            elif rule == "HUMAN_CENTRIC":
                if SHOP_FLOW == "undirected":
                    self.env.process(self.Human_Centric_release_with_IM(480, workload_norm))
                elif SHOP_FLOW == "directed":
                    self.env.process(self.Human_Centric_release_directed_with_IM(480, workload_norm))
            elif rule == "WL_DIRECT":
                if SHOP_FLOW == "undirected":
                    self.env.process(self.WL_Direct_release_with_IM(480, workload_norm))
                elif SHOP_FLOW == "directed":
                    self.env.process(self.WL_Direct_release_directed_with_IM(480, workload_norm))

            else:

                print("Release algorithm not recognised in Orders_release")

                exit()

        def _get_adjusted_workload_norm(self, base_workload_norm):
            """Adjusts the workload norm to account for machine reliability.

        This method modifies the base workload norm to reflect the current reliability
        of the machines. If a downtime manager is active, it queries the manager for an
        adjusted norm, which may be lower to account for reduced capacity. This allows
        the release mechanism to be responsive to machine disruptions.

        Args:
            base_workload_norm (float): The baseline workload norm.

        Returns:
            float: The adjusted workload norm.
        """
            """Get workload norm adjusted for machine reliability"""
            if self.downtime_manager and base_workload_norm > 0:
                return self.downtime_manager.get_adjusted_workload_norm(base_workload_norm)
            return base_workload_norm

        def Immediate_release(self):
            """Releases jobs immediately upon arrival in the pre-shop pool.

        This method implements the immediate release (IM) rule, which is the simplest
        release mechanism. It continuously monitors the upstream pool and, as soon as
        a job is available, releases it to the first machine in its routing. This
        approach does not consider system status or workload levels.

        Args:
            None

        Yields:
            simpy.events.Event: An event that is triggered when new jobs are available.
        """

            while True:

                while len(self.PoolUpstream) > 0:

                    job = self.PoolUpstream.get()

                    job.ReleaseDate = self.env.now

                    self.Pools[job.Routing[0]].append(job)

                # generate the event that trigger the next release

                waiting_new_jobs_event = self.env.event()

                self.PoolUpstream.waiting_new_jobs.append(waiting_new_jobs_event)

                yield waiting_new_jobs_event

        def Human_Centric_release(self, period, workload_norm):
            """Implements the LUMS-COR human-centric release rule.

        This method uses a two-stage gating process to release jobs. In the first
        stage, it releases collaborative jobs (those requiring human-machine interaction)
        as long as they do not violate human workload limits. In the second stage, it
        releases independent jobs, considering only machine workload limits. This approach
        prioritizes human-centric operations while still maintaining control over machine workloads.

        Args:
            period (int): The time interval between release decisions.
            workload_norm (float): The workload norm for controlling releases.

        Yields:
            simpy.events.Timeout: A timeout event that pauses the process until the next
                                  release decision point.
        """
            """
            LUMS-COR Human Centric rule: Two-stage gating sequence
            Stage 1: Release collaborative jobs with human workload limits
            Stage 2: Release independent jobs with machine workload limits (human loads fixed)
            NO FORCED RELEASE - jobs wait until constraints are satisfied
            """
            def evaluateJobStage1(self, job, machine_phases_load, human_phases_load):
                """Stage 1: Check human workload constraints for collaborative jobs"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                for station_id in job.Routing:
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > (workload_norm)/N_PHASES):
                        return False
                
                # Check human load constraints for human-machine stations ONLY
                for station_id in job.Routing:
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        if (human_phases_load[station_id] + job_human_load[station_id] > (workload_norm)/N_PHASES):
                            return False
                
                # If human constraints satisfied, release the job and update both loads
                job.ReleaseDate = self.env.now
                
                # Update workload only for stations in the job's routing
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    human_phases_load[station_id] += job_human_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True
                
            def evaluateJobStage2(self, job, machine_phases_load, human_phases_load):
                """Stage 2: Check machine workload constraints for independent jobs (human loads fixed)"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                # Check machine load constraints for all stations in routing
                for station_id in job.Routing:
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > (workload_norm)/N_PHASES):
                        return False
                
                # If machine constraints satisfied, release the job
                job.ReleaseDate = self.env.now
                
                # Update only machine workload (human loads are fixed from Stage 1)
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    # Do NOT update human_phases_load in Stage 2
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            while True:
                machine_phases_load, human_phases_load = get_corrected_shop_load_with_ratios(self.Pools)
                
                # Sort PSP by PLANNED RELEASE DATE instead of Due Date
                all_jobs = []
                for i in range(len(self.PoolUpstream)):
                    all_jobs.append(self.PoolUpstream.get(0))

                if DISPATCH_RULE == 'FIFO': all_jobs.sort(key=lambda x: x.ArrivalDate)
                elif DISPATCH_RULE == 'EDD': all_jobs.sort(key=lambda x: x.DueDate)
                else: all_jobs.sort(key=lambda x: x.ArrivalDate)

                # Separate collaborative and independent jobs while maintaining planned release order
                collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                independent_jobs = [job for job in all_jobs if not job.is_collaborative()]
                
                # STAGE 1: Process collaborative jobs with human workload constraints
                remaining_collaborative = []
                for job in collaborative_jobs:
                    if not evaluateJobStage1(self, job, machine_phases_load, human_phases_load):
                        remaining_collaborative.append(job)
                
                # STAGE 2: With human loads fixed, process independent jobs with machine constraints
                remaining_independent = []
                for job in independent_jobs:
                    if not evaluateJobStage2(self, job, machine_phases_load, human_phases_load):
                        remaining_independent.append(job)
                
                # Return unreleased jobs to pool (maintain planned release order)
                for job in remaining_collaborative + remaining_independent:
                    self.PoolUpstream.append(job)
                    
                yield env.timeout(period)

        def Human_Centric_release_directed(self, period, workload_norm):
            """Implements the human-centric release rule for directed flow shops.

        This method is a variation of the human-centric release rule, specifically
        adapted for directed flow manufacturing systems. It follows the same two-stage
        gating logic as the standard version but may incorporate flow-specific
        optimizations. Jobs are released based on human and machine workload limits,
        ensuring a balanced flow through the system.

        Args:
            period (int): The time interval for release decisions.
            workload_norm (float): The workload norm for release control.

        Yields:
            simpy.events.Timeout: A timeout event to schedule the next release decision.
        """
            """
            LUMS-COR Human Centric rule for directed flow: Two-stage gating sequence
            Stage 1: Release collaborative jobs with human workload limits
            Stage 2: Release independent jobs with machine workload limits (human loads fixed)
            NO FORCED RELEASE - jobs wait until constraints are satisfied
            """
            def evaluateJobStage1(self, job, machine_phases_load, human_phases_load):
                """Stage 1: Check human workload constraints for collaborative jobs"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                for station_id in job.Routing:
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > (workload_norm)/N_PHASES):
                        return False
                
                # Check human load constraints for human-machine stations ONLY
                for station_id in job.Routing:
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        if (human_phases_load[station_id] + job_human_load[station_id] > (workload_norm)/N_PHASES):
                            return False
                
                # If human constraints satisfied, release the job and update both loads
                job.ReleaseDate = self.env.now
                
                # Update workload only for stations in the job's routing
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    human_phases_load[station_id] += job_human_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True
                
            def evaluateJobStage2(self, job, machine_phases_load, human_phases_load):
                """Stage 2: Check machine workload constraints for independent jobs (human loads fixed)"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                # Check machine load constraints for all stations in routing
                for station_id in job.Routing:
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > (workload_norm)/N_PHASES):
                        return False
                
                # If machine constraints satisfied, release the job
                job.ReleaseDate = self.env.now
                
                # Update only machine workload (human loads are fixed from Stage 1)
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    # Do NOT update human_phases_load in Stage 2
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            while True:
                machine_phases_load, human_phases_load = get_corrected_shop_load_with_ratios(self.Pools)
                
                # Sort PSP by PLANNED RELEASE DATE instead of Due Date
                all_jobs = []
                for i in range(len(self.PoolUpstream)):
                    all_jobs.append(self.PoolUpstream.get(0))
                
                if DISPATCH_RULE == 'FIFO': all_jobs.sort(key=lambda x: x.ArrivalDate)
                elif DISPATCH_RULE == 'EDD': all_jobs.sort(key=lambda x: x.DueDate)
                else: all_jobs.sort(key=lambda x: x.ArrivalDate)
                
                # Separate collaborative and independent jobs while maintaining planned release order
                collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                independent_jobs = [job for job in all_jobs if not job.is_collaborative()]
                
                # STAGE 1: Process collaborative jobs with human workload constraints
                remaining_collaborative = []
                for job in collaborative_jobs:
                    if not evaluateJobStage1(self, job, machine_phases_load, human_phases_load):
                        remaining_collaborative.append(job)
                
                # STAGE 2: With human loads fixed, process independent jobs with machine constraints
                remaining_independent = []
                for job in independent_jobs:
                    if not evaluateJobStage2(self, job, machine_phases_load, human_phases_load):
                        remaining_independent.append(job)
                
                # Return unreleased jobs to pool (maintain planned release order)
                for job in remaining_collaborative + remaining_independent:
                    self.PoolUpstream.append(job)
                    
                yield env.timeout(period)

        def Human_Centric_release_with_IM(self, period, workload_norm):
            """Implements the human-centric release rule with immediate release for collaborative jobs.

        This method adapts the human-centric approach to handle scenarios with worker
        absenteeism. It includes logic to switch to an immediate release mode for
        collaborative jobs when necessary, ensuring that production continues even
        with reduced human capacity. The release decisions are also adjusted based on
        machine reliability and absenteeism levels.

        Args:
            period (int): The time interval for release decisions.
            workload_norm (float): The workload norm for release control.

        Yields:
            simpy.events.Timeout: A timeout event for periodic release decisions.
        """
            """Human Centric release with both absenteeism approaches"""
            
            def evaluateJobStage1_IM(self, job, machine_phases_load, human_phases_load):
                """Stage 1 for immediate mode: Always release collaborative jobs"""
                job_machine_load, job_human_load = job.get_CAW_with_ratios_routing_only()
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    human_phases_load[station_id] += job_human_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            def evaluateJobStage1_Normal(self, job, machine_phases_load, human_phases_load):
                """Stage 1 for normal workload control with absenteeism"""
                job_machine_load, job_human_load = job.get_CAW_with_ratios_routing_only()
                
                base_station_wln = workload_norm / N_PHASES
                
                for station_id in job.Routing:
                    # Machine constraint with machine reliability
                    machine_factor = 1.0
                    if self.downtime_manager:
                        machine_factor = self.downtime_manager.get_station_reliability_factor(station_id)
                    
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > base_station_wln * machine_factor):
                        return False
                    
                    # Human constraint with absenteeism (human-machine stations only)
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        human_factor = 1.0
                        if self.absenteeism_manager:
                            human_factor = self.absenteeism_manager.get_station_wln_factor(station_id)
                        
                        if (human_phases_load[station_id] + job_human_load[station_id] > base_station_wln * human_factor):
                            return False
                
                # Release job
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    human_phases_load[station_id] += job_human_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            def evaluateJobStage2_IM(self, job, machine_phases_load, human_phases_load):
                """Stage 2 for immediate mode: Release independent jobs"""
                job_machine_load, job_human_load = job.get_CAW_with_ratios_routing_only()
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            def evaluateJobStage2_Normal(self, job, machine_phases_load, human_phases_load):
                """Stage 2 with machine constraints only"""
                job_machine_load, job_human_load = job.get_CAW_with_ratios_routing_only()
                
                base_station_wln = workload_norm / N_PHASES
                
                for station_id in job.Routing:
                    # Only machine constraints in Stage 2
                    machine_factor = 1.0
                    if self.downtime_manager:
                        machine_factor = self.downtime_manager.get_station_reliability_factor(station_id)
                    
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > base_station_wln * machine_factor):
                        return False
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            while True:
                machine_phases_load, human_phases_load = get_corrected_aggregated_workload_with_ratios(self.Pools)
                
                # Sort PSP by Due Date
                all_jobs = []
                for i in range(len(self.PoolUpstream)):
                    all_jobs.append(self.PoolUpstream.get(0))
                
                if DISPATCH_RULE == 'FIFO': all_jobs.sort(key=lambda x: x.ArrivalDate)
                elif DISPATCH_RULE == 'EDD': all_jobs.sort(key=lambda x: x.DueDate)
                else: all_jobs.sort(key=lambda x: x.ArrivalDate)
                
                # Separate collaborative and independent jobs
                collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                independent_jobs = [job for job in all_jobs if not job.is_collaborative()]
                
                # Choose evaluation functions based on mode
                if workload_norm==0:
                    evaluateStage1 = evaluateJobStage1_IM
                    evaluateStage2 = evaluateJobStage2_IM
                else:
                    evaluateStage1 = evaluateJobStage1_Normal
                    evaluateStage2 = evaluateJobStage2_Normal
                
                # STAGE 1: Process collaborative jobs
                remaining_collaborative = []
                for job in collaborative_jobs:
                    if not evaluateStage1(self, job, machine_phases_load, human_phases_load):
                        remaining_collaborative.append(job)
                
                # STAGE 2: Process independent jobs
                remaining_independent = []
                for job in independent_jobs:
                    if not evaluateStage2(self, job, machine_phases_load, human_phases_load):
                        remaining_independent.append(job)
                
                # Return unreleased jobs to pool
                for job in remaining_collaborative + remaining_independent:
                    self.PoolUpstream.append(job)
                    
                yield self.env.timeout(period)

        def Human_Centric_release_directed_with_IM(self, period, workload_norm):
            """Implements an event-driven human-centric release for directed flow shops.

        This method is designed for directed flow systems and incorporates an event-driven
        release mechanism. Instead of relying solely on periodic checks, it can be
        triggered by system events, such as machine downtimes or changes in worker
        availability. This allows for a more responsive and adaptive release process,
        which is crucial in dynamic environments.

        Args:
            period (int): The periodic timeout for release decisions.
            workload_norm (float): The workload norm for release control.

        Yields:
            simpy.events.AnyOf: An event that can be either a periodic timeout or a
                                system-triggered event.
        """
            """Human Centric release with absenteeism affecting both WLN AND processing times (DIRECTED)"""

            def evaluateJobStage1_IM(self, job, machine_phases_load, human_phases_load):
                """Stage 1 for immediate mode"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    human_phases_load[station_id] += job_human_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            def evaluateJobStage1_Normal(self, job, machine_phases_load, human_phases_load):
                """Stage 1 for directed flow with absenteeism affecting both WLN AND processing times"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                base_station_wln = workload_norm / N_PHASES
                
                for station_id in job.Routing:
                    # Machine constraint with machine reliability
                    machine_factor = 1.0
                    if self.downtime_manager:
                        machine_factor = self.downtime_manager.get_station_reliability_factor(station_id)
                    
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > base_station_wln * machine_factor):
                        return False
                    
                    # Human constraint with absenteeism (human-machine stations only)
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        human_factor = 1.0
                        
                        if self.absenteeism_manager:
                            human_factor = self.absenteeism_manager.get_station_wln_factor(station_id)
                        
                        human_capacity = base_station_wln * human_factor
                        
                        if (human_phases_load[station_id] + job_human_load[station_id] > human_capacity):
                            return False
                
                # Release job - Update loads with productivity-adjusted values
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        human_phases_load[station_id] += job_human_load[station_id]
                
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            def evaluateJobStage2_IM(self, job, machine_phases_load, human_phases_load):
                """Stage 2 for immediate mode"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            def evaluateJobStage2_Normal(self, job, machine_phases_load, human_phases_load):
                """Stage 2 for directed flow with machine constraints only"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()
                
                base_station_wln = workload_norm / N_PHASES
                
                for station_id in job.Routing:
                    # Only machine constraints in Stage 2
                    machine_factor = 1.0
                    if self.downtime_manager:
                        machine_factor = self.downtime_manager.get_station_reliability_factor(station_id)
                    
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > base_station_wln * machine_factor):
                        return False
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            # EVENT-DRIVEN MAIN LOOP FOR DIRECTED FLOW
            next_periodic_time = self.env.now + period
            
            while True:
                try:
                    # Wait for (periodic timeout) OR (release trigger from downtime events OR absenteeism planning)
                    periodic_timeout = self.env.timeout(next_periodic_time - self.env.now)
                    trigger_event = self.system.release_trigger
                    
                    result = yield simpy.events.AnyOf(self.env, [periodic_timeout, trigger_event])
                    
                    if periodic_timeout in result:
                        wake_reason = "periodic"
                        next_periodic_time = self.env.now + period
                    else:
                        wake_reason = "event"
                        self.system.release_trigger = self.env.event()
                    
                    # PROCESS JOBS - DIRECTED FLOW: Use corrected shop load
                    machine_phases_load, human_phases_load = get_corrected_shop_load_with_ratios(self.Pools)
                    
                    # Sort by arrival date for directed flow
                    all_jobs = []
                    for i in range(len(self.PoolUpstream)):
                        all_jobs.append(self.PoolUpstream.get(0))
                    
                    if DISPATCH_RULE == 'FIFO': all_jobs.sort(key=lambda x: x.ArrivalDate)
                    elif DISPATCH_RULE == 'EDD': all_jobs.sort(key=lambda x: x.DueDate)
                    else: all_jobs.sort(key=lambda x: x.ArrivalDate)
                    
                    # Separate collaborative and independent jobs
                    collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                    independent_jobs = [job for job in all_jobs if not job.is_collaborative()]
                    
                    # Choose evaluation functions
                    if workload_norm == 0:
                        evaluateStage1 = evaluateJobStage1_IM
                        evaluateStage2 = evaluateJobStage2_IM
                    else:
                        evaluateStage1 = evaluateJobStage1_Normal
                        evaluateStage2 = evaluateJobStage2_Normal
                    
                    # Stage 1: Process collaborative jobs
                    remaining_collaborative = []
                    for job in collaborative_jobs:
                        if not evaluateStage1(self, job, machine_phases_load, human_phases_load):
                            remaining_collaborative.append(job)
                    
                    # Stage 2: Process independent jobs
                    remaining_independent = []
                    for job in independent_jobs:
                        if not evaluateStage2(self, job, machine_phases_load, human_phases_load):
                            remaining_independent.append(job)
                    
                    # Return unreleased jobs to pool
                    for job in remaining_collaborative + remaining_independent:
                        self.PoolUpstream.append(job)
                        
                except simpy.Interrupt:
                    continue

        def WL_Direct_release_with_IM(self, period, workload_norm):
            """Implements the direct workload control (WL_Direct) release rule.

        This method releases jobs based on the corrected aggregated workload (CAW) at
        each station. It includes adaptations for handling worker absenteeism, adjusting
        the effective workload norm based on machine reliability and human resource
        availability. This allows the system to maintain stable performance even in the
        presence of disruptions.

        Args:
            period (int): The time interval for release decisions.
            workload_norm (float): The workload norm for release control.

        Yields:
            simpy.events.Timeout: A timeout event for periodic release decisions.
        """
            """WL_Direct rule with both absenteeism approaches"""
            
            def evaluateJob_IM(self, job, phases_load):
                """Immediate mode: Always release jobs"""
                job_CAW = job.get_CAW_routing_only()
                
                job.ReleaseDate = self.env.now
                
                for station_id in job.Routing:
                    phases_load[station_id] += job_CAW[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True
            
            def evaluateJob_Normal(self, job, phases_load):
                """Normal evaluation with absenteeism"""
                job_CAW = job.get_CAW_routing_only()
                
                base_station_wln = workload_norm / N_PHASES
                
                for station_id in job.Routing:
                    # Start with base station WLN
                    effective_station_wln = base_station_wln
                    
                    # Apply machine reliability factor
                    if self.downtime_manager:
                        machine_factor = self.downtime_manager.get_station_reliability_factor(station_id)
                        effective_station_wln *= machine_factor
                    
                    # Apply absenteeism factor (human-machine stations only)
                    if HUMAN_MACHINE_PHASES.get(station_id, False) and self.absenteeism_manager:
                        human_factor = self.absenteeism_manager.get_station_wln_factor(station_id)
                        effective_station_wln *= human_factor
                    
                    # Calculate load to compare
                    load_to_compare = job_CAW[station_id]
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        load_to_compare *= MACHINE_RATIO
                    
                    # Check constraint for this station
                    if (phases_load[station_id] + load_to_compare > effective_station_wln):
                        return False
                
                # Release job
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    phases_load[station_id] += job_CAW[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            while True:
                phases_load = get_corrected_aggregated_workload_routing_only(self.Pools)
                
                # Sort PSP by Due Date
                jobs_to_evaluate = []
                for i in range(len(self.PoolUpstream)):
                    jobs_to_evaluate.append(self.PoolUpstream.get(0))
                
                if DISPATCH_RULE == 'FIFO': jobs_to_evaluate.sort(key=lambda x: x.ArrivalDate)
                elif DISPATCH_RULE == 'EDD': jobs_to_evaluate.sort(key=lambda x: x.DueDate)
                else: jobs_to_evaluate.sort(key=lambda x: x.ArrivalDate)
                
                # Choose evaluation function based on mode
                if workload_norm==0:
                    evaluateJob = evaluateJob_IM
                else:
                    evaluateJob = evaluateJob_Normal
                
                # Process jobs in PRD order
                for job in jobs_to_evaluate:
                    if not evaluateJob(self, job, phases_load):
                        self.PoolUpstream.append(job)
                    
                yield self.env.timeout(period)

        def WL_Direct_release_directed_with_IM(self, period, workload_norm):
            """Implements an event-driven direct workload control for directed flow shops.

        This method combines the WL_Direct rule with an event-driven mechanism for
        directed flow systems. It is responsive to system events, such as machine
        downtimes and absenteeism, allowing for dynamic adjustments to the release
        process. This ensures that the workload is managed effectively, even in a
        changing production environment.

        Args:
            period (int): The periodic timeout for release decisions.
            workload_norm (float): The workload norm for release control.

        Yields:
            simpy.events.AnyOf: An event that can be either a periodic timeout or a
                                system-triggered event.
        """
            """Event-driven WL_Direct release for directed flow with both absenteeism types"""

            def evaluateJob_IM(self, job, phases_load):
                """Immediate mode: Always release jobs"""
                job_CSL = job.get_CSL_routing_only()
                
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    phases_load[station_id] += job_CSL[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True
            
            def evaluateJob_Normal(self, job, phases_load):
                """Normal evaluation for directed flow with absenteeism"""
                job_CSL = job.get_CSL_routing_only()
                
                base_station_wln = workload_norm / N_PHASES
                
                for station_id in job.Routing:
                    # Calculate effective WLN for this station
                    effective_station_wln = base_station_wln
                    
                    # Apply machine reliability factor
                    if self.downtime_manager:
                        machine_factor = self.downtime_manager.get_station_reliability_factor(station_id)
                        effective_station_wln *= machine_factor
                    
                    # Apply absenteeism factor (human-machine stations only)
                    if HUMAN_MACHINE_PHASES.get(station_id, False) and self.absenteeism_manager:
                        human_factor = self.absenteeism_manager.get_station_wln_factor(station_id)
                        effective_station_wln *= human_factor
                    
                    # Calculate load to compare
                    load_to_compare = job_CSL[station_id]
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        load_to_compare *= MACHINE_RATIO
                    
                    # Check constraint for this station
                    if (phases_load[station_id] + load_to_compare > effective_station_wln):
                        return False
                
                # Release job
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    phases_load[station_id] += job_CSL[station_id]
                    
                self.Pools[job.Routing[0]].append(job)
                
                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG 
                    JOBS_RELEASED_DEBUG.append(job)
                
                return True

            # EVENT-DRIVEN MAIN LOOP FOR DIRECTED FLOW
            next_periodic_time = self.env.now + period
            
            while True:
                try:
                    # Wait for (periodic timeout) OR (release trigger from downtime/absenteeism)
                    periodic_timeout = self.env.timeout(next_periodic_time - self.env.now)
                    trigger_event = self.system.release_trigger
                    
                    result = yield simpy.events.AnyOf(self.env, [periodic_timeout, trigger_event])
                    
                    if periodic_timeout in result:
                        wake_reason = "periodic"
                        next_periodic_time = self.env.now + period
                    else:
                        wake_reason = "event"
                        self.system.release_trigger = self.env.event()
                    
                    # PROCESS JOBS - DIRECTED FLOW: Use corrected shop load
                    phases_load = get_corrected_shop_load_routing_only(self.Pools)
                    
                    # Sort by ARRIVAL DATE (FIFO)
                    jobs_to_evaluate = []
                    for i in range(len(self.PoolUpstream)):
                        jobs_to_evaluate.append(self.PoolUpstream.get(0))
                    
                    if DISPATCH_RULE == 'FIFO': jobs_to_evaluate.sort(key=lambda x: x.ArrivalDate)
                    elif DISPATCH_RULE == 'EDD': jobs_to_evaluate.sort(key=lambda x: x.DueDate)
                    else: jobs_to_evaluate.sort(key=lambda x: x.ArrivalDate)
                    
                    # Choose evaluation function
                    if workload_norm == 0:
                        evaluateJob = evaluateJob_IM
                    else:
                        evaluateJob = evaluateJob_Normal
                    
                    # Process jobs in FIFO order
                    for job in jobs_to_evaluate:
                        if not evaluateJob(self, job, phases_load):
                            self.PoolUpstream.append(job)
                            
                except simpy.Interrupt:
                    continue

    class Pool(object):
        """Represents a buffer or queue for storing jobs.

    Pools are used throughout the simulation to hold jobs at different stages,
    such as the pre-shop pool (PSP) for jobs awaiting release, and the individual
    queues for each machine. The class provides methods for adding, removing, and
    accessing jobs, as well as mechanisms for notifying other system components
    when the pool's state changes.

    Attributes:
        env (simpy.Environment): The simulation environment.
        Pools (list): A reference to the list of all pools in the system.
        id (int): A unique identifier for the pool.
        array (list): The list of `Job` objects currently in the pool.
        waiting_new_jobs (list): A list of events to be triggered when a new job arrives.
        workload_limit_triggers (list): A list of events to be triggered when workload
                                        limits are exceeded.
    """

        def __init__(self, env, id, Pools):
            """Initializes a new Pool object.

        Args:
            env (simpy.Environment): The simulation environment.
            id (int): The unique identifier for the pool.
            Pools (list): A reference to the list of all pools in the system.
        """

            self.env = env

            self.Pools=Pools

            self.id = id

            self.array = list()  #list of jobs

            # self.waiting_new_jobs is an external trigger (used in Machine::Machine_loop() and Worker::_ReactiveWorker) to notify the availability of new jobs

            self.waiting_new_jobs = list()

            # self.workload_limit_triggers is an external trigger (used in Worker::_Flexible_loop()) to notify whether the lower workload limit has been reached

            self.workload_limit_triggers = list()   # ([id_worker,limit_trigger,event])
            
        def __getitem__(self, key):
            """Enables accessing jobs in the pool using index notation (e.g., `pool[0]`).

        Args:
            key (int): The index of the job to retrieve.

        Returns:
            Job: The job at the specified index.
        """

            # Used to implement "Pool[x]"

            return self.array[key]

        def __len__(self):
            """Enables using the `len()` function to get the number of jobs in the pool.

        Args:
            None

        Returns:
            int: The number of jobs in the pool.
        """

            # Used to implement "len(Pool)"

            return len(self.array)        

        def append(self, job):
            """Adds a job to the pool and triggers relevant events.

        When a job is added, this method also triggers any pending events that are
        waiting for new jobs to arrive. This is a key mechanism for signaling to
        machines and workers that new work is available. It also checks and triggers
        any workload limit events.

        Args:
            job (Job): The job to be added to the pool.
        """

            self.array.append(job)

            # trigger starving machines and workers

            while len(self.waiting_new_jobs) > 0:

                self.waiting_new_jobs[0].succeed()

                del(self.waiting_new_jobs[0])

            # trigger workes whether the workload limit has been exceeded

            i = 0

            while i < len(self.workload_limit_triggers):

                if (get_corrected_aggregated_workload(self.Pools)[self.id] > self.workload_limit_triggers[i][1]):

                    self.workload_limit_triggers[i][2].succeed()

                    del(self.workload_limit_triggers[i])

                else:

                    i += 1

        def get(self, index = 0):
            """Retrieves and removes a job from the pool.

        This method extracts a job from the specified index in the pool. If no index
        is provided, it defaults to the first job in the list. The job is removed from
        the pool as part of this operation.

        Args:
            index (int, optional): The index of the job to retrieve. Defaults to 0.

        Returns:
            Job: The job that was removed from the pool.
        """
            """ 

            The function extracts(then deletes too) the item at the location 'index'. If no location is defined it returns the first element of the list.

            Every time a job extraction happens the pool verifies whether it is necessary to comunicate the stations that the trigger level has been exceeded

            """

            temp = self.array[index]

            del(self.array[index])

            return temp

        def get_list(self):
            """Returns the list of all jobs currently in the pool.

        This method provides read-only access to the list of jobs in the pool without
        modifying the pool's contents.

        Args:
            None

        Returns:
            list: The list of `Job` objects in the pool.
        """

            return self.array            

        def delete(self,index):
            """Deletes a job from the pool at a specified index.

        Args:
            index (int): The index of the job to be deleted.
        """

            del(self.array[index])

        def sort(self):
            """Sorts the jobs in the pool based on the current dispatching rule.

        This method reorders the jobs in the pool according to the globally defined
        `DISPATCH_RULE`. This can be either 'FIFO' (First-In, First-Out), based on
        arrival date, or 'EDD' (Earliest Due Date).
        """

            if DISPATCH_RULE == 'FIFO': self.array.sort(key=lambda x: x.ArrivalDate)
            elif DISPATCH_RULE == 'EDD': self.array.sort(key=lambda x: x.DueDate)
            else: self.array.sort(key=lambda x: x.ArrivalDate)

    class Machine(object):
        """Represents a processing station or machine on the shop floor.

    This class models a machine that processes jobs from its upstream pool. It handles
    the logic for job processing, including waiting for resources like workers and
    handling disruptions such as downtime. The machine can operate in different modes
    (e.g., machine-centric or human-centric) and can be configured to require a worker
    for its operation.

    Attributes:
        env (simpy.Environment): The simulation environment.
        id (int): A unique identifier for the machine.
        has_worker (bool): A flag indicating if the machine requires a worker.
        processing_mode (str): The processing mode (e.g., 'machine_centric').
        JobsProcessed (int): A counter for the number of jobs processed.
        WorkloadProcessed (float): The total workload processed by the machine.
        current_job (Job): The job currently being processed.
        absenteeism_manager (DailyPlanningAbsenteeismManager): A manager for worker absenteeism.
        downtime_manager (MachineDowntimeManager): A manager for machine downtime.
        Workers (list): A list of workers currently assigned to the machine.
        PoolUpstream (Pool): The input queue for the machine.
        Jobs_delivered (Pool): The pool for completed jobs.
    """

        def __init__(self, env, id, PoolUpstream, Jobs_delivered, Pools, PSP, 
                    has_worker=False, processing_mode='machine_centric',
                    absenteeism_manager=None, downtime_manager=None):
            """Initializes a new Machine object.

        This constructor sets up the machine with its configuration and references to
        other system components. It initializes the machine's state, including its
        disruption managers, and starts the main processing loop as a SimPy process.

        Args:
            env (simpy.Environment): The simulation environment.
            id (int): The unique identifier for the machine.
            PoolUpstream (Pool): The input queue for the machine.
            Jobs_delivered (Pool): The pool for completed jobs.
            Pools (list): A list of all pools in the system.
            PSP (Pool): The pre-shop pool.
            has_worker (bool, optional): Whether the machine requires a worker. Defaults to False.
            processing_mode (str, optional): The machine's processing mode. Defaults to 'machine_centric'.
            absenteeism_manager (DailyPlanningAbsenteeismManager, optional): The absenteeism manager. Defaults to None.
            downtime_manager (MachineDowntimeManager, optional): The downtime manager. Defaults to None.
        """
            
            self.env = env
            self.id = id
            self.has_worker = has_worker
            self.processing_mode = processing_mode
            
            self.JobsProcessed = 0
            self.WorkloadProcessed = 0.0
            self.current_job = None
            self.efficiency = 0
            
            # Disruption managers
            self.absenteeism_manager = absenteeism_manager
            self.downtime_manager = downtime_manager
            
            # Worker state (for machine-centric mode)
            self.worker_busy = False
            self.worker_current_job = None
            self.human_busy_until = 0
            
            # Workers
            self.Workers = list()
            
            # External references
            self.PoolUpstream = PoolUpstream
            self.PSP = PSP
            self.Pools = Pools
            self.Jobs_delivered = Jobs_delivered
            
            # Internal triggers
            self.waiting_new_workers = self.env.event()
            self.waiting_new_jobs = self.env.event()
            PoolUpstream.waiting_new_jobs.append(self.waiting_new_jobs)
            
            # External triggers
            self.waiting_end_job = list()
            self.worker_done_events = []
            
            # Start processes
            self.process = self.env.process(self.Machine_loop())
        
        def Machine_loop(self):
            """The main processing loop for the machine.

        This method runs as a SimPy process and orchestrates the machine's operations.
        It waits for the necessary resources (jobs and, if required, workers), checks for
        downtime, processes the job, and handles job completion and routing to the next
        station. The processing logic varies based on the machine's configured mode.

        Args:
            None

        Yields:
            simpy.events.Event: Events for resource availability, processing time,
                                and other simulation delays.
        """
            """Main processing loop"""
            
            while True:
                try:
                    # Wait for resources
                    if self.has_worker:
                        yield (self.waiting_new_jobs & self.waiting_new_workers)
                    else:
                        yield self.waiting_new_jobs
                        
                except simpy.Interrupt:
                    continue
                
                while True:
                    
                    # **ADD: Check for machine downtime (blocking)**
                    if self.downtime_manager and not self.downtime_manager.is_machine_available(self.id):
                        repair_time = self.downtime_manager.get_machine_repair_time(self.id) - self.env.now
                        if repair_time > 0:
                            yield self.env.timeout(repair_time)
                        continue
                    
                    # Check for jobs
                    if len(self.PoolUpstream) == 0 and self.current_job == None:
                        self.waiting_new_jobs = self.env.event()
                        self.PoolUpstream.waiting_new_jobs.append(self.waiting_new_jobs)
                        break
                    
                    # Check for workers (only if machine needs them)
                    if self.has_worker and len(self.Workers) == 0:
                        self.efficiency = 0
                        self.waiting_new_workers = self.env.event()
                        break
                    
                    # In machine-centric mode, wait for worker to be available
                    if self.has_worker and self.processing_mode == 'machine_centric':
                        while self.worker_busy:
                            worker_done = self.env.event()
                            self.worker_done_events.append(worker_done)
                            yield worker_done
                    
                    # Get next job
                    if self.current_job == None:
                        self.current_job = self.PoolUpstream.get()
                        self.current_job.ArrivalDateMachines[self.id] = self.env.now
                    
                    # Calculate processing times
                    machine_time = self.current_job.RemainingTime[self.id]
                    
                    if self.has_worker:
                        worker = self.Workers[0]
                        
                        # **MODIFIED: Apply human variability ratio**
                        worker_time = self.current_job.RemainingTime[self.id] * HUMAN_RATIO
                        
                        # **ADD: Apply absenteeism productivity factor**
                        productivity_factor = 1.0
                        if self.absenteeism_manager:
                            productivity_factor = self.absenteeism_manager.get_station_productivity_factor(self.id)
                        
                        adjusted_worker_time = worker_time * productivity_factor
                        
                        self.efficiency = 1
                    else:
                        adjusted_worker_time = 0
                        self.efficiency = 1
                    
                    # Process based on mode
                    start_job = self.env.now
                    interrupted = False
                    
                    try:
                        if not self.has_worker:
                            # AUTOMATED MACHINE
                            yield self.env.timeout(machine_time)
                            
                        elif self.processing_mode == 'machine_centric':
                            # MACHINE CENTRIC - decoupled processing
                            yield self.env.process(self._machine_centric_processing(
                                self.current_job, machine_time, adjusted_worker_time))
                            
                        elif self.processing_mode == 'human_centric':
                            # HUMAN CENTRIC - coupled processing
                            processing_time = max(machine_time, adjusted_worker_time)
                            yield self.env.timeout(processing_time)
                    
                    except simpy.Interrupt:
                        interrupted = True
                    
                    # Track worker time
                    if self.has_worker and len(self.Workers) > 0:
                        for worker in self.Workers:
                            worker.WorkingTime[self.id] += (self.env.now - start_job)
                    
                    self.WorkloadProcessed += ((self.env.now - start_job) * self.efficiency)
                    
                    if interrupted:
                        self.current_job.RemainingTime[self.id] -= ((self.env.now - start_job) * self.efficiency)
                        continue
                    
                    # For machine-centric, job completion is handled in separate process
                    if self.processing_mode == 'machine_centric' and self.has_worker:
                        self.current_job = None
                        self.JobsProcessed += 1
                    else:
                        self._complete_job_at_machine()
                    
                    # Trigger workers waiting for job end
                    while len(self.waiting_end_job) > 0:
                        self.waiting_end_job[0].succeed()
                        del self.waiting_end_job[0]
        
        def _machine_centric_processing(self, job, machine_time, worker_time):
            """Handles job processing in machine-centric mode.

        In this mode, the machine and worker start processing the job simultaneously.
        However, the machine becomes free as soon as its part of the work is done,
        allowing it to start a new job. The original job is only considered complete
        when the worker also finishes their task. This method manages this decoupled
        processing flow.

        Args:
            job (Job): The job to be processed.
            machine_time (float): The processing time for the machine.
            worker_time (float): The processing time for the worker.

        Yields:
            simpy.events.Timeout: A timeout event for the machine's processing time.
        """
            """
            Machine-centric mode:
            - Machine and worker start together
            - Machine finishes → can take new job
            - Worker continues → job completes when worker done
            """
            start_time = self.env.now
            
            # Mark worker as busy with this job
            self.worker_busy = True
            self.worker_current_job = job
            
            # Start worker completion process (runs independently)
            self.env.process(self._worker_completion_process(job, worker_time, start_time))
            
            # Wait for machine to finish
            yield self.env.timeout(machine_time)
            
            # Machine is now free (but worker may still be processing)
        
        def _worker_completion_process(self, job, worker_time, start_time):
            """Manages the worker's part of the job processing in machine-centric mode.

        This method runs as an independent SimPy process to handle the worker's task.
        It waits for the worker's processing time to elapse and then finalizes the
        job's completion. This is where the job is officially marked as complete at
        the machine and routed to its next destination.

        Args:
            job (Job): The job being processed by the worker.
            worker_time (float): The worker's processing time for the job.
            start_time (float): The simulation time when the processing started.

        Yields:
            simpy.events.Timeout: A timeout event for the worker's processing time.
        """
            """
            Independent process for worker to complete job
            Job completion happens here, not in main loop
            """
            # Wait for worker to finish
            yield self.env.timeout(worker_time)
            
            # Worker is done - mark completion
            self.worker_busy = False
            
            # Update job completion (this is the TRUE completion time)
            job.CompletationDateMachines[self.id] = self.env.now
            job.RemainingTime[self.id] = 0
            job.Position += 1
            
            # Check if job is fully complete
            if job.Position == len(job.Routing):
                job.CompletationDate = self.env.now
                self.Jobs_delivered.append(job)
            else:
                # Route to next machine
                self.Pools[job.get_current_machine()].append(job)
            
            self.worker_current_job = None
            
            # Signal that worker is free
            while len(self.worker_done_events) > 0:
                self.worker_done_events[0].succeed()
                del self.worker_done_events[0]
        
        def _complete_job_at_machine(self):
            """Finalizes a job's processing at the machine.

        This method is called for automated or human-centric processing modes. It updates
        the job's completion time at the machine, advances its position in its routing,
        and then either moves it to the next machine's pool or to the completed jobs
        pool if it has finished its routing.

        Args:
            None
        """
            """Complete job in normal mode (automated or human-centric)"""
            self.current_job.CompletationDateMachines[self.id] = self.env.now
            self.current_job.RemainingTime[self.id] = 0
            self.current_job.Position += 1
            
            if self.current_job.Position == len(self.current_job.Routing):
                self.current_job.CompletationDate = self.env.now
                self.Jobs_delivered.append(self.current_job)
            else:
                self.Pools[self.current_job.get_current_machine()].append(self.current_job)
            
            self.current_job = None
            self.JobsProcessed += 1

    class Worker(object):
        """Represents a worker who can be assigned to different machines.

    This class models a worker's behavior, including their skills, flexibility,
    and movement between machines. Workers can have different skill sets, which
    determine their efficiency at various machines. Their behavior can be static
    (fixed to one machine), reactive (moving to other machines when idle), or
    flexible (proactively moving based on workload).

    Attributes:
        env (simpy.Environment): The simulation environment.
        id (int): A unique identifier for the worker.
        Default_machine (int): The worker's primary or home machine.
        skillperphase (list): A list representing the worker's skill level at each machine.
        current_machine_id (int): The ID of the machine the worker is currently at.
        relocation (list): A list tracking the number of relocations to each machine.
        WorkingTime (list): A list of the total time the worker has spent at each machine.
    """

        def __init__(self, env, id, Pools, Machines,Default_machine, skills = None):
            """Initializes a new Worker object.

        This constructor sets up a worker with a specific skill profile and behavior
        mode. It determines the worker's skills based on the global `WORKER_FLEXIBILITY`
        setting and starts the appropriate process for their behavior mode (e.g.,
        static, reactive, or flexible).

        Args:
            env (simpy.Environment): The simulation environment.
            id (int): The unique identifier for the worker.
            Pools (list): A list of all job pools in the system.
            Machines (list): A list of all machines in the system.
            Default_machine (int): The worker's home machine.
            skills (list, optional): A predefined list of skills. Defaults to None.
        """

            self.env = env

            self.skillperphase = list()

            self.id = id

            self.current_machine_id = -1

            # None raises error while printing 

            self.Default_machine=Default_machine

            # self.relocation --> vector to count the # of times a relocation happens

            self.relocation = [0] * N_PHASES

            if WORKER_FLEXIBILITY == 'triangular':

                self._SetTriangularSkills(WORKER_EFFICIENCY_DECREMENT)

                print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))

            elif WORKER_FLEXIBILITY == 'chain':

                self._SetChainSkills()

                print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))

            elif WORKER_FLEXIBILITY == 'chain upstream':

                self._SetChainSkills(upstream=True)

                print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))

            elif WORKER_FLEXIBILITY == 'flat':

                self._SetPlainSkills()

                print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))

            else:

                exit("wrong worker flexibility ")

            self.waiting_events = None

            #self.WorkloadProcessed = list(0 for i in range(N_PHASES))

            self.WorkingTime = list(0 for i in range(N_PHASES))

            # capacity adjstment for the current period

            self.Capacity_adjustment = list(0 for i in range(N_PHASES)) 

            if 'WORKER_MODE' not in globals():

                exit("Worker mode not initialised")            

            if WORKER_MODE == 'static':

                self._StaticicWorker(Machines)												# Fix to the machine

            elif WORKER_MODE == 'reactive':

                self.process = self.env.process(self._ReactiveWorker(Machines,None))

            elif WORKER_MODE == 'flexible':

                # output control

                    self.process = self.env.process(self._Flexible_loop(Machines,None))

            else:

                exit("Worker mode not recognised")  

        def _SetMonoSkill(self):
            """Sets a single skill for the worker at their default machine."""

            for i in range(0,N_PHASES):

                if i == self.Default_machine:

                    self.skillperphase.append(1)

                else:

                    self.skillperphase.append(0)

        #def _SetFlatSkills(self):

        #   self.skillperphase = [1,1,1,1,1]

        def _SetTriangularSkills(self, decrement):
            """Sets a triangular skill profile for the worker.

        The worker has the highest skill at their default machine, with skills
        decreasing linearly for adjacent machines.

        Args:
            decrement (float): The amount by which the skill level decreases for
                               each step away from the default machine.
        """
            self.skillperphase = [max(0, 1 - abs(i - self.Default_machine) * decrement) for i in range(N_PHASES)]

        def _SetExponentialSkills(self, decrement):
            """Sets an exponential skill profile for the worker.

        The worker's skill level decreases exponentially with the distance from
        their default machine.

        Args:
            decrement (float): The base of the exponential decay.
        """
            self.skillperphase = [max(0, pow(decrement, abs(self.Default_machine - i))) for i in range(N_PHASES)]

        def _SetChainSkills(self, upstream=False):
            """Sets a chain-like skill profile for the worker.
        The worker is skilled at their default machine and an adjacent machine,
        allowing them to move along a production line.
        Args:
            upstream (bool): If True, the adjacent machine is upstream, otherwise downstream.
        """
            self.skillperphase = [0] * N_PHASES
            self.skillperphase[self.Default_machine] = 1
            if upstream:
                adjacent_machine = (self.Default_machine - 1 + N_PHASES) % N_PHASES
            else:  # downstream
                adjacent_machine = (self.Default_machine + 1) % N_PHASES
            self.skillperphase[adjacent_machine] = 1 - WORKER_EFFICIENCY_DECREMENT


        def _SetPlainSkills(self):
            """Sets a flat skill profile, making the worker proficient at all machines.

        The worker has a high skill level at all machines, with a slightly higher
        efficiency at their default machine. This represents a highly flexible workforce.
        """
            self.skillperphase = [1 - WORKER_EFFICIENCY_DECREMENT] * N_PHASES
            self.skillperphase[self.Default_machine] = 1

        def _StaticicWorker(self, Machines):
            """Assigns the worker to their default machine for the entire simulation.

        This method implements the static worker behavior, where the worker is
        permanently assigned to a single machine and does not move.

        Args:
            Machines (list): The list of all machines in the system.
        """

            """

            The worker is assigned to its default station and never works at other departments

            """  

            self.current_machine_id = self.Default_machine

            Machines[self.Default_machine].Workers.append(self)

            Machines[self.Default_machine].process.interrupt()

            if Machines[self.current_machine_id].waiting_new_workers.triggered == False:

                Machines[self.current_machine_id].waiting_new_workers.succeed()    

        def _ReactiveWorker(self, Machines,Pools):
            """Implements the reactive worker behavior.

        A reactive worker stays at their home machine as long as there is work to do.
        If their home machine becomes idle, they will look for other machines that
        have pending jobs and where they have the necessary skills. They will then
        relocate to the most suitable machine.

        Args:
            Machines (list): The list of all machines.
            Pools (list): The list of all job pools.

        Yields:
            simpy.events.AnyOf: An event that is triggered by various conditions,
                                such as job completion or new job arrivals, prompting
                                the worker to re-evaluate their position.
        """

            """

            Worker are assigned to their home stations by default. They can be transferred to external departments only if

            there are no work to process at home dep or the workload at home department does not exceed beta. 

            """  

            def _Next_machine1(self):

                # workers can be transferred to external station only if there is no work to process at home dep. 

                if(len(Machines[self.Default_machine].PoolUpstream) > 0 ):

                     return self.Default_machine

                # 

                possible_external_machines = list()

                for i in range(N_PHASES):

                    if( i == self.Default_machine ): 

                        #Skip home department

                        continue 

                    if (self.skillperphase[i] > 0 and len(Machines[i].PoolUpstream) > 0):

                        possible_external_machines.append((i,self.skillperphase[i]))

                if len(possible_external_machines)>0:       

                    possible_external_machines.sort(key=lambda x: float(x[1]), reverse=True)

                    return possible_external_machines[0][0]

                return self.Default_machine

            while True:

                #define the next machine

                next_machine = _Next_machine1(self)

                # Unload the worker from its current station

                if self.current_machine_id != -1 and self.current_machine_id != next_machine:

                    for i in range(len(Machines[self.current_machine_id].Workers)):

                        if Machines[self.current_machine_id].Workers[i].id == self.id:

                            del(Machines[self.current_machine_id].Workers[i])

                            Machines[self.current_machine_id].process.interrupt()

                            break

                # Transfer to the next machine

                if self.current_machine_id != next_machine:

                    yield env.timeout(TRANSFER_TIME)

                    self.current_machine_id=next_machine

                    self.relocation[next_machine] += 1

                    Machines[self.current_machine_id].Workers.append(self)

                    Machines[self.current_machine_id].process.interrupt()

                    if Machines[self.current_machine_id].waiting_new_workers.triggered == False:

                        Machines[self.current_machine_id].waiting_new_workers.succeed()

                waiting_events=list()

                # keep the worker for at least WAITING_TIME minutes

                # waiting_events.append(self.env.timeout(60))

                #if self.current_machine_id == self.Default_machine:

                # wait the end of the current job

                end_current_job=self.env.event()

                Machines[self.current_machine_id].waiting_end_job.append(end_current_job)

                waiting_events.append(end_current_job)

                """

                if self.current_machine_id!=self.Default_machine and beta != None :

                    # critical workload(beta) at home department                

                    critical_workload_trigger=self.env.event()

                    Pools[self.Default_machine].workload_limit_triggers.append([self.id,beta,critical_workload_trigger])

                    waiting_events.append(critical_workload_trigger)

                elif self.current_machine_id!=self.Default_machine and beta == None:

                """

                # waiting new jobs at home department

                waiting_new_jobs = self.env.event()

                Machines[self.Default_machine].PoolUpstream.waiting_new_jobs.append(waiting_new_jobs)

                waiting_events.append(waiting_new_jobs)

                yield env.timeout(PERMANENCE_TIME)

                yield AnyOf(self.env,waiting_events)

        def _Flexible_loop(self,Machines,Pools):
            """Implements the flexible worker behavior.

        Flexible workers can be proactively transferred between machines based on
        capacity adjustments determined at the order release stage. This allows the
        system to dynamically allocate labor resources to where they are most needed,
        based on upcoming workload.

        Args:
            Machines (list): The list of all machines.
            Pools (list): The list of all job pools.

        Yields:
            simpy.events.AnyOf: An event that triggers the worker to re-evaluate
                                their position, based on job completion, capacity
                                adjustments, or new job arrivals.
        """

            """

            Worker are assigned to their home stations by default. They can be transferred to external departments only if

            there are no work to process at home dep or the workload at home department does not exceed beta, and the 

            vector of capacity adjustments defined at order release stage allows the transfer. 

            """  

            def _Next_machine1(self):

                # workers can be transferred to external station only if there is no work to process at home dep. 

                if(len(Machines[self.Default_machine].PoolUpstream) > 0 ):

                     return self.Default_machine

                possible_external_machines = list()

                for i in range(N_PHASES):

                    if( i == self.Default_machine ): 

                        #Skip home department

                        continue 

                    if (self.skillperphase[i] > 0 and len(Machines[i].PoolUpstream) > 0 and self.Capacity_adjustment[i]>1):

                        possible_external_machines.append((i,self.skillperphase[i]))

                if len(possible_external_machines)>0:

                    possible_external_machines.sort(key=lambda x: float(x[1]), reverse=True)

                    return possible_external_machines[0][0]

                return self.Default_machine

            while True:

                next_machine = _Next_machine1(self)

                # Unload the worker from the current station

                if self.current_machine_id != -1 and self.current_machine_id != next_machine:

                    for i in range(len(Machines[self.current_machine_id].Workers)):

                        if Machines[self.current_machine_id].Workers[i].id == self.id:

                            del(Machines[self.current_machine_id].Workers[i])

                            Machines[self.current_machine_id].process.interrupt()

                            break

                # Transfer to the next machine

                if self.current_machine_id != next_machine:

                    yield env.timeout(TRANSFER_TIME)

                    self.current_machine_id=next_machine

                    Machines[self.current_machine_id].Workers.append(self)

                    Machines[self.current_machine_id].process.interrupt()

                    if Machines[self.current_machine_id].waiting_new_workers.triggered == False:

                        Machines[self.current_machine_id].waiting_new_workers.succeed()

                start=self.env.now

                waiting_events=list()

                # check again next machine in WAITING_TIME minutes

                # waiting_events.append(self.env.timeout(WAITING_TIME))

                # wait the end of the current job

                end_current_job=self.env.event()

                Machines[self.current_machine_id].waiting_end_job.append(end_current_job)

                waiting_events.append(end_current_job)

                # end capacity adjustment

                # due to a bug for Capacity_adjustment[self.current_machine_id] infinitesimally small the program loops without processing job

                if self.current_machine_id!=self.Default_machine and self.Capacity_adjustment[self.current_machine_id]>1:

                    end_capacity_adjustment=self.env.timeout(self.Capacity_adjustment[self.current_machine_id])

                    waiting_events.append(end_capacity_adjustment)

                """

                if self.current_machine_id!=self.Default_machine and beta != None :

                    # critical workload(beta) at home department                

                    critical_workload_trigger=self.env.event()

                    Pools[self.Default_machine].workload_limit_triggers.append([self.id,beta,critical_workload_trigger])

                    waiting_events.append(critical_workload_trigger)

                elif self.current_machine_id!=self.Default_machine and beta == None:

                """    

                # waiting new jobs at home department

                waiting_new_jobs = self.env.event()

                Machines[self.Default_machine].PoolUpstream.waiting_new_jobs.append(waiting_new_jobs)

                waiting_events.append(waiting_new_jobs)

                yield env.timeout(PERMANENCE_TIME)

                yield AnyOf(self.env,waiting_events)

                if self.current_machine_id!=self.Default_machine:

                    # update capacity adjustment 

                    self.Capacity_adjustment[self.current_machine_id] = self.Capacity_adjustment[self.current_machine_id] - (self.env.now-start)

    class DailyPlanningAbsenteeismManager:
        """Manages worker absenteeism based on a daily planning framework.

    This class simulates the impact of worker absenteeism on the manufacturing system.
    It follows a dual-impact framework, where absenteeism leads to both a reduction
    in the workload norm (capacity) and an increase in processing times due to
    reduced productivity. The absenteeism levels are determined by global settings
    and are applied on a daily basis.

    Attributes:
        env (simpy.Environment): The simulation environment.
        absenteeism_level_key (str): The level of absenteeism ('low', 'medium', 'high').
        system (System): A reference to the main system object.
        station_wln_factors (list): A list of workload norm reduction factors for each station.
        station_productivity_factors (list): A list of productivity loss factors for each station.
    """
        """
        Planning-based absenteeism following Chapter 3 dual-impact framework
        """
        def __init__(self, env, absenteeism_level_key, system):
            """Initializes the DailyPlanningAbsenteeismManager.

        This constructor sets up the absenteeism manager with the specified level of
        absenteeism and a reference to the system. It initializes the factors for
        workload norm and productivity and starts the daily planning process if
        absenteeism is enabled.

        Args:
            env (simpy.Environment): The simulation environment.
            absenteeism_level_key (str): The key for the absenteeism level.
            system (System): The main system object.
        """
            self.env = env
            self.absenteeism_level_key = absenteeism_level_key
            self.system = system
            
            # For dynamic WLN (varies daily)
            self.station_wln_factors = [1.0] * N_PHASES
            
            # For productivity loss (processing time increases)
            self.station_productivity_factors = [1.0] * N_PHASES
            
            # Human-machine workstations
            self.human_stations = [i for i in range(N_PHASES) if HUMAN_MACHINE_PHASES.get(i, False)]
            
            # Get absenteeism range
            self.min_absenteeism, self.max_absenteeism = self._get_absenteeism_range()
            
            # Get scenario parameters
            self.p_s = self._get_compensation_fraction()  # Based on level
            self.delta_s = 0.28  # Stable, reflects job characteristics (near Brouwer's 27.8%)
            
            if absenteeism_level_key != "none":
                env.process(self.daily_planning_process())
        
        def _get_absenteeism_range(self):
            """Gets the minimum and maximum absenteeism range for the current level.

        Args:
            None

        Returns:
            tuple: A tuple containing the minimum and maximum absenteeism rates.
        """
            """Get absenteeism range (min, max) for each level"""
            if self.absenteeism_level_key == "low":
                return (0.0, 0.03)
            elif self.absenteeism_level_key == "medium":
                return (0.03, 0.06)
            elif self.absenteeism_level_key == "high":
                return (0.06, 0.10)
            else:
                return (0.0, 0.0)
        
        def _get_compensation_fraction(self):
            """Gets the fraction of operations that require compensation due to absenteeism.

        This value is fixed based on the absenteeism level and represents the
        proportion of tasks that are affected by the absence of a worker.

        Args:
            None

        Returns:
            float: The compensation fraction.
        """
            """
            p: Fraction of operations requiring compensation - FIXED by level
            """
            if self.absenteeism_level_key == "low":
                return 0.10
            elif self.absenteeism_level_key == "medium":
                return 0.20
            elif self.absenteeism_level_key == "high":
                return 0.30
            else:
                return 0.0
        
        def _calculate_productivity_factor(self):
            """Calculates the productivity factor based on the compensation fraction.

        This method uses a formula to determine the increase in processing times
        as a result of absenteeism. The formula accounts for the fact that not all
        tasks are equally affected by a worker's absence.

        Args:
            None

        Returns:
            float: The calculated productivity factor.
        """
            """
            Calculate productivity factor using Chapter 3 Eq 3.2:
            PT_h,s(t) = PT_h,s^0 / (1 - p_s * delta_s)
            """
            productivity_factor = 1.0 * (1.0 + self.p_s * self.delta_s)
            return productivity_factor
        
        def daily_planning_process(self):
            """Simulates the daily planning process for distributing absenteeism.

        This method runs as a SimPy process with a daily cycle. Each day, it randomly
        determines the total absenteeism level and distributes it among the human-machine
        workstations. It then updates the workload norm and productivity factors for
        the affected stations and triggers a release wake-up to allow the system to
        react to the changes.

        Args:
            None

        Yields:
            simpy.events.Timeout: A timeout event for the daily planning cycle.
        """
            """Daily planning: distribute total absenteeism across affected stations"""
            while True:
                yield self.env.timeout(480)  # Daily planning cycle
                
                # Reset all stations to normal
                for i in range(N_PHASES):
                    self.station_wln_factors[i] = 1.0
                    self.station_productivity_factors[i] = 1.0
                
                # Randomly select number of affected workstations (1 to 3)
                num_affected_stations = np.random.randint(1, 4)
                
                if num_affected_stations > 0:
                    # Generate random total absenteeism within level range
                    min_absenteeism, max_absenteeism = self._get_absenteeism_range()
                    total_absenteeism_today = np.random.uniform(min_absenteeism, max_absenteeism)
                    
                    # Randomly select affected workstations
                    affected_stations = np.random.choice(
                        self.human_stations, 
                        size=num_affected_stations, 
                        replace=False
                    )
                    
                    # Distribute using Dirichlet
                    if num_affected_stations == 1:
                        station_absenteeisms = [total_absenteeism_today]
                    else:
                        weights = np.random.dirichlet(np.ones(num_affected_stations))
                        station_absenteeisms = weights * total_absenteeism_today
                    
                    # Calculate productivity factor once (same for all stations at this level)
                    productivity_factor = self._calculate_productivity_factor()
                    
                    for i, station_id in enumerate(affected_stations):
                        absenteeism_pct = station_absenteeisms[i]
                        
                        # WLN reduction (proportional to absence) - Chapter 3 Eq 3.1
                        self.station_wln_factors[station_id] = 1.0 - absenteeism_pct
                        
                        # Processing time increase (same factor for all) - Chapter 3 Eq 3.2
                        self.station_productivity_factors[station_id] = productivity_factor
                        
                        #if SCREEN_DEBUG:
                           # print(f"Day {self.env.now/480:.0f}: Station {station_id} - "
                                #f"{absenteeism_pct*100:.1f}% absent → "
                                #f"WLN factor={self.station_wln_factors[station_id]:.3f}, "
                                #f"PT factor={self.station_productivity_factors[station_id]:.3f} "
                                #f"(p_s={self.p_s:.2f}, δ_s={self.delta_s:.2f})")
                
                self._trigger_release_wake_up()
        
        def _trigger_release_wake_up(self):
            """Triggers the order release process to re-evaluate releases.

        This method sends a signal to the `Orders_release` class, prompting it to
        wake up and make new release decisions. This is important for ensuring that
        the system can adapt quickly to changes in capacity caused by absenteeism.
        """
            """Fire the release trigger"""
            if (hasattr(self.system, 'release_trigger') and 
                self.system.release_trigger is not None and 
                not self.system.release_trigger.triggered):
                self.system.release_trigger.succeed()
        
        def get_station_wln_factor(self, station_id):
            """Gets the workload norm reduction factor for a specific station.

        Args:
            station_id (int): The ID of the station.

        Returns:
            float: The workload norm reduction factor.
        """
            """Get WLN factor - Chapter 3 Eq 3.1"""
            return self.station_wln_factors[station_id]
        
        def get_station_productivity_factor(self, station_id):
            """Gets the productivity loss factor for a specific station.

        Args:
            station_id (int): The ID of the station.

        Returns:
            float: The productivity loss factor.
        """
            """Get productivity factor - Chapter 3 Eq 3.2"""
            return self.station_productivity_factors[station_id]

    class MachineDowntimeManager:
        """Manages machine downtime events, including degradation and breakdowns.

    This class simulates machine failures and their impact on the system. It can model
    two types of failures: degradation, which reduces the machine's processing capacity
    (workload norm), and breakdowns, which make the machine completely unavailable.
    The manager uses an event-driven approach to trigger release wake-ups, allowing
    the system to react immediately to changes in machine availability.

    Attributes:
        env (simpy.Environment): The simulation environment.
        downtime_level_key (str): The level of downtime ('low', 'medium', 'high').
        system (System): A reference to the main system object.
        reliability_factors (list): A list of reliability factors for each machine.
        machine_available (list): A list of booleans indicating if each machine is available.
        machine_repair_time (list): A list of scheduled repair times for each machine.
    """
        """
        Manages machine downtime with event-driven release wake-ups
        - Degradation: Reduced WLN, no blocking
        - Breakdown: WLN=0, workstation blocking
        """
        def __init__(self, env, downtime_level_key, system):
            """Initializes the MachineDowntimeManager.

        This constructor sets up the downtime manager with the specified downtime level
        and a reference to the system. It initializes the reliability parameters and
        starts a reliability process for each machine if downtime is enabled.

        Args:
            env (simpy.Environment): The simulation environment.
            downtime_level_key (str): The key for the downtime level.
            system (System): The main system object.
        """
            self.env = env
            self.downtime_level_key = downtime_level_key
            self.system = system  # Reference to trigger release wake-ups
            
            # For dynamic WLN (varies by failure type)
            self.reliability_factors = [1.0] * N_PHASES
            
            # For workstation blocking (breakdown only)
            self.machine_available = [True] * N_PHASES
            self.machine_repair_time = [0] * N_PHASES
            
            self.station_status = ["up"] * N_PHASES
            self.failure_in_progress = [False] * N_PHASES
            
            self.event_params = self._get_event_parameters()
            
            if downtime_level_key != "none":
                for station_id in range(N_PHASES):
                    env.process(self.station_reliability_process(station_id))
        
        def _get_event_parameters(self):
            """Gets the reliability parameters for the current downtime level.

        This method returns a dictionary of parameters, such as mean time between
        failures (MTBF) and mean time to repair (MTTR), based on the selected
        downtime level.

        Args:
            None

        Returns:
            dict: A dictionary of reliability parameters.
        """
            """Reliability parameters"""
            if self.downtime_level_key == "low":
                return {
                    'mtbf_minutes': 60.0 * 480,           # 30 days MTBF
                    'mttr_mean_minutes': 4.0 * 60,        # 4 hours MTTR
                    'mttr_std_minutes': 2.0 * 60,
                    'planning_reduction': (0.1, 0.2),     # Reduce WLN by 10-20%
                    'breakdown_reduction': 0.0,           # WLN=0 for breakdowns
                    'degradation_chance': 0.7,            # 70% degradation vs breakdown
                }
            elif self.downtime_level_key == "medium":
                return {
                    'mtbf_minutes': 60.0 * 480,
                    'mttr_mean_minutes': 4.0 * 60,        # 4 hours MTTR
                    'mttr_std_minutes': 2.0 * 60,
                    'planning_reduction': (0.15, 0.3),    # Reduce WLN by 15-30%
                    'breakdown_reduction': 0.0,           # WLN=0 for breakdowns
                    'degradation_chance': 0.7,
                }
            elif self.downtime_level_key == "high":
                return {
                    'mtbf_minutes': 60.0 * 480,
                    'mttr_mean_minutes': 4.0 * 60,        # 4 hours MTTR
                    'mttr_std_minutes': 2.0 * 60,
                    'planning_reduction': (0.2, 0.4),     # Reduce WLN by 20-40%
                    'breakdown_reduction': 0.0,           # WLN=0 for breakdowns
                    'degradation_chance': 0.7,
                }
            else:
                return None
        
        def station_reliability_process(self, station_id):
            """Simulates the failure and repair process for a single station.

        This method runs as a SimPy process for each machine. It generates random
        failure events based on the MTBF, determines the type of failure (degradation
        or breakdown), and schedules a repair. It also triggers the release wake-up
        mechanism at the start and end of each failure event.

        Args:
            station_id (int): The ID of the station to be simulated.

        Yields:
            simpy.events.Timeout: Timeout events for the time to failure and the
                                  time to repair.
        """
            """Generate failure/repair events with immediate release wake-ups"""
            while True:
                if self.event_params is None:
                    return
                
                # Wait for next failure
                time_to_failure = np.random.exponential(self.event_params['mtbf_minutes'])
                yield self.env.timeout(time_to_failure)
                
                if self.failure_in_progress[station_id]:
                    continue
                
                self.failure_in_progress[station_id] = True
                
                # Determine failure type
                if np.random.random() < self.event_params['degradation_chance']:
                    # DEGRADATION - Reduced WLN, no blocking
                    min_reduction, max_reduction = self.event_params['planning_reduction']
                    reduction = np.random.uniform(min_reduction, max_reduction)
                    reliability_factor = 1.0 - reduction
                    status = "degraded"
                    blocks_processing = False
                else:
                    # BREAKDOWN - WLN=0, blocking
                    reliability_factor = self.event_params['breakdown_reduction']  # 0.0
                    status = "down"
                    blocks_processing = True
                
                # Calculate repair time
                mttr_mean = self.event_params['mttr_mean_minutes']
                mttr_std = self.event_params['mttr_std_minutes']
                
                variance = mttr_std ** 2
                mu = np.log((mttr_mean ** 2) / np.sqrt(variance + mttr_mean ** 2))
                sigma = np.sqrt(np.log(variance / (mttr_mean ** 2) + 1))
                
                repair_time = max(30.0, np.random.lognormal(mu, sigma))
                
                # START FAILURE
                self.reliability_factors[station_id] = reliability_factor
                self.station_status[station_id] = status
                
                if blocks_processing:
                    self.machine_available[station_id] = False
                    self.machine_repair_time[station_id] = self.env.now + repair_time
                
                #if SCREEN_DEBUG:
                 #   effect_str = f"WLN={reliability_factor:.2f}" if not blocks_processing else "WLN=0, BLOCKING"
                  #  print(f"Time {self.env.now/480:.1f}: Machine {station_id} {status} - "
                   #     f"{effect_str} for {repair_time/60:.1f}h")
                
                # FIRE RELEASE TRIGGER - Wake up release process immediately
                self._trigger_release_wake_up("failure_start", station_id, status)
                
                # Wait for repair
                yield self.env.timeout(repair_time)
                
                # END FAILURE - Restore normal operation
                self.reliability_factors[station_id] = 1.0
                self.machine_available[station_id] = True
                self.machine_repair_time[station_id] = 0
                self.station_status[station_id] = "up"
                self.failure_in_progress[station_id] = False
                
                #if SCREEN_DEBUG:
                 #   print(f"Time {self.env.now/480:.1f}: Machine {station_id} repaired - normal operation")
                
                # FIRE RELEASE TRIGGER - Wake up release process immediately
                self._trigger_release_wake_up("repair_complete", station_id, "up")
        
        def _trigger_release_wake_up(self, event_type, station_id, status):
            """Triggers the order release process to re-evaluate releases.

        This method sends a signal to the `Orders_release` class, allowing it to
        respond immediately to changes in machine availability.

        Args:
            event_type (str): The type of event that triggered the wake-up.
            station_id (int): The ID of the station involved.
            status (str): The new status of the station.
        """
            """Fire the release trigger to wake up job release process"""
            if (self.system.release_trigger is not None and 
                not self.system.release_trigger.triggered):
                
                #if SCREEN_DEBUG:
                 #   print(f"Time {self.env.now/480:.1f}: Triggering release wake-up ({event_type} at station {station_id})")
                
                self.system.release_trigger.succeed()
        
        def get_station_reliability_factor(self, station_id):
            """Gets the current reliability factor for a station.

        This factor can be used to adjust the workload norm for the station.

        Args:
            station_id (int): The ID of the station.

        Returns:
            float: The reliability factor.
        """
            """Get dynamic WLN reliability factor"""
            return self.reliability_factors[station_id]
        
        def is_machine_available(self, station_id):
            """Checks if a machine is available for processing.

        Args:
            station_id (int): The ID of the machine.

        Returns:
            bool: True if the machine is available, False otherwise.
        """
            """Check if machine is available for processing (not broken down)"""
            return self.machine_available[station_id]
        
        def get_machine_repair_time(self, station_id):
            """Gets the scheduled repair time for a machine.

        Args:
            station_id (int): The ID of the machine.

        Returns:
            float: The simulation time when the machine is expected to be repaired.
        """
            """Get when machine will be repaired"""
            return self.machine_repair_time[station_id]

    """class StaticAbsenteeismManager:
        
        #Static absenteeism: Total percentage distributed across 3 human-machine stations as precautionary measure
       
        def __init__(self, env, absenteeism_level_key, system):
            self.env = env
            self.absenteeism_level_key = absenteeism_level_key
            self.system = system
            
            # Static WLN reduction
            self.station_wln_factors = [1.0] * N_PHASES
            
            # Static productivity factors
            self.station_productivity_factors = [1.0] * N_PHASES
            
            # Human-machine workstations (from your config: stations 2, 3, 4)
            self.human_stations = [i for i in range(N_PHASES) if HUMAN_MACHINE_PHASES.get(i, False)]
            
            # Apply static reduction based on level
            self._apply_static_reduction()

        def _apply_static_reduction(self):
            #Distribute total absenteeism across the 3 human-machine stations
            if self.absenteeism_level_key == "low":
                total_absenteeism = 0.03  # 3% total
            elif self.absenteeism_level_key == "medium":
                total_absenteeism = 0.06  # 6% total (FIXED: was 0.6)
            elif self.absenteeism_level_key == "high":
                total_absenteeism = 0.10  # 10% total
            else:
                total_absenteeism = 0.0
            
            if total_absenteeism > 0 and len(self.human_stations) > 0:
                # Distribute total absenteeism equally across human-machine stations
                absenteeism_per_station = total_absenteeism / len(self.human_stations)
                
                for station_id in self.human_stations:
                    # Apply WLN reduction
                    self.station_wln_factors[station_id] = 1.0 - absenteeism_per_station
                    
                    # Apply productivity loss
                    productivity_loss = self._calculate_productivity_loss(absenteeism_per_station)
                    self.station_productivity_factors[station_id] = 1.0 + productivity_loss
                    
                    if SCREEN_DEBUG:
                        print(f"Static absenteeism: Station {station_id} - "
                            f"WLN factor = {self.station_wln_factors[station_id]:.3f}, "
                            f"Productivity factor = {self.station_productivity_factors[station_id]:.3f} "
                            f"({absenteeism_per_station*100:.1f}% absenteeism)")
                
                if SCREEN_DEBUG:
                    print(f"Static absenteeism: Total {total_absenteeism*100:.0f}% distributed across {len(self.human_stations)} stations "
                        f"= {absenteeism_per_station*100:.1f}% per station")
        
        def _calculate_productivity_loss(self, absenteeism_pct):
            
            #Calculate productivity loss based on absenteeism percentage
            #Research shows productivity loss is often higher than absenteeism percentage
            #due to stress, overtime fatigue, and disrupted teamwork
            
            # Conservative approach: productivity loss = 1.3 * absenteeism rate
            # Based on van den Hout et al. (2023) and Pauly et al. (2017) research
            productivity_multiplier = 1.3  # Evidence-based multiplier
            
            return absenteeism_pct * productivity_multiplier

        def get_station_wln_factor(self, station_id):
            #Get static WLN factor
            return self.station_wln_factors[station_id]
        
        def get_station_productivity_factor(self, station_id):
            Get productivity factor for processing time calculations
            return self.station_productivity_factors[station_id]
    """

    class System(object):
        """Orchestrates the entire manufacturing simulation.

    This class serves as the central hub for the simulation, responsible for
    initializing and connecting all the other components, such as machines, workers,
    and job pools. It sets up the simulation environment and manages the main
    processes, including job generation and order release. The system can be
    configured with different levels of absenteeism and downtime to simulate
    various production scenarios.

    Attributes:
        env (simpy.Environment): The simulation environment.
        release_trigger (simpy.events.Event): An event used to trigger the order release process.
        downtime_manager (MachineDowntimeManager): The manager for machine downtimes.
        absenteeism_manager (DailyPlanningAbsenteeismManager): The manager for worker absenteeism.
        Pools (list): A list of all job pools.
        PSP (Pool): The pre-shop pool.
        Jobs_delivered (Pool): The pool for completed jobs.
        Machines (list): A list of all machine objects.
        Workers (list): A list of all worker objects.
        generator (Jobs_generator): The job generator.
        OR (Orders_release): The order release mechanism.
    """

        def __init__(self, env, absenteeism_level_key="none", downtime_level_key="none", 
                    machine_absenteeism_type="daily"):
            """Initializes the System object.

        This constructor sets up the entire simulation environment. It creates the
        necessary number of machines, workers, and job pools, and initializes the
        disruption managers for absenteeism and downtime. It also creates and starts
        the main system processes for job generation and order release.

        Args:
            env (simpy.Environment): The simulation environment.
            absenteeism_level_key (str, optional): The level of worker absenteeism.
                                                     Defaults to "none".
            downtime_level_key (str, optional): The level of machine downtime.
                                                  Defaults to "none".
            machine_absenteeism_type (str, optional): The type of absenteeism model
                                                        to use. Defaults to "daily".
        """
            
            self.env = env
            self.release_trigger = None  # Will be set by Orders_release

            # Initialize downtime manager
            self.downtime_manager = MachineDowntimeManager(env, downtime_level_key, self) if downtime_level_key != "none" else None

            # Initialize absenteeism manager in System class (consistent with original architecture)
            if absenteeism_level_key != "none":
                # HUMAN_CENTRIC always uses daily planning
                if RELEASE_RULE == "HUMAN_CENTRIC":
                    self.absenteeism_manager = DailyPlanningAbsenteeismManager(env, absenteeism_level_key, self)
                    
                # WL_DIRECT can use either approach
                elif RELEASE_RULE == "WL_DIRECT":
                    if MACHINE_ABSENTEEISM_TYPE == "static":
                        self.absenteeism_manager = StaticAbsenteeismManager(env, absenteeism_level_key, self)
           
                    else:  # "daily" or default
                        self.absenteeism_manager = DailyPlanningAbsenteeismManager(env, absenteeism_level_key, self)
                        
                # Default for other rules
                else:
                    self.absenteeism_manager = DailyPlanningAbsenteeismManager(env, absenteeism_level_key, self)
            else:
                self.absenteeism_manager = None

            # Pools setup
            self.Pools=list()   
            self.PSP = Pool(env, -1,self.Pools)
            
            for i in range(N_PHASES):
                self.Pools.append(Pool(env,i,self.Pools))
            
            self.Jobs_delivered = Pool(env,N_PHASES,self.Pools)
                
            # Map release rule to processing mode
            if RELEASE_RULE == "HUMAN_CENTRIC":
                processing_mode = 'human_centric'
            else:
                processing_mode = 'machine_centric'

            # MACHINES - Create based on HUMAN_MACHINE_PHASES
            self.Machines = list()

            for i in range(N_PHASES):
                has_worker = HUMAN_MACHINE_PHASES.get(i, False)
                
                self.Machines.append(Machine(
                    env, i, self.Pools[i], self.Jobs_delivered, self.Pools, self.PSP,
                    has_worker=has_worker,
                    processing_mode=processing_mode,
                    absenteeism_manager=self.absenteeism_manager,
                    downtime_manager=self.downtime_manager))

            # WORKERS
            self.Workers=list()
            for i in range(N_WORKERS):
                self.Workers.append(Worker(env, i, self.Pools, self.Machines, i))  

            # SET UP SYSTEM PROCESSES
            self.generator = Jobs_generator(env, self.PSP)
            # Orders_release gets the managers from system
            self.OR = Orders_release(env, RELEASE_RULE, self, WORKLOAD_NORMS[WLIndex])

 #####WORKLOAD COMPUTATION FUNCTIONS#####

    def get_direct_workload(pools):
        """Calculates the direct workload for each station.

    Direct workload is the sum of the processing times of the jobs currently
    waiting in the queue of each station. This provides a measure of the immediate
    workload at each machine.

    Args:
        pools (list): A list of job pools, one for each station.

    Returns:
        list: A list of the direct workload at each station.
    """

        # The result is a vector of workloads

        # each cell represent the direct load at each station

        direct_WL = list(0 for i in range(N_PHASES))

        for pool in pools:

            for job in pool:

                for i in range(N_PHASES):

                    if job.RemainingTime[i] > 0:

                        direct_WL[i] += job.RemainingTime[i]

                        break

        return direct_WL        

    def get_aggregated_workload(pools):
        """Calculates the aggregated workload for each station.

    Aggregated workload includes the direct workload at a station plus all the
    work that will eventually arrive at that station from jobs currently at
    upstream stations. This provides a more forward-looking measure of workload.

    Args:
        pools (list): A list of job pools.

    Returns:
        list: A list of the aggregated workload at each station.
    """

        # The result is a vector of workloads. The values represent the aggregated load of the stations (direct + indirect)

        # It does not depends on the job routing but only on the relmaining time 

        aggregated_WL = list(0 for i in range(N_PHASES))

        for pool in pools:

            for job in pool:

                for i in range(N_PHASES):

                    if job.RemainingTime[i] > 0:

                        aggregated_WL[i] += job.RemainingTime[i]

        return aggregated_WL 
        
    def get_corrected_aggregated_workload(pools):
        """Calculates the corrected aggregated workload (CAW).

    CAW adjusts the aggregated workload by considering the position of each job
    in its routing sequence. Work that is scheduled to be done sooner is given a
    higher weight, providing a more accurate picture of the imminent workload.

    Args:
        pools (list): A list of job pools.

    Returns:
        list: A list of the corrected aggregated workload at each station.
    """

        # The result is a vector of workloads. The values represent the aggregated load of the stations (direct + indirect)

        # corrected by the position of the job into the shop floor.

        aggregated_WL = list(0 for i in range(N_PHASES))

        for pool in pools:

            for job in pool:

                job_contribution = job.get_CAW()

                #print("CAW",job.get_CAW())

                for i in range(N_PHASES):

                    aggregated_WL[i] += job_contribution[i]

        #print()

        #print("pt",self.ProcessingTime )

        #

        #print(aggregated_WL)

        #time.sleep(5)

        return aggregated_WL         

    def get_shop_load(pools):
        """Calculates the total shop load.

    Shop load is the sum of all processing times for all jobs currently on the
    shop floor, regardless of their current location. This provides a high-level
    overview of the total amount of work in the system.

    Args:
        pools (list): A list of job pools.

    Returns:
        list: A list of the total shop load contributed by each station.
    """

        # The result is a vector of workloads. It considers also the contribution of processing times already processed of jobs

        # that have not been delivered yet

        shop_load = list(0 for i in range(N_PHASES))

        for pool in pools:

            for job in pool:

                for contribution_index in range(N_PHASES):

                    shop_load[contribution_index]+=job.ProcessingTime[contribution_index]        

        return shop_load   

    def get_shop_load2(pools):
        """Calculates the total shop load as a single value.

    This is an alternative method for calculating the shop load, which sums the
    `ShopLoad` attribute of each job to get a single aggregate value for the
    entire system.

    Args:
        pools (list): A list of job pools.

    Returns:
        float: The total shop load.
    """

        # The result is a vector of workloads. It considers also the contribution of task already processed of jobs

        # that have not been delivered yet

        shop_load = 0#list(0 for i in range(N_PHASES))

        for pool in pools:

            for job in pool:

                shop_load+=job.ShopLoad

        return shop_load 

    def get_corrected_shop_load(pools):
        """Calculates the corrected shop load (CSL).

    CSL distributes the total processing time of each job across the machines
    in its routing, providing a more balanced measure of shop load. This helps
    to avoid situations where a single long job can skew the workload perception.

    Args:
        pools (list): A list of job pools.

    Returns:
        list: A list of the corrected shop load at each station.
    """

        # The result is a vector of workloads. It considers also the contribution of task already processed of jobs

        # that have not been delivered yet

        shop_load = list(0 for i in range(N_PHASES))

        for pool in pools:

            for job in pool:

                job_CSL=job.get_CSL()

                for contribution_index in range(N_PHASES):

                    shop_load[contribution_index]+=job_CSL[contribution_index]

        return shop_load 

    def ResetStatistics(env, WARMUP,system):
        """Resets the simulation statistics after the warm-up period.

    This function is executed as a SimPy process and is scheduled to run after the
    specified warm-up period. It clears all statistical counters, such as the
    number of jobs processed and the workload processed, to ensure that the final
    results are not skewed by the initial transient phase of the simulation.

    Args:
        env (simpy.Environment): The simulation environment.
        WARMUP (int): The duration of the warm-up period.
        system (System): The main system object.

    Yields:
        simpy.events.Timeout: A timeout event that pauses the process until the
                              warm-up period is over.
    """

        """

        Machines:

        -> Set JobsProcessed to 0

        -> Set WorkloadProcessed to 0    

        Pools:

        -> Set Jobs_delivered to void 

        Workers:

        -> Set WorkloadProcessed to void 

        -> Set WorkingTime to void 

        """

        yield env.timeout(WARMUP)

        #for i in range(1000):

        #    print("RESETTING")

        in_pools = sum(len(pool) for pool in system.Pools)
        in_psp = len(system.PSP)
        in_proc = sum(1 for m in system.Machines if m.current_job is not None)
        
        system.generator.generated_orders = in_pools + in_psp + in_proc 

        for machine in system.Machines:

            machine.JobsProcessed=0

            machine.WorkloadProcessed=0.0

        global JOBS_WARMUP

        while len(system.Jobs_delivered)>0:

            system.Jobs_delivered.delete(0)

            JOBS_WARMUP += 1

        for worker in system.Workers:

            worker.WorkloadProcessed = list(0 for i in range(N_PHASES))

            worker.WorkingTime = list(0 for i in range(N_PHASES))

        system.OR.released_workload = list(0 for i in range(N_PHASES))

        system.generator.generated_processing_time = list(0 for i in range(N_PHASES))

        return

    def RunDebug5(env, run,system):
        """Collects and stores debug information at regular intervals during the simulation.

    This function runs as a SimPy process and periodically gathers a wide range of
    performance metrics, such as workload levels, job completion statistics, and
    worker utilization. The collected data is stored in a global list for later
    analysis and output to a CSV file. This is useful for tracking the system's
    behavior over time and identifying any trends or issues.

    Args:
        env (simpy.Environment): The simulation environment.
        run (int): The current simulation run number.
        system (System): The main system object.

    Yields:
        simpy.events.Timeout: Timeout events that schedule the data collection
                              at regular intervals.
    """

        if(WARMUP!=0):

            yield env.timeout(WARMUP)

            while(len(JOBS_DELIVERED_DEBUG)>0):

                del(JOBS_DELIVERED_DEBUG[0])

            while(len(JOBS_RELEASED_DEBUG)>0):

                del(JOBS_RELEASED_DEBUG[0])

            while(len(JOBS_ENTRY_DEBUG)>0):

                del(JOBS_ENTRY_DEBUG[0]) 

        yield env.timeout(480/2)

        global results_DEBUG

        results_index=0

        while(env.now<SIMULATION_LENGTH-TIME_BTW_DEBUGS):

            yield env.timeout(TIME_BTW_DEBUGS)

            units = 0
            if len(JOBS_DELIVERED_DEBUG) > 0:
                units = len(JOBS_DELIVERED_DEBUG)
            
            gtt = 0.0
            sft = 0.0
            tardiness = 0.0
            lateness = 0.0
            tardy = 0.0
            std_lateness = 0.0
            
            if units > 0:
                gtt = sum(job.get_GTT() for job in JOBS_DELIVERED_DEBUG) / units
                sft = sum(job.get_SFT() for job in JOBS_DELIVERED_DEBUG) / units
                tardiness = sum(job.get_tardiness() for job in JOBS_DELIVERED_DEBUG) / units
                lateness = sum(job.get_lateness() for job in JOBS_DELIVERED_DEBUG) / units
                tardy = sum(job.get_tardy() for job in JOBS_DELIVERED_DEBUG) / float(units)
                std_lateness = np.std(np.array(list(job.get_lateness() for job in JOBS_DELIVERED_DEBUG)))

            if run == 0:
                result = {
                    "time": (env.now - TIME_BTW_DEBUGS),
                    "WL entry": (sum(sum(job.ProcessingTime) for job in JOBS_ENTRY_DEBUG)),
                    "WL released": (sum(sum(job.ProcessingTime) for job in JOBS_RELEASED_DEBUG)),
                    "WL processed": (sum(sum(job.ProcessingTime) for job in JOBS_DELIVERED_DEBUG)),
                    "Jobs processed": units,
                    "GTT": gtt,
                    "SFT": sft,
                    "Tardiness": tardiness,
                    "Lateness": lateness,
                    "Tardy": tardy,
                    "STDLateness": std_lateness,
                }
                # Queues information
                result["PSP Shop Load"] = (sum(sum(job.ProcessingTime) for job in system.PSP))
                sl = get_shop_load(system.Pools)
                for i in range(N_PHASES):
                    result["Shop Load-" + str(i)] = sl[i]
                result["Total Shop Load"] = (sum(sl))
                # Workers information
                worker_total_working_time = list()
                for i in range(N_WORKERS):
                    if ((sum(system.Workers[i].WorkingTime)) > 0):
                        worker_total_working_time.append(sum(system.Workers[i].WorkingTime))
                    else:
                        worker_total_working_time.append(-1)
                for i in range(N_WORKERS):
                    result["Worker " + str(i) + " Idleness(%)"] = (env.now - WARMUP - worker_total_working_time[i]) / (
                                env.now - WARMUP) * 100
                    result["Total Idle Time"] = ((env.now - WARMUP) * 5 - sum(
                        worker_total_working_time[worker.id] for worker in system.Workers))
                # workers extra load %
                for i in range(N_WORKERS):
                    result["Worker " + str(i) + " out(%)"] = ((worker_total_working_time[i] -
                                                               system.Workers[i].WorkingTime[i]) /
                                                              worker_total_working_time[i]) * 100
                # workers relocations
                for i in range(N_WORKERS):
                    result["W" + str(i) + " Relocations"] = system.Workers[i].relocation[i]
                for i in range(N_PHASES):
                    result["Workers on M" + str(i)] = len(system.Machines[i].Workers)
                #Queues Length
                ql = get_direct_workload(system.Pools)
                for i in range(N_PHASES):
                    result["Queue Length-" + str(i)] = ql[i]
                results_DEBUG.append(result)
            else:
                results_DEBUG[results_index]["WL entry"] += (sum(sum(job.ProcessingTime) for job in JOBS_ENTRY_DEBUG))
                results_DEBUG[results_index]["WL released"] += (
                    sum(sum(job.ProcessingTime) for job in JOBS_RELEASED_DEBUG))
                results_DEBUG[results_index]["WL processed"] += (
                    sum(sum(job.ProcessingTime) for job in JOBS_DELIVERED_DEBUG))
                results_DEBUG[results_index]["Jobs processed"] += (units)
                results_DEBUG[results_index]["GTT"] += gtt
                results_DEBUG[results_index]["SFT"] += sft
                results_DEBUG[results_index]["Tardiness"] += tardiness
                results_DEBUG[results_index]["Lateness"] += lateness
                results_DEBUG[results_index]["Tardy"] += tardy
                results_DEBUG[results_index]["STDLateness"] += std_lateness
                # Queues information
                results_DEBUG[results_index]["PSP Shop Load"] += (sum(sum(job.ProcessingTime) for job in system.PSP))
                sl = get_shop_load(system.Pools)
                for i in range(N_PHASES):
                    results_DEBUG[results_index]["Shop Load-" + str(i)] += sl[i]
                results_DEBUG[results_index]["Total Shop Load"] += (sum(sl))
                # Workers information
                worker_total_working_time = list()
                for i in range(N_WORKERS):
                    if ((sum(system.Workers[i].WorkingTime)) > 0):
                        worker_total_working_time.append(sum(system.Workers[i].WorkingTime))
                    else:
                        worker_total_working_time.append(-1)
                for i in range(N_WORKERS):
                    results_DEBUG[results_index]["Worker " + str(i) + " Idleness(%)"] += (
                                env.now - WARMUP - worker_total_working_time[i]) / (env.now - WARMUP) * 100
                    # print ("Worker "+str(i)+" Idleness(%)", (env.now-WARMUP-worker_total_working_time[i])/(env.now-WARMUP)*100)
                    results_DEBUG[results_index]["Total Idle Time"] += (
                                (env.now - WARMUP) * 5 - sum(worker_total_working_time[worker.id] for worker in system.Workers))
                # workers extra load %
                for i in range(N_WORKERS):
                    results_DEBUG[results_index]["Worker " + str(i) + " out(%)"] += (
                                (worker_total_working_time[i] - system.Workers[i].WorkingTime[i]) /
                                worker_total_working_time[i]) * 100
                # workers relocations
                for i in range(N_WORKERS):
                    results_DEBUG[results_index]["W" + str(i) + " Relocations"] += system.Workers[i].relocation[i]

                results_index+=1

            while(len(JOBS_DELIVERED_DEBUG)>0):

                del(JOBS_DELIVERED_DEBUG[0])

            while(len(JOBS_RELEASED_DEBUG)>0):

                del(JOBS_RELEASED_DEBUG[0])

            while(len(JOBS_ENTRY_DEBUG)>0):

                del(JOBS_ENTRY_DEBUG[0])        

    def debug5_write():
        """Writes the collected debug data to a CSV file.

    This function takes the data gathered by `RunDebug5` and writes it to a CSV
    file. The file is named based on the current simulation parameters, allowing
    for easy identification and comparison of results from different scenarios.
    """

        global  results_DEBUG      

        # Record one performance over time 

        access_type='w'

        with open('daily_RUN_DEBUG5_'+RELEASE_RULE+"_"+WORKER_MODE+ "_" + WORKER_FLEXIBILITY +"_"+str(WORKLOAD_NORMS[WLIndex])+'.csv', access_type) as csvfile:

            fieldnames = [

            "RELEASE_RULE",

            "WORKER_MODE",

            "Workload",

            "time",

            "WL entry",

            "WL released",

            "WL processed",

            "Jobs processed",

            "GTT",

            "SFT",

            "Tardiness",

            "Lateness",

            "Tardy",

            "STDLateness",

            'Constraint Scenario',
            
            'Current Human Norm',
            
            'Current Machine Norm', 
            
            'Constraint Switches'
            ]

            # Queues information

            fieldnames.append("PSP Shop Load")

            for i in range(N_PHASES):

                fieldnames.append("Shop Load-"+str(i))

            fieldnames.append("Total Shop Load")

            # Workers information

            # worker idleness

            for i in range(N_WORKERS):

                fieldnames.append("Worker "+str(i)+" Idleness(%)")

            fieldnames.append("Total Idle Time")

            # workers extra load %

            for i in range(N_WORKERS):

                fieldnames.append("Worker "+str(i)+" out(%)")

            # workers relocations

            for i in range(N_WORKERS):

                fieldnames.append("W" + str(i) + " Relocations")

            # workers on each machine

            for i in range(N_PHASES):

                fieldnames.append("Workers on M" + str(i))

            #queues information

            for i in range(N_PHASES):

                fieldnames.append("Queue Length-" + str(i))

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #if(run==0):

            writer.writeheader()

            CURTIME = 0

            CURTIME += WARMUP 

            CURTIME += 480/2

            index=0

            while(CURTIME<SIMULATION_LENGTH-TIME_BTW_DEBUGS):

                #print(CURTIME,SIMULATION_LENGTH,TIME_BTW_DEBUGS)

                CURTIME+=TIME_BTW_DEBUGS

                N_SIMULATION_RUNS = LAST_RUN - FIRST_RUN

                #print(N_SIMULATION_RUNS)

                row = {}

                #print(results_DEBUG)

                row = {

                    "RELEASE_RULE":RELEASE_RULE,

                    "WORKER_MODE":WORKER_MODE,

                    "Workload":WORKLOAD_NORMS[WLIndex],

                    "time":results_DEBUG[index]["time"],

                    "WL entry":(results_DEBUG[index]["WL entry"]/N_SIMULATION_RUNS),

                    "WL released":(results_DEBUG[index]["WL released"]/N_SIMULATION_RUNS),

                    "WL processed":(results_DEBUG[index]["WL processed"]/N_SIMULATION_RUNS),

                    "Jobs processed":(results_DEBUG[index]["Jobs processed"]/N_SIMULATION_RUNS),

                    "GTT":(results_DEBUG[index]["GTT"]/N_SIMULATION_RUNS),

                    "SFT":(results_DEBUG[index]["SFT"]/N_SIMULATION_RUNS),

                    "Tardiness":(results_DEBUG[index]["Tardiness"]/N_SIMULATION_RUNS),

                    "Lateness":(results_DEBUG[index]["Lateness"]/N_SIMULATION_RUNS),

                    "Tardy":(results_DEBUG[index]["Tardy"]/N_SIMULATION_RUNS),

                    "STDLateness":(results_DEBUG[index]["STDLateness"]/N_SIMULATION_RUNS),

                }

                #print(results_DEBUG[index]["GTT"])

                # Queues information

                row["PSP Shop Load"]=(results_DEBUG[index]["PSP Shop Load"]/N_SIMULATION_RUNS)

                for i in range(N_PHASES):

                    row["Shop Load-"+str(i)]=(results_DEBUG[index]["Shop Load-"+str(i)]/N_SIMULATION_RUNS)

                row["Total Shop Load"]=(results_DEBUG[index]["Total Shop Load"]/N_SIMULATION_RUNS)

                # Workers information

                # worker idleness

                for i in range(N_WORKERS):

                    row["Worker "+str(i)+" Idleness(%)"]=(results_DEBUG[index]["Worker "+str(i)+" Idleness(%)"]/N_SIMULATION_RUNS)

                    row["Total Idle Time"]=(results_DEBUG[index]["Total Idle Time"]/N_SIMULATION_RUNS)

                # workers extra load %

                for i in range(N_WORKERS):

                    row["Worker "+str(i)+" out(%)"]=(results_DEBUG[index]["Worker "+str(i)+" out(%)"]/N_SIMULATION_RUNS)

                # workers relocation

                for i in range(N_WORKERS):

                    row["W" + str(i) + " Relocations"] = (

                            results_DEBUG[index]["W" + str(i) + " Relocations"] / N_SIMULATION_RUNS)

                # workers on each machine

                for i in range(N_PHASES):

                    row["Workers on M" + str(i)] = results_DEBUG[index]["Workers on M" + str(i)]

                #queues information

                for i in range(N_PHASES):

                    row["Queue Length-" + str(i)] = (results_DEBUG[index]["Queue Length-" + str(i)] / N_SIMULATION_RUNS)

                writer.writerow(row)

                index+=1

    def screenDebug(env, run, system ):
        """Prints a snapshot of the system's status to the console.

    This function is intended for debugging and monitoring purposes. It runs as a
    SimPy process and, at regular intervals, it clears the screen and prints a
    detailed summary of the current state of the simulation. This includes information
    about job pools, machine and worker utilization, and overall system performance.

    Args:
        env (simpy.Environment): The simulation environment.
        run (int): The current simulation run number.
        system (System): The main system object.

    Yields:
        simpy.events.Timeout: A timeout event to schedule the next screen update.
    """

        while 1:

            yield env.timeout(TIME_BTW_SCREEN_DEBUGS)

            FinishedUnits = -1

            if len(system.Jobs_delivered)>0:

                FinishedUnits=len(system.Jobs_delivered)

            print("\n" * 20) # Clear the screen

            #   run info

            total_simulations = (len(WORKLOAD_NORMS)*(LAST_RUN-FIRST_RUN))

            done_simulations = max(1,(WLIndex*(LAST_RUN-FIRST_RUN) + run))

            print ("### Day:%d - Rel.rule: %s - Work.Mode: %s - Run: %d - WLN:%d - runs: %d/%d \t remaining time:%d h\t%d h\t###\n"%(env.now/480,RELEASE_RULE,WORKER_MODE,run,WORKLOAD_NORMS[WLIndex],done_simulations, total_simulations, (total_simulations-done_simulations)*(time.time()-start)/done_simulations/60/60,(time.time()-start)/60/60))

            print("\n")

            #   pools info

            SL = get_shop_load(system.Pools)

            CAW = get_corrected_aggregated_workload(system.Pools)     

            print("PSP - %d jobs"%(len(system.PSP)))

            for index_pool in range(len(system.Pools)):

                print("Pool: %d \t %d jobs \t SL:%d \t CAW - %d"%(system.Pools[index_pool].id, len(system.Pools[index_pool]),SL[index_pool],CAW[index_pool]))

            print("Jobs_delivered - %d jobs\n"%(FinishedUnits))

            print((sum(sum(worker.WorkingTime) for worker in system.Workers))/(env.now*5))

            print()

            net_time = env.now
            if env.now > WARMUP:
                net_time -= WARMUP

            # Machines info
            for machine in system.Machines:
                print("Machine %d \t Jobs processed %d \t WL processed %f \t Eff. %f \t Utilization %f" % (
                machine.id, machine.JobsProcessed, machine.WorkloadProcessed, machine.efficiency,
                machine.WorkloadProcessed / net_time))
            print("\n")

            # Workers info
            for worker in system.Workers:
                sumworkingtime = sum(worker.WorkingTime)
                sWorkingTime = [str(round(wt / net_time * 100, 2)) for wt in worker.WorkingTime]
                print("Worker:%d \t Current machine:%d\t" % (worker.id, worker.current_machine_id) + "\t ".join(
                    sWorkingTime) + "\t Home:" + str(
                    round(worker.WorkingTime[worker.Default_machine] / net_time, 2)) + "\t Idle:" + str(
                    round((net_time - sumworkingtime) / float(net_time) * 100, 2)) + "\t% out: " + str(
                    round((sum(worker.WorkingTime) - worker.WorkingTime[worker.Default_machine]) / net_time * 100, 2)))
                if env.now > WARMUP:
                    print(sumworkingtime / net_time)

            if RELEASE_RULE == "AL_MOD":
                print("MAX extra cap per phase: ",
                      "\t".join(list(str(sum(worker.Capacity_adjustment[i] for worker in system.Workers)) for i in
                                     range(N_PHASES))))

            # System info
            print("\nAvg. Job generator output rate: %f" % (system.generator.generated_orders / float(net_time)))
            print("Avg. Jobs delivered rate[jobs/day]: %f" % (FinishedUnits / float(net_time)))
            print("Jobs delivered/Jobs generated: %f" % ((FinishedUnits) / float(system.generator.generated_orders)))

            #print("GTT", sum(job.get_GTT() for job in system.Jobs_delivered)/len(system.Jobs_delivered))

            #print("SFT", sum(job.get_SFT() for job in system.Jobs_delivered)/len(system.Jobs_delivered))

    def analyze_dual_constraint_results():
        """Analyzes the results of simulations with dual constraints.

    This function loads the output CSV files from the simulations, groups the data
    by release rule and constraint scenario, and then calculates summary statistics
    for key performance indicators like GTT, SFT, and tardiness. The results are
    printed to the console for easy comparison.

    Returns:
        pandas.DataFrame: A DataFrame containing the aggregated results.
    """
        # Load results from all your CSV files
        df_list = []
        
        # Read all SystemOutput CSV files
        for rule in ['HUMAN_CENTRIC', 'WL']:
            try:
                filename = f'SystemOutput_{rule}_static_False_flat_0.csv'
                temp_df = pd.read_csv(filename)
                df_list.append(temp_df)
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
        
        if not df_list:
            print("No CSV files found")
            return None
            
        df = pd.concat(df_list, ignore_index=True)
        
        # Analysis code from previous response...
        grouped = df.groupby(['Release rule', 'Constraint Scenario'])
        
        results_summary = grouped.agg({
            'Av. GTT': ['mean', 'std'],
            'Av. SFT': ['mean', 'std'], 
            'Tardy': ['mean', 'std']
        }).round(2)
        
        print("=== RELEASE RULE PERFORMANCE BY SCENARIO ===")
        print(results_summary)
        
        return df

    # MAIN

    for WLIndex in range(0,len(WORKLOAD_NORMS)):             

        for run in range(FIRST_RUN, LAST_RUN):

            np.random.seed(54363*run)

            env = simpy.Environment()

            system = System(env, ABSENTEEISM_LEVEL, DOWNTIME_LEVEL, MACHINE_ABSENTEEISM_TYPE)

            if WARMUP!=0:

                # Reset statistics after WARMUP time units

                env.process(ResetStatistics(env, WARMUP,system))    

            # <if run debug>

            if RUN_DEBUG:

                env.process(RunDebug5(env, run,system))

            # </>

            if SCREEN_DEBUG:

                env.process(screenDebug(env, run,system))            

            env.run(until = SIMULATION_LENGTH)

            FinishedUnits = -1

            if len(system.Jobs_delivered) > 0:

                FinishedUnits = len(system.Jobs_delivered)

            if len(system.Jobs_delivered) > 0:
                jobs = system.Jobs_delivered.get_list()
                net_time = env.now - WARMUP
                
                run_result = {
                    'GTT': sum(job.get_GTT() for job in jobs)/FinishedUnits,
                    'SFT': sum(job.get_SFT() for job in jobs)/FinishedUnits,
                    'Tardiness': sum(job.get_tardiness() for job in jobs)/FinishedUnits,
                    'Tardy': sum(job.get_tardy() for job in jobs)/float(FinishedUnits),
                    'WLN': WORKLOAD_NORMS[WLIndex],
                    'Release_Rule': RELEASE_RULE,
                    'Shop_Flow': SHOP_FLOW,
                    'Shop_Length': SHOP_LENGTH,
                    'Absenteeism_Level': ABSENTEEISM_LEVEL,
                    'Downtime_Level': DOWNTIME_LEVEL,
                    'Worker_Mode': WORKER_MODE,
                    'Starvation_Avoidance': STARVATION_AVOIDANCE,
                    'Worker_Flexibility': WORKER_FLEXIBILITY,
                    'Human_Variability': HUMAN_RATIO,
                }
                
                UNIFIED_RESULTS.append(run_result)
                    
            if CSV_OUTPUT_JOBS is True or CSV_OUTPUT_SYSTEM is True:

                access_type = 'w'

                if run > 0 or WLIndex > 0:

                    access_type = 'a'

                if CSV_OUTPUT_JOBS is True:

                    # write the infomation of all completed job

                    with open('JobsOutput_' + RELEASE_RULE + "_" + WORKER_MODE + "_" + WORKER_FLEXIBILITY + "_" + str(WORKLOAD_NORMS[WLIndex]) + "_" + str(run) + '.csv', access_type) as csvfile:

                        fieldnames = ['Workload','nrun','id', 'Arrival Date', 'Due Date', 'Completation Date', 'GTT','SFT','Tardiness','Lateness']

                        for i in range(N_PHASES):

                            fieldnames.append("PT("+str(i)+")")

                        for i in range(N_PHASES):

                            fieldnames.append("SFT(" + str(i) + ")")

                        for i in range(N_PHASES):

                            fieldnames.append("Lead Time(" + str(i) + ")")

                        for i in range(N_PHASES):

                            fieldnames.append("Arrival date Queue M(" + str(i) + ")")

                        for i in range(N_PHASES):

                            fieldnames.append("Arrival date on M(" + str(i) + ")")

                        for i in range(N_PHASES):

                            fieldnames.append("Completion date on M(" + str(i) + ")")

                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        if  access_type=='w':

                            writer.writeheader()

                        for job in system.Jobs_delivered.get_list():

                            row={

                                'Workload':WORKLOAD_NORMS[WLIndex],

                                'nrun':run,

                                'id':job.id, 

                                'Arrival Date':job.ArrivalDate, 

                                'Due Date':job.DueDate, 

                                'Completation Date':job.CompletationDate, 

                                'GTT':job.get_GTT(),

                                'SFT':job.get_SFT(),

                                'Tardiness':job.get_tardiness(),

                                'Lateness':job.get_lateness()

                                }

                            for i in range(N_PHASES):

                                row["PT("+str(i)+")"] = job.ProcessingTime[i]

                            SFT_each_Machine = job.get_SFT_Machines()

                            for i in range(N_PHASES):

                                row["SFT(" + str(i) + ")"] = SFT_each_Machine[i]

                            LT_each_Machine = job.get_LT_Machines()

                            for i in range(N_PHASES):

                                row["Lead Time(" + str(i) + ")"] = LT_each_Machine[i]

                            for i in range(N_PHASES):

                                row["Arrival date Queue M(" + str(i) + ")"] = job.ArrivalDateQueue[i]

                            for i in range(N_PHASES):

                                row["Arrival date on M(" + str(i) + ")"] = job.ArrivalDateMachines[i]

                            for i in range(N_PHASES):

                                row["Completion date on M(" + str(i) + ")"] = job.CompletationDateMachines[i]

                            writer.writerow(row)

                if CSV_OUTPUT_SYSTEM is True:
                    
                    #access_type = 'w' if run == 0 and CONSTRAINT_SCENARIO == "baseline" else 'a'
                    # write the final output of every simulation

                    #'SystemOutput_'+RELEASE_RULE+"_"+WORKER_MODE+"_"+str(WORKLOAD_NORMS[WLIndex])+'.csv'

                    with open('SystemOutput_' + RELEASE_RULE + "_" + SHOP_FLOW + "_" + str(SHOP_LENGTH) + "_" + ABSENTEEISM_LEVEL + "_" + DOWNTIME_LEVEL + "_" + str(HUMAN_RATIO) + "_" +
                    str(STARVATION_AVOIDANCE) + "_" + WORKER_FLEXIBILITY + "_" + 
                    str(WORKER_EFFICIENCY_DECREMENT) +  '.csv',
                    access_type) as csvfile:

                        fieldnames = [

                        # Run/system information

                        'Workload',

                        'Release rule',

                        'Worker mode',

                        'StarvationAvoidance',

                        'Shopflow',

                        'Shoplength',

                        'nrun', 

                        #'Job Entry',

                        'Exit Rate',

                        'Total Processed Workload',

                        #Jobs intormation

                        'Av. GTT','Av. SFT','Av. Tardiness','Av. Lateness','Tardy','STD Lateness',

                        'Constraint Scenario', 'Current Human Norm', 'Current Machine Norm', 'Constraint Switches'

                        ]

                        # Queues information

                        fieldnames.append("PSP Shop Load")

                        for i in range(N_PHASES):

                            fieldnames.append("Shop Load-"+str(i))

                        fieldnames.append("Total Shop Load")

                        # Workers information

                        # worker working time

                        for i in range(N_WORKERS):

                            for j in range(N_PHASES):   

                                fieldnames.append("W"+str(i)+"-M"+str(j))

                        # worker idleness

                        for i in range(N_WORKERS):

                            fieldnames.append("W"+str(i)+" Idleness")

                        # worker relocations

                        for i in range(N_WORKERS):

                            fieldnames.append("W" + str(i) + " Relocations")

                        # workers relocation to a specific machine

                        for i in range(N_WORKERS):

                            for j in range(N_PHASES):

                                fieldnames.append("Rel-W" + str(i) + "-M" + str(j))

                        # Machines

                        for i in range(N_PHASES):

                            fieldnames.append("Machine "+str(i)+" eff.(%)")

                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        if  access_type=='w':

                            writer.writeheader()

                        net_time=env.now-WARMUP

                        jobs = system.Jobs_delivered.get_list()

                        row = {

                            'Workload':WORKLOAD_NORMS[WLIndex],

                            'Release rule':RELEASE_RULE,

                            'Worker mode':WORKER_MODE,

                            'StarvationAvoidance':str(STARVATION_AVOIDANCE),

                            'Shopflow':SHOP_FLOW,

                            'Shoplength':str(SHOP_LENGTH),

                            'nrun':run,

                            #'Job Entry':((generator.generated_orders-len(Rejected_orders))/float(generator.generated_orders)),

                            'Exit Rate':(FinishedUnits/float(net_time)),

                            'Total Processed Workload':(sum(sum(job.ProcessingTime) for job in system.Jobs_delivered)),                    

                            'Av. GTT': (sum(job.get_GTT() for job in jobs)/FinishedUnits),

                            'Av. SFT': (sum(job.get_SFT() for job in jobs)/FinishedUnits),                

                            'Av. Tardiness':(sum(job.get_tardiness() for job in jobs)/FinishedUnits),

                            'Av. Lateness':(sum(job.get_lateness() for job in jobs)/FinishedUnits),

                            'Tardy':(sum(job.get_tardy() for job in jobs)/float(FinishedUnits)),

                            'STD Lateness':(np.std(np.array(list(job.get_lateness() for job in jobs)))),

                            }

                        # Queues information

                        sl = get_shop_load(system.Pools)

                        row["PSP Shop Load"]=(sum(sum(job.ProcessingTime) for job in system.PSP))

                        for i in range(N_PHASES):

                            row["Shop Load-"+str(i)]=sl[i]

                        row["Total Shop Load"]=(sum(sl))

                        # Workers information

                        for i in range(N_WORKERS):

                            for j in range(N_PHASES):   

                                row["W"+str(i)+"-M"+str(j)]=(system.Workers[i].WorkingTime[j])/(net_time)

                        # worker idleness

                        for i in range(N_WORKERS):

                            row["W"+str(i)+" Idleness"]=(net_time-sum(system.Workers[i].WorkingTime))/(net_time)

                        # worker total relocations

                        for i in range(N_WORKERS):

                            row["W" + str(i) + " Relocations"] = system.Workers[i].relocation[i]

                        # subdivision of workers relocations

                        for i in range(N_WORKERS):

                            for j in range(N_PHASES):

                                row["Rel-W" + str(i) + "-M" + str(j)] = system.Workers[i].relocation[j]

                        # Machines

                        for machine in system.Machines:

                            row["Machine "+str(machine.id)+" eff.(%)"]=(machine.WorkloadProcessed/(net_time))

                        writer.writerow(row)

        # <if run debug>

        if RUN_DEBUG:

            debug5_write()

            results_DEBUG=list()

        #

    print("\nEnd of the simulation")

    print("Simulation time: " + str(time.time()-start) + " sec")

def write_unified_results():
    """Writes a master summary file with the aggregated results of all simulations.

    This function is called at the end of all simulation runs. It takes the
    `UNIFIED_RESULTS` list, which contains the key performance indicators from
    each run, and calculates the average performance for each combination of
    simulation parameters. The results are then written to a master CSV file,
    providing a comprehensive overview of the entire experiment.
    """
    """Write the master unified summary file with ALL combinations"""
    if not UNIFIED_RESULTS:
        print("No unified results to write")
        return
        
    # Group results by parameter combination and calculate averages
    from collections import defaultdict
    grouped_results = defaultdict(list)
    
    for result in UNIFIED_RESULTS:
        key = (result['Release_Rule'], result['Shop_Flow'], result['Shop_Length'], 
               result['WLN'], result['Absenteeism_Level'], result['Downtime_Level'],
               result['Worker_Mode'], result['Starvation_Avoidance'], result['Worker_Flexibility'],result['Human_Variability'])
        grouped_results[key].append(result)
    
    filename = 'MASTER_UNIFIED_SUMMARY_ALL_COMBINATIONS.csv'
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Release_Rule', 'WLN', 
            'Shop_Flow', 'Shop_Length', 'Human_Variability', 
            'Absenteeism_Level', 'Downtime_Level', 'Worker_Flexibility',
            'Avg_GTT', 'Avg_SFT', 'Avg_Tardiness', 'Avg_Tardy_Percentage', 
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for key, runs in grouped_results.items():
            if runs:
                avg_row = {
                    'Release_Rule': runs[0]['Release_Rule'],
                    'WLN': runs[0]['WLN'],
                    'Shop_Flow': runs[0]['Shop_Flow'],
                    'Shop_Length': runs[0]['Shop_Length'],
                    'Human_Variability': runs[0]['Human_Variability'],  # ✅ FIXED
                    'Absenteeism_Level': runs[0]['Absenteeism_Level'],
                    'Downtime_Level': runs[0]['Downtime_Level'],
                    'Worker_Flexibility': runs[0]['Worker_Flexibility'],  # Also add this
                    'Avg_GTT': sum(r['GTT'] for r in runs) / len(runs),
                    'Avg_SFT': sum(r['SFT'] for r in runs) / len(runs),
                    'Avg_Tardiness': sum(r['Tardiness'] for r in runs) / len(runs),
                    'Avg_Tardy_Percentage': sum(r['Tardy'] for r in runs) / len(runs),
                }
                writer.writerow(avg_row)
    
    print(f"Master unified results written to {filename} with {len(grouped_results)} parameter combinations")

# Call the function
write_unified_results()
