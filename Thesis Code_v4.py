import random
import simpy
from simpy.events import AnyOf, AllOf, Event
import csv
import pulp
import math
import gurobipy
import time
import numpy as np
from itertools import permutations
import pandas as pd
from optimizer_v4 import run_wb_mod_optimizer

# =============================================================================
# # ORR PARAMETERS
# =============================================================================
# Defines the order release mechanism used in the simulation.
# Available options:
# "IM": Immediate Release - Jobs are released as soon as they arrive.
# "WL_DIRECT": Workload Limiting - Releases jobs only if workload norms are not exceeded.
# "HUMAN_CENTRIC": Human-Centric - Prioritizes jobs requiring human interaction.
# "WB_MOD": Workload Balancing (optimization-driven)
RELEASE_RULE="WB_MOD"
# If True, a starving machine can pull a job from the pre-shop pool, bypassing the release rule.
STARVATION_AVOIDANCE = False

# =============================================================================
# # WORKERS PARAMETERS
# =============================================================================
# Defines how workers behave and are managed in the system.
# "static": Workers are fixed to their primary station.
# "reactive": Workers can move to other stations based on simple rules (e.g., queue length).
# "flexible": Workers can be reallocated based on more complex logic (output control), often tied to WL_MOD/WB_MOD.
# "plan_following": Workers follow a centrally computed labor adjustment plan from WB_MOD.
WORKER_MODE = "plan_following"
# Defines the skill set of workers across different stations.
# "mono": Workers are skilled at only one station.
# "exponential": Skills decrease exponentially as the distance from the home station increases.
# "flat": Workers are skilled at all stations (with a possible efficiency decrement).
# "chain": Workers are skilled at their home station and the next one downstream.
# "chain upstream": Workers are skilled at their home station and the next one upstream.
# "triangular": Skills decrease as the distance from the home station increases.
WORKER_FLEXIBILITY  = "triangular"
# Efficiency loss when a worker operates at a non-primary station (e.g., 0.1 for 10% loss).
WORKER_EFFICIENCY_DECREMENT  = 0.25
# Time required for a worker to move from one station to another.
TRANSFER_TIME = 3
# Minimum time a worker must stay at a station after being transferred.
PERMANENCE_TIME = 60

# =============================================================================
# # SHOP PARAMETERS
# =============================================================================
# Defines the physical layout and flow of the job shop.
# "directed": Jobs follow a unidirectional flow (e.g., M1 -> M2 -> M3).
# "undirected": Jobs can move between any two machines (pure job shop).
SHOP_FLOW = "directed"
# Defines the number of operations in a job's routing.
# 5: All jobs have a fixed number of operations.
# "variable": The number of operations varies for each job.
SHOP_LENGTH = "variable"


# =============================================================================
# # JOBS PARAMETERS
# =============================================================================
# Parameters for the log-normal distribution of job processing times.
JOBS_MEAN = 30
JOBS_VARIANCE = 900
# Pre-calculated parameters for the log-normal distribution based on mean and variance.
JOBS_MU = np.log((JOBS_MEAN**2) / math.sqrt(JOBS_MEAN**2 + JOBS_VARIANCE))
JOBS_SIGMA = math.sqrt(np.log((JOBS_MEAN**2 + JOBS_VARIANCE) / (JOBS_MEAN**2)))

# =============================================================================
# # JOBS GENERATOR PARAMETERS
# =============================================================================
# The INPUT_RATE is calculated dynamically based on TARGET_UTILIZATION.
# It is initially set to 0 and should not be manually changed.
INPUT_RATE = 0
# The desired average utilization level of the machines in the shop.
# This is used to calculate the job arrival rate to load the system appropriately.
TARGET_UTILIZATION = 0.9375


# =============================================================================
# # JOBS RELEASE PARAMETERS
# =============================================================================
# Defines the level of worker absenteeism, can be set by a parameter sweep.
ABSENTEEISM_LEVEL = 0.0
# Dictionary defining different levels of absenteeism.
ABSENTEEISM_LEVELS = {
    "none": 0.0,
    "low": 0.05,     # 5% of the workforce is absent.
    "medium": 0.10,  # 10% of the workforce is absent.
    "high": 0.15     # 15% of the workforce is absent.
}

# =============================================================================
# # NEW PARAMETERS FOR WB_MOD, DYNAMIC CAPS, URGENT VALVE
# =============================================================================
# Horizon for the WB_MOD optimizer (e.g., 5-10 periods)
REVIEW_HORIZON_T = 5
# Parameters for the WB_MOD optimizer objective function
WB_MOD_PARAMS = {
    "rho": 1.5,             # Penalty for overload
    "kappa": 0.1,           # Penalty for labor adjustments
    "weights_decay": 0.5    # Geometric decay for time-weighted objective
}
# Enable/disable dynamic capacity adjustments
DYNAMIC_CAPS_ENABLED = False
# Enable/disable the urgent job release valve
URGENT_VALVE_ENABLED = False

# --- WB_MOD Feature Flags ---
WB_MOD_USE_OPTIMIZER = True
WB_MOD_MAX_PSP = 20          # Max jobs to consider in optimizer for speed
WB_MOD_SOLVER_TIMEOUT = 10   # Seconds

### USE IF DYNAMIC SYSTEM ###


# DUAL WORKLOAD NORM PARAMETERS
CONSTRAINT_SCENARIOS = {
    "baseline": {"type": "static", "human_norm": 1000, "machine_norm": 1000},
    "human_constrained": {"type": "static", "human_norm": 800, "machine_norm": 1200},
    "machine_constrained": {"type": "static", "human_norm": 1200, "machine_norm": 800},
    "dynamic_switching": {"type": "dynamic", "base_human": 1000, "base_machine": 1000, "switch_frequency": 960}  # Every 2 days
}

# Enhanced parameter structure
PAR1 = [
    # [Release_Rule, Worker_Mode, Starvation_Avoidance, Constraint_Scenario, Absenteeism_Level]
    # ["HUMAN_CENTRIC", "reactive", False, "baseline", "high"],
    # ["HUMAN_CENTRIC", "reactive", False, "human_constrained", "high"],
    # ["HUMAN_CENTRIC", "reactive", False, "dynamic_switching", "high"],
    ["WB_MOD", "plan_following", False, "baseline", "high"],
    ["WB_MOD", "plan_following", False, "human_constrained", "high"],
    ["WB_MOD", "plan_following", False, "dynamic_switching", "high"],
    # ["WL_DIRECT", "static", False, "baseline", "none"],
    # ["WL_DIRECT", "static", False, "human_constrained", "none"],
    # ["WL_DIRECT", "static", False, "dynamic_switching", "none"],
]

# Global variables for current constraints
CURRENT_HUMAN_NORM = 1000
CURRENT_MACHINE_NORM = 1000
CONSTRAINT_SCENARIO = "dynamic_switching"
"""
# =============================================================================
# # SIMULATION CONFIGURATION
# =============================================================================
# This section defines the different simulation scenarios to be run.
# PAR1 defines the core settings for each simulation run.
PAR1=[
    # algorith, workforce, starv av.
    ["IM","static", False],
    ["HUMAN_CENTRIC","static", False],
    ["WL_DIRECT", "static", False],
    ]
"""
# PAR2 expands on PAR1 by adding shop flow and length configurations to each scenario.
PAR2 = []
for config in PAR1:
    temp = config + [SHOP_FLOW, SHOP_LENGTH]
    PAR2.append(temp)

for config in PAR2:
    # Single, consistent mapping from config to named fields
    RELEASE_RULE = config[0]
    STARVATION_AVOIDANCE = config[2] 
    WORKER_MODE = config[1]
    SHOP_FLOW = config[3]
    SHOP_LENGTH = config[4]

    ### USE IF DYNAMIC SYSTEM ###

    CONSTRAINT_SCENARIO = config[3] 
    ABSENTEEISM_LEVEL = ABSENTEEISM_LEVELS[config[4]]  
    SHOP_FLOW=config[5]
    SHOP_LENGTH=config[6]

    # Log the parsed scenario for verification
    print("\n" + "="*80)
    print(f"PARSING SCENARIO CONFIG: \n"
          f"  Release Rule: {RELEASE_RULE}\n"
          f"  Worker Mode: {WORKER_MODE}\n"
          f"  Starvation Avoidance: {STARVATION_AVOIDANCE}\n"
          f"  Constraint Scenario: {CONSTRAINT_SCENARIO}\n"
          f"  Absenteeism Level: {ABSENTEEISM_LEVEL}\n"
          f"  Shop Flow: {SHOP_FLOW}\n"
          f"  Shop Length: {SHOP_LENGTH}")
    print("="*80)
    
    # Set initial constraint values
    scenario_config = CONSTRAINT_SCENARIOS[CONSTRAINT_SCENARIO]
    if scenario_config["type"] == "static":
        CURRENT_HUMAN_NORM = scenario_config["human_norm"]
        CURRENT_MACHINE_NORM = scenario_config["machine_norm"]
    else:
        CURRENT_HUMAN_NORM = scenario_config["base_human"]
        CURRENT_MACHINE_NORM = scenario_config["base_machine"]

    WORKLOAD_NORMS = [1000]  # Single value for dual constraint system
    WLIndex = 0

    if   RELEASE_RULE=="PR" or RELEASE_RULE=="IM":
        WORKLOAD_NORMS=[0]
    else:
        if SHOP_FLOW=="directed" and SHOP_LENGTH==5:
            WORKLOAD_NORMS = [2700, 3000, 3600, 4800, 6600]
        elif SHOP_FLOW=="directed" and SHOP_LENGTH=="variable":
            WORKLOAD_NORMS=[950,1000,1050,1100,1200,2000]
            #WORKLOAD_NORMS=[1000]
            #WORKLOAD_NORMS = [2700, 3000, 3600, 4800, 9600]
        elif SHOP_FLOW=="undirected" and SHOP_LENGTH==5:
            WORKLOAD_NORMS = [5400, 6600, 7800, 9600, 28800]
        elif SHOP_FLOW=="undirected" and SHOP_LENGTH=="variable":
            WORKLOAD_NORMS = [2700, 3600, 4800, 7200, 28800]

    N_WORKERS = 5

    N_PHASES = 5                # Machines/Workers

    HUMAN_MACHINE_PHASES = {
        0: False,  # Machine 1 - Pure machine station
        1: False,  # Machine 2 - Pure machine station
        2: True,   # Machine 3 - Human-machine collaborative station
        3: True,   # Machine 4 - Human-machine collaborative station
        4: True    # Machine 5 - Human-machine collaborative station
    }

    # Human parameters for H-M stations
    HUMAN_VARIABILITY = 1.0  # Start with no variability, can increase later (1.025, 1.05, etc.)
    HUMAN_INTERACTION_RANGE = (0, 1)  # Range of human involvement in H-M stations
    MACHINE_RATIO = 0.5  # 60% of station capacity for machines at H-M stations
    HUMAN_RATIO = 1 - MACHINE_RATIO  # 40% of station capacity for humans at H-M stations

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

        INPUT_RATE = (480*N_PHASES*TARGET_UTILIZATION) / (JOBS_MEAN*(JOBS_ROUTINGS_AVG_MACHINES))/480

        print("INPUT RATE", INPUT_RATE)
        print(INPUT_RATE*480, "jobs per day")
        time.sleep(2)
    getShopConfigurations()

    if (SHOP_FLOW=="directed" and SHOP_LENGTH=="variable"):
        DUE_DATE_MAX = 2250
    elif (SHOP_FLOW=="directed" and SHOP_LENGTH==5):
        DUE_DATE_MAX = 3600
    elif (SHOP_FLOW=="undirected" and SHOP_LENGTH==5):
        DUE_DATE_MAX = 3450
    elif (SHOP_FLOW=="undirected" and SHOP_LENGTH=="variable"):
        DUE_DATE_MAX = 3150
    else:
        print("shop configuration not recognised. Setiing an aribitrary due date between 2000 and 4000 mins")
        time.sleep(10)
        DUE_DATE_MIN = 2000
        DUE_DATE_MAX = 4000

    class Job(object):
        """
        Represents a single job that moves through the shop floor.
        Each job has a unique ID, a specific routing through the machines (phases),
        and processing times for each operation. It also tracks various time-related
        metrics like arrival, release, and completion dates.
        For human-centric simulations, it tracks the degree of human interaction
        required at each phase and calculates separate workloads for humans and machines.
        Attributes:
            id (int): A unique identifier for the job.
            ArrivalDate (float): The simulation time when the job arrived in the system (at the PSP).
            ReleaseDate (float): The simulation time when the job is released to the shop floor.
            CompletationDate (float): The simulation time when the job finishes its last operation.
            Routing (list): A sequence of machine IDs representing the job's path.
            ProcessingTime (list): The processing time required at each machine in its routing.
            HumanInteraction (dict): A dictionary mapping phase ID to the percentage of human
                                     involvement (0.0 to 1.0) for that phase.
            TotalHumanWork (float): The total processing time allocated to humans for this job.
            TotalMachineWork (float): The total processing time allocated to machines for this job.
            is_collaborative (bool): True if the job requires human interaction at any H-M station.
            RemainingTime (list): The remaining processing time at each phase.
            Position (int): The current step in the job's routing.
            ... and other statistical tracking attributes.
        """
        def __init__(self,env, id):
            #self.env = env
            global DUE_DATE_MIN
            global DUE_DATE_MAX

            self.id = id
            self.ArrivalDate = env.now
            self.ReleaseDate = None
            self.CompletationDate = None
            self.ArrivalDateMachines = list(0 for i in range(N_PHASES))
            self.CompletationDateMachines = list(0 for i in range(N_PHASES))
            self.ArrivalDateQueue = list(0 for i in range(N_PHASES))

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

            DUE_DATE_MIN = 83.686*len(self.Routing)
            self.DueDate = self.ArrivalDate + np.random.uniform(DUE_DATE_MIN,DUE_DATE_MAX)

            #Processing Time
            self.ProcessingTime = list(0 for i in range(N_PHASES))
            for i in self.Routing:
                x = np.random.lognormal(JOBS_MU, JOBS_SIGMA)
                while x>360:
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
            self.force_released = False

        # ... rest of your methods remain the same ...
        def get_current_machine(self):
            if self.Position >= len(self.Routing):
                self.Position = len(self.Routing) - 1
            return(self.Routing[self.Position])

        def get_CAW(self):
            """Calculates the Contribution to the Corrected Aggregated Workload (CAW)."""
            load = [0] * N_PHASES
            weight = 1
            for i in range(self.Position, len(self.Routing)):
                load[self.Routing[i]] = self.RemainingTime[self.Routing[i]] / weight
                weight += 1
            return load

        def get_CSL(self):
            """Calculates the Contribution to the Corrected Shop Load (CSL)."""
            load = [0] * N_PHASES
            for i in range(N_PHASES):
                load[i] = self.ProcessingTime[i] / len(self.Routing)
            return load

        def is_collaborative(self):
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
            """
            LUMS-COR compliant: Contribution to corrected aggregated workload
            Only considers stations in the job's actual routing with position correction
            """
            load = list(0 for i in range(N_PHASES))

            for pos_in_routing in range(self.Position, len(self.Routing)):
                station_id = self.Routing[pos_in_routing]
                # Position correction: divide by (position in remaining routing + 1)
                position_correction = pos_in_routing - self.Position + 1
                load[station_id] = self.RemainingTime[station_id] / position_correction

            return load

        def get_CSL_routing_only(self):
            """
            LUMS-COR compliant: Contribution to corrected shop load
            Only considers stations in the job's actual routing
            """
            load = list(0 for i in range(N_PHASES))

            for station_id in self.Routing:
                load[station_id] = self.ProcessingTime[station_id] / len(self.Routing)

            return load

        def get_CSL_with_ratios_routing_only(self):
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

        def get_dual_CSL_routing_only(self):
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

            """ total_delivery_time """
            if self.__GTT__ == None:
                self.__GTT__ = self.CompletationDate - self.ArrivalDate
            return self.__GTT__

        def get_SFT(self):
            """total_manufacturing_lead_time"""
            if self.__SFT__ == None:
                self.__SFT__= self.CompletationDate - self.ReleaseDate
            return self.__SFT__

        def get_SFT_Machines(self):
            '''array with the SFT on each machine'''
            SFT_Machines = list(0 for i in range(N_PHASES))
            for i in range(N_PHASES):
                SFT_Machines[i] = self.CompletationDateMachines[i] - self.ArrivalDateMachines[i]
            return SFT_Machines

        def get_LT_Machines(self):
            '''array with the SFT on each machine'''
            LT_Machines = list(0 for i in range(N_PHASES))
            for i in range(N_PHASES):
                LT_Machines[i] = self.CompletationDateMachines[i] - self.ArrivalDateQueue[i]
            return LT_Machines

        def get_tardy(self):
            if self.__tardy__ == None:
                if self.DueDate < self.CompletationDate:
                    self.__tardy__=  1

                else:
                    self.__tardy__=  0
            return self.__tardy__

        def get_tardiness(self):
            if self.__tardiness__ == None:
                self.__tardiness__= max(0, self.CompletationDate - self.DueDate)
            return self.__tardiness__

        def get_lateness(self):
            if self.__lateness__ == None:
                self.__lateness__= self.CompletationDate-self.DueDate
            return self.__lateness__



    class Jobs_generator(object):
        """
        Generates jobs and introduces them into the system.
        This class is responsible for creating new `Job` objects at a specified rate
        and placing them into the initial Pre-Shop Pool (PSP), from where they await
        release to the main shop floor. It models the arrival of new orders.
        Attributes:
            env (simpy.Environment): The simulation environment.
            PoolDownstream (Pool): The Pre-Shop Pool where newly generated jobs are placed.
            input_rate (float): The rate at which jobs arrive (jobs per time unit).
            generated_orders (int): A counter for the total number of jobs created.
        """

        def __init__(self, env, PoolDownstream):
            """
            Initializes the job generator.
            Args:
                env (simpy.Environment): The simulation environment.
                PoolDownstream (Pool): The pool to which new jobs will be added (typically the PSP).
            """
            self.env = env
            self.PoolDownstream = PoolDownstream
            self.input_rate = INPUT_RATE
            self.generated_orders = 0

            # Start the continuous job generation process.
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
            """
            A SimPy process that continuously creates new jobs.
            This generator runs in an infinite loop, creating a new job, adding it
            to the downstream pool, and then waiting for a period of time determined
            by an exponential distribution based on the `input_rate`. This models
            a Poisson arrival process.
            """
            while True:
                # Create a new job instance with a unique ID.
                job = Job(self.env, self.generated_orders + 1)

                # Add the new job to the Pre-Shop Pool.
                self.PoolDownstream.append(job)

                # For debugging purposes, track jobs as they enter the system.
                if RUN_DEBUG:
                    global JOBS_ENTRY_DEBUG
                    JOBS_ENTRY_DEBUG.append(job)

                # Increment the count of generated jobs.
                self.generated_orders += 1

                # Wait for the next job arrival, following an exponential distribution.
                yield env.timeout(np.random.exponential(1/self.input_rate))

    ###JOBS###

    def get_corrected_shop_load_with_ratios(pools):
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

    def get_corrected_aggregated_workload_routing_only(pools):
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

    class Orders_release(object):
        """
        Manages the release of jobs from the Pre-Shop Pool (PSP) to the shop floor.

        This class acts as a dispatcher, implementing various order release mechanisms
        based on the global `RELEASE_RULE` configuration. It contains a collection of
        SimPy processes, each representing a different release strategy, from simple
        immediate or periodic releases to complex workload-based and optimization models.

        The `__init__` method functions as a router, starting the appropriate process
        for the selected release rule.
        """
        def __init__(self, env, rule, system, workload_norm=-1):
            """
            Initializes the order release mechanism.

            Args:
                env (simpy.Environment): The simulation environment.
                rule (str): The release rule to be used (e.g., "IM", "PR", "WL_HUMAN").
                system (System): A reference to the main system object.
                workload_norm (float): The workload norm parameter, used by WL-based rules.
            """
            self.env = env
            # --- References to other system components ---
            self.PoolUpstream = system.PSP  # The Pre-Shop Pool where jobs are waiting.
            self.Pools = system.Pools       # The WIP pools for the first machines.
            self.system = system

            self.released_workload = list(0 for i in range(N_PHASES))
            self.Forced_releases_count = 0

            # --- Router for selecting the release algorithm ---
            # Based on the 'rule' parameter, the corresponding SimPy process is started.
            if rule == "IM":
                self.env.process(self.Immediate_release())
            elif rule == "WL_DIRECT":
                if SHOP_FLOW == "undirected":
                    self.env.process(self.WL_Direct_release(480, workload_norm))
                elif SHOP_FLOW == "directed" and ABSENTEEISM_LEVELS==True:
                    self.env.process(self.WL_release_dual(480))
                else:
                    self.env.process(self.WL_Direct_release_directed(480, workload_norm))
            elif rule == "HUMAN_CENTRIC":
                if SHOP_FLOW == "undirected":
                    self.env.process(self.Human_Centric_release(480, workload_norm))
                elif SHOP_FLOW == "directed":
                    self.env.process(self.Human_Centric_release_directed(480,workload_norm))
            elif rule == "WB_MOD":
                self.env.process(self.WB_MOD_release(480)) # Period can be configured
            else:
                print("Release algorithm not recognised in Orders_release")
                exit()

        def Immediate_release(self):
            """
            Releases jobs from the PSP as soon as they arrive.

            This is a continuous process that checks for jobs in the Pre-Shop Pool.
            If any jobs are found, it immediately gets them, sets their release date,
            and moves them to the WIP pool of their first machine. It then waits
            for a signal that a new job has arrived in the PSP.
            """
            while True:
                # Process all jobs currently in the upstream pool.
                while len(self.PoolUpstream) > 0:
                    job = self.PoolUpstream.get()
                    job.ReleaseDate = self.env.now

                    # For data analysis, mark if the job's routing includes any H-M stations.
                    job.uses_hm_stations = any(
                        HUMAN_MACHINE_PHASES.get(station, False)
                        for station in job.Routing
                    )

                    # Append the job to the queue of its first designated machine.
                    self.Pools[job.Routing[0]].append(job)

                # After clearing the pool, wait for a new job to be generated.
                waiting_new_jobs_event = self.env.event()
                self.PoolUpstream.waiting_new_jobs.append(waiting_new_jobs_event)
                yield waiting_new_jobs_event

        def _urgent_job_valve(self):
            if not URGENT_VALVE_ENABLED:
                return

            # Create a temporary list to iterate over, as we will modify the pool
            unreleased_jobs = list(self.PoolUpstream.get_list())

            # Use a list to store jobs that will be force-released to avoid modifying the list while iterating
            jobs_to_force_release = []

            for job in unreleased_jobs:
                # Trigger: PRD (DueDate) is in the past or present
                if job.DueDate <= self.env.now:
                    # Policy: Station-idle valve
                    first_station_id = job.get_current_machine()
                    first_station = self.system.Machines[first_station_id]

                    # A station is idle if it has no job being processed and its WIP queue is empty
                    if first_station.current_job is None and len(first_station.PoolUpstream) == 0:
                        jobs_to_force_release.append(job)

            # Now, iterate through the jobs marked for force-release
            for job in jobs_to_force_release:
                # Remove the job from the upstream pool
                # This is inefficient, but necessary as we don't have job IDs in the pool
                for i, pool_job in enumerate(self.PoolUpstream.array):
                    if pool_job.id == job.id:
                        self.PoolUpstream.delete(i)
                        break

                # Release the job
                job.ReleaseDate = self.env.now
                job.force_released = True # Tag the job
                self.Forced_releases_count += 1
                self.Pools[job.get_current_machine()].append(job)

                if SCREEN_DEBUG:
                    print(f"URGENT VALVE: Force-releasing job {job.id} at time {self.env.now:.2f}")

        def WL_Direct_release(self, period, workload_norm):
            """
            WL_Direct rule: LUMS-COR compliant with routing-only loads and PRD sorting
            """
            def evaluateJob(self, job, phases_load, system):
                job_CAW = job.get_CAW_routing_only()

                # Check only stations in the job's routing
                for station_id in job.Routing:
                    # Apply machine ratio for human-machine stations
                    load_to_compare = job_CAW[station_id]
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        load_to_compare *= MACHINE_RATIO
                    if DYNAMIC_CAPS_ENABLED:
                        limit = system.Cap_mach[station_id]
                    else:
                        limit = workload_norm / N_PHASES
                    if (phases_load[station_id] + load_to_compare > limit):
                        return False

                job.ReleaseDate = self.env.now

                # Only update workload for stations in the job's routing
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

                jobs_to_evaluate.sort(key=lambda x: x.DueDate)

                # Process jobs in PRD order
                for job in jobs_to_evaluate:
                    if not evaluateJob(self, job, phases_load,self.system):
                        self.PoolUpstream.append(job)

                self._urgent_job_valve()
                yield env.timeout(period)

        def WL_Direct_release_directed(self, period, workload_norm):
            """
            WL_Direct rule: LUMS-COR compliant for directed flow
            """
            def evaluateJob(self, job, phases_load, system):
                job_CSL = job.get_CSL_routing_only()

                # Check only stations in the job's routing
                for station_id in job.Routing:
                    # Apply machine ratio for human-machine stations
                    load_to_compare = job_CSL[station_id]
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        load_to_compare *= MACHINE_RATIO

                    if DYNAMIC_CAPS_ENABLED:
                        limit = system.Cap_mach[station_id]
                    else:
                        limit = workload_norm / N_PHASES

                    if (phases_load[station_id] + load_to_compare > limit):
                        return False

                job.ReleaseDate = self.env.now

                # Only update workload for stations in the job's routing
                for station_id in job.Routing:
                    phases_load[station_id] += job_CSL[station_id]

                self.Pools[job.Routing[0]].append(job)

                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG
                    JOBS_RELEASED_DEBUG.append(job)

                return True

            while True:
                phases_load = get_corrected_shop_load_routing_only(self.Pools)

                # Sort PSP by Due Date
                jobs_to_evaluate = []
                for i in range(len(self.PoolUpstream)):
                    jobs_to_evaluate.append(self.PoolUpstream.get(0))

                jobs_to_evaluate.sort(key=lambda x: x.DueDate)

                # Process jobs in PRD order
                for job in jobs_to_evaluate:
                    if not evaluateJob(self, job, phases_load, self.system):
                        self.PoolUpstream.append(job)

                self._urgent_job_valve()

                yield env.timeout(period)

        def WL_release_dual(self, period):

            ### USE IF DYNAMIC SYSTEM ###

            def evaluateJob(self, job, machine_phases_load, human_phases_load):
                job_machine_load, job_human_load = job.get_dual_CSL_routing_only()

                global CURRENT_HUMAN_NORM, CURRENT_MACHINE_NORM

                for station_id in job.Routing:
                    # Check machine constraint
                    if (machine_phases_load[station_id] + job_machine_load[station_id] >
                        CURRENT_MACHINE_NORM/N_PHASES):
                        return False

                    # Check human constraint for human-machine stations
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        if (human_phases_load[station_id] + job_human_load[station_id] >
                            CURRENT_HUMAN_NORM/N_PHASES):
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

            while True:
                machine_phases_load, human_phases_load = get_corrected_shop_load_with_ratios(self.Pools)

                # Process jobs in arrival order
                for _ in range(len(self.PoolUpstream)):
                    temp_job = self.PoolUpstream.get()
                    if not evaluateJob(self, temp_job, machine_phases_load, human_phases_load):
                        self.PoolUpstream.append(temp_job)

                yield env.timeout(period)

        def Human_Centric_release(self, period, workload_norm):
            ### USE IF DYNAMIC SYSTEM ###
            """
            LUMS-COR Human Centric rule: Two-stage gating sequence
            Stage 1: Release collaborative jobs with human workload limits
            Stage 2: Release independent jobs with machine workload limits (human loads fixed)
            """
            def evaluateJobStage1(self, job, machine_phases_load, human_phases_load, system):
                """Stage 1: Check human workload constraints for collaborative jobs"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()

                for station_id in job.Routing:
                    if DYNAMIC_CAPS_ENABLED:
                        limit = system.Cap_mach[station_id]
                    else:
                        limit = workload_norm / N_PHASES
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > limit):
                        return False


                # Check human load constraints for human-machine stations ONLY
                for station_id in job.Routing:
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        if DYNAMIC_CAPS_ENABLED:
                            limit = system.Cap_hum[station_id]
                        else:
                            limit = workload_norm / N_PHASES
                        if (human_phases_load[station_id] + job_human_load[station_id] > limit):
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

            def evaluateJobStage2(self, job, machine_phases_load, human_phases_load, system):
                """Stage 2: Check machine workload constraints for independent jobs (human loads fixed)"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()

                # Check machine load constraints for all stations in routing
                for station_id in job.Routing:
                    if DYNAMIC_CAPS_ENABLED:
                        limit = system.Cap_mach[station_id]
                    else:
                        limit = workload_norm / N_PHASES
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > limit):
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

                # Sort PSP by Due Date
                all_jobs = []
                for i in range(len(self.PoolUpstream)):
                    all_jobs.append(self.PoolUpstream.get(0))

                all_jobs.sort(key=lambda x: x.DueDate)

                # Separate collaborative and independent jobs while maintaining due date order
                collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                independent_jobs = [job for job in all_jobs if not job.is_collaborative()]

                # STAGE 1: Process collaborative jobs with human workload constraints
                remaining_collaborative = []
                for job in collaborative_jobs:
                    if not evaluateJobStage1(self, job, machine_phases_load, human_phases_load, self.system):
                        remaining_collaborative.append(job)

                # STAGE 2: With human loads fixed, process independent jobs with machine constraints
                remaining_independent = []
                for job in independent_jobs:
                    if not evaluateJobStage2(self, job, machine_phases_load, human_phases_load, self.system):
                        remaining_independent.append(job)

                # Return unreleased jobs to pool (maintain PRD order)
                for job in remaining_collaborative + remaining_independent:
                    self.PoolUpstream.append(job)

                self._urgent_job_valve()
                yield env.timeout(period)

        def Human_Centric_release_directed(self, period, workload_norm):
            """
            LUMS-COR Human Centric rule for directed flow: Two-stage gating sequence
            Stage 1: Release collaborative jobs with human workload limits
            Stage 2: Release independent jobs with machine workload limits (human loads fixed)
            """
            def evaluateJobStage1(self, job, machine_phases_load, human_phases_load, system):
                """Stage 1: Check human workload constraints for collaborative jobs"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()

                for station_id in job.Routing:
                    if DYNAMIC_CAPS_ENABLED:
                        limit = system.Cap_mach[station_id]
                    else:
                        limit = workload_norm / N_PHASES
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > limit):
                        return False

                # Check human load constraints for human-machine stations ONLY
                for station_id in job.Routing:
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        if DYNAMIC_CAPS_ENABLED:
                            limit = system.Cap_hum[station_id]
                        else:
                            limit = workload_norm / N_PHASES
                        if (human_phases_load[station_id] + job_human_load[station_id] > limit):
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

            def evaluateJobStage2(self, job, machine_phases_load, human_phases_load, system):
                """Stage 2: Check machine workload constraints for independent jobs (human loads fixed)"""
                job_machine_load, job_human_load = job.get_CSL_with_ratios_routing_only()

                # Check machine load constraints for all stations in routing
                for station_id in job.Routing:
                    if DYNAMIC_CAPS_ENABLED:
                        limit = system.Cap_mach[station_id]
                    else:
                        limit = workload_norm / N_PHASES
                    if (machine_phases_load[station_id] + job_machine_load[station_id] > limit):
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

                # Sort PSP by Due Date
                all_jobs = []
                for i in range(len(self.PoolUpstream)):
                    all_jobs.append(self.PoolUpstream.get(0))

                all_jobs.sort(key=lambda x: x.DueDate)

                # Separate collaborative and independent jobs while maintaining due date order
                collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                independent_jobs = [job for job in all_jobs if not job.is_collaborative()]

                # STAGE 1: Process collaborative jobs with human workload constraints
                remaining_collaborative = []
                for job in collaborative_jobs:
                    if not evaluateJobStage1(self, job, machine_phases_load, human_phases_load, self.system):
                        remaining_collaborative.append(job)

                # STAGE 2: With human loads fixed, process independent jobs with machine constraints
                remaining_independent = []
                for job in independent_jobs:
                    if not evaluateJobStage2(self, job, machine_phases_load, human_phases_load, self.system):
                        remaining_independent.append(job)

                # Return unreleased jobs to pool (maintain PRD order)
                for job in remaining_collaborative + remaining_independent:
                    self.PoolUpstream.append(job)

                self._urgent_job_valve()
                yield env.timeout(period)

        def Human_Centric_release_dual(self, period):
            """
            Enhanced Human Centric rule with dual workload norms
            """
            def evaluateJobStage1(self, job, machine_phases_load, human_phases_load):
                """Stage 1: Check constraints for collaborative jobs"""
                job_machine_load, job_human_load = job.get_dual_CSL_routing_only()

                global CURRENT_HUMAN_NORM, CURRENT_MACHINE_NORM

                for station_id in job.Routing:
                    # Check machine constraint
                    if (machine_phases_load[station_id] + job_machine_load[station_id] >
                        CURRENT_MACHINE_NORM/N_PHASES):
                        return False

                    # Check human constraint for human-machine stations
                    if HUMAN_MACHINE_PHASES.get(station_id, False):
                        if (human_phases_load[station_id] + job_human_load[station_id] >
                            CURRENT_HUMAN_NORM/N_PHASES):
                            return False

                # Release job and update loads
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]
                    human_phases_load[station_id] += job_human_load[station_id]

                self.Pools[job.Routing[0]].append(job)

                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG
                    JOBS_RELEASED_DEBUG.append(job)

                return True

            def evaluateJobStage2(self, job, machine_phases_load, human_phases_load):
                """Stage 2: Check constraints for independent jobs"""
                job_machine_load, job_human_load = job.get_dual_CSL_routing_only()

                global CURRENT_MACHINE_NORM

                # Only check machine constraints (human loads are fixed)
                for station_id in job.Routing:
                    if (machine_phases_load[station_id] + job_machine_load[station_id] >
                        CURRENT_MACHINE_NORM/N_PHASES):
                        return False

                # Release job
                job.ReleaseDate = self.env.now
                for station_id in job.Routing:
                    machine_phases_load[station_id] += job_machine_load[station_id]

                self.Pools[job.Routing[0]].append(job)

            def forceReleaseJob(self, job):
                """Force release a job regardless of constraints"""
                job.ReleaseDate = self.env.now

                # Check if job has valid position
                if job.Position >= len(job.Routing):
                    print(f"WARNING: Force releasing job {job.id} with invalid position {job.Position}")
                    job.Position = 0  # Reset to beginning

                # Use current position, not always [0]
                target_machine = job.Routing[job.Position]
                self.Pools[target_machine].append(job)

                if RUN_DEBUG:
                    global JOBS_RELEASED_DEBUG
                    JOBS_RELEASED_DEBUG.append(job)

            while True:
                machine_phases_load, human_phases_load = get_corrected_shop_load_with_ratios(self.Pools)

                # Sort jobs by due date
                all_jobs = []
                for i in range(len(self.PoolUpstream)):
                    all_jobs.append(self.PoolUpstream.get(0))

                all_jobs.sort(key=lambda x: x.DueDate)

                # Separate jobs into categories
                collaborative_jobs = [job for job in all_jobs if job.is_collaborative()]
                independent_jobs = [job for job in all_jobs if not job.is_collaborative()]
                urgent_jobs = []

                # Check for jobs that need forced release
                for job_list in [collaborative_jobs, independent_jobs]:
                    i = 0
                    while i < len(job_list):
                        job = job_list[i]
                        time_in_psp = self.env.now - job.ArrivalDate
                        due_date_urgency = job.DueDate - self.env.now

                        # Same forced release conditions
                        if (due_date_urgency < 0 or time_in_psp > 2000 or due_date_urgency < 480):
                            urgent_jobs.append(job_list.pop(i))
                        else:
                            i += 1

                # Force release urgent jobs first
                for job in urgent_jobs:
                    forceReleaseJob(self, job)

                if urgent_jobs and SCREEN_DEBUG:
                    print(f"HUMAN_CENTRIC forced release: {len(urgent_jobs)} jobs at time {self.env.now/480:.1f}")

                # Stage 1: Process remaining collaborative jobs
                remaining_collaborative = []
                for job in collaborative_jobs:
                    if not evaluateJobStage1(self, job, machine_phases_load, human_phases_load):
                        remaining_collaborative.append(job)

                # Stage 2: Process remaining independent jobs
                remaining_independent = []
                for job in independent_jobs:
                    if not evaluateJobStage2(self, job, machine_phases_load, human_phases_load):
                        remaining_independent.append(job)

                # Return unreleased jobs
                for job in remaining_collaborative + remaining_independent:
                    self.PoolUpstream.append(job)

                yield env.timeout(period)

        def WB_MOD_release(self, period):
            """Periodically calls the WB_MOD optimizer and releases jobs based on its plan."""
            print("WB_MOD release process started.")
            first_run = True

            while True:
                # 1. Defer on first run to avoid race condition with capacity setup
                if first_run:
                    yield self.env.timeout(1)
                    first_run = False

                # Default telemetry values for this cycle
                self.system.WBMOD_stats = {
                    'solve_status': "Skipped", 'psp_size': 0, 'subset_size': 0,
                    'solve_time_sec': 0.0, 'jobs_released': 0, 'adj_minutes_planned_t0': 0.0
                }

                # 2. Check feature flag
                if not WB_MOD_USE_OPTIMIZER:
                    self.system.WBMOD_stats['solve_status'] = "SkippedByFlag"
                    print(f"Time {self.env.now:.2f}: WB_MOD skipped by flag.")
                    # Fallback test logic: release a random 20% of jobs
                    psp_jobs = self.PoolUpstream.get_list()
                    jobs_to_release_ids = [j.id for j in psp_jobs if random.random() < 0.2]

                    remaining_jobs_in_pool = list(self.PoolUpstream.get_list())
                    self.PoolUpstream.array.clear()

                    for job in remaining_jobs_in_pool:
                        if job.id in jobs_to_release_ids:
                            job.ReleaseDate = self.env.now
                            self.Pools[job.get_current_machine()].append(job)
                        else:
                            self.PoolUpstream.append(job)

                    self.system.WBMOD_stats['jobs_released'] = len(jobs_to_release_ids)
                    yield self.env.timeout(period)
                    continue

                # 3. Prepare inputs for the optimizer
                all_psp_jobs = self.PoolUpstream.get_list()
                self.system.WBMOD_stats['psp_size'] = len(all_psp_jobs)

                if not all_psp_jobs:
                    print(f"Time {self.env.now:.2f}: WB_MOD review, PSP is empty.")
                    yield self.env.timeout(period)
                    continue

                # Sort by due date and take a subset
                all_psp_jobs.sort(key=lambda j: j.DueDate)
                psp_subset = all_psp_jobs[:WB_MOD_MAX_PSP]
                self.system.WBMOD_stats['subset_size'] = len(psp_subset)

                # 4. Call the optimizer
                jobs_to_release_ids, adj_plan_t0, status, solve_time = run_wb_mod_optimizer(
                    psp_subset, self.system, REVIEW_HORIZON_T, WB_MOD_SOLVER_TIMEOUT
                )

                # 5. Update telemetry
                self.system.WBMOD_stats['solve_status'] = status
                self.system.WBMOD_stats['solve_time_sec'] = solve_time
                self.system.WBMOD_stats['jobs_released'] = len(jobs_to_release_ids)
                self.system.WBMOD_stats['adj_minutes_planned_t0'] = sum(adj_plan_t0.values())

                # 6. Process optimizer results
                use_fallback = True
                if status in {"Optimal", "TimeLimit", "Feasible"}:
                    if jobs_to_release_ids:
                        print(f"Time {self.env.now:.2f}: WB_MOD releasing {len(jobs_to_release_ids)} jobs from optimizer plan.")
                        self.system.adj_plan = adj_plan_t0

                        # Efficiently release jobs
                        jobs_to_process = {job.id: job for job in self.PoolUpstream.get_list()}
                        self.PoolUpstream.array.clear()

                        for job_id, job in jobs_to_process.items():
                            if job_id in jobs_to_release_ids:
                                job.ReleaseDate = self.env.now
                                self.Pools[job.get_current_machine()].append(job)
                            else:
                                self.PoolUpstream.append(job)

                        use_fallback = False
                    else:
                        print(f"Time {self.env.now:.2f}: WB_MOD optimizer returned valid status but no jobs to release.")

                if use_fallback:
                    print(f"Time {self.env.now:.2f}: WB_MOD falling back to WL_DIRECT logic for this cycle.")
                    # Fallback to WL_DIRECT logic (simplified version)
                    phases_load = get_corrected_shop_load_routing_only(self.Pools)
                    jobs_to_evaluate = list(self.PoolUpstream.get_list())
                    jobs_to_evaluate.sort(key=lambda x: x.DueDate)
                    self.PoolUpstream.array.clear()

                    released_count = 0
                    for job in jobs_to_evaluate:
                        job_CSL = job.get_CSL_routing_only()
                        can_release = True
                        for station_id in job.Routing:
                            if (phases_load[station_id] + job_CSL[station_id] > CURRENT_MACHINE_NORM / N_PHASES):
                                can_release = False
                                break

                        if can_release:
                            job.ReleaseDate = self.env.now
                            for station_id in job.Routing:
                                phases_load[station_id] += job_CSL[station_id]
                            self.Pools[job.Routing[0]].append(job)
                            released_count += 1
                        else:
                            self.PoolUpstream.append(job)

                    self.system.WBMOD_stats['jobs_released'] = released_count
                    self.system.WBMOD_stats['solve_status'] = f"Fallback_{status}"

                # 7. Signal workers about the new plan
                if hasattr(self.system, 'new_adj_plan_event'):
                    if not self.system.new_adj_plan_event.triggered:
                        print(f"Time {self.env.now:.2f}: WB_MOD triggering new_adj_plan_event.")
                        self.system.new_adj_plan_event.succeed()
                    self.system.new_adj_plan_event = self.env.event()

                # 8. Wait for the next review period
                yield self.env.timeout(period)

    class Pool(object):
        """Represents a buffer or queue for jobs in the simulation."""
        def __init__(self, env, id, Pools):
            self.env = env
            self.id = id
            self.Pools = Pools
            self.array = list()  # The list of jobs in the pool.
            # List of events to trigger when a new job arrives.
            self.waiting_new_jobs = list()
            # List of triggers for workload limits, used in flexible worker mode.
            self.workload_limit_triggers = list()

        def __getitem__(self, key):
            """Allows accessing jobs in the pool by index."""
            return self.array[key]

        def __len__(self):
            """Returns the number of jobs in the pool."""
            return len(self.array)

        def append(self, job):
            """Adds a job to the pool and triggers any waiting processes."""
            self.array.append(job)
            # Trigger events for starving machines or workers waiting for new jobs.
            while len(self.waiting_new_jobs) > 0:
                self.waiting_new_jobs.pop(0).succeed()
            # Trigger events for workers if a workload limit has been exceeded.
            i = 0
            while i < len(self.workload_limit_triggers):
                if get_corrected_aggregated_workload(self.Pools)[self.id] > self.workload_limit_triggers[i][1]:
                    self.workload_limit_triggers.pop(i)[2].succeed()
                else:
                    i += 1

        def get(self, index=0):
            """Removes and returns a job from the pool at a given index."""
            return self.array.pop(index)

        def get_list(self):
            """Returns the raw list of jobs in the pool."""
            return self.array

        def delete(self, index):
            """Deletes a job from the pool at a given index."""
            del self.array[index]

        def sort(self):
            """Sorts the jobs in the pool by their arrival date."""
            self.array.sort(key=lambda x: x.ArrivalDate)

    class Machine(object):
        """Represents a machine or work station in the shop."""
        def __init__(self, env, id, PoolUpstream, Jobs_delivered, Pools, PSP):
            self.env = env
            self.id = id
            self.JobsProcessed = 0
            self.WorkloadProcessed = 0.0
            self.current_job = None
            self.efficiency = 0
            self.Workers = list()
            self.empty_station_minutes = 0
            self.overmanned_minutes = 0
            # References to other system components.
            self.PoolUpstream = PoolUpstream
            self.PSP = PSP
            self.Pools = Pools
            self.Jobs_delivered = Jobs_delivered
            # SimPy events for process synchronization.
            self.waiting_new_workers = self.env.event()
            self.waiting_new_jobs = self.env.event()
            PoolUpstream.waiting_new_jobs.append(self.waiting_new_jobs)
            self.HumanWaitingPool = []
            self.waiting_end_job = list()
            # Start the machine's main processing loop.
            self.process = self.env.process(self.Machine_loop())



        def _process_human_machine_collaborative(self, job):
            """
            Process job on human-machine station with human centric rule
            Takes the maximum of human and machine processing times
            """
            base_time = job.RemainingTime[self.id]
            machine_time = base_time * MACHINE_RATIO / float(self.efficiency)
            human_time = base_time * HUMAN_RATIO  # Human efficiency assumed to be 1

            return max(machine_time, human_time)

        def _process_human_machine_non_collaborative(self, job):
            """
            Process job on human-machine station with non-human-centric rules
            More complex logic based on which component (human/machine) finishes first
            """
            base_time = job.RemainingTime[self.id]
            machine_time = base_time * MACHINE_RATIO / float(self.efficiency)
            human_time = base_time * HUMAN_RATIO  # Human efficiency assumed to be 1

            if human_time <= machine_time:
                # Human finishes first or at same time - same as collaborative case
                return max(machine_time, human_time)
            else:
                # Machine finishes first - set flag and start human completion process
                self._machine_finished_first = True
                self._remaining_human_time = human_time - machine_time

                # Start a separate process to handle the human completion
                self.env.process(self._handle_machine_first_completion(job, human_time - machine_time))

                return machine_time

        def _handle_machine_first_completion(self, job, remaining_human_time):
            """
            Handle case where machine completes before human in non-collaborative rule
            The job waits for human to complete before it can proceed
            """
            # Wait for the remaining human processing time
            yield self.env.timeout(remaining_human_time)

            # After human processing is complete, move job from waiting pool to next stage
            if job in self.HumanWaitingPool:
                self.HumanWaitingPool.remove(job)

                # Update job completion status
                job.Position += 1

                if job.Position == len(job.Routing):
                    # Job is complete
                    job.CompletationDate = self.env.now

                    if RUN_DEBUG:
                        global JOBS_DELIVERED_DEBUG
                        JOBS_DELIVERED_DEBUG.append(job)

                    self.Jobs_delivered.append(job)
                else:
                    # Move to next station
                    self.Pools[job.get_current_machine()].append(job)

                self.JobsProcessed += 1

                # Trigger any waiting workers
                while len(self.waiting_end_job) > 0:
                    self.waiting_end_job[0].succeed()
                    del self.waiting_end_job[0]

        def Machine_loop(self):
            while True:
                try:
                    # Wait for jobs or workers as needed
                    while len(self.PoolUpstream) == 0 and self.current_job == None and len(self.HumanWaitingPool) == 0:
                        if STARVATION_AVOIDANCE:
                            # Try to get job from PSP
                            found_job = False
                            for i in range(len(self.PSP)):
                                if self.PSP[i].Routing[0] == self.id:
                                    self.current_job = self.PSP.get(i)
                                    self.current_job.ReleaseDate = self.env.now
                                    found_job = True
                                    break
                            if not found_job:
                                # Wait for new jobs
                                self.waiting_new_jobs = self.env.event()
                                self.PoolUpstream.waiting_new_jobs.append(self.waiting_new_jobs)
                                yield self.waiting_new_jobs
                        else:
                            # Wait for new jobs
                            self.waiting_new_jobs = self.env.event()
                            self.PoolUpstream.waiting_new_jobs.append(self.waiting_new_jobs)
                            yield self.waiting_new_jobs

                    while len(self.Workers) == 0:
                        self.efficiency = 0
                        self.waiting_new_workers = self.env.event()
                        yield self.waiting_new_workers

                    # Inner processing loop
                    while True:
                        try:
                            # Get job to process
                            if self.current_job == None:
                                if len(self.HumanWaitingPool) > 0:
                                    # Process job from human waiting pool - no machine processing needed
                                    self.current_job = self.HumanWaitingPool.pop(0)

                                    # This job is waiting for human completion only
                                    # Calculate remaining human processing time
                                    if hasattr(self.current_job, '_remaining_human_time'):
                                        remaining_human_time = self.current_job._remaining_human_time
                                        delattr(self.current_job, '_remaining_human_time')
                                    else:
                                        # Fallback - shouldn't happen
                                        base_time = self.current_job.ProcessingTime[self.id] - self.current_job.RemainingTime[self.id]
                                        remaining_human_time = base_time * HUMAN_RATIO

                                    # Wait for human completion
                                    yield self.env.timeout(remaining_human_time)

                                    # Human processing completed
                                    self._complete_job()
                                    continue

                                elif len(self.PoolUpstream) > 0:
                                    self.current_job = self.PoolUpstream.get()
                                    self.current_job.ArrivalDateMachines[self.id] = self.env.now
                                    # ADD THESE LINES:

                                    #if job_tracker is not None:
                                     #   job_tracker.update_station_arrival(self.current_job, self.id, self.env.now)

                                else:
                                    break  # Exit inner loop to wait for more jobs

                            # Check if we still have workers
                            if len(self.Workers) == 0:
                                self.efficiency = 0
                                break  # Exit inner loop to wait for workers

                            # Calculate efficiency for machine processing
                            workers_current_job = list(self.Workers)
                            self.efficiency = sum(worker.skillperphase[self.id] for worker in workers_current_job)

                            # Determine processing logic
                            machine_finishes_first = False

                            if HUMAN_MACHINE_PHASES.get(self.id, False):
                                if RELEASE_RULE == "HUMAN_CENTRIC":
                                    # Collaborative processing - both work together
                                    machine_time = self.current_job.RemainingTime[self.id] * MACHINE_RATIO / float(self.efficiency)
                                    human_time = self.current_job.RemainingTime[self.id] * HUMAN_RATIO
                                    processing_time = max(machine_time, human_time)
                                else:
                                    # Non-collaborative - check who finishes first
                                    machine_time = self.current_job.RemainingTime[self.id] * MACHINE_RATIO / float(self.efficiency)
                                    human_time = self.current_job.RemainingTime[self.id] * HUMAN_RATIO

                                    if machine_time < human_time:
                                        # Machine finishes first
                                        processing_time = machine_time
                                        machine_finishes_first = True
                                    else:
                                        # Human finishes first or same time
                                        processing_time = human_time
                            else:
                                # Pure machine station
                                processing_time = self.current_job.RemainingTime[self.id] / float(self.efficiency)

                            # Process the job
                            start_time = self.env.now

                            if processing_time < 0: processing_time = 0
                            yield self.env.timeout(processing_time)

                           # if job_tracker is not None:
                            #    job_tracker.update_station_processing(
                             #       self.current_job, self.id, start_time, self.env.now, self.efficiency
                              #  )
                            # Update worker times
                            actual_processing_time = self.env.now - start_time
                            for worker in workers_current_job:
                                worker.WorkingTime[self.id] += actual_processing_time

                            self.WorkloadProcessed += (actual_processing_time * self.efficiency)

                            # Job processing completed
                            self.current_job.CompletationDateMachines[self.id] = self.env.now
                            self.current_job.RemainingTime[self.id] = 0

                            # Handle completion based on human-machine logic
                            if machine_finishes_first:
                                # Machine finished first - job waits for human completion
                                remaining_human_time = human_time - machine_time
                                self.current_job._remaining_human_time = remaining_human_time

                                self.HumanWaitingPool.append(self.current_job)
                                self.current_job = None
                                self.JobsProcessed += 1

                                # Trigger workers waiting for end of job
                                while len(self.waiting_end_job) > 0:
                                    self.waiting_end_job[0].succeed()
                                    del self.waiting_end_job[0]
                            else:
                                # Normal completion
                                self._complete_job()

                        except simpy.Interrupt:
                            # Handle interruption during job processing
                            if self.current_job is not None:
                                # Update remaining time based on partial processing
                                if 'start_time' in locals():
                                    partial_time = self.env.now - start_time
                                    for worker in workers_current_job:
                                        worker.WorkingTime[self.id] += partial_time
                                    self.WorkloadProcessed += (partial_time * self.efficiency)
                                    self.current_job.RemainingTime[self.id] -= (partial_time * self.efficiency)
                            # Re-raise to exit inner loop
                            raise

                except simpy.Interrupt:
                    # Outer interrupt handler - restart the entire loop
                    continue

        def _complete_job(self):
            """Helper method to complete a job and move it to next stage"""
            self.current_job.Position += 1

            if self.current_job.Position == len(self.current_job.Routing):
                # Check if job already delivered (add this check!)
                if not hasattr(self.current_job, '_already_delivered'):
                    self.current_job.CompletationDate = self.env.now
                    self.current_job._already_delivered = True  # Mark as delivered

                    if RUN_DEBUG:
                        global JOBS_DELIVERED_DEBUG
                        JOBS_DELIVERED_DEBUG.append(self.current_job)

                    self.Jobs_delivered.append(self.current_job)
            else:
                # Move to next station
                next_machine_id = self.current_job.get_current_machine()
                self.Pools[next_machine_id].append(self.current_job)

            self.current_job = None
            self.JobsProcessed += 1

            # Trigger waiting workers
            while len(self.waiting_end_job) > 0:
                self.waiting_end_job[0].succeed()
                del self.waiting_end_job[0]

    class Worker(object):
        """
        Represents a worker in the job shop.

        Workers are the primary resource for processing jobs at machines. Each worker
        has a unique ID, a default (home) machine, and a specific skill set that
        determines their efficiency at different machines.

        The behavior of a worker is determined by the `WORKER_MODE` parameter, which
        can be:
        - "static": The worker is permanently assigned to their default machine.
        - "reactive": The worker can move to other machines based on simple rules,
                      such as queue lengths, to help with bottlenecks.
        - "flexible": The worker's movement is governed by a more complex output
                      control system, often tied to workload balancing algorithms.

        Attributes:
            id (int): A unique identifier for the worker.
            Default_machine (int): The ID of the worker's home machine.
            skillperphase (list): A list where each index `i` represents a machine and
                                  the value is the worker's efficiency (0.0 to 1.0) at that machine.
            current_machine_id (int): The ID of the machine where the worker is currently located.
            WorkingTime (list): Tracks the total time the worker has spent working at each machine.
            Capacity_adjustment (list): Used in flexible modes to carry capacity sharing information.
            relocation (list): A counter for the number of times the worker has moved to each machine.
        """

        def __init__(self, env, id, Pools, Machines, Default_machine, system, skills = None):
            """
            Initializes a worker.

            Args:
                env (simpy.Environment): The simulation environment.
                id (int): The worker's unique ID.
                Pools (list): A reference to all pools in the system.
                Machines (list): A reference to all machines in the system.
                Default_machine (int): The ID of the worker's home machine.
                skills: (Not currently used) Placeholder for future skill definitions.
            """

            self.env = env
            self.id = id
            self.Default_machine = Default_machine # The worker's home station ID.
            self.system = system

            # Initialize attributes for tracking state and performance.
            self.skillperphase = list() # Stores worker's efficiency at each machine.
            self.current_machine_id = -1 # Current machine location, -1 means not yet assigned.
            self.relocation = list(0 for i in range(N_PHASES)) # Tracks number of moves to each machine.
            self.WorkingTime = list(0 for i in range(N_PHASES)) # Tracks time spent at each machine.
            self.time_in_transfer = 0 # Tracks time spent moving between stations.

            # This attribute is used for advanced flexible strategies to adjust capacity.
            self.Capacity_adjustment = list(0 for i in range(N_PHASES))

            # --- Set Skill Set based on Global Configuration ---
            # The following block determines the worker's skills based on the `WORKER_FLEXIBILITY` parameter.
            if WORKER_FLEXIBILITY == 'triangular': self._SetTriangularSkills(WORKER_EFFICIENCY_DECREMENT)
            elif WORKER_FLEXIBILITY == 'chain': self._SetChainSkills()
            elif WORKER_FLEXIBILITY == 'chain upstream': self._SetChainUpstreamSkills()
            elif WORKER_FLEXIBILITY == 'flat': self._SetPlainSkills()
            elif WORKER_FLEXIBILITY == 'mono': self._SetMonoSkill()
            elif WORKER_FLEXIBILITY == 'exponential': self._SetExponentialSkills(WORKER_EFFICIENCY_DECREMENT)
            else: exit("wrong worker flexibility ")
            print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))
            self.waiting_events = None

            # --- Set Behavior Mode based on Global Configuration ---
            # This block starts the appropriate SimPy process for the worker's behavior.
            if 'WORKER_MODE' not in globals(): exit("Worker mode not initialised")
            # Worker is fixed to their default machine.
            if WORKER_MODE == 'static': self._StaticicWorker(Machines)
            # Worker moves based on simple queue-based rules.
            elif WORKER_MODE == 'reactive': self.process = self.env.process(self._ReactiveWorker(Machines,None))
            # Worker moves based on an output control mechanism.
            elif WORKER_MODE == 'flexible': self.process = self.env.process(self._Flexible_loop(Machines,None))
            elif WORKER_MODE == 'plan_following': self.process = self.env.process(self._PlanFollowingWorker(Machines, system))
            else: exit("Worker mode not recognised")

        def _SetMonoSkill(self):
            """Assigns a skill of 1.0 to the default machine and 0 to all others."""
            for i in range(0,N_PHASES):
                if i == self.Default_machine:
                    self.skillperphase.append(1)
                else:
                    self.skillperphase.append(0)

        def _SetTriangularSkills(self, decrement):
            """
            Sets skills that decrease linearly with distance from the home machine.
            This method seems complex and might have a bug or be designed for a
            specific topology not immediately obvious from the code. It appears to
            set skills for the immediate neighbors of the default machine.
            """
            temp=list()
            for i in range(3):  # 0, 1, 2
                for i in range(N_PHASES):
                    temp.append(0)

            # Set skill for the default machine and its neighbors
            temp[self.Default_machine + N_PHASES] = 1
            temp[self.Default_machine + N_PHASES+1] = 1 - decrement
            temp[self.Default_machine + N_PHASES-1] = 1 - decrement

            # Sum up the skills from the temporary list.
            for i in range(N_PHASES):
                temp_sum = 0
                for j in range(3):
                    temp_sum += temp[i+N_PHASES*j]
                self.skillperphase.append(temp_sum)

        def _SetExponentialSkills(self, decrement):
            """Sets skills that decrease exponentially with distance from the home machine."""
            for i in range(0, N_PHASES):
                self.skillperphase.append(max(pow(decrement, abs(self.Default_machine - i)), 0))

        def _SetChainSkills(self):
            """
            Assigns skills for a 'chain' or flow-shop layout.
            The worker is fully skilled at their home machine and the next machine
            downstream. If they are at the last machine, they are skilled at the first.
            """
            self.skillperphase = list(0 for i in range(N_PHASES))
            self.skillperphase[self.Default_machine] = 1

            if self.Default_machine == N_PHASES-1:
                # Last machine, skilled at the first machine (circular flow)
                self.skillperphase[0] = 1 - WORKER_EFFICIENCY_DECREMENT
            else:
                # Skilled at the next machine downstream
                self.skillperphase[self.Default_machine+1] = 1 - WORKER_EFFICIENCY_DECREMENT

        def _SetChainUpstreamSkills(self):
            """
            Assigns skills for a 'chain' layout, but for the upstream machine.
            The worker is fully skilled at their home machine and the machine
            immediately upstream.
            """
            self.skillperphase = list(0 for i in range(N_PHASES))
            self.skillperphase[self.Default_machine] = 1

            if self.Default_machine == N_PHASES - 1:
                # If at the last machine, skilled at the second to last.
                self.skillperphase[3] = 1 - WORKER_EFFICIENCY_DECREMENT
            else:
                # Skilled at the previous machine upstream.
                self.skillperphase[self.Default_machine - 1] = 1 - WORKER_EFFICIENCY_DECREMENT

        def _SetPlainSkills(self):
            """
            Assigns skills uniformly across all machines ('flat' flexibility).
            The worker is fully skilled at their home machine and has a slightly
            decremented skill at all other machines.
            """
            self.skillperphase=list()
            for i in range(N_PHASES):
                self.skillperphase.append(1-WORKER_EFFICIENCY_DECREMENT)
            # Full skill at the home machine.
            self.skillperphase[self.Default_machine] = 1



        def _StaticicWorker(self, Machines):
            """
            Assigns the worker to their default station permanently.

            This is the simplest worker mode. The worker is added to the list of
            workers at their default machine and an interrupt is triggered to let
            the machine know a new worker is available.
            """
            self.current_machine_id = self.Default_machine
            Machines[self.Default_machine].Workers.append(self)

            # Interrupt the machine's process to re-evaluate its state with the new worker.
            Machines[self.Default_machine].process.interrupt()

            # If the machine was waiting for a worker, succeed the event.
            if Machines[self.current_machine_id].waiting_new_workers.triggered == False:
                Machines[self.current_machine_id].waiting_new_workers.succeed()

        def _get_next_machine_reactive(self, Machines):
            """Determines the best machine for a reactive worker to move to."""
            # Priority 1: Stay at the home machine if there is work to do.
            if len(Machines[self.Default_machine].PoolUpstream) > 0:
                 return self.Default_machine

            # Priority 2: Find a different machine with work.
            possible_external_machines = []
            for i in range(N_PHASES):
                if i == self.Default_machine:
                    continue # Skip home machine

                # Check if the worker is skilled and if there's a queue of jobs.
                if self.skillperphase[i] > 0 and len(Machines[i].PoolUpstream) > 0:
                    possible_external_machines.append((i, self.skillperphase[i]))

            if possible_external_machines:
                # Sort potential machines by skill level (descending) and choose the best.
                possible_external_machines.sort(key=lambda x: float(x[1]), reverse=True)
                return possible_external_machines[0][0]

            # Priority 3: If no other machine has work, return to the default machine.
            return self.Default_machine

        def _ReactiveWorker(self, Machines, Pools):
            """
            A SimPy process for a worker who reactively moves between stations.

            The worker stays at their home station if it has work. If not, they look
            for other stations with work where they have skills. They prioritize
            stations where they have higher skills.
            """
            while True:
                # Determine the next target machine based on reactive logic.
                next_machine = self._get_next_machine_reactive(Machines)

                # --- Handle Relocation ---
                if self.current_machine_id != -1 and self.current_machine_id != next_machine:
                    # Unload the worker from their current machine.
                    for i in range(len(Machines[self.current_machine_id].Workers)):
                        if Machines[self.current_machine_id].Workers[i].id == self.id:
                            del Machines[self.current_machine_id].Workers[i]
                            Machines[self.current_machine_id].process.interrupt()
                            break

                if self.current_machine_id != next_machine:
                    # If moving, incur a transfer time penalty.
                    yield self.env.timeout(TRANSFER_TIME)
                    self.current_machine_id = next_machine
                    self.relocation[next_machine] += 1

                    # Add the worker to the new machine's workforce.
                    Machines[self.current_machine_id].Workers.append(self)
                    Machines[self.current_machine_id].process.interrupt()
                    if not Machines[self.current_machine_id].waiting_new_workers.triggered:
                        Machines[self.current_machine_id].waiting_new_workers.succeed()

                # --- Wait for a trigger to re-evaluate position ---
                # The worker will stay at the current machine until one of these events occurs.
                waiting_events = []

                # Event 1: The current job finishes.
                end_current_job = self.env.event()
                Machines[self.current_machine_id].waiting_end_job.append(end_current_job)
                waiting_events.append(end_current_job)

                # Event 2: A new job arrives at the worker's home station (pulling them back).
                waiting_new_jobs = self.env.event()
                Machines[self.Default_machine].PoolUpstream.waiting_new_jobs.append(waiting_new_jobs)
                waiting_events.append(waiting_new_jobs)

                # The worker must stay for a minimum permanence time before moving again.
                yield self.env.timeout(PERMANENCE_TIME)

                # Wait for any of the defined trigger events.
                if "HUMAN" in RELEASE_RULE and HUMAN_MACHINE_PHASES.get(self.current_machine_id, False):
                    yield end_current_job
                else:
                    yield AnyOf(self.env, waiting_events)

        def _Flexible_loop(self, Machines, Pools):
            """
            A SimPy process for a worker whose movement is controlled by an output
            control mechanism (e.g., workload balancing).
            """
            def _get_next_machine_flexible(self):
                """
                Determines the next machine based on both local needs and centrally
                planned capacity adjustments.
                """
                # A worker will not move from their home station if it has a queue.
                if len(Machines[self.Default_machine].PoolUpstream) > 0:
                     return self.Default_machine

                # Look for other machines that need help, according to the capacity adjustment plan.
                possible_external_machines = []
                for i in range(N_PHASES):
                    if i == self.Default_machine:
                        continue

                    # Check for skill, a queue, and a capacity adjustment signal > 1.
                    if (self.skillperphase[i] > 0 and
                        len(Machines[i].PoolUpstream) > 0 and
                        self.Capacity_adjustment[i] > 1):
                        possible_external_machines.append((i, self.skillperphase[i]))

                if possible_external_machines:
                    # Choose the best machine based on skill.
                    possible_external_machines.sort(key=lambda x: float(x[1]), reverse=True)
                    return possible_external_machines[0][0]

                # If no other options, stay at the default machine.
                return self.Default_machine

            while True:
                next_machine = _get_next_machine_flexible(self)
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
                if "HUMAN" in RELEASE_RULE and HUMAN_MACHINE_PHASES.get(self.current_machine_id, False):
                    # Regla por defecto human-centric: el operario espera a la máquina
                    yield end_current_job
                else:
                    yield AnyOf(self.env, waiting_events)
                if self.current_machine_id!=self.Default_machine:
                    # update capacity adjustment
                    self.Capacity_adjustment[self.current_machine_id] = self.Capacity_adjustment[self.current_machine_id] - (self.env.now-start)

        def _PlanFollowingWorker(self, Machines, system):
            """
            A SimPy process for a worker that follows a centrally computed plan.
            Falls back to reactive behavior if no plan is available.
            """
            while True:
                print(f"Time {self.env.now:.2f}: Worker {self.id} starting loop.")
                # Default behavior is reactive
                next_machine = self._get_next_machine_reactive(Machines)

                # Check for a plan to follow
                # The plan is in system.adj_plan, format: {(to_station, from_station): minutes}
                r = self.Default_machine
                if system.adj_plan:
                    for (i, from_r), minutes in system.adj_plan.items():
                        if from_r == r and minutes > 0:
                            if self.skillperphase[i] > 0:
                                next_machine = i
                                # To prevent all workers from moving, we 'consume' the plan.
                                # This is a simplification; a real system might need specific worker assignments.
                                system.adj_plan[(i, from_r)] = 0
                                break

                # --- Handle Relocation (similar to _ReactiveWorker) ---
                if self.current_machine_id != -1 and self.current_machine_id != next_machine:
                    for i in range(len(Machines[self.current_machine_id].Workers)):
                        if Machines[self.current_machine_id].Workers[i].id == self.id:
                            del Machines[self.current_machine_id].Workers[i]
                            Machines[self.current_machine_id].process.interrupt()
                            break

                if self.current_machine_id != next_machine:
                    yield self.env.timeout(TRANSFER_TIME)
                    self.time_in_transfer += TRANSFER_TIME # Log telemetry
                    self.current_machine_id = next_machine
                    self.relocation[next_machine] += 1

                    Machines[self.current_machine_id].Workers.append(self)
                    Machines[self.current_machine_id].process.interrupt()
                    if not Machines[self.current_machine_id].waiting_new_workers.triggered:
                        Machines[self.current_machine_id].waiting_new_workers.succeed()

                # --- Wait for a trigger to re-evaluate position ---
                waiting_events = [system.new_adj_plan_event]

                # Also wait for the current job to finish
                end_current_job = self.env.event()
                if self.current_machine_id != -1:
                    Machines[self.current_machine_id].waiting_end_job.append(end_current_job)
                    waiting_events.append(end_current_job)

                # Also wait for a new job to arrive at the home station
                waiting_new_jobs = self.env.event()
                Machines[self.Default_machine].PoolUpstream.waiting_new_jobs.append(waiting_new_jobs)
                waiting_events.append(waiting_new_jobs)

                print(f"Time {self.env.now:.2f}: Worker {self.id} waiting for events.")
                yield self.env.timeout(PERMANENCE_TIME)
                yield AnyOf(self.env, waiting_events)
                print(f"Time {self.env.now:.2f}: Worker {self.id} woke up.")

    class DynamicConstraintManager:
        def __init__(self, env, scenario_config, system):
            self.env = env
            self.config = scenario_config
            self.system = system
            self.switch_count = 0
            # self.current_human_norm = scenario_config.get("base_human", scenario_config.get("human_norm", 1000))
            # self.current_machine_norm = scenario_config.get("base_machine", scenario_config.get("machine_norm", 1000))

            # global CURRENT_HUMAN_NORM, CURRENT_MACHINE_NORM
            # CURRENT_HUMAN_NORM = self.current_human_norm
            # CURRENT_MACHINE_NORM = self.current_machine_norm
            # The old global norms are now superseded by the station-specific capacities
            # but we can keep them for legacy rules if needed.

            if scenario_config["type"] == "dynamic" and DYNAMIC_CAPS_ENABLED:
                env.process(self.availability_switching_process())

        def availability_switching_process(self):
            """Switch availability factors dynamically."""
            switch_freq = self.config["switch_frequency"]
            # base_human = self.config["base_human"]
            # base_machine = self.config["base_machine"]

            while True:
                yield self.env.timeout(switch_freq)
                self.switch_count += 1

                # Example of dynamic availability: simulate a machine breakdown or absenteeism
                # This can be made more sophisticated later.
                for i in range(N_PHASES):
                    # Randomly dip availability
                    self.system.A_mach[i] = random.choice([1.0, 1.0, 1.0, 0.8]) # 25% chance of 20% downtime
                    self.system.A_hum[i] = random.choice([1.0, 1.0, 0.9, 0.85]) # Chance of absenteeism

                # Update the effective capacities in the system
                self.system.update_effective_capacities()

                if SCREEN_DEBUG:
                    print(f"Time {self.env.now/480:.1f}: Availability updated.")
                    print(f"  A_mach: {[round(x, 2) for x in self.system.A_mach]}")
                    print(f"  Cap_mach: {[round(x, 1) for x in self.system.Cap_mach]}")
                    print(f"  A_hum: {[round(x, 2) for x in self.system.A_hum]}")
                    print(f"  Cap_hum: {[round(x, 1) for x in self.system.Cap_hum]}")

    class System(object):
        """
        The main class that sets up and manages the entire simulation environment,
        including pools, machines, workers, and system-level processes.
        """
        def __init__(self, env):
            self.env = env
            self.Pools=list()
            # Create WIP pools for each machine.
            self.Pools = [Pool(env, i, self.Pools) for i in range(N_PHASES)]
            # Create the Pre-Shop Pool (PSP) where jobs first arrive.
            self.PSP = Pool(env, -1, self.Pools)
            # Create the pool for completed jobs.
            self.Jobs_delivered = Pool(env, N_PHASES, self.Pools)
            # Create the machines.
            self.Machines = [Machine(self.env, i, self.Pools[i], self.Jobs_delivered, self.Pools, self.PSP) for i in range(N_PHASES)]
            # Create the workers.
            self.Workers = [Worker(env, i, self.Pools, self.Machines, i, self) for i in range(N_WORKERS)]
            # Start the job generator.
            self.generator = Jobs_generator(env, self.PSP)

            # Initialize base and effective capacities before setting up order release
            self.BaseCap_mach = [0] * N_PHASES
            self.BaseCap_hum = [0] * N_PHASES
            self.A_mach = [1.0] * N_PHASES # Availability factors, default to 1.0
            self.A_hum = [1.0] * N_PHASES
            self.Cap_mach = [0] * N_PHASES
            self.Cap_hum = [0] * N_PHASES
            self.setup_capacities()

            # Start the order release mechanism.
            self.OR = Orders_release(env, RELEASE_RULE, self, WORKLOAD_NORMS[WLIndex])

            self.adj_plan = {}
            self.new_adj_plan_event = env.event()

            # Store parameters for the optimizer to access
            self.WB_MOD_PARAMS = WB_MOD_PARAMS
            self.N_PHASES = N_PHASES

            # Initialize telemetry storage
            self.WBMOD_stats = {
                'solve_status': "N/A",
                'psp_size': 0,
                'subset_size': 0,
                'solve_time_sec': 0.0,
                'jobs_released': 0,
                'adj_minutes_planned_t0': 0.0
            }

            ### USE IF DYNAMIC SYSTEM ###

            global CONSTRAINT_SCENARIO
            scenario_config = CONSTRAINT_SCENARIOS[CONSTRAINT_SCENARIO]

            if scenario_config["type"] == "static":
                global CURRENT_HUMAN_NORM, CURRENT_MACHINE_NORM
                CURRENT_HUMAN_NORM = scenario_config["human_norm"]
                CURRENT_MACHINE_NORM = scenario_config["machine_norm"]

            self.constraint_manager = DynamicConstraintManager(env, scenario_config, self)
            self.env.process(self.monitor_stations(480)) # Monitor once per day


        def monitor_stations(self, period):
            while True:
                yield self.env.timeout(period)
                for machine in self.Machines:
                    if len(machine.Workers) == 0:
                        machine.empty_station_minutes += period
                    if len(machine.Workers) > 1: # Assuming 1 worker is "manned"
                        machine.overmanned_minutes += period

        def setup_capacities(self):
            """Calculates and sets the base and effective capacities."""
            workload_norm = WORKLOAD_NORMS[WLIndex]
            for i in range(N_PHASES):
                if HUMAN_MACHINE_PHASES.get(i, False):
                    # H-M station
                    self.BaseCap_mach[i] = (workload_norm / N_PHASES) * MACHINE_RATIO
                    self.BaseCap_hum[i] = (workload_norm / N_PHASES) * HUMAN_RATIO
                else:
                    # Pure machine station
                    self.BaseCap_mach[i] = (workload_norm / N_PHASES) * 1.0
                    self.BaseCap_hum[i] = 0

            self.update_effective_capacities()

        def update_effective_capacities(self):
            """Updates effective capacities based on current availability factors."""
            for i in range(N_PHASES):
                self.Cap_mach[i] = self.BaseCap_mach[i] * self.A_mach[i]
                self.Cap_hum[i] = self.BaseCap_hum[i] * self.A_hum[i]

    # --- Standalone Workload Calculation Functions ---
    def get_direct_workload(pools):
        """Calculates the direct workload (remaining time of the first job in queue) at each station."""
        direct_WL = [0] * N_PHASES
        for pool in pools:
            for job in pool:
                for i in range(N_PHASES):
                    if job.RemainingTime[i] > 0:
                        direct_WL[i] += job.RemainingTime[i]
                        break
        return direct_WL

    def get_aggregated_workload(pools):
        """Calculates the aggregated workload (sum of all remaining processing times) at each station."""
        aggregated_WL = [0] * N_PHASES
        for pool in pools:
            for job in pool:
                for i in range(N_PHASES):
                    if job.RemainingTime[i] > 0:
                        aggregated_WL[i] += job.RemainingTime[i]
        return aggregated_WL

    def get_corrected_aggregated_workload(pools):
        """Calculates the corrected aggregated workload using the CAW method for each job."""
        aggregated_WL = list(0 for i in range(N_PHASES))
        for pool in pools:
            for job in pool:
                # Original job
                job_contribution = job.get_CAW()
                for i in range(N_PHASES):
                    aggregated_WL[i] += job_contribution[i]

        return aggregated_WL

    def get_shop_load(pools):
        """Calculates the total shop load, including work already processed on jobs in the pools."""
        shop_load = [0] * N_PHASES
        for pool in pools:
            for job in pool:
                for i in range(N_PHASES):
                    shop_load[i] += job.ProcessingTime[i]
        return shop_load

    def get_shop_load2(pools):
        # The result is a vector of workloads. It considers also the contribution of task already processed of jobs
        # that have not been delivered yet
        shop_load = 0#list(0 for i in range(N_PHASES))
        for pool in pools:
            for job in pool:
                shop_load+=job.ShopLoad
        return shop_load

    def get_corrected_shop_load(pools):
        """Calculates the corrected shop load using the CSL method for each job."""
        shop_load = [0] * N_PHASES
        for pool in pools:
            for job in pool:
                job_CSL = job.get_CSL()
                for i in range(N_PHASES):
                    shop_load[i] += job_CSL[i]
        return shop_load

    def ResetStatistics(env, WARMUP, system):
        """
        A SimPy process that resets key simulation statistics after the warm-up period.
        """
        yield env.timeout(WARMUP)
        # Reset machine and job statistics.
        system.generator.generated_orders = sum(len(pool) for pool in system.Pools)
        for machine in system.Machines:
            machine.JobsProcessed = 0
            machine.WorkloadProcessed = 0.0
        global JOBS_WARMUP
        JOBS_WARMUP += len(system.Jobs_delivered)
        system.Jobs_delivered.array.clear()
        # Reset worker statistics.
        for worker in system.Workers:
            worker.WorkingTime = [0] * N_PHASES
            worker.WorkloadProcessed = [0] * N_PHASES
        system.OR.released_workload = [0] * N_PHASES
        system.generator.generated_processing_time = [0] * N_PHASES
        return

    def RunDebug5(env, run,system):
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
            if run == 0 :
                units = -1
                if len(JOBS_DELIVERED_DEBUG)>0:
                    units=len(JOBS_DELIVERED_DEBUG)
                result = {
                    "time":(env.now-TIME_BTW_DEBUGS),
                    "WL entry":(sum(sum(job.ProcessingTime) for job in JOBS_ENTRY_DEBUG)),
                    "WL released":(sum(sum(job.ProcessingTime) for job in JOBS_RELEASED_DEBUG)),
                    "WL processed":(sum(sum(job.ProcessingTime) for job in JOBS_DELIVERED_DEBUG)),
                    "Jobs processed":units,
                    "GTT":(sum(job.get_GTT() for job in JOBS_DELIVERED_DEBUG)/units),
                    "SFT":(sum(job.get_SFT() for job in JOBS_DELIVERED_DEBUG)/units),
                    "Tardiness":(sum(job.get_tardiness() for job in JOBS_DELIVERED_DEBUG)/units),
                    "Lateness":(sum(job.get_lateness() for job in JOBS_DELIVERED_DEBUG)/units),
                    "Tardy":(sum(job.get_tardy() for job in JOBS_DELIVERED_DEBUG)/float(units)),
                    "STDLateness":(np.std(np.array(list(job.get_lateness() for job in JOBS_DELIVERED_DEBUG)))),
                    }
                # Queues information
                result["PSP Shop Load"]=(sum(sum(job.ProcessingTime) for job in system.PSP))
                sl = get_shop_load(system.Pools)
                for i in range(N_PHASES):
                    result["Shop Load-"+str(i)]=sl[i]
                result["Total Shop Load"]=(sum(sl))
                # Workers information
                # worker idleness
                worker_total_working_time=list()
                for i in range(N_WORKERS):
                    if((sum(system.Workers[i].WorkingTime)) > 0):
                        worker_total_working_time.append(sum(system.Workers[i].WorkingTime))
                    else:
                        worker_total_working_time.append(-1)
                for i in range(N_WORKERS):
                    result["Worker "+str(i)+" Idleness(%)"]=(env.now-WARMUP-worker_total_working_time[i])/(env.now-WARMUP)*100
                    result["Total Idle Time"]=((env.now-WARMUP)*5-sum(worker_total_working_time[worker.id] for worker in system.Workers))
                # workers extra load %
                for i in range(N_WORKERS):
                    result["Worker "+str(i)+" out(%)"]=((worker_total_working_time[i]-system.Workers[i].WorkingTime[i])/worker_total_working_time[i])*100
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
                units = -1
                if len(JOBS_DELIVERED_DEBUG)>0:
                    units=len(JOBS_DELIVERED_DEBUG)
                results_DEBUG[results_index]["WL entry"]+=(sum(sum(job.ProcessingTime) for job in JOBS_ENTRY_DEBUG))
                results_DEBUG[results_index]["WL released"]+=(sum(sum(job.ProcessingTime) for job in JOBS_RELEASED_DEBUG))
                results_DEBUG[results_index]["WL processed"]+=(sum(sum(job.ProcessingTime) for job in JOBS_DELIVERED_DEBUG))
                results_DEBUG[results_index]["Jobs processed"]+=(units)
                results_DEBUG[results_index]["GTT"]+=(sum(job.get_GTT() for job in JOBS_DELIVERED_DEBUG)/units)
                results_DEBUG[results_index]["SFT"]+=(sum(job.get_SFT() for job in JOBS_DELIVERED_DEBUG)/units)
                results_DEBUG[results_index]["Tardiness"]+=(sum(job.get_tardiness() for job in JOBS_DELIVERED_DEBUG)/units)
                results_DEBUG[results_index]["Lateness"]+=(sum(job.get_lateness() for job in JOBS_DELIVERED_DEBUG)/units)
                results_DEBUG[results_index]["Tardy"]+=(sum(job.get_tardy() for job in JOBS_DELIVERED_DEBUG)/float(units))
                results_DEBUG[results_index]["STDLateness"]+=(np.std(np.array(list(job.get_lateness() for job in JOBS_DELIVERED_DEBUG))))
                # Queues information
                results_DEBUG[results_index]["PSP Shop Load"]+=(sum(sum(job.ProcessingTime) for job in system.PSP))
                sl = get_shop_load(system.Pools)
                for i in range(N_PHASES):
                    results_DEBUG[results_index]["Shop Load-"+str(i)]+=sl[i]
                results_DEBUG[results_index]["Total Shop Load"]+=(sum(sl))
                # Workers information
                # worker idleness
                worker_total_working_time=list()
                for i in range(N_WORKERS):
                    if((sum(system.Workers[i].WorkingTime)) > 0):
                        worker_total_working_time.append(sum(system.Workers[i].WorkingTime))
                    else:
                        worker_total_working_time.append(-1)
                for i in range(N_WORKERS):
                    results_DEBUG[results_index]["Worker "+str(i)+" Idleness(%)"]+=(env.now-WARMUP-worker_total_working_time[i])/(env.now-WARMUP)*100
                    #print ("Worker "+str(i)+" Idleness(%)", (env.now-WARMUP-worker_total_working_time[i])/(env.now-WARMUP)*100)
                    results_DEBUG[results_index]["Total Idle Time"]+=((env.now-WARMUP)*5-sum(worker_total_working_time[worker.id] for worker in system.Workers))
                # workers extra load %
                for i in range(N_WORKERS):
                    results_DEBUG[results_index]["Worker "+str(i)+" out(%)"]+=((worker_total_working_time[i]-system.Workers[i].WorkingTime[i])/worker_total_working_time[i])*100
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
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames,lineterminator="\r")
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
                    ### USE IF DYNAMIC SYSTEM ###

                    'Constraint Scenario': CONSTRAINT_SCENARIO,
                    'Current Human Norm': CURRENT_HUMAN_NORM,
                    'Current Machine Norm': CURRENT_MACHINE_NORM,
                    'Constraint Switches': getattr(system.constraint_manager, 'switch_count', 0)

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
            ### USE IF DYNAMIC SYSTEM ###

            if len(system.PSP) > 0:
                print(f"PSP size: {len(system.PSP)}")
                print(f"Current constraints - Human: {CURRENT_HUMAN_NORM:.0f}, Machine: {CURRENT_MACHINE_NORM:.0f}")
                # Sample a few jobs to see why they're not being released
                if len(system.PSP) > 5:
                    sample_job = system.PSP[0]
                    machine_load, human_load = get_corrected_shop_load_with_ratios(system.Pools)
                    job_m_load, job_h_load = sample_job.get_dual_CSL_routing_only()
                    print("Sample job blocking reasons:")
                    for station_id in sample_job.Routing:
                        m_limit = CURRENT_MACHINE_NORM/N_PHASES
                        h_limit = CURRENT_HUMAN_NORM/N_PHASES
                        m_would_exceed = (machine_load[station_id] + job_m_load[station_id]) > m_limit
                        h_would_exceed = (human_load[station_id] + job_h_load[station_id]) > h_limit if HUMAN_MACHINE_PHASES.get(station_id, False) else False
                        if m_would_exceed or h_would_exceed:
                            print(f"  Station {station_id}: M_exceed={m_would_exceed}, H_exceed={h_would_exceed}")


            if (env.now > WARMUP):
                net_time=env.now-WARMUP
                #   machines info
                for machine in system.Machines:
                    print("Machine %d \t Jobs processed %d \t WL processed %f \t Eff. %f \t Utilization %f" %(machine.id, machine.JobsProcessed, machine.WorkloadProcessed,machine.efficiency, machine.WorkloadProcessed/(net_time)))
                print("\n")
                #   Workers info
                for worker in system.Workers:
                    sumworkingtime = sum(worker.WorkingTime)
                    sWorkingTime = [str(round(wt/(net_time)*100,2)) for wt in worker.WorkingTime]
                    print("Worker:%d \t Current machine:%d\t"%(worker.id, worker.current_machine_id) + "\t ".join(sWorkingTime) +"\t Home:"+ str(round(worker.WorkingTime[worker.Default_machine]/net_time,2))+"\t Idle:"+ str(round((net_time-sumworkingtime)/float(net_time)*100,2)) +"\t% out: " + str(round((sum(worker.WorkingTime) - worker.WorkingTime[worker.Default_machine])/(net_time)*100,2)))
                    print(sumworkingtime/net_time)
                if RELEASE_RULE=="AL_MOD":
                    print("MAX extra cap per phase: ", "\t".join(list(str(sum(worker.Capacity_adjustment[i] for worker in system.Workers)) for i in range(N_PHASES))))
                #   System info
                print("\nAvg. Job generator output rate: %f" %(system.generator.generated_orders/float(net_time)))
                print("Avg. Jobs delivered rate[jobs/day]: %f" %(FinishedUnits/float(net_time)))
                print("Jobs delivered/Jobs generated: %f"%((FinishedUnits)/float(system.generator.generated_orders)))
                """
                z = (str(PT) for PT in system.generator.generated_processing_time)
                print("WL generated:\t", "\t".join(z))
                print("WL released:\t", "\t".join((str(PT) for PT in system.OR.released_workload)))
                print("Machine WL:\t", "\t".join(str(machine.WorkloadProcessed) for machine in system.Machines))
                y = (str(round(sum(worker.WorkloadProcessed[i] for worker in system.Workers),6)) for i in range(N_PHASES))
                print("Worker WL:\t", "\t".join(y))
                x = (str(sum(job.ProcessingTime[i] for job in system.Jobs_delivered)) for i in range(N_PHASES))
                print("Jobs total WL:\t", "\t".join(x))
                print(sum(sum(job.ProcessingTime) for job in system.Jobs_delivered)/sum(sum(worker.WorkloadProcessed) for worker in system.Workers))
                print()
                print("RELEASED/GENERATED\t", (sum(PT for PT in system.OR.released_workload)/(sum(PT for PT in system.generator.generated_processing_time))))
                print("PROC. Work/RELEASED\t", (sum(sum(worker.WorkloadProcessed[i] for worker in system.Workers)for i in range(N_PHASES)) /sum(PT for PT in system.OR.released_workload)))
                print("PROC. Mach/RELEASED\t", (sum(machine.WorkloadProcessed for machine in system.Machines) /sum(PT for PT in system.OR.released_workload)))
                print("DELIVERED/PROC. Mach\t", (sum(sum(job.ProcessingTime) for job in system.Jobs_delivered)/sum(machine.WorkloadProcessed for machine in system.Machines)))
                """
            else:
                #   machines info
                for machine in system.Machines:
                    print("Machine %d \t Jobs processed %d \t WL processed %f\t Eff. %f  \t Utilization %f" %(machine.id, machine.JobsProcessed, machine.WorkloadProcessed,machine.efficiency, machine.WorkloadProcessed/(env.now)))
                print("\n")
                #   Workers info
                for worker in system.Workers:
                    sumworkingtime = sum(worker.WorkingTime)
                    sWorkingTime = [str(round(wt/env.now*100,2)) for wt in worker.WorkingTime]
                    print("Worker:%d \t Current machine:%d\t"%(worker.id, worker.current_machine_id) + "\t ".join(sWorkingTime) +"\t Home:"+"\t Idle:"+ str(round((env.now-sumworkingtime)/float(env.now)*100,2)) +"\t% out: " + str(round((sum(worker.WorkingTime) - worker.WorkingTime[worker.Default_machine])/env.now*100,2)))
                if RELEASE_RULE=="AL_MOD":
                    print("MAX extra cap per phase: ", "\t".join(list(str(sum(worker.Capacity_adjustment[i] for worker in system.Workers)) for i in range(N_PHASES))))
                #   System info
                print("\nAvg. Job generator output rate: %f" %(system.generator.generated_orders/float(env.now)))
                print("Avg. Jobs delivered rate[jobs/day]: %f" %(FinishedUnits/float(env.now)))
                print("Jobs delivered/Jobs generated: %f"%((FinishedUnits)/float(system.generator.generated_orders)))
                """
                z = (str(PT) for PT in system.generator.generated_processing_time)
                print("WL generated:\t", "\t".join(z))
                print("WL released:\t", "\t".join((str(PT) for PT in system.OR.released_workload)))
                print("Machine WL:\t", "\t".join(str(machine.WorkloadProcessed) for machine in system.Machines))
                y = (str(round(sum(worker.WorkloadProcessed[i] for worker in system.Workers),6)) for i in range(N_PHASES))
                print("Worker WL:\t", "\t".join(y))
                #a = ((float(z[i]) - float(y[i])) for i in range(N_PHASES))
                #print("Generated-processed total WL:", "\t".join(str(i) for i in a))
                #print("Jobs total WL:", sum(sum(job.ProcessingTime) for job in system.Jobs_delivered))
                x = (str(sum(job.ProcessingTime[i] for job in system.Jobs_delivered)) for i in range(N_PHASES))
                print("Jobs total WL:\t", "\t".join(x))
                print(sum(sum(job.ProcessingTime) for job in system.Jobs_delivered)/sum(sum(worker.WorkloadProcessed) for worker in system.Workers))
                print()
                print("RELEASED/GENERATED\t", (sum(PT for PT in system.OR.released_workload)/(sum(PT for PT in system.generator.generated_processing_time))))
                print("PROC. Work/RELEASED\t", (sum(sum(worker.WorkloadProcessed[i] for worker in system.Workers)for i in range(N_PHASES)) /sum(PT for PT in system.OR.released_workload)))
                print("PROC. Mach/RELEASED\t", (sum(machine.WorkloadProcessed for machine in system.Machines) /sum(PT for PT in system.OR.released_workload)))
                print("DELIVERED/PROC. Mach\t", (sum(sum(job.ProcessingTime) for job in system.Jobs_delivered)/sum(machine.WorkloadProcessed for machine in system.Machines)))
                """
            #print("GTT", sum(job.get_GTT() for job in system.Jobs_delivered)/len(system.Jobs_delivered))
            #print("SFT", sum(job.get_SFT() for job in system.Jobs_delivered)/len(system.Jobs_delivered))

    def analyze_dual_constraint_results():
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
            system = System(env)
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
            if CSV_OUTPUT_JOBS is True or CSV_OUTPUT_SYSTEM is True:
                access_type = 'w'
                if run > 0 or WLIndex > 0:
                    access_type = 'a'
                if CSV_OUTPUT_JOBS is True:
                    with open('JobsOutput_' + RELEASE_RULE + "_" + WORKER_MODE + "_" +
                            WORKER_FLEXIBILITY + "_" + str(WORKLOAD_NORMS[WLIndex]) +
                            "_" + str(run) + '.csv', access_type) as csvfile:
                        fieldnames = ['Workload','nrun','id', 'Arrival Date', 'Due Date',
                                    'Completation Date', 'GTT','SFT','Tardiness','Lateness', 'ForceReleased']
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
                        # ... rest of your existing fieldnames ...
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,lineterminator="\r")
                        if access_type == 'w':
                            writer.writeheader()
                        for job in system.Jobs_delivered.get_list():
                            row = {
                                'Workload': WORKLOAD_NORMS[WLIndex],
                                'nrun': run,
                                'id': job.id,
                                'Arrival Date': job.ArrivalDate,
                                'Due Date': job.DueDate,
                                'Completation Date': job.CompletationDate,
                                'GTT': job.get_GTT(),
                                'SFT': job.get_SFT(),
                                'Tardiness': job.get_tardiness(),
                                'Lateness': job.get_lateness(),
                                'ForceReleased': job.force_released
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
                    with open('SystemOutput_' + RELEASE_RULE + "_" + WORKER_MODE + "_" + str(
                        STARVATION_AVOIDANCE) + "_" + WORKER_FLEXIBILITY + "_" + str(WORKER_EFFICIENCY_DECREMENT) + '.csv',
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
                        # 'Job Entry',
                        'Exit Rate',
                        'Total Processed Workload',
                        'Forced_releases_count',
                        # 'Jobs intormation',
                        'Av. GTT','Av. SFT','Av. Tardiness','Av. Lateness','Tardy','STD Lateness',
                        'Constraint Scenario', 'Current Human Norm', 'Current Machine Norm', 'Constraint Switches'
                        ]
                        # Capacity information
                        for i in range(N_PHASES):
                            fieldnames.append("Cap_mach_"+str(i))
                            fieldnames.append("Cap_hum_"+str(i))
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
                        for i in range(N_WORKERS):
                            fieldnames.append("W" + str(i) + " TimeInTransfer")
                        for i in range(N_WORKERS):
                            fieldnames.append("W" + str(i) + " PctTimeOffHome")
                        # workers relocation to a specific machine
                        for i in range(N_WORKERS):
                            for j in range(N_PHASES):
                                fieldnames.append("Rel-W" + str(i) + "-M" + str(j))
                        # Machines
                        for i in range(N_PHASES):
                            fieldnames.append("Machine "+str(i)+" eff.(%)")
                        for i in range(N_PHASES):
                            fieldnames.append("M"+str(i)+" OvermannedMins")
                            fieldnames.append("M"+str(i)+" EmptyMins")

                        # WB_MOD Telemetry
                        fieldnames.extend([
                            'WBMOD_solve_status', 'WBMOD_psp_size', 'WBMOD_subset_size',
                            'WBMOD_solve_time_sec', 'WBMOD_jobs_released', 'WBMOD_adj_minutes_planned_t0'
                        ])

                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\r")
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
                            'Forced_releases_count':system.OR.Forced_releases_count,
                            'Av. GTT': (sum(job.get_GTT() for job in jobs)/FinishedUnits),
                            'Av. SFT': (sum(job.get_SFT() for job in jobs)/FinishedUnits),
                            'Av. Tardiness':(sum(job.get_tardiness() for job in jobs)/FinishedUnits),
                            'Av. Lateness':(sum(job.get_lateness() for job in jobs)/FinishedUnits),
                            'Tardy':(sum(job.get_tardy() for job in jobs)/float(FinishedUnits)),
                            'STD Lateness':(np.std(np.array(list(job.get_lateness() for job in jobs)))),
                            ### USE IF DYNAMIC SYSTEM ###
                            'Constraint Scenario': CONSTRAINT_SCENARIO,
                            'Current Human Norm': CURRENT_HUMAN_NORM,
                            'Current Machine Norm': CURRENT_MACHINE_NORM,
                            'Constraint Switches': getattr(system.constraint_manager, 'switch_count', 0)
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
                        # Capacity information
                        for i in range(N_PHASES):
                            row["Cap_mach_"+str(i)] = system.Cap_mach[i]
                            row["Cap_hum_"+str(i)] = system.Cap_hum[i]
                        # Worker idleness
                        for i in range(N_WORKERS):
                            row["W"+str(i)+" Idleness"]=(net_time-sum(system.Workers[i].WorkingTime))/(net_time)
                        # Worker total relocations
                        for i in range(N_WORKERS):
                            row["W" + str(i) + " Relocations"] = sum(system.Workers[i].relocation)
                            row["W" + str(i) + " TimeInTransfer"] = system.Workers[i].time_in_transfer
                            total_working_time = sum(system.Workers[i].WorkingTime)
                            if total_working_time > 0:
                                time_off_home = total_working_time - system.Workers[i].WorkingTime[system.Workers[i].Default_machine]
                                row["W" + str(i) + " PctTimeOffHome"] = (time_off_home / total_working_time) * 100
                            else:
                                row["W" + str(i) + " PctTimeOffHome"] = 0
                        # Subdivision of workers relocations
                        for i in range(N_WORKERS):
                            for j in range(N_PHASES):
                                row["Rel-W" + str(i) + "-M" + str(j)] = system.Workers[i].relocation[j]
                        # Machines
                        for machine in system.Machines:
                            row["Machine "+str(machine.id)+" eff.(%)"]=(machine.WorkloadProcessed/(net_time))
                            row["M"+str(machine.id)+" OvermannedMins"] = machine.overmanned_minutes
                            row["M"+str(machine.id)+" EmptyMins"] = machine.empty_station_minutes

                        # Add new telemetry values
                        row.update({
                            'WBMOD_solve_status': system.WBMOD_stats['solve_status'],
                            'WBMOD_psp_size': system.WBMOD_stats['psp_size'],
                            'WBMOD_subset_size': system.WBMOD_stats['subset_size'],
                            'WBMOD_solve_time_sec': system.WBMOD_stats['solve_time_sec'],
                            'WBMOD_jobs_released': system.WBMOD_stats['jobs_released'],
                            'WBMOD_adj_minutes_planned_t0': system.WBMOD_stats['adj_minutes_planned_t0']
                        })

                        writer.writerow(row)
        # <if run debug>
        if RUN_DEBUG:
            debug5_write()
            results_DEBUG=list()
        #

    print("\nEnd of the simulation")
    print("Simulation time: " + str(time.time()-start) + " sec")
