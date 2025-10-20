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

#TARGET_UTILIZATION = 0.9375
TARGET_UTILIZATION = 0.89
UNIFIED_RESULTS = []



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
                            #1.025, 
                            #1.05, 
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
            WORKLOAD_NORMS = [1600,1700,1800,2000,2100, 0]
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

        

        INPUT_RATE = (480*N_PHASES*TARGET_UTILIZATION)/ (JOBS_MEAN*(JOBS_ROUTINGS_AVG_MACHINES))/480

        

        print("INPUT RATE", INPUT_RATE)

        

        print(INPUT_RATE*480, "jobs per day")

        

        time.sleep(2)

    getShopConfigurations()

     

    if (SHOP_FLOW=="directed" and SHOP_LENGTH=="variable"):

        DUE_DATE_MAX = 3000

    elif (SHOP_FLOW=="directed" and SHOP_LENGTH==5):

        DUE_DATE_MAX = 12500

    elif (SHOP_FLOW=="undirected" and SHOP_LENGTH==5):

        DUE_DATE_MAX = 12500

    elif (SHOP_FLOW=="undirected" and SHOP_LENGTH=="variable"):

        DUE_DATE_MAX = 1500

    else:

        print("shop configuration not recognised. Setiing an aribitrary due date between 2000 and 4000 mins")

        time.sleep(10)

        DUE_DATE_MIN = 2000

        DUE_DATE_MAX = 4000


#####CLASES#####

    class Job(object):



        def __init__(self,env, id): #

            

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



            DUE_DATE_MIN = 83.686*len(self.Routing)*HUMAN_RATIO
            

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
            # Add bounds checking to prevent IndexError
            if self.Position >= len(self.Routing):
                #print(f"ERROR: Job {self.id} Position {self.Position} exceeds Routing length {len(self.Routing)}")
                #print(f"Job Routing: {self.Routing}")
                #print(f"Job arrived at: {self.ArrivalDate}, released at: {self.ReleaseDate}")
                # Reset position to last valid position as emergency fix
                self.Position = len(self.Routing) - 1
                
            return self.Routing[self.Position]

        def get_CAW(self):

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

            """ contribution to the corrected shop load """

            

            load = list(0 for i in range(N_PHASES))

            

            for i in range(N_PHASES):

                

                load[i]=self.ProcessingTime[i]/len(self.Routing)

            

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
            load = list(0 for i in range(N_PHASES))

            for i, station_id in enumerate(self.Routing[self.Position:]):
                # i starts from 0 for remaining stations
                position_correction = i + 1  # First remaining station gets weight 1, second gets 1/2, etc.
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

        def get_CAW_with_ratios_routing_only(self):
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



        def __init__(self, env, PoolDownstream):



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

 ###JOBS###
   
    class Orders_release(object):



        def __init__(self, env, rule, system, workload_norm =- 1):

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
            """Get workload norm adjusted for machine reliability"""
            if self.downtime_manager and base_workload_norm > 0:
                return self.downtime_manager.get_adjusted_workload_norm(base_workload_norm)
            return base_workload_norm

        def Immediate_release(self):

            

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
                
                all_jobs.sort(key=lambda x: x.ArrivalDate)  # Changed from x.DueDate
                
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
                
                all_jobs.sort(key=lambda x: x.DueDate)  # Changed from x.DueDate
                
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
                
                all_jobs.sort(key=lambda x: x.DueDate)
                
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
                    
                    all_jobs.sort(key=lambda x: x.DueDate)
                    
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
                
                jobs_to_evaluate.sort(key=lambda x: x.DueDate)
                
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
                    
                    jobs_to_evaluate.sort(key=lambda x: x.DueDate)
                    
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


        def __init__(self, env, id, Pools):

            

            self.env = env

            

            self.Pools=Pools

            

            self.id = id

            

            self.array = list()  #list of jobs

            

            # self.waiting_new_jobs is an external trigger (used in Machine::Machine_loop() and Worker::_ReactiveWorker) to notify the availability of new jobs

            self.waiting_new_jobs = list()

            

            # self.workload_limit_triggers is an external trigger (used in Worker::_Flexible_loop()) to notify whether the lower workload limit has been reached

            self.workload_limit_triggers = list()   # ([id_worker,limit_trigger,event])
            
        def __getitem__(self, key):

            # Used to implement "Pool[x]"

            

            return self.array[key]

        def __len__(self):

            # Used to implement "len(Pool)"

            

            return len(self.array)        

        def append(self, job):

                

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

            """ 

            The function extracts(then deletes too) the item at the location 'index'. If no location is defined it returns the first element of the list.

            Every time a job extraction happens the pool verifies whether it is necessary to comunicate the stations that the trigger level has been exceeded

            """

                

            temp = self.array[index]



            del(self.array[index])

     

            return temp

        def get_list(self):

            

            return self.array            

        def delete(self,index):



            del(self.array[index])

        def sort(self):

            

            self.array.sort(key=lambda x: x.ArrivalDate)

    class Machine(object):

    
        def __init__(self, env, id, PoolUpstream, Jobs_delivered, Pools, PSP, 
                    has_worker=False, processing_mode='machine_centric',
                    absenteeism_manager=None, downtime_manager=None):
            
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

        

        def __init__(self, env, id, Pools, Machines,Default_machine, skills = None):

            

            self.env = env

            

            self.skillperphase = list()

            

            self.id = id



            self.current_machine_id = -1

            # None raises error while printing 

            

            self.Default_machine=Default_machine



            # self.relocation --> vector to count the # of times a relocation happens

            self.relocation = list(0 for i in range(N_PHASES))



            if WORKER_FLEXIBILITY == 'triangular':

                

                self._SetTriangularSkills(WORKER_EFFICIENCY_DECREMENT)



                print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))

                

            elif WORKER_FLEXIBILITY == 'chain':

                

                self._SetChainSkills()



                print("Vector of flexibility for worker %d: " %self.id +str(self.skillperphase))



            elif WORKER_FLEXIBILITY == 'chain upstream':



                self._SetChainUpstreamSkills()



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



            for i in range(0,N_PHASES):

                

                if i == self.Default_machine:



                    self.skillperphase.append(1)



                else:



                    self.skillperphase.append(0)

        #def _SetFlatSkills(self):

        #   self.skillperphase = [1,1,1,1,1]

        def _SetTriangularSkills(self, decrement):

            

            temp=list()

            

            for i in range(3):  # 0, 1 , 2 

                

                for i in range(N_PHASES):

                    

                    temp.append(0)

                    

            #for i in range(len(temp)):  # 0,1,2,3,4,    5,6,7,8,9,    10,11,12,13,14        

                

            #    temp[i] = max(1 - abs(self.Default_machine + N_PHASES - i) * decrement, 0)

             

            temp[self.Default_machine + N_PHASES] = 1 

            temp[self.Default_machine + N_PHASES+1] = 1 - decrement

            temp[self.Default_machine + N_PHASES-1] = 1 - decrement

            

            for i in range(N_PHASES):

                

                temp_sum = 0

                

                for j in range(3):

                    

                    temp_sum += temp[i+N_PHASES*j]

                    

                self.skillperphase.append(temp_sum)

        def _SetExponentialSkills(self, decrement):

               

            for i in range(0, N_PHASES):

                

                self.skillperphase.append(max(pow(decrement, abs(self.Default_machine - i)), 0))

        def _SetChainSkills(self):

            

            self.skillperphase = list(0 for i in range(N_PHASES))

            

            self.skillperphase[self.Default_machine] = 1

            

            if self.Default_machine == N_PHASES-1:

                

                self.skillperphase[0] = 1-WORKER_EFFICIENCY_DECREMENT

            

            else:

                

                self.skillperphase[self.Default_machine+1] = 1-WORKER_EFFICIENCY_DECREMENT

        def _SetChainUpstreamSkills(self):



            self.skillperphase = list(0 for i in range(N_PHASES))



            self.skillperphase[self.Default_machine] = 1



            if self.Default_machine == N_PHASES - 1:



                self.skillperphase[3] = 1 - WORKER_EFFICIENCY_DECREMENT



            else:



                self.skillperphase[self.Default_machine - 1] = 1 - WORKER_EFFICIENCY_DECREMENT

        def _SetPlainSkills(self):

            

            self.skillperphase=list()

            for i in range(N_PHASES):

                

                self.skillperphase.append(1-WORKER_EFFICIENCY_DECREMENT)

            

            self.skillperphase[self.Default_machine] = 1

        def _StaticicWorker(self, Machines):

            """

            The worker is assigned to its default station and never works at other departments

            """  

            

            self.current_machine_id = self.Default_machine

            

            Machines[self.Default_machine].Workers.append(self)

            

            Machines[self.Default_machine].process.interrupt()

            

            if Machines[self.current_machine_id].waiting_new_workers.triggered == False:

                

                Machines[self.current_machine_id].waiting_new_workers.succeed()    

        def _ReactiveWorker(self, Machines,Pools):

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
        """
        Planning-based absenteeism following Chapter 3 dual-impact framework
        """
        def __init__(self, env, absenteeism_level_key, system):
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
            """
            Calculate productivity factor using Chapter 3 Eq 3.2:
            PT_h,s(t) = PT_h,s^0 / (1 - p_s * delta_s)
            """
            productivity_factor = 1.0 * (1.0 + self.p_s * self.delta_s)
            return productivity_factor
        
        def daily_planning_process(self):
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
            """Fire the release trigger"""
            if (hasattr(self.system, 'release_trigger') and 
                self.system.release_trigger is not None and 
                not self.system.release_trigger.triggered):
                self.system.release_trigger.succeed()
        
        def get_station_wln_factor(self, station_id):
            """Get WLN factor - Chapter 3 Eq 3.1"""
            return self.station_wln_factors[station_id]
        
        def get_station_productivity_factor(self, station_id):
            """Get productivity factor - Chapter 3 Eq 3.2"""
            return self.station_productivity_factors[station_id]

    class MachineDowntimeManager:
        """
        Manages machine downtime with event-driven release wake-ups
        - Degradation: Reduced WLN, no blocking
        - Breakdown: WLN=0, workstation blocking
        """
        def __init__(self, env, downtime_level_key, system):
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
            """Fire the release trigger to wake up job release process"""
            if (self.system.release_trigger is not None and 
                not self.system.release_trigger.triggered):
                
                #if SCREEN_DEBUG:
                 #   print(f"Time {self.env.now/480:.1f}: Triggering release wake-up ({event_type} at station {station_id})")
                
                self.system.release_trigger.succeed()
        
        def get_station_reliability_factor(self, station_id):
            """Get dynamic WLN reliability factor"""
            return self.reliability_factors[station_id]
        
        def is_machine_available(self, station_id):
            """Check if machine is available for processing (not broken down)"""
            return self.machine_available[station_id]
        
        def get_machine_repair_time(self, station_id):
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

    
        def __init__(self, env, absenteeism_level_key="none", downtime_level_key="none", 
                    machine_absenteeism_type="daily"):
            
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

        # The result is a vector of workloads. It considers also the contribution of processing times already processed of jobs

        # that have not been delivered yet



        shop_load = list(0 for i in range(N_PHASES))

        

        for pool in pools:

            

            for job in pool:

                

                for contribution_index in range(N_PHASES):

                    

                    shop_load[contribution_index]+=job.ProcessingTime[contribution_index]        



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
