import pulp
import math
import time

def run_wb_mod_optimizer(psp_jobs, system, horizon_T, time_limit_sec=10):
    """
    Runs the Workload Balancing (WB_MOD) optimizer.

    This function formulates and solves a linear programming model to decide which
    jobs to release from the Pre-Shop Pool (PSP) and how to allocate labor
    resources over a planning horizon.

    Args:
        psp_jobs (list): A list of Job objects currently in the PSP.
        system (System): The main system object, providing access to capacities
                         and other system-wide parameters.
        horizon_T (int): The number of time periods in the planning horizon.
        time_limit_sec (int): The maximum time in seconds allowed for the solver.

    Returns:
        tuple: A tuple containing:
            - jobs_to_release_ids (list): A list of IDs of jobs to be released in the
                                          first period (t=0).
            - adj_plan_t0 (dict): A dictionary representing the labor adjustment plan
                                  for the first period, e.g., {(to_station, from_station): minutes}.
            - status_string (str): The status of the solver solution (e.g., "Optimal",
                                   "TimeLimit", "Infeasible").
            - solve_time_sec (float): The total time taken to solve the model.
    """
    start_time = time.time()
    print("Optimizer: Starting.")

    # --- Parameters ---
    # Parameters are now accessed from the system object or passed directly.
    T = horizon_T
    RHO = system.WB_MOD_PARAMS["rho"]
    KAPPA = system.WB_MOD_PARAMS["kappa"]
    WEIGHTS_DECAY = system.WB_MOD_PARAMS["weights_decay"]
    N_PHASES = system.N_PHASES
    print(f"Optimizer: Parameters set (T={T}, RHO={RHO}, KAPPA={KAPPA}, TIMEOUT={time_limit_sec}s).")

    # --- Problem Definition ---
    model = pulp.LpProblem("WB_MOD_Release_Planning", pulp.LpMinimize)

    # --- Sets and Indices ---
    job_ids = [j.id for j in psp_jobs]
    stations = range(N_PHASES)
    time_periods = range(T)

    # --- Model Parameters (pre-calculated from job and system data) ---
    csl_mach = {j.id: j.get_CSL_with_ratios_routing_only()[0] for j in psp_jobs}
    csl_hum = {j.id: j.get_CSL_with_ratios_routing_only()[1] for j in psp_jobs}

    # Capacities are assumed constant over the horizon for this model version
    cap_mach = {(t, i): system.Cap_mach[i] for t in time_periods for i in stations}
    cap_hum = {(t, i): system.Cap_hum[i] for t in time_periods for i in stations}

    w = {t: math.pow(WEIGHTS_DECAY, t + 1) for t in time_periods}
    print("Optimizer: Model parameters calculated.")

    # --- Decision Variables ---
    x = pulp.LpVariable.dicts("Release", (job_ids, time_periods), cat='Binary')
    U_mach = pulp.LpVariable.dicts("Underutil_Mach", (time_periods, stations), lowBound=0)
    O_mach = pulp.LpVariable.dicts("Overutil_Mach", (time_periods, stations), lowBound=0)
    U_hum = pulp.LpVariable.dicts("Underutil_Hum", (time_periods, stations), lowBound=0)
    O_hum = pulp.LpVariable.dicts("Overutil_Hum", (time_periods, stations), lowBound=0)
    Adj = pulp.LpVariable.dicts("Adjust", (time_periods, stations, stations), lowBound=0)
    print("Optimizer: Variables defined.")

    # --- Objective Function ---
    objective = pulp.lpSum(
        w[t] * (U_mach[t][i] + U_hum[t][i] + RHO * (O_mach[t][i] + O_hum[t][i]))
        for t in time_periods for i in stations
    ) + pulp.lpSum(
        KAPPA * Adj[t][i][r] for t in time_periods for i in stations for r in stations if i != r
    )
    model += objective
    print("Optimizer: Objective defined.")

    # --- Constraints ---
    for t in time_periods:
        for i in stations:
            load_mach = pulp.lpSum(x[j_id][t] * csl_mach[j_id][i] for j_id in job_ids)
            load_hum = pulp.lpSum(x[j_id][t] * csl_hum[j_id][i] for j_id in job_ids)
            adj_in = pulp.lpSum(Adj[t][i][r] for r in stations if r != i)
            adj_out = pulp.lpSum(Adj[t][r][i] for r in stations if r != i)
            model += load_mach - cap_mach[t, i] == O_mach[t][i] - U_mach[t][i]
            model += load_hum + adj_in - adj_out - cap_hum[t, i] == O_hum[t][i] - U_hum[t][i]

    # A job can be released at most once over the entire horizon
    for j_id in job_ids:
        model += pulp.lpSum(x[j_id][t] for t in time_periods) <= 1
    print("Optimizer: Constraints defined.")

    # --- Solve ---
    # Use the passed time limit and suppress solver messages for cleaner logs
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=0)
    print("Optimizer: Calling solver...")
    model.solve(solver)

    solve_time_sec = time.time() - start_time
    status_string = pulp.LpStatus[model.status]
    print(f"Optimizer: Solver finished in {solve_time_sec:.2f}s with status: {status_string}")

    # --- Extract Results ---
    jobs_to_release_ids = []
    adj_plan_t0 = {}

    # Only extract results if a valid solution was found
    if status_string in ["Optimal", "TimeLimit", "Feasible"]:
        # Extract job release decisions for the first period (t=0)
        for j_id in job_ids:
            if pulp.value(x[j_id][0]) > 0.99: # Check if the binary variable is 1
                jobs_to_release_ids.append(j_id)

        # Extract labor adjustment plan for the first period (t=0)
        for i in stations:
            for r in stations:
                if i != r and pulp.value(Adj[0][i][r]) > 0:
                    adj_plan_t0[(i, r)] = pulp.value(Adj[0][i][r])

        print(f"WB_MOD Optimizer: Solution found. Releasing {len(jobs_to_release_ids)} jobs. Plan: {adj_plan_t0}")
    else:
        print(f"WB_MOD Optimizer: No usable solution found. Status: {status_string}")

    return jobs_to_release_ids, adj_plan_t0, status_string, solve_time_sec
