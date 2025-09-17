import pulp
import math

def run_wb_mod_optimizer(psp_jobs, system, config):
    """
    Runs the Workload Balancing (WB_MOD) optimizer.
    """
    print("Optimizer: Starting.")
    # --- Parameters ---
    T = config.get("review_horizon_T", 5)
    RHO = config.get("wb_mod_rho", 1.5)
    KAPPA = config.get("wb_mod_kappa", 0.1)
    WEIGHTS_DECAY = config.get("wb_mod_weights_decay", 0.5)
    N_PHASES = config.get("n_phases")
    print("Optimizer: Parameters set.")

    # --- Problem Definition ---
    model = pulp.LpProblem("WB_MOD_Release_Planning", pulp.LpMinimize)

    # --- Sets and Indices ---
    job_ids = [j.id for j in psp_jobs]
    stations = range(N_PHASES)
    time_periods = range(T)

    # --- Model Parameters ---
    csl_mach = {j.id: j.get_CSL_with_ratios_routing_only()[0] for j in psp_jobs}
    csl_hum = {j.id: j.get_CSL_with_ratios_routing_only()[1] for j in psp_jobs}

    cap_mach = { (t, i): system.Cap_mach[i] for t in time_periods for i in stations }
    cap_hum = { (t, i): system.Cap_hum[i] for t in time_periods for i in stations }

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

    for j_id in job_ids:
        model += pulp.lpSum(x[j_id][t] for t in time_periods) <= 1
    print("Optimizer: Constraints defined.")

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(timeLimit=10, msg=1) # Reduced time limit, increased verbosity
    print("Optimizer: Calling solver.")
    model.solve(solver)
    print("Optimizer: Solver finished.")

    # --- Extract Results ---
    jobs_to_release = []
    adj_plan = {}
    if pulp.LpStatus[model.status] == 'Optimal':
        print("WB_MOD Optimizer: Optimal solution found.")
        for j_id in job_ids:
            if pulp.value(x[j_id][0]) > 0.99:
                jobs_to_release.append(j_id)

        adj_plan_t0 = {}
        for i in stations:
            for r in stations:
                if i != r and pulp.value(Adj[0][i][r]) > 0:
                    adj_plan_t0[(i, r)] = pulp.value(Adj[0][i][r])
        adj_plan = adj_plan_t0
    else:
        print(f"WB_MOD Optimizer: No optimal solution found. Status: {pulp.LpStatus[model.status]}")
        return [], {}

    return jobs_to_release, adj_plan
