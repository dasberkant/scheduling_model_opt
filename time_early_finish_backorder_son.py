import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

def solve_rental_scheduling_time_indexed(num_orders, num_vehicles,
                                         p_list, d_list, f_list, l_list,
                                         A_list, time_granularity=1):
    """
    Solves the one-day rental scheduling model using a time-indexed formulation
    with EARLIEST FINISH logic, plus the UPDATED constraint for backorderable orders:
    
      If 24 + p_i <= d_i (i.e. order i is backorderable), then
      we require t + p_i < 24 + p_i -> t < 24.
      In code: t_max = 24 - time_granularity for those orders.

    Other conditions:
      - If 24 + p_i > d_i => order is MANDATORY (must finish by d_i).
      - Earliest finish F_i and latest finish L_i => F_i <= t + p_i <= L_i if scheduled.
      - Objective: maximize sum of leftover margins = Σ [d_i - (t + p_i)].
    """

    # 24-hour planning horizon discretized by "time_granularity"
    H = 24
    T = list(range(0, H + 1, time_granularity))

    # Index sets for orders and vehicles
    I = list(range(1, num_orders + 1))
    V = list(range(1, num_vehicles + 1))

    # Build Gurobi model
    model = gp.Model("TimeIndexedRentalScheduling")
    model.setParam("OutputFlag", 0)  # Mute solver output

    # Decision var: y[i,t,v] = 1 if order i starts at time t on vehicle v
    y = {}
    for i in I:
        p_i = p_list[i - 1]
        d_i = d_list[i - 1]
        F_i = f_list[i - 1]  # earliest finish
        L_i = l_list[i - 1]  # latest finish

        for v in V:
            A_v = A_list[v - 1]  # vehicle availability time

            # Distinguish mandatory vs. backorderable
            if 24 + p_i > d_i:
                # MANDATORY -> must finish by d_i => t_max = d_i - p_i
                t_max = d_i - p_i
            else:
                # BACKORDERABLE -> updated: t + p_i < 24 + p_i => t < 24
                # so the largest feasible t is 24 - time_granularity
                t_max = 24 - time_granularity

            for t in T:
                # Must not start before the vehicle is available
                if t < A_v:
                    continue
                # Must not start after t_max
                if t > t_max:
                    continue

                finish_time = t + p_i
                # Enforce earliest/ latest finish: F_i <= finish_time <= L_i
                if finish_time < F_i or finish_time > L_i:
                    continue

                # If all conditions pass, add a binary variable
                y[(i, t, v)] = model.addVar(vtype=GRB.BINARY,
                                            name=f"y_{i}_{t}_{v}")

    model.update()

    # z[i] = 1 if order i is backordered
    z = model.addVars(I, vtype=GRB.BINARY, name="z")

    # 1. Each order either scheduled exactly once or backordered
    for i in I:
        expr = gp.quicksum(y[k] for k in y if k[0] == i)
        model.addConstr(expr + z[i] == 1, name=f"AssignBack_{i}")

    # 2. Must-schedule constraint for mandatory orders
    for i in I:
        p_i = p_list[i - 1]
        d_i = d_list[i - 1]
        if 24 + p_i > d_i:
            model.addConstr(z[i] == 0, name=f"MustSchedule_{i}")

    # 3. Non-overlap: each vehicle can only process one order at a time
    for v in V:
        for tau in T:
            expr = gp.LinExpr()
            for (i_k, t_k, v_k) in y:
                if v_k == v:
                    p_i = p_list[i_k - 1]
                    if t_k <= tau < t_k + p_i:
                        expr.add(y[(i_k, t_k, v_k)])
            model.addConstr(expr <= 1, name=f"NonOverlap_v{v}_tau{tau}")

    # 4. Objective: maximize sum of leftover margin = Σ [d_i - (t + p_i)] * y[i,t,v]
    objective_expr = gp.LinExpr()
    for (i_k, t_k, v_k) in y:
        p_i = p_list[i_k - 1]
        d_i = d_list[i_k - 1]
        margin = d_i - (t_k + p_i)
        objective_expr.add(margin * y[(i_k, t_k, v_k)])

    model.setObjective(objective_expr, GRB.MAXIMIZE)

    # Solve model
    model.optimize()
    if model.status == GRB.OPTIMAL:
        # Extract the solution
        schedule = {}
        for i in I:
            scheduled = False
            for (i_k, t_k, v_k) in y:
                if i_k == i and y[(i_k, t_k, v_k)].X > 0.5:
                    schedule[i] = (t_k, v_k, p_list[i - 1])
                    scheduled = True
                    break
            if not scheduled:
                schedule[i] = None

        z_sol = {i: int(z[i].X + 0.5) for i in I}

        # Prepare Gantt data
        gantt_data = {v: [] for v in V}
        for i in I:
            if schedule[i] is not None:
                start_time, vehicle, duration = schedule[i]
                gantt_data[vehicle].append((i, start_time, duration))

        return {
            "objective": model.ObjVal,
            "schedule": schedule,
            "z": z_sol,
            "gantt_data": gantt_data
        }
    else:
        raise ValueError("No feasible or optimal solution found.")

def main():
    st.title("One-Day Rental Scheduling")
    st.write("""
    """)

    # === Sidebar: Model Settings ===
    st.sidebar.header("Model Settings")
    num_orders = st.sidebar.number_input("Number of Orders", min_value=1, max_value=100, value=5)
    num_vehicles = st.sidebar.number_input("Number of Vehicles", min_value=1, max_value=50, value=2)
    time_granularity = st.sidebar.number_input("Time Granularity (hours)",
                                               min_value=1, max_value=4, value=1)

    # === Vehicle Availability ===
    st.subheader("Vehicle Availability")
    st.write("Enter the availability time (in hours) for each vehicle:")
    A_list = []
    for v in range(num_vehicles):
        val = st.number_input(f"Availability of Vehicle {v+1}", min_value=0.0, value=0.0, step=1.0)
        A_list.append(val)

    st.markdown("---")

    # === Order Data ===
    st.subheader("Order Data")
    st.write("""
    For each order, enter:
      - pᵢ (Processing Time),
      - dᵢ (Deadline),
      - Fᵢ (Earliest Finish),
      - Lᵢ (Latest Finish).
    """)

    p_list = []
    d_list = []
    f_list = []
    l_list = []
    for i in range(num_orders):
        cols = st.columns(4)
        with cols[0]:
            p_val = st.number_input(f"p[{i+1}] (hours)", min_value=0.0, value=2.0, step=1.0)
        with cols[1]:
            d_val = st.number_input(f"d[{i+1}] (deadline)", min_value=0.0, value=24.0, step=1.0)
        with cols[2]:
            f_val = st.number_input(f"F[{i+1}] (earliest finish)", min_value=0.0, value=0.0, step=1.0)
        with cols[3]:
            l_val = st.number_input(f"L[{i+1}] (latest finish)", min_value=0.0, value=999.0, step=1.0)

        p_list.append(p_val)
        d_list.append(d_val)
        f_list.append(f_val)
        l_list.append(l_val)

    # === Solve ===
    if st.button("Solve"):
        try:
            result = solve_rental_scheduling_time_indexed(
                num_orders, num_vehicles,
                p_list, d_list, f_list, l_list,
                A_list, time_granularity=time_granularity
            )
            st.success("Solution found!")
            st.write(f"**Objective (Total Leftover Margin):** {result['objective']:.2f}")

            # Show results
            st.write("### Schedule/Backorders")
            for i in range(1, num_orders + 1):
                sched = result["schedule"][i]
                if sched is not None:
                    start_t, veh, dur = sched
                    st.write(f"- Order {i}: Start={start_t}, Vehicle={veh}, Duration={dur}, Backordered=0")
                else:
                    st.write(f"- Order {i}: Backordered (z[{i}]=1)")

            # Gantt charts
            for v in range(1, num_vehicles + 1):
                tasks = result["gantt_data"][v]
                if not tasks:
                    st.write(f"**Vehicle {v}**: No assigned orders.")
                    continue

                st.write(f"## Gantt Chart: Vehicle {v}")
                tasks = sorted(tasks, key=lambda x: x[1])  # sort by start time

                fig, ax = plt.subplots()
                max_end = 24
                y_labels = []

                for idx, (order_id, start_time, duration) in enumerate(tasks):
                    ax.barh(idx, duration, left=start_time)
                    label_text = f"Order {order_id}\nStart={start_time}, Dur={duration}"

                    if duration < 1.5:
                        tx = start_time + duration + 0.1
                        ha = "left"
                        color = "black"
                    else:
                        tx = start_time + duration / 2
                        ha = "center"
                        color = "white"
                    ax.text(tx, idx, label_text, ha=ha, va="center", color=color, fontsize=8)

                    end_time = start_time + duration
                    if end_time > max_end:
                        max_end = end_time

                    y_labels.append(f"Order {order_id}")

                ax.set_xlim(0, max_end)
                ax.set_xlabel("Time (hours)")
                ax.set_ylabel("Orders")
                ax.set_title(f"Vehicle {v} Gantt")
                ax.set_yticks(range(len(tasks)))
                ax.set_yticklabels(y_labels)

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
