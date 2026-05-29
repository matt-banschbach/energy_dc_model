import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ortools.math_opt.python import mathopt

st.set_page_config(page_title="Energy DC Optimizer", layout="wide")
st.title("Energy DC Procurement Optimizer")
st.caption("Monthly MWh procurement model · costs in $/MWh · emissions in metric tons CO₂e/MWh")

# ── DEFAULT DATA ──────────────────────────────────────────────────────────────

DEFAULT_DC = pd.DataFrame({
    "data_center": ["DC_Texas", "DC_Virginia", "DC_Iowa", "DC_Oregon", "DC_Georgia"],
    "demand_mwh": [42000, 55000, 30000, 26000, 38000],
    "carbon_intensity_cap": [0.44, 0.48, 0.52, 0.50, 0.45],
})

DEFAULT_PROVIDERS = pd.DataFrame({
    "provider": ["Coal_A", "Coal_B", "Gas_A", "Gas_B", "Oil_A", "Wind_A", "Wind_B", "Solar_A", "Solar_B"],
    "source": ["coal", "coal", "lng", "lng", "oil", "wind", "wind", "solar", "solar"],
    "base_cost_per_mwh": [65, 72, 80, 95, 140, 45, 55, 40, 50],
    "emissions_tco2e_per_mwh": [0.95, 1.05, 0.42, 0.50, 0.78, 0.01, 0.02, 0.02, 0.03],
    "capacity_mwh": [60000, 45000, 65000, 45000, 25000, 40000, 35000, 37000, 32000],
})

DEFAULT_LOCATION_ADDERS = pd.DataFrame({
    "data_center": ["DC_Texas", "DC_Virginia", "DC_Iowa", "DC_Oregon", "DC_Georgia"],
    "location_adder": [0, 8, -3, 4, 6],
})

SOURCE_COLORS = {
    "coal": "#4a4a4a",
    "lng": "#f4a261",
    "oil": "#e76f51",
    "wind": "#57cc99",
    "solar": "#ffd166",
}

# ── SIDEBAR: PARAMETERS ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Parameters")

    max_intake_factor = st.slider(
        "Max Intake Factor",
        min_value=1.00, max_value=1.50, value=1.15, step=0.01,
        help="Maximum energy intake as a multiplier of demand (1.15 = 15% buffer above demand)",
    )

    # random_seed = st.number_input(
    #     "Cost Random Seed",
    #     min_value=0, max_value=9999, value=42,
    #     help="Seed for the random delivery-basis adder in cost generation",
    # )

    st.subheader("Data Centers")
    dc_edited = st.data_editor(
        DEFAULT_DC,
        width="stretch",
        hide_index=True,
        disabled=["data_center"],
        column_config={
            "data_center": st.column_config.TextColumn("Data Center"),
            "demand_mwh": st.column_config.NumberColumn("Demand (MWh)", min_value=0, step=1000),
            "carbon_intensity_cap": st.column_config.NumberColumn(
                "Carbon Cap (tCO₂e/MWh)", min_value=0.0, max_value=2.0, format="%.3f",
                help="Max allowed average emissions intensity for this DC. Click a cell to edit.",
            ),
        },
        key="dc_editor",
    )

    st.subheader("Providers")
    prov_edited = st.data_editor(
        DEFAULT_PROVIDERS,
        width="stretch",
        hide_index=True,
        disabled=["provider", "source"],
        column_config={
            "provider": st.column_config.TextColumn("Provider"),
            "source": st.column_config.TextColumn("Source"),
            "base_cost_per_mwh": st.column_config.NumberColumn("Base Cost ($/MWh)", min_value=0),
            "emissions_tco2e_per_mwh": st.column_config.NumberColumn(
                "Emissions (tCO₂e/MWh)", min_value=0.0, format="%.3f"
            ),
            "capacity_mwh": st.column_config.NumberColumn("Capacity (MWh)", min_value=0, step=1000),
        },
        key="prov_editor",
    )

    st.subheader("Location Adders ($/MWh)")
    loc_edited = st.data_editor(
        DEFAULT_LOCATION_ADDERS,
        width="stretch",
        hide_index=True,
        disabled=["data_center"],
        column_config={
            "data_center": st.column_config.TextColumn("Data Center"),
            "location_adder": st.column_config.NumberColumn("Adder ($/MWh)", format="%.0f"),
        },
        key="loc_editor",
    )

    st.divider()
    run_btn = st.button("Run Optimization", type="primary", width="stretch")


# ── SOLVER ────────────────────────────────────────────────────────────────────

def solve(dc_df, prov_df, loc_df, max_intake_factor):
    # rng = np.random.default_rng(int(seed))

    dc = dc_df.copy()
    dc["max_energy_mwh"] = (dc["demand_mwh"] * max_intake_factor).round(0).astype(int)
    dc = dc.set_index("data_center")

    prov = prov_df.copy().set_index("provider")
    loc_adders = loc_df.set_index("data_center")["location_adder"].to_dict()
    base_cost = prov["base_cost_per_mwh"].to_dict()

    DC = dc.index.tolist()
    P = prov.index.tolist()

    cost = {
        (i, j): round(base_cost[j] + loc_adders[i], 2)
        for i in DC
        for j in P
    }

    m = mathopt.Model(name="energy_dc_v1")
    x = {(i, j): m.add_variable(lb=0.0, name=f"x[{i},{j}]") for i in DC for j in P}

    for i in DC:
        m.add_linear_constraint(
            expr=mathopt.fast_sum(x[i, j] for j in P),
            lb=dc["demand_mwh"][i],
            ub=dc["max_energy_mwh"][i],
        )
    for i in DC:
        m.add_linear_constraint(
            expr=mathopt.fast_sum(prov["emissions_tco2e_per_mwh"][j] * x[i, j] for j in P)
                 - dc["carbon_intensity_cap"][i] * mathopt.fast_sum(x[i, j] for j in P),
            ub=0.0,
        )
    for j in P:
        m.add_linear_constraint(
            expr=mathopt.fast_sum(x[i, j] for i in DC),
            ub=prov["capacity_mwh"][j],
        )

    m.minimize(mathopt.fast_sum(cost[i, j] * x[i, j] for i in DC for j in P))
    result = mathopt.solve(m, mathopt.SolverType.GLOP)

    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        return {
            "feasible": False,
            "reason": result.termination.reason.name,
        }

    vals = result.variable_values()

    alloc = pd.DataFrame(
        [[vals.get(x[i, j], 0.0) for j in P] for i in DC],
        index=DC, columns=P,
    ).round(2)

    cost_df = pd.DataFrame(
        [[cost[(i, j)] * vals.get(x[i, j], 0.0) for j in P] for i in DC],
        index=DC, columns=P,
    ).round(2)

    actual_tco2e = [
        round(sum(prov["emissions_tco2e_per_mwh"][j] * vals.get(x[i, j], 0.0) for j in P), 4)
        for i in DC
    ]
    actual_mwh_per_dc = [round(alloc.loc[i].sum(), 4) for i in DC]
    emissions_df = pd.DataFrame({
        "data_center": DC,
        "actual_tco2e": actual_tco2e,
        "actual_mwh": actual_mwh_per_dc,
        "actual_intensity": [
            round(e / m, 4) if m > 0 else 0.0
            for e, m in zip(actual_tco2e, actual_mwh_per_dc)
        ],
        "cap_intensity": [dc["carbon_intensity_cap"][i] for i in DC],
    })

    return {
        "feasible": True,
        "total_cost": result.objective_value(),
        "alloc": alloc,
        "cost_df": cost_df,
        "emissions_df": emissions_df,
        "prov": prov,
        "dc": dc,
    }


# ── MAIN AREA ─────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner("Solving…"):
        result = solve(dc_edited, prov_edited, loc_edited, max_intake_factor)
    st.session_state["result"] = result

if "result" not in st.session_state:
    st.info("Set parameters in the sidebar, then click **Run Optimization**.")
    st.stop()

r = st.session_state["result"]

if not r["feasible"]:
    reason = r["reason"]
    st.error(f"**Model is infeasible** — solver returned `{reason}`")
    with st.expander("Troubleshooting hints"):
        total_renewable = sum(DEFAULT_PROVIDERS.loc[DEFAULT_PROVIDERS["source"].isin(["wind", "solar"]), "capacity_mwh"])
        total_demand = dc_edited["demand_mwh"].sum()
        st.markdown(f"""
**What to check:**

- **Carbon caps too tight?** Lower carbon intensity caps force the model to rely more on renewables.
  Total renewable capacity is **{total_renewable:,} MWh** against total demand of **{total_demand:,} MWh**
  ({100 * total_renewable / total_demand:.0f}% renewable coverage max). If caps require more than
  that share of clean energy, the model cannot feasibly meet demand.

- **Insufficient provider capacity?** Total supply across all providers must cover total demand.

- **Max Intake Factor too low?** Raising it above 1.00 gives the solver more flexibility.

_Relax one or more of the above constraints and re-run._
""")
    st.stop()

total_cost = r["total_cost"]
alloc = r["alloc"]
cost_df = r["cost_df"]
emissions_df = r["emissions_df"]
prov = r["prov"]
dc = r["dc"]
prov_source = prov["source"].to_dict()

# ── KPI METRICS ───────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
total_mwh = alloc.values.sum()
total_emissions = emissions_df["actual_tco2e"].sum()
avg_intensity = total_emissions / total_mwh if total_mwh > 0 else 0.0

k1.metric("Total Cost", f"${total_cost:,.0f}")
k2.metric("Total MWh Procured", f"{total_mwh:,.0f}")
k3.metric("Total Emissions (tCO₂e)", f"{total_emissions:,.0f}")
k4.metric("Avg Emissions Intensity", f"{avg_intensity:.3f} tCO₂e/MWh")

st.divider()

# ── ROW 1: Allocation heatmap + Cost by DC ────────────────────────────────────

c1, c2 = st.columns(2)

with c1:
    st.subheader("Allocation Heatmap (MWh)")
    fig_heat = px.imshow(
        alloc,
        labels={"x": "Provider", "y": "Data Center", "color": "MWh"},
        color_continuous_scale="Blues",
        text_auto=".0f",
        aspect="auto",
    )
    fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_heat, width="stretch")

with c2:
    st.subheader("Total Cost by Data Center ($)")
    cost_by_dc = cost_df.sum(axis=1).reset_index()
    cost_by_dc.columns = ["Data Center", "Total Cost ($)"]
    fig_cost = px.bar(
        cost_by_dc, x="Data Center", y="Total Cost ($)",
        color="Data Center",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=",.0f",
    )
    fig_cost.update_traces(textposition="outside")
    fig_cost.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cost, width="stretch")

# ── ROW 2: Energy mix + Carbon compliance ─────────────────────────────────────

c3, c4 = st.columns(2)

with c3:
    st.subheader("Energy Mix by Source per Data Center")
    source_map = prov["source"].to_dict()
    mix_rows = [
        {"Data Center": dc_name, "Source": source_map[p], "MWh": alloc.loc[dc_name, p]}
        for dc_name in alloc.index
        for p in alloc.columns
        if alloc.loc[dc_name, p] > 0.01
    ]
    mix_df = pd.DataFrame(mix_rows)
    if not mix_df.empty:
        mix_agg = mix_df.groupby(["Data Center", "Source"], as_index=False)["MWh"].sum()
        fig_mix = px.bar(
            mix_agg, x="Data Center", y="MWh", color="Source",
            color_discrete_map=SOURCE_COLORS,
            barmode="stack",
        )
        fig_mix.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_mix, width="stretch")

with c4:
    st.subheader("Carbon Compliance (tCO₂e/MWh intensity)")
    fig_carbon = go.Figure()
    fig_carbon.add_bar(
        name="Actual Intensity",
        x=emissions_df["data_center"],
        y=emissions_df["actual_intensity"],
        marker_color="#e76f51",
        text=emissions_df["actual_intensity"].round(3),
        textposition="outside",
    )
    fig_carbon.add_scatter(
        name="Cap",
        x=emissions_df["data_center"],
        y=emissions_df["cap_intensity"],
        mode="markers",
        marker=dict(symbol="line-ew", size=24, color="black", line=dict(width=3, color="black")),
    )
    fig_carbon.update_layout(
        yaxis_title="tCO₂e/MWh",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_carbon, width="stretch")

# ── ROW 3: Cost per MWh by provider + Supplier utilisation ───────────────────

c5, c6 = st.columns(2)

with c5:
    st.subheader("Avg Cost per MWh by Provider")
    total_mwh_by_prov = alloc.sum(axis=0)
    total_cost_by_prov = cost_df.sum(axis=0)
    avg_cost_prov = (total_cost_by_prov / total_mwh_by_prov.replace(0, np.nan)).dropna().reset_index()
    avg_cost_prov.columns = ["Provider", "Avg $/MWh"]
    avg_cost_prov = avg_cost_prov.sort_values("Avg $/MWh")
    avg_cost_prov["Source"] = avg_cost_prov["Provider"].map(prov_source)
    fig_prov_cost = px.bar(
        avg_cost_prov, x="Provider", y="Avg $/MWh",
        color="Source", color_discrete_map=SOURCE_COLORS,
        text_auto=".1f",
    )
    fig_prov_cost.update_traces(textposition="outside")
    fig_prov_cost.update_layout(showlegend=True, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_prov_cost, width="stretch")

with c6:
    st.subheader("Supplier Capacity Utilisation (%)")
    capacity = prov["capacity_mwh"].to_dict()
    util_rows = []
    for p in alloc.columns:
        used = alloc[p].sum()
        cap = capacity[p]
        util_rows.append({
            "Provider": p,
            "Used (MWh)": round(used, 0),
            "Capacity (MWh)": cap,
            "Utilisation (%)": round(100 * used / cap, 1) if cap > 0 else 0,
            "Source": prov_source[p],
        })
    util_df = pd.DataFrame(util_rows).sort_values("Utilisation (%)", ascending=False)
    fig_util = px.bar(
        util_df, x="Provider", y="Utilisation (%)",
        color="Source", color_discrete_map=SOURCE_COLORS,
        text="Utilisation (%)",
    )
    fig_util.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Capacity limit")
    fig_util.update_traces(textposition="outside")
    fig_util.update_layout(showlegend=True, margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[0, 115])
    st.plotly_chart(fig_util, width="stretch")

# ── DETAIL TABLES ─────────────────────────────────────────────────────────────

st.divider()

tab1, tab2, tab3 = st.tabs(["Allocation (MWh)", "Cost Breakdown ($)", "Carbon Detail"])

with tab1:
    st.dataframe(
        alloc.style.format("{:.1f}").background_gradient(cmap="Blues", axis=None),
        width="stretch",
    )

with tab2:
    st.dataframe(
        cost_df.style.format("${:,.0f}").background_gradient(cmap="Oranges", axis=None),
        width="stretch",
    )

with tab3:
    carbon_detail = emissions_df.copy()
    carbon_detail["headroom_intensity"] = (
        carbon_detail["cap_intensity"] - carbon_detail["actual_intensity"]
    ).round(4)
    carbon_detail["utilisation_%"] = (
        100 * carbon_detail["actual_intensity"] / carbon_detail["cap_intensity"]
    ).round(1)
    st.dataframe(
        carbon_detail[["data_center", "actual_tco2e", "actual_mwh", "actual_intensity", "cap_intensity", "headroom_intensity", "utilisation_%"]].rename(columns={
            "data_center": "Data Center",
            "actual_tco2e": "Total Emissions (tCO₂e)",
            "actual_mwh": "Total MWh",
            "actual_intensity": "Actual Intensity (tCO₂e/MWh)",
            "cap_intensity": "Cap (tCO₂e/MWh)",
            "headroom_intensity": "Headroom (tCO₂e/MWh)",
            "utilisation_%": "Utilisation (%)",
        }).style.format({
            "Total Emissions (tCO₂e)": "{:,.1f}",
            "Total MWh": "{:,.0f}",
            "Actual Intensity (tCO₂e/MWh)": "{:.4f}",
            "Cap (tCO₂e/MWh)": "{:.4f}",
            "Headroom (tCO₂e/MWh)": "{:.4f}",
            "Utilisation (%)": "{:.1f}%",
        }).background_gradient(subset=["Utilisation (%)"], cmap="RdYlGn_r"),
        width="stretch",
        hide_index=True,
    )
