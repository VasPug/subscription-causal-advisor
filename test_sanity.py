"""Self-check: oracle coherence, estimator behavior, business-template calibration."""
import numpy as np

import sim
from methods import att_tau


def main():
    rng = np.random.default_rng(0)
    # 1) advisor's adjusted estimate beats the naive contrast when only
    #    engagement confounds (conf=0: h hits outcomes but not adoption)
    pop = sim.draw_population(20000, dict(tau_x=8.0, conf=0.0), rng)
    orc = sim.oracle(pop, 1)
    data = sim.observational_dataset(pop, 2)
    t = data["D"] > 0.5
    tau_naive = data["Y"][t].mean() - data["Y"][~t].mean()
    tau_adj = att_tau(data["X"], data["D"], data["Y"])
    print(f"conf=0.0: tau_true={orc['tau_true']:.2f} naive={tau_naive:.2f} "
          f"adjusted={tau_adj:.2f}")
    assert abs(tau_adj - orc["tau_true"]) < abs(tau_naive - orc["tau_true"])

    # 2) hidden confounding (conf>0) inflates the naive contrast further
    pop_c = sim.draw_population(20000, dict(tau_x=8.0, conf=1.8), rng)
    orc_c = sim.oracle(pop_c, 1)
    data_c = sim.observational_dataset(pop_c, 2)
    tc = data_c["D"] > 0.5
    tau_naive_c = data_c["Y"][tc].mean() - data_c["Y"][~tc].mean()
    tau_adj_c = att_tau(data_c["X"], data_c["D"], data_c["Y"])
    print(f"conf=1.8: tau_true={orc_c['tau_true']:.2f} naive={tau_naive_c:.2f} "
          f"adjusted={tau_adj_c:.2f}")
    assert tau_naive_c - orc_c["tau_true"] > tau_naive - orc["tau_true"]

    # 3) business template is oracle-sign-consistent at the true tau, and the
    #    two regimes sit on opposite sides of the decision
    gates = {}
    for tau_x in (1.0, 8.0):
        pop2 = sim.draw_population(20000, dict(tau_x=tau_x, conf=0.0), rng)
        orc2 = sim.oracle(pop2, 3)
        data2 = sim.observational_dataset(pop2, 4)
        gate, dest = sim.decide_from_tau(orc2["tau_true"], data2)
        gates[tau_x] = orc2["delta_true"] > 0
        print(f"tau_x={tau_x}: delta_true={orc2['delta_true']:+.3f} tau_true="
              f"{orc2['tau_true']:+.2f} template@truth: gate={gate} dest={dest:+.3f}")
        assert gate == (orc2["delta_true"] > 0), (
            f"template disagrees with oracle at tau_x={tau_x}")
    assert gates[1.0] != gates[8.0], "regimes should disagree on the best action"
    print("sanity OK")


if __name__ == "__main__":
    main()
