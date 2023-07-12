import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "/Users/shuotaodiao/Documents/Research/PostDoc/experiments/calbrienstick_sample_paths_analysis_v2"
t = np.array([0.1 * _ for _ in range(300)])

p1_param1 = 87.2317
p1_param2 = 5
p1_vsp = 10 ** (p1_param1 * t / (p1_param2 ** 2 + np.square(t)))

p2_param1 = 53.2020
p2_param2 = 3
p2_vsp = 10 ** (p2_param1 * t / (p2_param2 ** 2 + np.square(t)))

p3_param1 = 56.5276
p3_param2 = 3.1368
p3_vsp = 10 ** (p3_param1 * t / (p3_param2 ** 2 + np.square(t)))

p4_param1 = 55.7357
p4_param2 = 3.1160
p4_vsp = 10 ** (p4_param1 * t / (p4_param2 ** 2 + np.square(t)))


plt.plot(t, p1_vsp, marker='', color="tab:green", linewidth=1, alpha=1.0, label="Before Feb. 20, 2021")
plt.plot(t, p2_vsp, marker='', color="tab:orange", linewidth=1, alpha=1.0, label="Feb. 20, 2021 - May 19, 2021")
plt.plot(t, p3_vsp, marker='', color="tab:cyan", linewidth=1, alpha=1.0, label="May 20, 2021 - Aug. 19, 2021")
plt.plot(t, p4_vsp, marker='', color="tab:purple", linewidth=1, alpha=1.0, label="Aug. 20, 2021 - Nov. 20, 2021")

true_viral_param1 = 71.97
true_viral_param2 = 4
v_true = 10 ** (true_viral_param1 * t / (true_viral_param2 ** 2 + np.square(t)))

plt.plot(t, v_true, marker='', color="k", linewidth=2, alpha=1.0, label="Viral Shedding Profile by Phan, et al. 2023\n(Based on Stool Samples Collected in Jan. 2020)")
plt.xlabel("Days after Exposed")
plt.ylabel("Viral Shedding (Copies per Gram)")
plt.title("Summary of Fitted Viral Shedding Parameters")
plt.legend()
#plt.show()
plt.savefig(os.path.join(output_dir, "vsp.png"), dpi=300, bbox_inches='tight')