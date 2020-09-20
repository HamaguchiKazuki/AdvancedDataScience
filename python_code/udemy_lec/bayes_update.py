import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np
import pandas as pd

plt.style.use("ggplot")
plt.close('all')

# param
p_a = 3.0 / 10
p_b = 5.0 / 9
p_prior = 0.5  # 袋を選ぶ事前分布
# 0:bule, 1:red
data = [0, 1, 0, 0, 1, 1, 1]


N_data = 7
likehood_a = bernoulli.pmf(data[:N_data], p_a)  # 尤度関数
likehood_b = bernoulli.pmf(data[:N_data], p_b)  # 尤度関数


# 事後分布
pa_posterior = p_prior
pb_posterior = p_prior

pa_posterior *= np.prod(likehood_a)
pb_posterior *= np.prod(likehood_b)
# 規格化
norm = pa_posterior + pb_posterior
df = pd.DataFrame([pa_posterior / norm, pb_posterior / norm], columns=["post"])
x = np.arange(df.shape[0])
plt.bar(x, df["post"])
plt.xticks(x, ["bag a", "bag b"])
plt.show()
