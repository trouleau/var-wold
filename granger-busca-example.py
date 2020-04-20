import numpy as np
from sklearn import preprocessing
from gb import GrangerBusca, simulate

np.random.seed(1234)

# Define parameters
mu_rates = np.array([0.1, 0.1, 0.1])
Alpha_ba = np.array([[0.0, 1.0, 0.0],
                     [0.7, 0.3, 0.0],
                     [0.7, 0.1, 0.2]])
Beta_ba = np.ones((3, 3))

# Simulate timestamps
sim = simulate.GrangeBuscaSimulator(mu_rates=mu_rates, Alpha_ba=Alpha_ba, Beta_ba=Beta_ba)
timestamps = sim.simulate(forward=1e5)
print(f'Simulated {sum(map(len, timestamps)):d} timestamps')
print()

# Fit the model
granger_model = GrangerBusca(
    alpha_prior=1.0/len(timestamps),
    num_iter=1000, metropolis=True, beta_strategy=1)
granger_model.fit(timestamps)

alpha_ = preprocessing.normalize(granger_model.Alpha_.toarray(),"l1")
mu_ = granger_model.mu_
beta_ = granger_model.beta_

# Print results
print()
print('Ground truth baseline rates:')
print(mu_rates)
print('Estimate baseline rates:')
print(mu_.round(2))
print()
print('Ground truth alphas:')
print(Alpha_ba)
print('Estimate alphas:')
print(alpha_.round(2))
print()
print('Ground truth betas:')
print(Beta_ba)
print('Estimate betas:')
print(beta_.round(2))
