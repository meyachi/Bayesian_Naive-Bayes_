"""
Naive Bayes Simulation

Author: Charlotte Mae M. De La Cruz; un:@meyachi

Date: April 17, 2026

This program demonstrates the Naive Bayes algorithm using a simple fitness classification dataset.
It shows how the model learns from training data and predicts outcomes using probability.
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np

print("Naive Bayes Simulation")
print("\nThis program classifies whether an individual is FIT or NOT FIT")
print("based on Physical and Mental conditions using Naive Bayes.\n")

print("Feature Meaning:")
print("Physical: 1 = Fit, 0 = Not Fit")
print("Mental: 1 = Stable, 0 = Unstable\n")

X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

y = np.array([1, 1, 0, 1, 0])

model = GaussianNB()
model.fit(X, y)

test_data = [[1, 1]]

prediction = model.predict(test_data)
probabilities = model.predict_proba(test_data)

print("Input:", test_data)

print("\nClass Probabilities:")
print("P(Not Fit):", round(probabilities[0][0], 4))
print("P(Fit):", round(probabilities[0][1], 4))

print("\nFinal Prediction:", prediction[0])

if prediction[0] == 1:
    print("\nInterpretation:")
    print("The model predicts that the individual is FIT for flying duties.")
else:
    print("\nInterpretation:")
    print("The model predicts that the individual is NOT FIT for flying duties.")


# References
# Scikit-learn Developers (2024). Naive Bayes Documentation
# https://scikit-learn.org/

# Naïve Bayes Classifier Lecture Notes (2026)
# Solver (n.d.)