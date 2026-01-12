import pandas as pa
import numpy as nu

def load_data():
    
    df = pa.read_excel("Lab Session Data.xlsx", sheet_name=0)

   
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values

    return X, y

def find_cost(X, y):
    
    X_inv = nu.linalg.pinv(X)

  
    cost = X_inv @ y
    return cost

def main():
    X, y = load_data()

    
    rank = nu.linalg.matrix_rank(X)

   
    cost = find_cost(X, y)

    print("feature matrix X:\n", X)
    print("\noutput vector y:\n", y)
    print("\nrank of the feature matrix:", rank)

    print("\nestimate cost of the products:")
    print("Candies (Rs per unit):", cost[0])
    print("Mangoes (Rs per kg):", cost[1])
    print("Milk Packets (Rs per packet):", cost[2])

if __name__ == "__main__":
    main()