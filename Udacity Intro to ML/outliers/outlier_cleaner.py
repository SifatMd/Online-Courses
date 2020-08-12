#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    differences = []
    for p, a, n in zip(predictions, ages, net_worths):
        differences.append((p, a, n, (p-n)**2))
    
    print(differences[0])
    differences.sort(key=lambda x: x[3])
    print(differences[0])

    size = int(len(differences)*0.9)
    for i in range(0, size, 1):
        cleaned_data.append((differences[i][1], differences[i][2], differences[i][3]))

    
    return cleaned_data

