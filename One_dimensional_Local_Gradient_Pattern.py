import numpy as np

def ODLGP(numpy_array, neighbouring_points,stride):
    feature_vector = []
    if(neighbouring_points%2==1):
        print("Pick an even number for neighbouring_points")
    else:
        if (stride >= neighbouring_points + 1):
            print(
                "Number for stride too large! You'll lose data samples. Pick a number smaller then neighbouring points ")
        for i in range(0, len(numpy_array)-neighbouring_points,stride+1):
            sequence = numpy_array[i:i+neighbouring_points+1]
           # print(sequence)

            gradient_value = np.zeros(neighbouring_points+1)
            diff_sequence = np.zeros(neighbouring_points+1)
            for k in range(neighbouring_points+1):
                gradient_value[k] = np.abs(sequence[k] - sequence[neighbouring_points//2])
            sum = np.sum(gradient_value)
            mean = sum/neighbouring_points
           # print(gradient_value)
           # print(mean)
            for k in range(neighbouring_points+1):

                diff_sequence[k] = 0 if gradient_value[k]-mean<0 else 1

            diff_sequence = np.delete(diff_sequence, neighbouring_points//2)
           # print(diff_sequence)
            code = diff_sequence.dot(2 ** np.arange(diff_sequence.size)[::-1])
            feature_vector.append(code)


    feature_vector =np.asanyarray(feature_vector)
    feature_vector= feature_vector.astype(int)
    #print(feature_vector)

    histogram=np.zeros(np.power(2,neighbouring_points))
    for element in feature_vector:
          histogram[element]= histogram[element]+1
    return histogram



#print(ODLGP(np.array([50,35,32,18,10,3,-1,-5,-6]),8,1))

