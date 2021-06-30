import numpy as np
def LNDP(numpy_array, neighbouring_points,stride):
    feature_vector = []
    if(neighbouring_points%2==1):
        print("Pick an even number for neighbouring_points")
    else:
        if(stride>=neighbouring_points+1):
            print("Number for stride too large! You'll lose data samples. Pick a number smaller then neighbouring points")
        for i in range(0, len(numpy_array)-neighbouring_points,stride+1):
            sequence = numpy_array[i:i+neighbouring_points+1]
            #print(sequence)

            diff_sequence = np.zeros(neighbouring_points)
            for k in range(neighbouring_points):
                diff_sequence[k]= 0 if sequence[k+1]-sequence[k]<0 else 1
            #print(diff_sequence)
            code = diff_sequence.dot(2 ** np.arange(diff_sequence.size)[::-1])
            feature_vector.append(code)


        feature_vector =np.asanyarray(feature_vector)
        feature_vector= feature_vector.astype(int)
        #print(feature_vector)
        histogram=np.zeros(np.power(2,neighbouring_points))
        for element in feature_vector:
             histogram[element]= histogram[element]+1
        return histogram

#leave the prints uncommented to verify if the algorithm works
#print(LNDP(np.array([50,35,32,18,10,3,4,8,12]),2,3))