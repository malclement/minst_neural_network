from matplotlib import pyplot as plt
import numpy


# Find the most frequent value in a list and returns it
def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
    return num


# Visualise training
def visualise_training(epoch_list, accuracy_list):
    plt.plot(epoch_list, accuracy_list)
    plt.xlabel('epoch')  # Add an x-label to the axes.
    plt.ylabel('accuracy in %')  # Add a y-label to the axes.
    plt.grid()
    plt.show()


# Create a vector size 9x1 corresponding to perfect output vector
def create_solution(solution_value):
    indent = solution_value
    output = numpy.array([0 for i in range(10)])
    output[indent] = 1
    return output


one = create_solution(1)
two = create_solution(2)
three = create_solution(3)
four = create_solution(4)
five = create_solution(5)
six = create_solution(6)
seven = create_solution(7)
eight = create_solution(8)
nine = create_solution(9)


def numbers_to_letters(i):
    if i == 1: return "one"
    elif i == 2: return "two"
    elif i == 3: return "three"
    elif i == 4: return "four"
    elif i == 5: return "five"
    elif i == 6: return "six"
    elif i == 7: return "seven"
    elif i == 8: return "eight"
    elif i == 9: return "nine"
    else:
        print("error wrong number")
        return -1