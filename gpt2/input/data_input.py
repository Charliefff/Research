
def data_size(text, training_size):
    training = str()
    lines = text.split('\n')
    for i, j in enumerate(lines):
        if i < training_size:
            # print(j)
            training += j
        else:
            return training

    return training
