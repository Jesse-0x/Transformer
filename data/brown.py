# load the Brown corpus from the brown1.txt file


data = []
passage = ""
with open('brown1.txt', 'r') as f:
    # Separate the data into paragraphs
    # A01 0010    The Fulton County Grand Jury said Friday an investigation
    for line in f:
        c = line.split(' ')
        if passage != c[0]:
            passage = c[0]
            data.append(line[9:][:-1])
        else:
            data[-1] += line[9:][:-1]
