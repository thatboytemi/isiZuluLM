# Class that will be used to ensure that training data is of the same format as data used in research by Moeng et al. (2021)
# Ensures that data is compatible with their segmentation model

def cleaner(filename):
    file = open(filename, "r")
    res = []
    for i in range(3): # ignore first 3 comment lines of file
        file.readline()
    lines = file.readlines()
    word =""
    segmentation = ""
    i =0  
    while i<len(lines):
        line = lines[i].split("\t")
        if len(line)<10:
            i+=1
            continue
        if line[3]=="PUNCT":
            i+=1
            continue
        if "-" in line[0]:
            temp = line[0].split("-")
            temp = int(temp[1])-int(temp[0]) + 1
            word = line[1]
            for j in range(1, temp+1):
                line = lines[i+j].split("\t")
                parts = line[1].split("-")
                for part in parts:
                    if part!="":
                        segmentation+="-"+part
            res.append(word+" | "+ segmentation[1:]+"\n")
            word=""
            segmentation=""
            i += temp+1
            continue
        res.append(line[1]+" | "+ line[1]+ "\n")
        i+=1
    return res



def main():
    text = cleaner("./data/zu.test.conllu")
    file = open("./data/zu.clean.test.conllu", "w")
    res = []
    for word in text:
        parts = word.replace("\n", "").split(" | ")
        subwords = parts[0].replace("-", " ").replace("."," ").split(" ")
        if len(subwords) > 1:
            # print(subwords)
            segments = parts[1].replace("-", " ").replace("."," ").split(" ")
            # print(segments)
            for i in range(len(subwords)):
                currentseg = ""
                currentword = ""
                saved =""
                for j in range(len(segments)):
                    if (currentword+segments[j]) in subwords[i]:
                        currentword+=segments[j]
                        currentseg+=segments[j]+"-"
                        saved = currentseg
                    elif segments[j]in subwords[i]:
                        currentword=segments[j]
                        currentseg=segments[j]+"-"
                        saved = currentseg
                    else:
                        currentseg = ""
                        currentword = ""
                if(subwords[i]=="Provincial"):
                    print("ran")
                    print(saved)
                temp = (subwords[i] +" | "+saved).rstrip("-")
                res.append(temp+"\n")
        else:
            res.append(word)
    file.writelines(res)
    print("Done")
main()