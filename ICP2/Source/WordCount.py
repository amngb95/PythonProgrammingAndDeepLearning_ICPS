"""mylistInput = []
myListOutput = []
n = int(input("Enter size of the list:\n"))

for i in range(0, n):
    String = str(input("Enter the String to append:\n"))
    mylistInput.append(String)

for String in mylistInput :
    print(String)
    def word_count(String):
        counts = dict()
        words = String.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

        return counts
    print(word_count(String))"""

from collections import Counter
def word_count(fname):
        with open(fname) as f:
                return Counter(f.read().split())

print("Number of words in the file :",word_count("test.txt"))
