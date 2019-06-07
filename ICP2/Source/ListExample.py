mylistInput = []
myListOutput = []
#Give the Size of the list
n = int(input("Enter size of the list:\n"))

for i in range(0, n):
    num = int(input("Enter number to append:\n"))
    #Append the Numbers to myListInpu2t
    mylistInput.append(num)

#Iterate the Numbers in myListInput
for num in mylistInput:
    result = num*0.453592
    # Append the result to myListOutput
    myListOutput.append(result)

print("Required Output is :",myListOutput)