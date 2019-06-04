# an example for raw_input and int conversion

String1 = str(input('Please enter Sample Input: '));

String2 = str(input('Please Enter String need to be checked: '));

result = String1.find(String2);

if (String1.find(String2)!= -1) :
    print(String1.replace(String2,String2 + "s"))
else:
    print("Doesn't contains given substring ",String2)