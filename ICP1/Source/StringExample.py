# an example for raw_input and int conversion

String1 = str(input('Please enter Input String: '));

String2 = str(input('Please Enter the word to be check: '));

if (String1.find(String2)!= -1) :
    print("Required Output is ",String1.replace(String2,String2 + "s"))
else:
    print("Doesn't contains given substring ",String2)