# an example for raw_input and int conversion

print('Program for Reverse a String')

String1 = str(input('Please enter Sample Input: '));

String2 = str(input('Please enter First Character to Remove: '));

String3 = str(input('Please enter Second Character to Remove: '));

String4 = String1.replace(String2,"");

String5 = String4.replace(String3,"");

String6 = String5[::-1]

print ('Here is some output ' + String6)

print ('Thanks you. END')