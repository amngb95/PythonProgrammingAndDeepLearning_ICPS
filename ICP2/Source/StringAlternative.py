String  = str(input("Enter the String:\n"))

def string_alternative(String):
    String1 = String[::2]
    print(String1)

def main():
 string_alternative(String)

main()