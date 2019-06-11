
class Employee:
    count = 0
    totalSal = 0
    def __init__(self, name, family,salary,department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.count += 1
        Employee.totalSal += salary

    def averagesalary(self):
        print(self.totalSal,self.count)
        return self.totalSal/self.count

class FullTimeEmployee(Employee):
    def __init__(self,name,family,salary,department,employeeId):
        Employee.__init__(self, name, family,salary,department)
        self.employeeId = employeeId

    def getEmployeeId(self):
        id = self.employeeId
        name = self.name
        print("Name and EmployeeId : ",id,name)

userInput = 'yes'
while (userInput != 'no'):
    e1 = Employee("Anvesh", "Mandadi", 5000, "CSE")
    e2 = Employee("John", "Peter", 5000, "CSE")
    print(e1.salary)
    print(e2.salary)

    y = FullTimeEmployee("Anvesh", "Mandadi", 5000, "CSE", "16272447")
    y.getEmployeeId()
    print(e1.averagesalary())
    print(e2.averagesalary())
    userInput = input("Enter the Input:")











