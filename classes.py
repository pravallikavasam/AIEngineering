class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def display(self):
        print(self.name, self.salary)


#emp1 = Employee("John", 80000)
#emp2 = Employee("Sarah", 120000)

#emp1.display()
#emp2.display()

class Manager(Employee):
    def __init__(self,name,salary,department):
        super().__init__(name,salary)
        self.department=department
    def show_department(self):
        print(self.department)

manager1=Manager("John",100000,"Engineering")

manager1.display()
manager1.show_department()


