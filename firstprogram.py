import pandas as pd
data={
    "Name": ["John", "Sarah", "Mike"],
    "Experience":[3,6,9],
    "salary":[80000,120000,150000]
}
df = pd.DataFrame(data)
print(df)
print(df["salary"].mean())
high_salary = df[df["salary"] > 100000]
print(high_salary)
filtered_data=df[
    (df["Experience"]>5)& (df["salary"]>100000)
    ]
print(filtered_data)
df["bonus"]=df["salary"]*0.10
print(df)
selected_data= df[["Name","salary"]]
print(selected_data)
df["totalcompen"]=df["salary"]+df["bonus"]
print(df)

##def salary_level(salary):
#    if salary>=120000:
#        return "High"
##    else:
#        return "Normal"
#print(salary_level(120000))



#df["salary_level"]=df["salary"].apply(salary_level) 
#print(df)
#def add_bonus(salary):
#    return salary*0.10
#df["add_bonus"]=df["salary"].apply(add_bonus)

df["add_bonus"]= df["salary"].apply(lambda salary:salary*0.10)
print(df)

df["Tax"]=df["salary"].apply(lambda salary:salary*0.20)
print(df)

df["salary_level"]= df["salary"].apply(lambda salary: "high" if salary>=120000 else "normal")
print(df)

data2 = {
    "name" : ["John", "sarah", "Mike", "David"],
    "salary" : [80000,120000,None,100000]
}
df2 =pd.DataFrame(data2)
print(df2)
print(df2.isnull())
print(df2.isnull().sum())
#df2["salary"]=df2["salary"].fillna(0)
#print(df2)
avg_sal=df2["salary"].mean()
print(avg_sal)
#df2["salary"]=df2["salary"].fillna(avg_sal)
#
# print(df2)
df3=df2.dropna()
print(df3)
median_salary=df2["salary"].median()
df2["salary"]=df2["salary"].fillna(median_salary)
print(df2)
sortedvalues=df.sort_values("salary")
print(sortedvalues)

data3 = {
    "Name": ["John", "Sarah", "Mike", "David", "Emma"],
    "Department": ["IT", "HR", "IT", "HR", "IT"],
    "salary": [80000, 70000, 120000, 90000, 100000]
}

df3 = pd.DataFrame(data3)

print(df3)
avgsal=df3.groupby("Department")["salary"].mean()
print(avgsal)
employee_count=df3.groupby("Department")["Name"].count()
print(employee_count)

salary_stats=df3.groupby("Department")["salary"].agg(["mean","min","max"])
print(salary_stats)

employees={
    "ID":[1,2,3],
    "Name":["John","Sarah","Mike"]
}
Salaries={
    "ID":[1,2,3],
    "Salary":[80000,120000,150000]

}
df_emp=pd.DataFrame(employees)
df_sal=pd.DataFrame(Salaries)
print(df_emp)
print(df_sal)
combined=pd.merge(df_emp,df_sal,on="ID")
print(combined)
employees1={
    "ID":[1,2,3],
    "Name":["John","Sarah","Mike"]
}
Salaries1={
    "ID":[1,2],
    "Salary":[80000,120000]

}
df_emp1=pd.DataFrame(employees1)
df_sal1=pd.DataFrame(Salaries1)
combined1=pd.merge(df_emp1,df_sal1,on="ID",how="left")
print(combined1)
