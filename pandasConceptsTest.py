import pandas as pd

df = pd.DataFrame(
    [["cool","pandas"],
     [200,111]],
    # columns=["A","B"],
    # index=[0,1]
    # name="Prueba dataFrame"
);

df1 = pd.DataFrame(
    [["Pandas",2000],
     ["Pandas",2000]],
     columns=["A","B"],
     index=["Persona1","Persona2"]
);

data = pd.DataFrame({'A': [1, 2, 3],
                     'B': [1.1, 2.2, 3.3],
                     'C': ['foo', 'bar', 'baz']})

obj_cols = data.select_dtypes(include=['object'])


series = pd.Series([1,2,3,4]);
# print(series.to_string(index=False));

# print(df.to_string(index=False));

categoricas = df1.select_dtypes(include=["object"]);
numericas = df1.select_dtypes(exclude=["object"]);
# print(df1);
print("Categoricas:\n",list(categoricas));
print("Numericas\n",list(numericas));
# print(numericas);

df2 = pd.DataFrame(
    [
        ["Rojo",1],
        ["Amarillo",2],
        ["Azul",3]
    ],
    columns=["Color","Numero"]
);
print(df2);
df2_cat = df2.select_dtypes(include=["object"]);
df2_num = df2.select_dtypes(exclude=["object"]);
print("Dataframe categorico \n",df2_cat);
print(pd.get_dummies(df2_cat).columns);