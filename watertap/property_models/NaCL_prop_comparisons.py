import numpy as np
import pandas as pd

pitzer_enthalpy_file = "G:\My Drive\05_Active_Research_Projects\MVC\Properties\Pitzer Specific Enthalpy.csv"
df_h = pd.read_csv(pitzer_enthalpy_file)
print(df_h)