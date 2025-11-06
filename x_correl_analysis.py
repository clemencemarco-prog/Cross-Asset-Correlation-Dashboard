#import librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def xcorrel():
   #import data
   df = pd.read_csv('Marco-Ferreira-DataSet.csv', sep= ";")
   #observe data
   print(df.shape)
   #variables names
   print(df.columns)
   #overview
   print(df.head())
   print(df.tail()) 
   print(df.info()) 
   for col in df.columns:
    print(f"\n--- {col} ---")
    print(df[col].map(type).value_counts())

   #univariate analysis
   # Sélection colonnes 2 à 9 et lignes à partir de la 2e
   df_sub = df.iloc[1:, 2:10]  
   print(df.describe())
   #print(df.hist(figsize=(10, 8), bins=30))
   #plt.show()

   #Bi-variate analysis
      #Correlation Matrix
   #print(df.corr()) #La commande df.corr() ne fonctionne pas ici car certaines colonnes ne sont pas numériques. Il faut d'abord "nettoyer" les données.

   #Clean Data
      #Supprimer les colonnes inutiles
   df = df.drop(columns=["Unnamed: 0"])
   print(df.columns)
      # Suppression manuelle des lignes corrompues (3651 → 3727)
   df = df.drop(index=range(3651, 3728))  # 3728 car le dernier indice est exclusif
   df = df.reset_index(drop=True)
   print("Nouvelles dimensions du DataFrame :", df.shape)
      #Convertir la colonne en datetime
   df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
   print(df.head())
      #Traiter les valeurs manquantes de la colonne jour
   weekday_map_fr = {
     0: "lundi", 1: "mardi", 2: "mercredi", 3: "jeudi",
     4: "vendredi", 5: "samedi", 6: "dimanche"
   }
   df["Jour"] = df["Date"].dt.weekday.map(weekday_map_fr)
   df["Jour"] = df["Jour"].astype(str).str.strip().str.lower()
   print(df["Jour"].value_counts(dropna=False))

      #Supprimer les weekends
   df = df[~df["Jour"].isin(["samedi", "dimanche"])].reset_index(drop=True)
   print("Nouvelles dimensions :", df.shape)
   print(df["Jour"].value_counts())

      #Supprimer les données parasites
   colonnes_numeriques = df.loc[:, "DGS10":"DCOILWTICO"].columns
   df[colonnes_numeriques] = df[colonnes_numeriques].apply(
      lambda col: pd.to_numeric(col, errors="coerce")
   )
   print(df[colonnes_numeriques].isna().sum())

   print(df.head(2000))

   # Cherche le mot 'test' dans tout le DataFrame
   mask_test = df.apply(lambda col: col.astype(str).str.contains("test", case=False, na=False))

   # Affiche les lignes où au moins une cellule contient 'test'
   lignes_test = df[mask_test.any(axis=1)]

   print(f"Nombre de lignes contenant 'test' : {len(lignes_test)}")
   print(lignes_test)
   
   for col in df.columns:
     print(f"\n--- {col} ---")
     print(df[col].map(type).value_counts())

     df = df.ffill().bfill()

   print("Nombre de valeurs manquantes AVANT :", df.isna().sum().sum())
   df = df.ffill().bfill()
   print("Nombre de valeurs manquantes APRÈS :", df.isna().sum().sum())
   print(df.head())
   print(df.tail())
   print("Nombre de valeurs manquantes AVANT :", df.isna().sum().sum())
   df = df.ffill().bfill()
   print("Nombre de valeurs manquantes APRÈS :", df.isna().sum().sum())

   #Calculer les spreads
   df["chg_us10y_bps"] = df["DGS10"].diff() * 100   # 1 point = 100 bps
   df["chg_hy_bps"] = df["BAMLH0A0HYM2"].diff() * 100

   print(df[["chg_us10y_bps", "chg_hy_bps"]].head())

   #Calculer les rendements
   df["ret_sp500"] = df["SP500"].pct_change()
   df["ret_oil"] = df["DCOILWTICO"].pct_change()
   df["ret_fx"] = df["DEXUSEU"].pct_change()
   df["ret_vix"] = df["VIXCLS"].pct_change()
   df["dVIX"] = df["VIXCLS"].diff() 
   #We compute VIX variation using diff() instead of returns, 
   #because VIX is an implied volatility index and not a traded asset. 
   #Absolute changes (in volatility points) are more meaningful for market stress analysis.

   print(df[["ret_sp500", "ret_oil", "ret_fx", "ret_vix", "dVIX"]].head())

   df = df.dropna()
   print(df[["chg_us10y_bps", "chg_hy_bps"]].head())
   print(df[["ret_sp500", "ret_oil", "ret_fx", "ret_vix"]].head())

   print(df.head(40))
   
   #Matrice de corrélation
   corr_matrix = df[[
    "ret_sp500",
    "ret_oil",
    "ret_fx",
    "dVIX",
    "chg_us10y_bps",
    "chg_hy_bps"
]].corr()
   
   print(corr_matrix)
   
   plt.figure(figsize=(8, 6))
   sns.heatmap(
      corr_matrix,           # ta matrice
      annot=True,            # affiche les valeurs dans chaque case
      cmap="coolwarm",       # dégradé bleu -> rouge
      center=0,              # 0 au milieu du dégradé
      square=True,           # carrés proportionnels
      vmin=-1, vmax=1        # bornes fixes de -1 à 1
   )
   plt.title("Matrice de corrélation - Heatmap")
   plt.tight_layout()
   plt.show()

   #Rolling corrélation
      #Rolling SP500 vs VIX (en utilisant dVIX)
   df["roll_corr_sp500_vix"] = df["ret_sp500"].rolling(30).corr(df["dVIX"])
      #Rolling SP500 vs taux US (10Y)
   df["roll_corr_sp500_us10y"] = df["ret_sp500"].rolling(30).corr(df["chg_us10y_bps"])
      #Rolling SP500 vs pétrole (WTI)
   df["roll_corr_sp500_oil"] = df["ret_sp500"].rolling(30).corr(df["ret_oil"])

   print(df[[
    "roll_corr_sp500_vix",
    "roll_corr_sp500_us10y",
    "roll_corr_sp500_oil"
   ]].tail(35))

   df = df.dropna()
   
   return df
df = xcorrel()
df.to_csv("df.csv")