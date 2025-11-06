#import librairies
import pandas as pd
pd.__version__

#Read and Display CSV data
df = pd.read_csv('Marco-Ferreira-DataSet.csv', sep= ";")
print(df.head())

#Mettre un commentaire
print(df.shape)
print(df.dtypes)
print(df.isna().sum())

#Data Cleaning
    #Supprimer l'index inutile
df = df.drop(columns=["Unnamed: 0"])
print(df.columns)

    # Suppression manuelle des lignes corrompues (3651 → 3727)
df = df.drop(index=range(3651, 3728))  # 3728 car le dernier indice est exclusif
df = df.reset_index(drop=True)
print("Nouvelles dimensions du DataFrame :", df.shape)

    #Vérification des dates manquantes dans la série temporelle
        #Convertir la colonne en datetime
print("Colonne avant strip:", list(df.columns))
df.columns = df.columns.str.strip()
print("Colonnes après  strip:", list(df.columns))
print("Type 'Date' avant conversion:", df["Date"].dtype)
date_raw = df["Date"].astype(str).str.strip()
date_parsed = pd.to_datetime(date_raw, dayfirst=True, errors="coerce")
print(date_parsed.dtype)
print(date_parsed.isna().mean())
df["Date"] = date_parsed
print("Type 'Date' après conversion:", df["Date"].dtype)
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

